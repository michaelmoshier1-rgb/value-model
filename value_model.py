#!/usr/bin/env python3
"""
Sharp Moneyline Model (free data, no API key)  -  NBA · College Basketball · NFL · College Football
==================================================================================================
Predicts each team's win probability from its own ratings, then bets Kalshi moneylines where the
predicted probability beats Kalshi's price (fee included) by MIN_EDGE or more.

Factors in the prediction
  1. Efficiency ratings, opponent-adjusted (ridge regression):
       basketball - points per possession margin (pace-neutral), from ESPN box scores
       football   - blend of points margin and yards-per-play margin, from ESPN box scores
     Blowout margins capped; recent games weighted more (half-life per sport).
  2. Prior: last season's ratings, regressed toward average, so early-season ratings aren't noise.
  3. Home advantage (fitted from this season) and neutral sites.
  4. Rest and schedule: back-to-backs and 3-in-4 nights (basketball), byes and short weeks (football).
  5. Travel: extra miles travelled vs. the opponent (venues geocoded via Open-Meteo).
  6. Injuries: ESPN injury report x player value (basketball: per-minute Game Score above replacement x
     minutes; football: starting QB value from adjusted net yards per attempt), scaled by how often the
     player was actually in the games the ratings were built from.
  7. Weather (outdoor football): high wind / heavy rain pulls the projection toward 50-50.
  8. Pace (basketball): projected possessions scale the projected margin.
  9. Market prior: final win % = w x model + (1-w) x DraftKings no-vig, with w fitted on this season.
 10. Line-movement filter: skip a side when DraftKings has moved hard against it since the open (news).
 11. Calibration: the margin->win% spread (SD) and w are fitted on a point-in-time backtest of this season.

Plays lock once at fixed times (ET): NFL Tue 9am · CFB Sun 9am · NBA/CBB daily 8am. Flat 1u per bet.
W/L, profit and CLV use the locked Kalshi price. CLV = Kalshi closing mid x your payout - 1.

Run:  python3 value_model.py      Flags: --offline  --no-open  --now 2026-11-24T08:05
Standard library only (Python 3.9+).
"""
import json, math, os, re, sys, time, urllib.request, webbrowser
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date
from statistics import NormalDist

# ================================ CONFIG ====================================
SPORTS = {
    "nba": dict(label="NBA", kind="bball", path="basketball/nba", q="", kalshi=("KXNBAGAME",),
                lock=("daily", 8, 0), horizon_days=1,
                season=("2026-10-20", None), prev=("2025-10-21", "2026-04-13"),
                sd=12.0, hca=2.4, half_life=45, prior_games=10, regress=0.45, unknown_pct=None, cap=30,
                b2b=-1.6, three_in_four=-0.6, travel=0.35, inj_scale=0.45, inj_cap=7.0, team_inj_cap=11.0),
    "cbb": dict(label="College Basketball", kind="bball", path="basketball/mens-college-basketball",
                q="&groups=50&limit=500", kalshi=("KXNCAAMBGAME",), lock=("daily", 8, 0), horizon_days=1,
                season=("2026-11-02", None), prev=("2025-11-03", "2026-04-07"),
                sd=10.5, hca=3.0, half_life=40, prior_games=6, regress=0.55, unknown_pct=8, cap=30,
                b2b=-0.8, three_in_four=-0.4, travel=0.35, inj_scale=0.40, inj_cap=6.0, team_inj_cap=9.0),
    "nfl": dict(label="NFL", kind="fball", path="football/nfl", q="", kalshi=("KXNFLGAME",),
                lock=("tue", 9, 0), horizon_days=7,
                season=("2026-09-09", None), prev=("2025-09-04", "2026-01-05"),
                sd=13.5, hca=1.7, half_life=80, prior_games=4, regress=0.5, unknown_pct=None, cap=24,
                bye=0.8, short_week=-0.8, travel=0.25, qb_base=3.0, qb_slope=1.2, qb_cap=8.0),
    "cfb": dict(label="College Football", kind="fball", path="football/college-football", q="&groups=80&limit=500",
                kalshi=("KXNCAAFGAME",), lock=("sun", 9, 0), horizon_days=7,
                season=("2026-08-27", None), prev=("2025-08-23", "2026-01-21"),
                sd=15.5, hca=2.6, half_life=70, prior_games=3, regress=0.5, unknown_pct=5, cap=35,
                bye=0.5, short_week=-0.5, travel=0.25, qb_base=3.0, qb_slope=1.0, qb_cap=7.0),
}
MIN_EDGE = 0.03             # bet when model win% x Kalshi payout - 1 >= 3%  (also defines "min odds")
MAX_EDGE = 0.15             # bigger "edges" are usually missing information -> skipped and listed
MIN_P = 0.12                # skip teams the model gives under 12%
MOVE_SKIP = 0.06            # skip a side if DraftKings moved 6+ pts of win% against it since the open
KALSHI_MAX_SPREAD = 0.08    # ignore Kalshi markets with a bid/ask gap wider than 8 cents
KALSHI_FEE = 0.07           # Kalshi taker fee = 0.07 x P x (1-P)
W_DEFAULT = 0.5             # model weight vs market until the backtest has enough games
W_BOUNDS = (0.25, 0.85)     # fitted model weight is kept inside these bounds
MIN_BT_GAMES = 150          # games needed before fitted SD / weight are used
YPP_PTS = 7.0               # football: 1 yard-per-play of margin ~ 7 points
WIND_MPH, RAIN_MM = 15, 2.0 # football weather thresholds at kickoff
WEATHER_SHRINK = 0.93       # projected margin multiplier in bad weather (x0.88 in wind >= 22 mph)
INJ_STATUS = {"out": 1.0, "injured reserve": 1.0, "suspension": 1.0, "doubtful": 0.75,
              "questionable": 0.35, "day-to-day": 0.3}
CATCHUP_DATE = "2026-11-24" # one-time: on this date, sports that have never locked lock at current prices
LOCK_GRACE_MIN = 60
FINAL_AFTER_H = 2.5
WORKERS = 12
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "value_data.json")
CACHE_DIR = os.path.join(HERE, "cache")
OUT_HTML = os.path.join(HERE, "value_model.html")
TEMPLATE = os.path.join(HERE, "value_template.html")
SB = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard?dates={}{}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/{}/summary?event={}"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={}&status=open&limit=1000{}"
GEO = "https://geocoding-api.open-meteo.com/v1/search?name={}&count=10&language=en&format=json"
WX = ("https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&hourly=wind_speed_10m,precipitation"
      "&wind_speed_unit=mph&timezone=UTC&start_date={}&end_date={}")
N = NormalDist()
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-5))
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
STATES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut",
              DE="Delaware", DC="District of Columbia", FL="Florida", GA="Georgia", HI="Hawaii", ID="Idaho", IL="Illinois",
              IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine", MD="Maryland",
              MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana",
              NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey", NM="New Mexico", NY="New York",
              NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania",
              RI="Rhode Island", SC="South Carolina", SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah",
              VT="Vermont", VA="Virginia", WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming",
              PR="Puerto Rico", ON="Ontario", QC="Quebec", BC="British Columbia", AB="Alberta", MB="Manitoba")


# ------------------------------- network -----------------------------------
def http_get(url):
    err = ""
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sharp-model)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = str(e)
            time.sleep(1.0 * (i + 1))
    raise IOError(err)


FETCH = http_get  # swapped out in tests


def pmap(fn, items):
    out = {}
    items = list(items)
    if not items:
        return out
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = None
    return out


def dget(o, *path):
    for p in path:
        if isinstance(o, list):
            o = o[p] if isinstance(p, int) and len(o) > p else None
        elif isinstance(o, dict):
            o = o.get(p)
        else:
            return None
    return o


# ------------------------------- small math --------------------------------
def nv_home(h_am, a_am):
    """No-vig home win prob from two American prices."""
    dh, da = dec_from_american(h_am), dec_from_american(a_am)
    if not dh or not da:
        return None
    ih, ia = 1 / dh, 1 / da
    return ih / (ih + ia)


def miles(a, b):
    if not a or not b:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(min(1, h)))


def pct_rank(vals, pct):
    v = sorted(vals)
    if not v:
        return 0.0
    return v[min(len(v) - 1, max(0, int(len(v) * pct / 100)))]


def logloss(p, y):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


# ------------------------------- shared helpers ----------------------------
def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ET)


def to_num(x):
    try:
        s = str(x).strip().lower().lstrip("ou").replace("+", "")
        if s in ("even", "ev", "pk", "pick"):
            return 100.0 if "ev" in s else 0.0
        return float(s)
    except Exception:
        return None


def dec_from_american(a):
    if a is None or a == 0:
        return None
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def fmt_american(d):
    a = american(d)
    return f"+{a}" if a > 0 else str(a)


def kalshi_eff(p):
    """p = contract price (0-1). Effective decimal payout per $ incl. taker fee."""
    return 1 / (p + KALSHI_FEE * p * (1 - p))


def kalshi_price_for_cost(c):
    f = KALSHI_FEE
    return ((1 + f) - math.sqrt(max((1 + f) ** 2 - 4 * f * c, 0))) / (2 * f)


def clamp(q):
    return min(max(q, 1e-4), 1 - 1e-4)


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("st.", "state").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in ("the", "university", "of", "u")]
    return toks


def name_score(kname, team):
    k = norm_name(kname)
    best = 0.0
    for cand in (team["location"], team["name"], team["short"], team["abbr"]):
        c = norm_name(cand)
        if not c or not k:
            continue
        if k == c:
            return 1.0
        inter = len(set(k) & set(c))
        best = max(best, inter / max(len(set(k) | set(c)), 1))
    return best


def kalshi_markets(series):
    out, cursor = [], ""
    for _ in range(6):
        data = FETCH(KALSHI.format(series, f"&cursor={cursor}" if cursor else ""))
        out.extend(data.get("markets", []))
        cursor = data.get("cursor") or ""
        if not cursor:
            break
    return out


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def ev_parts(event_ticker):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:\d{4})?([A-Z0-9]+)$", event_ticker or "")
    if not m:
        return None, ""
    try:
        return date(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3))), m.group(4)
    except Exception:
        return None, ""


def match_event(event_ticker, title, games):
    """Kalshi event -> (game, {kalshi_name: 'home'|'away'})."""
    d, code = ev_parts(event_ticker)
    names = [x.strip() for x in re.split(r"\s+(?:vs\.?|at|@)\s+", title or "", maxsplit=1)]
    best, best_s, best_map = None, 0, None
    for g in games.values():
        gd = parse_t(g["commence"]).date()
        if d and abs((gd - d).days) > 1:
            continue
        H, A = g["home"], g["away"]
        if code and code in (A["abbr"] + H["abbr"], H["abbr"] + A["abbr"]) and len(names) == 2:
            first_home = code.startswith(H["abbr"]) and code != A["abbr"] + H["abbr"]
            return g, {names[0]: "home" if first_home else "away", names[1]: "away" if first_home else "home"}
        if len(names) != 2:
            continue
        s1 = name_score(names[0], A) + name_score(names[1], H)
        s2 = name_score(names[0], H) + name_score(names[1], A)
        s = max(s1, s2)
        if s > best_s:
            best, best_s = g, s
            best_map = {names[0]: "away", names[1]: "home"} if s1 >= s2 else {names[0]: "home", names[1]: "away"}
    return (best, best_map) if best_s >= 1.0 else (None, None)


def side_from_title(text, name_map):
    """'Utah wins by over 7.5 points' -> 'home'/'away' using the event's name map."""
    best, bs = None, 0
    for nm, side in name_map.items():
        if text.lower().startswith(nm.lower()):
            return side
        k = norm_name(nm)
        t = norm_name(text)[:len(k)]
        sc = len(set(k) & set(t)) / max(len(set(k)), 1)
        if sc > bs:
            best, bs = side, sc
    return best if bs >= 0.5 else None


def kalshi_for_games(cfg, games, notes):
    """Attach Kalshi quotes to ESPN games: g['k'] = {'h2h': {side: (bid,ask,nbid,nask)}, 'spreads': [...], 'totals': [...]}"""
    for g in games.values():
        g["k"] = {"h2h": {}, "spreads": [], "totals": []}
    ev_cache = {}
    titles = kalshi_set_event_titles(cfg, notes)
    for series, mkt in zip(cfg["kalshi"], ("h2h", "spreads", "totals")):
        try:
            ms = kalshi_markets(series)
        except Exception as e:
            notes.append(f"Kalshi {series} unavailable this run ({e}).")
            continue
        for m in ms:
            et = m.get("event_ticker")
            yb, ya = fnum(m.get("yes_bid_dollars")), fnum(m.get("yes_ask_dollars"))
            nb, na = fnum(m.get("no_bid_dollars")), fnum(m.get("no_ask_dollars"))
            if ya is None or na is None:
                continue
            if et not in ev_cache:
                title = titles.get(et) or m.get("event_title") or ""
                if not title:
                    # derive "A vs B" from rules text when the event title isn't on the market
                    rp = m.get("rules_primary") or ""
                    mm = re.search(r"in the (.+?) vs (.+?) (?:Pro|College|NCAA|Men|Football|Basketball)", rp)
                    title = f"{mm.group(1)} vs {mm.group(2)}" if mm else ""
                ev_cache[et] = (title, None, None)
            title, g, nmap = ev_cache[et]
            if g is None and title:
                g, nmap = match_event(et, title, games)
                ev_cache[et] = (title, g, nmap)
            if g is None:
                continue
            q = (yb or 0.0, ya, nb or 0.0, na)
            if mkt == "h2h":
                side = side_from_title(m.get("yes_sub_title") or m.get("title") or "", nmap)
                if side:
                    g["k"]["h2h"][side] = q
            elif mkt == "spreads":
                side = side_from_title(m.get("yes_sub_title") or m.get("title") or "", nmap)
                s = fnum(m.get("floor_strike"))
                if side and s is not None:
                    g["k"]["spreads"].append((side, s) + q)   # YES = side wins by more than s (side -s)
            else:
                s = fnum(m.get("floor_strike"))
                if s is not None:
                    g["k"]["totals"].append((s,) + q)         # YES = Over s
    return games


def kalshi_set_event_titles(cfg, notes):
    """Map event_ticker -> title via the events endpoint (titles like 'Detroit vs Orlando')."""
    titles = {}
    for series in cfg["kalshi"]:
        cursor = ""
        for _ in range(6):
            try:
                data = FETCH(f"https://api.elections.kalshi.com/trade-api/v2/events?series_ticker={series}&status=open&limit=200"
                             + (f"&cursor={cursor}" if cursor else ""))
            except Exception:
                break
            for e in data.get("events", []):
                titles[e["event_ticker"]] = e.get("title") or ""
            cursor = data.get("cursor") or ""
            if not cursor:
                break
    return titles


def mid(b, a):
    if b is None or a is None or b <= 0 or a >= 1 or a - b > KALSHI_MAX_SPREAD:
        return None
    return (b + a) / 2


def last_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(8):
        d = base - timedelta(days=k)
        if d <= now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


def next_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(9):
        d = base + timedelta(days=k)
        if d > now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


def lockable(cfg, commence, now):
    if commence <= now:
        return False
    if cfg["lock"][0] == "daily":
        return commence.date() == now.date()
    return commence - now <= timedelta(days=7)


# ------------------------------- ESPN parsing -------------------------------
def team_info(c):
    t = c.get("team") or {}
    return dict(id=str(t.get("id") or c.get("id") or ""), abbr=(t.get("abbreviation") or "").upper(),
                location=t.get("location") or "", name=t.get("displayName") or "", short=t.get("shortDisplayName") or "")


def ml_pair(o, which):
    """DraftKings moneyline (home, away) American odds from an ESPN odds/pickcenter object. which: 'open'|'close'."""
    if not o:
        return None
    h = to_num(dget(o, "moneyline", "home", which, "odds"))
    a = to_num(dget(o, "moneyline", "away", which, "odds"))
    if which == "close" and (h is None or a is None):
        h = to_num(dget(o, "homeTeamOdds", "moneyLine")) if h is None else h
        a = to_num(dget(o, "awayTeamOdds", "moneyLine")) if a is None else a
    return [h, a] if h and a else None


def dk_obj(lst):
    if not lst:
        return None
    o = next((x for x in lst if "draft" in str(dget(x, "provider", "name") or "").lower()), lst[0])
    op, cl = ml_pair(o, "open"), ml_pair(o, "close")
    return dict(open=op or cl, cur=cl or op) if (op or cl) else None


def parse_event(ev):
    comp = (ev.get("competitions") or [{}])[0]
    cs = comp.get("competitors") or []
    H = next((c for c in cs if c.get("homeAway") == "home"), None)
    A = next((c for c in cs if c.get("homeAway") == "away"), None)
    if not H or not A:
        return None
    st = (comp.get("status") or ev.get("status") or {}).get("type") or {}
    v = comp.get("venue") or {}
    ad = v.get("address") or {}
    return dict(id=str(ev["id"]), commence=parse_t(ev["date"]).isoformat(), home=team_info(H), away=team_info(A),
                neutral=bool(comp.get("neutralSite")), city=ad.get("city") or "", state=ad.get("state") or ad.get("country") or "",
                indoor=bool(v.get("indoor")),
                status="final" if st.get("completed") else ("live" if st.get("state") == "in" else
                                                              ("off" if st.get("state") == "post" else "pre")),
                hs=to_num(H.get("score")), as_=to_num(A.get("score")), dk=dk_obj(comp.get("odds")))


def stat_map(team_block):
    return {s.get("name"): s.get("displayValue") for s in (team_block.get("statistics") or [])}


def split2(s, sep="-"):
    try:
        a, b = str(s).replace("/", "-").split("-")[:2]
        return float(a), float(b)
    except Exception:
        return None, None


def parse_box(sm, kind):
    """Team efficiency stats, player lines, DraftKings open/close from a completed-game summary."""
    out = dict(teams={}, players={}, dk=dk_obj(sm.get("pickcenter")))
    for tb in dget(sm, "boxscore", "teams") or []:
        tid = str(dget(tb, "team", "id"))
        m = stat_map(tb)
        if kind == "bball":
            fgm, fga = split2(m.get("fieldGoalsMade-fieldGoalsAttempted"))
            ftm, fta = split2(m.get("freeThrowsMade-freeThrowsAttempted"))
            orb = to_num(m.get("offensiveRebounds"))
            tov = to_num(m.get("totalTurnovers")) if m.get("totalTurnovers") is not None else to_num(m.get("turnovers"))
            if None not in (fga, fta, orb, tov):
                out["teams"][tid] = dict(poss=round(fga - orb + tov + 0.44 * fta, 2))
        else:
            plays, yds = to_num(m.get("totalOffensivePlays")), to_num(m.get("totalYards"))
            if plays and yds is not None:
                out["teams"][tid] = dict(plays=plays, yds=yds, to=to_num(m.get("turnovers")) or 0)
    for pb in dget(sm, "boxscore", "players") or []:
        tid = str(dget(pb, "team", "id"))
        rows = []
        for cat in pb.get("statistics") or []:
            keys = cat.get("keys") or []
            if kind == "bball":
                if "minutes" not in keys:
                    continue
                ix = {k: i for i, k in enumerate(keys)}
                for a in cat.get("athletes") or []:
                    st = a.get("stats") or []
                    if a.get("didNotPlay") or len(st) < len(keys):
                        continue
                    mn = to_num(st[ix["minutes"]])
                    if not mn:
                        continue
                    g = lambda k: to_num(st[ix[k]]) if k in ix else 0.0
                    fgm, fga = split2(st[ix["fieldGoalsMade-fieldGoalsAttempted"]]) if "fieldGoalsMade-fieldGoalsAttempted" in ix else (0, 0)
                    ftm, fta = split2(st[ix["freeThrowsMade-freeThrowsAttempted"]]) if "freeThrowsMade-freeThrowsAttempted" in ix else (0, 0)
                    gs = ((g("points") or 0) + 0.4 * (fgm or 0) - 0.7 * (fga or 0) - 0.4 * ((fta or 0) - (ftm or 0))
                          + 0.7 * (g("offensiveRebounds") or 0) + 0.3 * (g("defensiveRebounds") or 0) + (g("steals") or 0)
                          + 0.7 * (g("assists") or 0) + 0.7 * (g("blocks") or 0) - 0.4 * (g("fouls") or 0) - (g("turnovers") or 0))
                    rows.append([str(dget(a, "athlete", "id")), dget(a, "athlete", "displayName") or "", mn, round(gs, 2)])
            elif cat.get("name") == "passing":
                ix = {k: i for i, k in enumerate(keys)}
                for a in cat.get("athletes") or []:
                    st = a.get("stats") or []
                    try:
                        comp, att = split2(st[ix["completions/passingAttempts"]])
                        yds = to_num(st[ix["passingYards"]]) or 0
                        td = to_num(st[ix["passingTouchdowns"]]) or 0
                        it = to_num(st[ix["interceptions"]]) or 0
                        sk, sky = split2(st[ix["sacks-sackYardsLost"]]) if "sacks-sackYardsLost" in ix else (0, 0)
                    except Exception:
                        continue
                    if att:
                        rows.append([str(dget(a, "athlete", "id")), dget(a, "athlete", "displayName") or "",
                                     att, yds, td, it, sk or 0, sky or 0])
        if rows:
            out["players"][tid] = rows
    return out


def parse_injuries(sm):
    """{team_id: [(athlete_id, name, status_weight, status_text)]} from a pre-game summary."""
    out = {}
    for tb in sm.get("injuries") or []:
        tid = str(dget(tb, "team", "id"))
        lst = []
        for inj in tb.get("injuries") or []:
            stt = str(inj.get("status") or dget(inj, "type", "description") or "").lower()
            w = next((v for k, v in INJ_STATUS.items() if k in stt), 0.0)
            if not w:
                continue
            a = inj.get("athlete") or {}
            aid = str(a.get("id") or "")
            if not aid:
                for l in a.get("links") or []:
                    m = re.search(r"/id/(\d+)", str(l.get("href") or ""))
                    if m:
                        aid = m.group(1)
                        break
            lst.append((aid, a.get("displayName") or "", w, inj.get("status") or stt))
        out[tid] = lst
    return out


# ------------------------------- cache -------------------------------------
def cache_path(sk):
    return os.path.join(CACHE_DIR, f"{sk}.json")


def load_cache(sk):
    p = cache_path(sk)
    c = {}
    if os.path.exists(p):
        with open(p) as f:
            c = json.load(f)
    for k, v in (("games", {}), ("done", []), ("box", {}), ("prior", None), ("bt", None)):
        c.setdefault(k, v)
    return c


def save_cache(sk, c):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(sk), "w") as f:
        json.dump(c, f, separators=(",", ":"))


def date_range(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def fetch_dates(cfg, dates):
    res = pmap(lambda d: FETCH(SB.format(cfg["path"], d.strftime("%Y%m%d"), cfg["q"])), dates)
    out = {}
    for d, data in res.items():
        if data is None:
            continue
        out[d] = [g for g in (parse_event(e) for e in data.get("events", [])) if g]
    return out


def refresh_games(sk, cfg, c, now, notes):
    start = date.fromisoformat(cfg["season"][0])
    end = now.date() + timedelta(days=cfg["horizon_days"])
    recent = now.date() - timedelta(days=3)
    need = [d for d in date_range(start, end) if d >= recent or d.isoformat() not in c["done"]]
    got = fetch_dates(cfg, need)
    if need and not got:
        notes.append(f"{cfg['label']}: ESPN unavailable this run.")
    done = set(c["done"])
    for d, games in got.items():
        for g in games:
            old = c["games"].get(g["id"])
            if old and not g.get("dk") and old.get("dk"):
                g["dk"] = old["dk"]
            c["games"][g["id"]] = g
        if d < now.date() - timedelta(days=1) and all(g["status"] in ("final", "off") for g in games):
            done.add(d.isoformat())
    c["done"] = sorted(done)


def refresh_boxes(sk, cfg, c, notes, limit=6000):
    need = [gid for gid, g in c["games"].items() if g["status"] == "final" and gid not in c["box"]][:limit]
    res = pmap(lambda gid: FETCH(SUMMARY.format(cfg["path"], gid)), need)
    for gid, sm in res.items():
        if sm:
            c["box"][gid] = parse_box(sm, cfg["kind"])


def build_prior(sk, cfg, c, notes):
    if c["prior"]:
        return
    a, b = (date.fromisoformat(x) for x in cfg["prev"])
    got = fetch_dates(cfg, list(date_range(a, b)))
    obs = []
    for games in got.values():
        for g in games:
            if g["status"] == "final" and g["hs"] is not None:
                m = max(-cfg["cap"], min(cfg["cap"], g["hs"] - g["as_"]))
                obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], m, 1.0))
    if len(obs) < 50:
        notes.append(f"{cfg['label']}: last season's results unavailable; starting ratings at average.")
        c["prior"] = {"ratings": {}, "unknown": 0.0, "n": len(obs)}
        return
    r, hca = ridge(obs, {}, 0.0, cfg["hca"], lam=1.0)
    vals = list(r.values())
    unk = pct_rank(vals, cfg["unknown_pct"]) * (1 - cfg["regress"]) if cfg["unknown_pct"] else 0.0
    c["prior"] = {"ratings": {t: round(v * (1 - cfg["regress"]), 3) for t, v in r.items()}, "unknown": round(unk, 3), "n": len(obs)}


# ------------------------------- geography ---------------------------------
def load_geo():
    p = os.path.join(CACHE_DIR, "geo.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_geo(geo):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "geo.json"), "w") as f:
        json.dump(geo, f, separators=(",", ":"))


def geocode_all(geo, places):
    need = [p for p in places if p not in geo and p.split("|")[0]]
    def one(p):
        city, st = p.split("|")
        data = FETCH(GEO.format(urllib.request.quote(city)))
        res = data.get("results") or []
        full = STATES.get(st, st)
        best = next((r for r in res if r.get("admin1") == full), None) or \
               next((r for r in res if r.get("country_code") in ("US", "CA")), None) or (res[0] if res else None)
        return [round(best["latitude"], 4), round(best["longitude"], 4)] if best else None
    for p, v in pmap(one, need).items():
        geo[p] = v
    return geo


def place(g):
    return f"{g.get('city','')}|{g.get('state','')}"


# ------------------------------- schedule features --------------------------
def schedule_features(c, cfg, geo):
    """Per-game rest and travel adjustments (points to ADD to the home margin) + detail text."""
    games = sorted(c["games"].values(), key=lambda g: g["commence"])
    homes = defaultdict(Counter)
    for g in games:
        if not g["neutral"] and g.get("city"):
            homes[g["home"]["id"]][place(g)] += 1
    home_loc = {t: geo.get(cnt.most_common(1)[0][0]) for t, cnt in homes.items()}
    diffs = []
    trav = {}
    for g in games:
        v = geo.get(place(g))
        mh = 0.0 if not g["neutral"] else (miles(home_loc.get(g["home"]["id"]), v) or 0.0)
        ma = miles(home_loc.get(g["away"]["id"]), v)
        if ma is None:
            continue
        trav[g["id"]] = (mh, ma)
        if not g["neutral"]:
            diffs.append(ma - mh)
    base = sum(diffs) / len(diffs) if diffs else 0.0
    last = defaultdict(list)
    feats = {}
    for g in games:
        t = parse_t(g["commence"])
        adj, det = 0.0, []
        for side, sign in (("home", 1), ("away", -1)):
            tid = g[side]["id"]
            prev = last[tid]
            a = 0.0
            if prev:
                rest = (t.date() - prev[-1].date()).days
                if cfg["kind"] == "bball":
                    if rest <= 1:
                        a += cfg["b2b"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} B2B")
                    if sum(1 for p in prev[-3:] if (t.date() - p.date()).days <= 3) >= 2:
                        a += cfg["three_in_four"]
                else:
                    if rest >= 12:
                        a += cfg["bye"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} off bye")
                    elif rest <= 5:
                        a += cfg["short_week"]
                        det.append(f"{g[side]['short'] or g[side]['abbr']} short week")
            adj += sign * a
            last[tid].append(t)
        if g["id"] in trav:
            mh, ma = trav[g["id"]]
            ta = cfg["travel"] * ((ma - mh) - (0.0 if g["neutral"] else base)) / 1000
            ta = max(-2.0, min(2.0, ta))
            adj += ta
            if abs(ta) >= 0.3:
                det.append(f"travel {'+' if ta > 0 else ''}{ta:.1f}")
        feats[g["id"]] = (round(adj, 3), det)
    return feats


# ------------------------------- ratings -----------------------------------
def ridge(obs, prior, unknown, hca0, lam, half_life=None, asof=None, iters=40):
    """obs: (iso_time, home_id, away_id, neutral, margin, weight). Returns ratings, hca."""
    if asof is not None and half_life:
        obs = [(t, h, a, n, m, w * 0.5 ** (max(0, (asof - parse_t(t)).days) / half_life)) for t, h, a, n, m, w in obs]
    teams = {x for o in obs for x in (o[1], o[2])}
    r = {t: prior.get(t, unknown) for t in teams}
    pr = dict(r)
    hca = hca0
    for _ in range(iters):
        num = den = 0.0
        for t, h, a, n, m, w in obs:
            if not n:
                num += w * (m - r[h] + r[a])
                den += w
        hca = (num + 30 * hca0) / (den + 30)
        S, W = defaultdict(float), defaultdict(float)
        for t, h, a, n, m, w in obs:
            e = 0.0 if n else hca
            S[h] += w * (m + r[a] - e); W[h] += w
            S[a] += w * (r[h] + e - m); W[a] += w
        for k in teams:
            r[k] = 0.5 * r[k] + 0.5 * (S[k] + lam * pr[k]) / (W[k] + lam)
    return r, hca


def pace_ratings(obs_p):
    """obs_p: (home, away, poss). poss = P + p_h + p_a."""
    if not obs_p:
        return 0.0, {}
    P = sum(o[2] for o in obs_p) / len(obs_p)
    p = defaultdict(float)
    for _ in range(25):
        S, W = defaultdict(float), defaultdict(float)
        for h, a, x in obs_p:
            S[h] += x - P - p[a]; W[h] += 1
            S[a] += x - P - p[h]; W[a] += 1
        for k in S:
            p[k] = S[k] / (W[k] + 5)
    return P, dict(p)


def game_margin(g, box, cfg, league_pace):
    """Observed home margin metric (pace-neutral efficiency for basketball, points+YPP for football)."""
    raw = g["hs"] - g["as_"]
    tb = (box or {}).get("teams") or {}
    H, A = tb.get(g["home"]["id"]), tb.get(g["away"]["id"])
    m = raw
    if cfg["kind"] == "bball" and H and A and H.get("poss") and A.get("poss") and league_pace:
        poss = (H["poss"] + A["poss"]) / 2
        m = raw / poss * league_pace
    elif cfg["kind"] == "fball" and H and A and H.get("plays") and A.get("plays"):
        m = 0.5 * raw + 0.5 * (H["yds"] / H["plays"] - A["yds"] / A["plays"]) * YPP_PTS
    return max(-cfg["cap"], min(cfg["cap"], m))


def league_pace_of(c):
    ps = [t["poss"] for b in c["box"].values() for t in (b.get("teams") or {}).values() if t.get("poss")]
    return sum(ps) / len(ps) if ps else None


def observations(c, cfg, feats, before=None):
    lp = league_pace_of(c) if cfg["kind"] == "bball" else None
    obs, obs_p = [], []
    for g in c["games"].values():
        if g["status"] != "final" or g["hs"] is None or (before and parse_t(g["commence"]) >= before):
            continue
        box = c["box"].get(g["id"])
        m = game_margin(g, box, cfg, lp) - feats.get(g["id"], (0.0, []))[0]
        obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], m, 1.0))
        tb = (box or {}).get("teams") or {}
        if cfg["kind"] == "bball" and g["home"]["id"] in tb and g["away"]["id"] in tb:
            obs_p.append((g["home"]["id"], g["away"]["id"], (tb[g["home"]["id"]]["poss"] + tb[g["away"]["id"]]["poss"]) / 2))
    return obs, obs_p, lp


def fit_ratings(c, cfg, feats, asof, before=None):
    obs, obs_p, lp = observations(c, cfg, feats, before)
    prior = (c["prior"] or {}).get("ratings", {})
    unk = (c["prior"] or {}).get("unknown", 0.0)
    r, hca = ridge(obs, prior, unk, cfg["hca"], cfg["prior_games"], cfg["half_life"], asof)
    P, p = pace_ratings(obs_p) if cfg["kind"] == "bball" else (None, {})
    games_n = Counter(x for o in obs for x in (o[1], o[2]))
    return dict(r=r, hca=hca, P=P, pace=p, lp=lp, prior=prior, unk=unk, n=games_n)


def rating(R, tid):
    return R["r"].get(tid, R["prior"].get(tid, R["unk"]))


def project(g, R, cfg, feats, extra=0.0, wx_mult=1.0):
    """Projected home margin and the pieces that make it up."""
    rh, ra = rating(R, g["home"]["id"]), rating(R, g["away"]["id"])
    sched = feats.get(g["id"], (0.0, []))[0]
    base = rh - ra + (0.0 if g["neutral"] else R["hca"])
    m = base + sched + extra
    if cfg["kind"] == "bball" and R["P"] and R["lp"]:
        poss = R["P"] + R["pace"].get(g["home"]["id"], 0) + R["pace"].get(g["away"]["id"], 0)
        m *= poss / R["lp"]
    m *= wx_mult
    return m, dict(rh=round(rh, 2), ra=round(ra, 2), hca=0.0 if g["neutral"] else round(R["hca"], 2),
                   sched=round(sched, 2), inj=round(extra, 2), wx=wx_mult)


# ------------------------------- players / injuries -------------------------
def player_values(c, cfg):
    """{team_id: {athlete_id: (name, value_pts, share_of_games)}} and a name index."""
    tg = Counter()
    for gid, b in c["box"].items():
        for tid in (b.get("players") or {}):
            tg[tid] += 1
    vals = defaultdict(dict)
    if cfg["kind"] == "bball":
        agg = defaultdict(lambda: [0, 0.0, 0.0, ""])
        for b in c["box"].values():
            for tid, rows in (b.get("players") or {}).items():
                for pid, name, mn, gs in rows:
                    x = agg[(tid, pid)]
                    x[0] += 1; x[1] += mn; x[2] += gs; x[3] = name
        rot = [x[2] / x[1] for x in agg.values() if x[0] >= 3 and x[1] / x[0] >= 12]
        repl = pct_rank(rot, 20) if rot else 0.0
        for (tid, pid), (gp, mn, gs, name) in agg.items():
            if gp < 2:
                continue
            v = max(0.0, (gs / mn - repl) * (mn / gp)) * cfg["inj_scale"]
            vals[tid][pid] = (name, round(min(v, cfg["inj_cap"]), 2), round(gp / max(tg[tid], 1), 3))
    else:
        qbs = defaultdict(lambda: [0, 0.0, 0.0, 0, ""])   # starts, att, value-yards, att+sacks, name
        all_any = [0.0, 0.0]
        for b in c["box"].values():
            for tid, rows in (b.get("players") or {}).items():
                if not rows:
                    continue
                lead = max(rows, key=lambda r: r[2])
                for pid, name, att, yds, td, it, sk, sky in rows:
                    ay = yds + 20 * td - 45 * it - sky
                    x = qbs[(tid, pid)]
                    x[0] += 1 if pid == lead[0] else 0
                    x[1] += att; x[2] += ay; x[3] += att + sk; x[4] = name
                    all_any[0] += ay; all_any[1] += att + sk
        lg = all_any[0] / all_any[1] if all_any[1] else 6.0
        for (tid, pid), (st, att, ay, dropbacks, name) in qbs.items():
            if st < 1 or dropbacks < 20:
                continue
            anya = ay / dropbacks
            v = max(1.5, min(cfg["qb_cap"], cfg["qb_base"] + cfg["qb_slope"] * (anya - lg)))
            vals[tid][pid] = (name, round(v, 2), round(st / max(tg[tid], 1), 3))
    names = {(tid, n.lower()): pid for tid, d in vals.items() for pid, (n, v, s) in d.items()}
    return vals, names


def injury_adjust(tid, inj, vals, names, cfg):
    """Points to subtract from a team for listed injuries; returns (points, [text])."""
    tot, det = 0.0, []
    for aid, name, w, status in inj.get(tid, []):
        pid = aid if aid in vals.get(tid, {}) else names.get((tid, name.lower()))
        if not pid or pid not in vals.get(tid, {}):
            continue
        nm, v, share = vals[tid][pid]
        pts = v * w * share
        if pts >= 0.2:
            tot += pts
            det.append(f"{nm} {status} -{pts:.1f}")
    cap = cfg.get("team_inj_cap") or cfg.get("qb_cap") or 10
    return min(tot, cap), det


# ------------------------------- weather -----------------------------------
def weather_mult(g, geo):
    if g.get("indoor"):
        return 1.0, None
    loc = geo.get(place(g))
    if not loc:
        return 1.0, None
    t = parse_t(g["commence"]).astimezone(timezone.utc)
    try:
        d = FETCH(WX.format(loc[0], loc[1], t.date().isoformat(), t.date().isoformat()))
        hrs = d["hourly"]["time"]
        i = min(range(len(hrs)), key=lambda k: abs((datetime.fromisoformat(hrs[k]).replace(tzinfo=timezone.utc) - t).total_seconds()))
        wind, rain = d["hourly"]["wind_speed_10m"][i], d["hourly"]["precipitation"][i]
    except Exception:
        return 1.0, None
    if wind is None:
        return 1.0, None
    if wind >= 22:
        return 0.88, f"wind {wind:.0f} mph"
    if wind >= WIND_MPH or (rain or 0) >= RAIN_MM:
        return WEATHER_SHRINK, f"wind {wind:.0f} mph, rain {rain or 0:.1f} mm/h"
    return 1.0, None


# ------------------------------- backtest / calibration ---------------------
def backtest(sk, cfg, c, feats, now):
    """Point-in-time predictions for every completed game this season (ratings from earlier games only)."""
    days = defaultdict(list)
    for g in c["games"].values():
        if g["status"] == "final" and g["hs"] is not None and g["hs"] != g["as_"]:
            days[parse_t(g["commence"]).date()].append(g)
    rows = []
    for d in sorted(days):
        cut = datetime(d.year, d.month, d.day, tzinfo=ET)
        R = fit_ratings(c, cfg, feats, cut, before=cut)
        for g in days[d]:
            m, _ = project(g, R, cfg, feats)
            box = c["box"].get(g["id"]) or {}
            dk = box.get("dk") or {}
            po = nv_home(*dk["open"]) if dk.get("open") else None
            pc = nv_home(*dk["cur"]) if dk.get("cur") else None
            rows.append(dict(id=g["id"], d=d.isoformat(), m=m, y=1 if g["hs"] > g["as_"] else 0, po=po, pc=pc,
                             oo=dk.get("open"), sd=None))
    sd0 = cfg["sd"]
    sd = sd0
    if len(rows) >= MIN_BT_GAMES:
        best = None
        for k in range(60, 151, 3):
            s = sd0 * k / 100
            ll = sum(logloss(N.cdf(r["m"] / s), r["y"]) for r in rows) / len(rows)
            if best is None or ll < best[0]:
                best = (ll, s)
        sd = best[1]
    for r in rows:
        r["pm"] = N.cdf(r["m"] / sd)
    mk = [r for r in rows if r["po"] is not None]
    w = W_DEFAULT
    w_raw = None
    if len(mk) >= MIN_BT_GAMES:
        best = None
        for k in range(0, 101, 5):
            ww = k / 100
            ll = sum(logloss(ww * r["pm"] + (1 - ww) * r["po"], r["y"]) for r in mk) / len(mk)
            if best is None or ll < best[0]:
                best = (ll, ww)
        w_raw = best[1]
        w = min(max(best[1], W_BOUNDS[0]), W_BOUNDS[1])
    # simulated early bets at DraftKings opening prices
    bets = []
    for r in mk:
        p = w * r["pm"] + (1 - w) * r["po"]
        for side, ps, am, pc in (("home", p, r["oo"][0], r["pc"]), ("away", 1 - p, r["oo"][1], None if r["pc"] is None else 1 - r["pc"])):
            d_ = dec_from_american(am)
            if not d_ or ps < MIN_P:
                continue
            e = ps * d_ - 1
            if MIN_EDGE <= e <= MAX_EDGE:
                won = (r["y"] == 1) == (side == "home")
                bets.append(dict(pl=(d_ - 1) if won else -1.0, clv=(pc * d_ - 1) if pc is not None else None))
    ll = lambda key: round(sum(logloss(key(r), r["y"]) for r in mk) / len(mk), 4) if mk else None
    cl = [b["clv"] for b in bets if b["clv"] is not None]
    return dict(date=now.date().isoformat(), games=len(rows), with_odds=len(mk), sd=round(sd, 2), w=round(w, 2),
                w_raw=None if w_raw is None else round(w_raw, 2),
                ll_model=ll(lambda r: r["pm"]), ll_blend=ll(lambda r: w * r["pm"] + (1 - w) * r["po"]),
                ll_open=ll(lambda r: r["po"]), ll_close=ll(lambda r: r["pc"] if r["pc"] is not None else r["po"]),
                acc=round(sum((r["pm"] > .5) == (r["y"] == 1) for r in rows) / len(rows), 3) if rows else None,
                bets=len(bets), wins=sum(b["pl"] > 0 for b in bets), units=round(sum(b["pl"] for b in bets), 2),
                roi=round(sum(b["pl"] for b in bets) / len(bets), 4) if bets else None,
                clv=round(sum(cl) / len(cl), 4) if cl else None, beat=round(sum(x > 0 for x in cl) / len(cl), 3) if cl else None)


# ------------------------------- Kalshi side --------------------------------
def kalshi_best(g, side):
    """Cheapest way to back `side` on Kalshi: (price, effective decimal payout) or None."""
    k = (g.get("k") or {}).get("h2h", {})
    own, oth = k.get(side), k.get("away" if side == "home" else "home")
    asks = []
    if own and 0.02 < own[1] < 0.98 and own[1] - own[0] <= KALSHI_MAX_SPREAD:
        asks.append(own[1])
    if oth and 0.02 < oth[3] < 0.98 and oth[3] - oth[2] <= KALSHI_MAX_SPREAD:
        asks.append(oth[3])
    if not asks:
        return None
    p = min(asks)
    return p, kalshi_eff(p)


def kalshi_mid(g, side):
    k = (g.get("k") or {}).get("h2h", {})
    own, oth = k.get(side), k.get("away" if side == "home" else "home")
    ests = []
    if own and mid(own[0], own[1]) is not None:
        ests.append(mid(own[0], own[1]))
    if oth and mid(oth[0], oth[1]) is not None:
        ests.append(1 - mid(oth[0], oth[1]))
    return sum(ests) / len(ests) if ests else None


def max_price(fair):
    return kalshi_price_for_cost(fair / (1 + MIN_EDGE))


def team_name(g, side, sk):
    t = g[side]
    return t["short"] if sk in ("nba", "nfl") and t["short"] else (t["location"] or t["name"])


# ------------------------------- store --------------------------------------
def load_store():
    st = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            st = json.load(f)
    for k, v in (("bets", {}), ("closes", {}), ("locks", {}), ("catchup", {}), ("boards", {}), ("models", {})):
        st.setdefault(k, v)
    st["notes"] = []
    return st


def save_store(st):
    with open(DATA_FILE, "w") as f:
        json.dump({k: st[k] for k in ("bets", "closes", "locks", "catchup", "boards", "models")}, f, indent=0)


# ------------------------------- daily evaluation ---------------------------
def evaluate(sk, cfg, c, st, now, geo, feats, games_today):
    """Price every lockable game; returns board rows and plays."""
    bt = c.get("bt") or {}
    sd = bt.get("sd") if (bt.get("games") or 0) >= MIN_BT_GAMES else cfg["sd"]
    w = bt.get("w") if (bt.get("with_odds") or 0) >= MIN_BT_GAMES else W_DEFAULT
    R = fit_ratings(c, cfg, feats, now)
    vals, names = player_values(c, cfg)
    sums = pmap(lambda gid: FETCH(SUMMARY.format(cfg["path"], gid)), [g["id"] for g in games_today])
    board, plays = [], []
    for g in games_today:
        sm = sums.get(g["id"]) or {}
        inj = parse_injuries(sm) if sm else {}
        dk = dk_obj(sm.get("pickcenter")) if sm else None
        dk = dk or g.get("dk")
        ih, dh = injury_adjust(g["home"]["id"], inj, vals, names, cfg)
        ia, da = injury_adjust(g["away"]["id"], inj, vals, names, cfg)
        wxm, wxt = weather_mult(g, geo) if cfg["kind"] == "fball" else (1.0, None)
        m0, parts = project(g, R, cfg, feats)                 # ratings + home + rest + travel (+ pace)
        m = (m0 * cfg["sd"] / sd + (ia - ih)) * wxm           # calibrated scale; injuries added in true points
        parts["inj"] = round(ia - ih, 2)
        pm_home = N.cdf(m / cfg["sd"])
        p_open = nv_home(*dk["open"]) if dk and dk.get("open") else None
        p_cur = nv_home(*dk["cur"]) if dk and dk.get("cur") else None
        p_mkt = p_cur if p_cur is not None else kalshi_mid(g, "home")
        p_home = w * pm_home + (1 - w) * p_mkt if p_mkt is not None else pm_home
        why = []
        rd = parts["rh"] - parts["ra"]
        why.append(f"rating {'+' if rd >= 0 else ''}{rd:.1f} (home)")
        if parts["hca"]:
            why.append(f"home +{parts['hca']:.1f}")
        why += feats.get(g["id"], (0.0, []))[1]
        why += dh + da
        if wxt:
            why.append(wxt)
        row = dict(sport=sk, id=g["id"], game=f"{team_name(g,'away',sk)} @ {team_name(g,'home',sk)}", commence=g["commence"],
                   home=team_name(g, "home", sk), away=team_name(g, "away", sk), margin=round(m, 2),
                   model_home=round(pm_home, 4), mkt_home=None if p_mkt is None else round(p_mkt, 4),
                   p_home=round(p_home, 4), why=why, best=None, status="")
        best = None
        for side in ("home", "away"):
            p = p_home if side == "home" else 1 - p_home
            kb = kalshi_best(g, side)
            if not kb:
                continue
            price, pay = kb
            e = p * pay - 1
            cand = dict(side=side, price=price, d_eff=pay, p=p, edge=e,
                        model=pm_home if side == "home" else 1 - pm_home)
            if best is None or e > best["edge"]:
                best = cand
        if best is None:
            row["status"] = "no Kalshi market"
        else:
            row["best"] = dict(team=team_name(g, best["side"], sk), price=round(best["price"], 4), edge=round(best["edge"], 4),
                               p=round(best["p"], 4))
            moved = None
            if p_open is not None and p_cur is not None:
                mv = (p_cur - p_open) if best["side"] == "home" else (p_open - p_cur)
                moved = mv
            if best["p"] < MIN_P:
                row["status"] = "long shot"
            elif best["edge"] < MIN_EDGE:
                row["status"] = "no edge"
            elif best["edge"] > MAX_EDGE:
                row["status"] = "edge too big - likely missing info"
            elif moved is not None and moved <= -MOVE_SKIP:
                row["status"] = f"market moved {abs(moved)*100:.0f} pts against - news"
            else:
                row["status"] = "PLAY"
                plays.append((g, best, row))
        board.append(row)
    st["models"][sk] = dict(sd=round(sd, 2), w=round(w, 2), hca=round(R["hca"], 2), at=now.isoformat())
    return board, plays, R


def update(st, now):
    geo = load_geo()
    for sk, cfg in SPORTS.items():
        c = load_cache(sk)
        lk = last_lock(cfg, now)
        due = lk and st["locks"].get(sk) != lk.isoformat() and (now - lk) <= timedelta(minutes=LOCK_GRACE_MIN)
        catchup = not due and sk not in st["locks"] and now.date().isoformat() == CATCHUP_DATE
        due = due or catchup
        open_bets = [b for b in st["bets"].values() if b["sport"] == sk and b["result"] is None]
        first = not c["games"]
        if not due and not open_bets and not first:
            continue
        refresh_games(sk, cfg, c, now, st["notes"])
        if due or first:
            refresh_boxes(sk, cfg, c, st["notes"])
            build_prior(sk, cfg, c, st["notes"])
            geocode_all(geo, {place(g) for g in c["games"].values() if g.get("city")})
            save_geo(geo)
        feats = schedule_features(c, cfg, geo)
        if due and (c.get("bt") or {}).get("date") != now.date().isoformat():
            c["bt"] = backtest(sk, cfg, c, feats, now)
        upcoming = {gid: g for gid, g in c["games"].items() if g["status"] == "pre" and parse_t(g["commence"]) > now}
        if upcoming:
            kalshi_for_games(cfg, upcoming, st["notes"])
        if due:
            todays = [g for g in upcoming.values() if lockable(cfg, parse_t(g["commence"]), now)]
            board, plays, R = evaluate(sk, cfg, c, st, now, geo, feats, todays)
            st["boards"][sk] = dict(at=now.isoformat(), rows=sorted(board, key=lambda r: r["commence"]))
            st["models"][sk]["ratings"] = sorted(
                [dict(team=next((g[s]["name"] or g[s]["location"] for g in c["games"].values() for s in ("home", "away")
                                 if g[s]["id"] == t), t), r=round(v, 2), n=R["n"].get(t, 0))
                 for t, v in R["r"].items() if R["n"].get(t, 0) >= 2], key=lambda x: -x["r"])[:400]
            for g, best, row in plays:
                bid = f"{g['id']}|h2h"
                if bid in st["bets"]:
                    continue
                st["bets"][bid] = dict(
                    id=bid, sport=sk, event_id=g["id"], commence=g["commence"], mkt="h2h", side=best["side"],
                    home=row["home"], away=row["away"], game=row["game"], label=team_name(g, best["side"], sk),
                    book="kalshi", price=round(best["price"], 4), american=None, d_eff=round(best["d_eff"], 5),
                    fair=round(best["p"], 4), model=round(best["model"], 4), edge=round(best["edge"], 4),
                    max_price=round(max_price(best["p"]), 4), units=1.0, why=row["why"],
                    locked_at=now.isoformat(), result=None, pl=None, home_score=None, away_score=None,
                    close_price=None, close_fair=None, clv=None, close_mins=None)
            priced = sum(1 for r in board if r["best"])
            if priced or not todays:
                st["locks"][sk] = lk.isoformat()
                if catchup:
                    st["catchup"][sk] = now.isoformat()
            else:
                st["notes"].append(f"{cfg['label']}: no Kalshi prices matched at lock time - will retry next run.")
        # closing prices for bet games, CLV, grading
        for g in upcoming.values():
            if any(b["event_id"] == g["id"] for b in st["bets"].values()):
                st["closes"][g["id"]] = dict(t=now.isoformat(), k=g.get("k"), dk=g.get("dk"))
        for b in [b for b in st["bets"].values() if b["sport"] == sk]:
            cmt = parse_t(b["commence"])
            cl = st["closes"].get(b["event_id"])
            if b["clv"] is None and cmt <= now and cl:
                fc = kalshi_mid(cl, b["side"])
                if fc is None and cl.get("dk") and cl["dk"].get("cur"):
                    ph = nv_home(*cl["dk"]["cur"])
                    fc = None if ph is None else (ph if b["side"] == "home" else 1 - ph)
                if fc is not None:
                    b["close_fair"] = round(fc, 5)
                    b["clv"] = round(fc * b["d_eff"] - 1, 5)
                    b["close_mins"] = round((cmt - parse_t(cl["t"])).total_seconds() / 60)
                    kb = kalshi_best(cl, b["side"])
                    if kb:
                        b["close_price"] = round(kb[0], 4)
            if b["result"] is None and (now - cmt).total_seconds() >= FINAL_AFTER_H * 3600:
                g = c["games"].get(b["event_id"])
                if g and g["status"] == "final" and g["hs"] is not None:
                    b["home_score"], b["away_score"] = g["hs"], g["as_"]
                    grade(b)
                elif g and g["status"] == "off":
                    b["result"], b["pl"] = "P", 0.0
        save_cache(sk, c)


def grade(b):
    mine = b["home_score"] - b["away_score"] if b["side"] == "home" else b["away_score"] - b["home_score"]
    b["result"] = "W" if mine > 0 else ("L" if mine < 0 else "P")
    b["pl"] = round(b["units"] * (b["d_eff"] - 1), 4) if b["result"] == "W" else (-b["units"] if b["result"] == "L" else 0.0)


# ------------------------------- output -------------------------------------
def build_page(st, now):
    sports = {}
    for sk, cfg in SPORTS.items():
        d, h, m = cfg["lock"]
        lt = datetime(2000, 1, 1, h, m).strftime("%-I:%M%p").lower()
        lk = last_lock(cfg, now)
        c = load_cache(sk)
        sports[sk] = dict(label=cfg["label"], lock=("Daily" if d == "daily" else d.capitalize()) + f" {lt} ET",
                          last_lock=lk.isoformat() if lk else None,
                          locked=st["locks"].get(sk) == (lk.isoformat() if lk else None),
                          next_lock=next_lock(cfg, now).isoformat(),
                          catchup=st["catchup"].get(sk) if lk and st["catchup"].get(sk, "") >= lk.isoformat() else None,
                          bt=c.get("bt"), model=st["models"].get(sk), board=st["boards"].get(sk))
    bets = sorted(st["bets"].values(), key=lambda b: (b["commence"], b["sport"]))
    payload = dict(generated=now.isoformat(), sports=sports, bets=bets, notes=sorted(set(st["notes"]))[:8], flags=[],
                   config=dict(MIN_EDGE=MIN_EDGE, MAX_EDGE=MAX_EDGE, MOVE_SKIP=MOVE_SKIP, KALSHI_FEE=KALSHI_FEE))
    with open(TEMPLATE) as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload))
    with open(OUT_HTML, "w") as f:
        f.write(html)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    now = datetime.now(ET)
    if "--now" in args:
        now = datetime.fromisoformat(args[args.index("--now") + 1]).replace(tzinfo=ET)
    st = load_store()
    if "--offline" not in args:
        update(st, now)
        save_store(st)
    build_page(st, now)
    bets = list(st["bets"].values())
    g = [b for b in bets if b["result"]]
    pl = sum(b["pl"] for b in g)
    print(f"{now:%a %b %-d %-I:%M %p} ET | {len(bets)} bets | {pl:+.2f}u")
    for sk in SPORTS:
        bt = load_cache(sk).get("bt") or {}
        bd = (st["boards"].get(sk) or {}).get("rows", [])
        print(f"  {sk}: backtest games {bt.get('games')} ll model {bt.get('ll_model')} blend {bt.get('ll_blend')} open {bt.get('ll_open')}"
              f" | board {len(bd)} | plays {sum(r['status']=='PLAY' for r in bd)}")
    for n in st["notes"]:
        print("  note:", n)
    if "--no-open" not in args:
        webbrowser.open("file://" + OUT_HTML)


if __name__ == "__main__":
    main()
