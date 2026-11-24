#!/usr/bin/env python3
"""
Early-Market Moneyline Value Model (free, no API key)  -  NFL · NBA · College Football · College Basketball
==========================================================================================================
Compares Kalshi prices with DraftKings moneylines (from ESPN) and takes the side priced
better than fair.

    Fair win %   = 50/50 blend of DraftKings no-vig % and Kalshi mid-price
    Edge (EV %)  = fair x payout - 1   (Kalshi payout includes the taker fee)
    Min odds     = highest price (as %) you can pay and still have a 2% edge
    Units        = flat 1u on every bet

Plays LOCK once, at a fixed time (the price at that run is the bet of record for W/L,
profit and CLV):
    NFL  Tuesday 9:00am ET    CFB  Sunday 9:00am ET    NBA / CBB  daily 8:00am ET
The first run at or after the lock time takes it (up to 60 min late if a run is missed).
NFL/CFB lock every game in the next 7 days; NBA/CBB lock that day's games.

CLV = fair closing % (last run before start) x your payout - 1. Results from ESPN finals.
Data: Kalshi public market data API + ESPN scoreboard (DraftKings). Standard library only.

Run:  python3 value_model.py      Flags: --offline  --no-open  --now 2026-11-24T08:05
"""
import json, math, os, re, sys, time, urllib.request, webbrowser
from datetime import datetime, timedelta, timezone, date
from statistics import NormalDist

# ================================ CONFIG ====================================
SPORTS = {   # lock: (day 'daily'|'mon'..'sun', hour, minute) ET -- plays are taken once, at this time
    "nfl": dict(label="NFL", espn="football/nfl", espn_q="", kalshi=("KXNFLGAME",),
                lock=("tue", 9, 0), horizon_days=7),
    "cfb": dict(label="College Football", espn="football/college-football", espn_q="&groups=80&limit=500",
                kalshi=("KXNCAAFGAME",), lock=("sun", 9, 0), horizon_days=7),
    "nba": dict(label="NBA", espn="basketball/nba", espn_q="", kalshi=("KXNBAGAME",),
                lock=("daily", 8, 0), horizon_days=1),
    "cbb": dict(label="College Basketball", espn="basketball/mens-college-basketball", espn_q="&groups=50&limit=500",
                kalshi=("KXNCAAMBGAME",), lock=("daily", 8, 0), horizon_days=1),
}
BOOK_PRIORITY = [["kalshi"], ["draftkings"]]   # Kalshi first; DraftKings if Kalshi has no qualifying play
W_DK, W_KALSHI = 0.5, 0.5   # weights for the fair price blend
MIN_EDGE = 0.02             # minimum EV to bet (2%); also defines "min odds"
MAX_EDGE = 0.15             # above this the price is probably stale/wrong -> skipped
MIN_FAIR_ML = 0.15          # skip teams with a fair win chance under 15%
MAX_POINT_SHIFT = 0.0       # (moneylines only; unused)
CLV_MAX_SHIFT = 0.0
KALSHI_MAX_SPREAD = 0.08    # ignore Kalshi markets whose bid/ask gap is wider than 8 cents (too thin)
KALSHI_FEE = 0.07           # taker fee rate: fee = 0.07 x P x (1-P) per contract
LOCK_GRACE_MIN = 60         # if the run at the lock time is missed, the next run within 60 min takes it
FINAL_AFTER_H = 2.5         # start checking for final scores this many hours after start
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "value_data.json")
OUT_HTML = os.path.join(HERE, "value_model.html")
TEMPLATE = os.path.join(HERE, "value_template.html")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard?dates={}{}"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={}&status=open&limit=1000{}"
N = NormalDist()
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-5))
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


# ------------------------------- network -----------------------------------
def http_get(url):
    err = ""
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (value-model)"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = str(e)
            time.sleep(1.5 * (i + 1))
    raise IOError(err)


FETCH = http_get  # swapped out in tests


# ------------------------------- helpers -----------------------------------
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


def shift_prob(q, mkt, side_is_over_or_team, from_pt, to_pt, cfg):
    """Move a win probability from one line to another with a normal model.
    spreads: points are the side's handicap (e.g. -3). totals: points are the total; side 'Over'/'Under'."""
    if mkt == "h2h" or from_pt is None or to_pt is None or from_pt == to_pt:
        return q
    z = N.inv_cdf(clamp(q))
    if mkt == "spreads":
        return N.cdf(z + (to_pt - from_pt) / cfg["sd_spread"])
    if side_is_over_or_team == "Over":
        return N.cdf(z - (to_pt - from_pt) / cfg["sd_total"])
    return N.cdf(z + (to_pt - from_pt) / cfg["sd_total"])


def norm_name(s):
    s = (s or "").lower().replace("&", " and ").replace("st.", "state").replace("'", "")
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


# ------------------------------- ESPN (DraftKings + scores) -----------------
def pick(o, *path):
    for p in path:
        if not isinstance(o, dict):
            return None
        o = o.get(p)
    return o


def parse_dk(odds_list, home):
    """DraftKings prices from an ESPN odds list -> dict of decimal prices + lines (home perspective)."""
    if not odds_list:
        return None
    od = next((o for o in odds_list if "draft" in str(pick(o, "provider", "name") or "").lower()), odds_list[0])
    def latest(side_obj, key):
        for t in ("close", "current", "open"):
            v = pick(side_obj, t, key)
            if v not in (None, ""):
                return v
        return None
    out = {}
    hml = to_num(latest(pick(od, "moneyline", "home"), "odds")) or to_num(pick(od, "homeTeamOdds", "moneyLine"))
    aml = to_num(latest(pick(od, "moneyline", "away"), "odds")) or to_num(pick(od, "awayTeamOdds", "moneyLine"))
    if hml and aml:
        out["h2h"] = dict(home=dec_from_american(hml), away=dec_from_american(aml))
    hl = to_num(latest(pick(od, "pointSpread", "home"), "line"))
    if hl is None and od.get("spread") is not None:
        hl = to_num(od.get("spread"))
    if hl is None:
        det = str(od.get("details") or "").split()
        if len(det) == 2 and to_num(det[1]) is not None:
            hl = to_num(det[1]) if det[0].upper() == home.upper() else -to_num(det[1])
    if hl is not None:
        ho = to_num(latest(pick(od, "pointSpread", "home"), "odds")) or to_num(pick(od, "homeTeamOdds", "spreadOdds")) or -110
        ao = to_num(latest(pick(od, "pointSpread", "away"), "odds")) or to_num(pick(od, "awayTeamOdds", "spreadOdds")) or -110
        out["spreads"] = dict(line=hl, home=dec_from_american(ho), away=dec_from_american(ao))
    tl = to_num(latest(pick(od, "total", "over"), "line"))
    if tl is None:
        tl = to_num(od.get("overUnder"))
    if tl:
        oo = to_num(latest(pick(od, "total", "over"), "odds")) or to_num(od.get("overOdds")) or -110
        uo = to_num(latest(pick(od, "total", "under"), "odds")) or to_num(od.get("underOdds")) or -110
        out["totals"] = dict(line=tl, over=dec_from_american(oo), under=dec_from_american(uo))
    return out or None


def espn_games(cfg, days):
    games = {}
    for d in days:
        try:
            data = FETCH(ESPN.format(cfg["espn"], d.strftime("%Y%m%d"), cfg["espn_q"]))
        except Exception:
            continue
        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            cs = comp.get("competitors") or []
            H = next((c for c in cs if c.get("homeAway") == "home"), None)
            A = next((c for c in cs if c.get("homeAway") == "away"), None)
            if not H or not A:
                continue
            def team(c):
                t = c.get("team") or {}
                return dict(abbr=(t.get("abbreviation") or "").upper(), location=t.get("location") or "",
                            name=t.get("displayName") or "", short=t.get("shortDisplayName") or "")
            st = (comp.get("status") or ev.get("status") or {}).get("type") or {}
            games[str(ev["id"])] = dict(
                id=str(ev["id"]), commence=parse_t(ev["date"]).isoformat(), home=team(H), away=team(A),
                status="final" if st.get("completed") else ("live" if st.get("state") == "in" else "pre"),
                home_score=to_num(H.get("score")), away_score=to_num(A.get("score")),
                dk=parse_dk(comp.get("odds"), team(H)["abbr"]))
    return games


# ------------------------------- Kalshi -------------------------------------
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


# ------------------------------- fair prices -------------------------------
def mid(b, a):
    if b is None or a is None or b <= 0 or a >= 1 or a - b > KALSHI_MAX_SPREAD:
        return None
    return (b + a) / 2


def dk_prob(g, mkt, side, point, cfg, max_shift):
    """side: 'home'/'away' for h2h/spreads (point = that side's handicap), 'Over'/'Under' for totals."""
    dk = (g.get("dk") or {}).get(mkt)
    if not dk:
        return None
    if mkt == "h2h":
        ih, ia = 1 / dk["home"], 1 / dk["away"]
        return (ih if side == "home" else ia) / (ih + ia)
    if mkt == "spreads":
        ih, ia = 1 / dk["home"], 1 / dk["away"]
        q = (ih if side == "home" else ia) / (ih + ia)
        base = dk["line"] if side == "home" else -dk["line"]
    else:
        io, iu = 1 / dk["over"], 1 / dk["under"]
        q = (io if side == "Over" else iu) / (io + iu)
        base = dk["line"]
    if abs(point - base) > max_shift + 1e-9:
        return None
    return shift_prob(q, mkt, side, base, point, cfg)


def kalshi_prob(g, mkt, side, point, cfg, max_shift):
    k = g.get("k") or {}
    if mkt == "h2h":
        ests = []
        own, oth = k.get("h2h", {}).get(side), k.get("h2h", {}).get("away" if side == "home" else "home")
        if own and mid(own[0], own[1]) is not None:
            ests.append(mid(own[0], own[1]))
        if oth and mid(oth[0], oth[1]) is not None:
            ests.append(1 - mid(oth[0], oth[1]))
        return sum(ests) / len(ests) if ests else None
    cands = []
    if mkt == "spreads":
        other = "away" if side == "home" else "home"
        for sd, s, yb, ya, nb, na in k.get("spreads", []):
            m_ = mid(yb, ya)
            if m_ is None:
                continue
            if sd == side:      # side -s with prob m_
                cands.append((-s, m_))
            elif sd == other:   # other -s  -> side +s with prob 1-m_
                cands.append((s, 1 - m_))
    else:
        for s, yb, ya, nb, na in k.get("totals", []):
            m_ = mid(yb, ya)
            if m_ is not None:
                cands.append((s, m_ if side == "Over" else 1 - m_))
    if not cands:
        return None
    pt, q = min(cands, key=lambda c: abs(c[0] - point))
    if abs(pt - point) > max_shift + 1e-9:
        return None
    return shift_prob(q, mkt, side, pt, point, cfg)


def fair_prob(g, mkt, side, point, cfg, max_shift=MAX_POINT_SHIFT):
    d = dk_prob(g, mkt, side, point, cfg, max_shift)
    k = kalshi_prob(g, mkt, side, point, cfg, max_shift)
    if d is None or k is None:
        return None
    return (W_DK * d + W_KALSHI * k) / (W_DK + W_KALSHI)



# ------------------------------- plays --------------------------------------
def team_name(g, side, sk):
    t = g[side]
    return t["short"] if sk in ("nba", "nfl") and t["short"] else (t["location"] or t["name"])


def offers(g):
    """Every way to back a team: (book, side, price_as_probability, payout_decimal)."""
    out = []
    k = (g.get("k") or {}).get("h2h", {})
    dkm = (g.get("dk") or {}).get("h2h")
    for side in ("home", "away"):
        own, oth = k.get(side), k.get("away" if side == "home" else "home")
        asks = []
        if own and 0.02 < own[1] < 0.98 and own[1] - own[0] <= KALSHI_MAX_SPREAD:
            asks.append(own[1])                      # buy YES on this team
        if oth and 0.02 < oth[3] < 0.98 and oth[3] - oth[2] <= KALSHI_MAX_SPREAD:
            asks.append(oth[3])                      # or buy NO on the other team
        if asks:
            p = min(asks)
            out.append(("kalshi", side, p, kalshi_eff(p)))
        if dkm:
            out.append(("draftkings", side, 1 / dkm[side], dkm[side]))
    return out


def max_price(book, fair):
    """Highest price (as %) you can pay and still have MIN_EDGE. Kalshi: contract price; DraftKings: implied %."""
    d_min = (1 + MIN_EDGE) / fair
    return kalshi_price_for_cost(1 / d_min) if book == "kalshi" else 1 / d_min


def units_for(fair, d_eff):
    """Flat staking: every bet is 1 unit."""
    return 1.0


def candidates(g):
    flagged = []
    for group in BOOK_PRIORITY:
        best = None
        for book, side, price, pay in offers(g):
            if book not in group:
                continue
            fair = fair_prob(g, "h2h", side, None, None)
            if fair is None or fair < MIN_FAIR_ML:
                continue
            edge = fair * pay - 1
            c = dict(side=side, book=book, price=price, d_eff=pay, fair=fair, edge=edge)
            if edge > MAX_EDGE:
                flagged.append(c)
            elif best is None or edge > best["edge"]:
                best = c
        if best and best["edge"] >= MIN_EDGE:
            return best, flagged
    return None, flagged


# ------------------------------- lock schedule ------------------------------
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


# ------------------------------- store / update -----------------------------
def load_store():
    st = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            st = json.load(f)
    st.setdefault("bets", {}); st.setdefault("closes", {}); st.setdefault("locks", {})
    st["notes"], st["flags"] = [], []
    return st


def save_store(st):
    with open(DATA_FILE, "w") as f:
        json.dump({k: st[k] for k in ("bets", "closes", "locks")}, f, indent=0)


def update(st, now):
    for sk, cfg in SPORTS.items():
        lk = last_lock(cfg, now)
        due = lk and st["locks"].get(sk) != lk.isoformat() and (now - lk) <= timedelta(minutes=LOCK_GRACE_MIN)
        open_bets = [b for b in st["bets"].values() if b["sport"] == sk and b["result"] is None]
        if not due and not open_bets:
            continue
        days = sorted({now.date() + timedelta(days=i) for i in range(cfg["horizon_days"] + 1)} |
                      {parse_t(b["commence"]).date() for b in open_bets})
        games = espn_games(cfg, days)
        if not games:
            continue
        kalshi_for_games(cfg, games, st["notes"])
        priced = 0
        for g in games.values():
            c = parse_t(g["commence"])
            if c <= now or g["status"] != "pre":
                continue
            if (g.get("dk") or {}).get("h2h") and (g.get("k") or {}).get("h2h"):
                priced += 1
            if due and lockable(cfg, c, now):
                p, flagged = candidates(g)
                for f_ in flagged:
                    st["flags"].append(f"{cfg['label']} {team_name(g, f_['side'], sk)} at {f_['book']}: "
                                       f"{f_['edge']*100:.0f}% edge looks stale, skipped")
                bid = f"{g['id']}|h2h"
                if p and bid not in st["bets"]:
                    st["bets"][bid] = dict(
                        id=bid, sport=sk, event_id=g["id"], commence=g["commence"], mkt="h2h", side=p["side"], point=None,
                        home=team_name(g, "home", sk), away=team_name(g, "away", sk),
                        game=f"{team_name(g,'away',sk)} @ {team_name(g,'home',sk)}",
                        label=team_name(g, p["side"], sk), book=p["book"], price=round(p["price"], 4),
                        american=fmt_american(1 / p["price"]) if p["book"] == "draftkings" else None,
                        d_eff=round(p["d_eff"], 5), fair=round(p["fair"], 4), edge=round(p["edge"], 4),
                        max_price=round(max_price(p["book"], p["fair"]), 4), units=units_for(p["fair"], p["d_eff"]),
                        locked_at=now.isoformat(), result=None, pl=None, home_score=None, away_score=None,
                        close_price=None, close_fair=None, clv=None, close_mins=None)
            if any(b["event_id"] == g["id"] for b in st["bets"].values()):
                st["closes"][g["id"]] = dict(t=now.isoformat(), dk=g.get("dk"), k=g.get("k"))
        if due:
            if priced:
                st["locks"][sk] = lk.isoformat()
            else:
                st["notes"].append(f"{cfg['label']}: no prices yet at lock time - will retry next run.")
        for b in [b for b in st["bets"].values() if b["sport"] == sk]:
            c = parse_t(b["commence"])
            cl = st["closes"].get(b["event_id"])
            if b["clv"] is None and c <= now and cl:
                fc = fair_prob(cl, "h2h", b["side"], None, None)
                if fc is not None:
                    b["close_fair"] = round(fc, 5)
                    b["clv"] = round(fc * b["d_eff"] - 1, 5)
                    b["close_mins"] = round((c - parse_t(cl["t"])).total_seconds() / 60)
                    same = [o for o in offers(cl) if o[0] == b["book"] and o[1] == b["side"]]
                    if same:
                        b["close_price"] = round(same[0][2], 4)
            if b["result"] is None and (now - c).total_seconds() >= FINAL_AFTER_H * 3600:
                g = games.get(b["event_id"])
                if g and g["status"] == "final" and g["home_score"] is not None:
                    b["home_score"], b["away_score"] = g["home_score"], g["away_score"]
                    grade(b)


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
        sports[sk] = dict(label=cfg["label"], lock=("Daily" if d == "daily" else d.capitalize()) + f" {lt} ET",
                          last_lock=lk.isoformat() if lk else None, locked=st["locks"].get(sk) == (lk.isoformat() if lk else None),
                          next_lock=next_lock(cfg, now).isoformat())
    bets = sorted(st["bets"].values(), key=lambda b: (b["commence"], b["sport"]))
    payload = dict(generated=now.isoformat(), sports=sports, bets=bets, notes=sorted(set(st["notes"]))[:8],
                   flags=st["flags"][:12], config=dict(MIN_EDGE=MIN_EDGE, KALSHI_FEE=KALSHI_FEE))
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
    risk = sum(b["units"] for b in g if b["result"] != "P")
    pl = sum(b["pl"] for b in g)
    cl = [b for b in bets if b["clv"] is not None]
    print(f"{now:%a %b %-d %-I:%M %p} ET | {len(bets)} bets | {pl:+.2f}u | ROI {100*pl/risk if risk else 0:.1f}% | "
          f"avg CLV {100*sum(b['clv'] for b in cl)/len(cl) if cl else 0:+.2f}%")
    for n in st["notes"]:
        print("  note:", n)
    if "--no-open" not in args:
        webbrowser.open("file://" + OUT_HTML)


if __name__ == "__main__":
    main()
