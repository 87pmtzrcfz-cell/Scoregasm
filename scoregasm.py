"""SCOREGASM: chance an NBA game passes through an exact tied score (default 69-69).
Run:  pip3 install streamlit requests numpy   then   streamlit run scoregasm.py
"""
import datetime
import math

import numpy as np
import requests
import streamlit as st

FEED = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# ---------------------------------------------------------------- the math
# One possession = (probability, running score after each step). Free-throw trips and
# and-ones pass through in-between scores, so a team can sit on the target mid-possession.
# Rates are tuned to league makes per game in 2018-21 play-by-play (backtested, see footer).
SEQ = [(.513, []), (.115, [3]), (.003, [3, 4]), (.254, [2]), (.033, [2, 3]), (.060, [1, 2]), (.022, [1])]
BASE_PPP = sum(p * (c[-1] if c else 0) for p, c in SEQ)  # about 1.106 points per possession


SC = SEQ[1:]                          # the scoring outcomes
SC_TOT = sum(p for p, _ in SC)        # chance a possession scores at all (about 0.49)
MARGIN_K, MARGIN_CAP = 0.002, 20      # trailing team scores ~0.2% faster per point behind (up to 20)


def tie_probability(a, b, minutes_left, ppa, ppb, pace, target):
    """Chance the game is exactly target-target at some point before regulation ends.
    ppa/ppb = points per possession for each team. The trailing team scores slightly faster and
    the leader slightly slower (comebacks, garbage time), a fix found by backtesting on real games."""
    if a > target or b > target:
        return 0.0
    if a == target and b == target:
        return 1.0
    n = round(minutes_left * 2 * pace / 48)  # possessions left, both teams combined
    da, db = target - a, target - b          # points each team is short of the target
    wa, wb = da + 1, db + 1
    ia, ib = np.meshgrid(np.arange(wa), np.arange(wb), indexing="ij")
    m = np.clip(ib - ia, -MARGIN_CAP, MARGIN_CAP)   # margin (A minus B) at each state = db - da
    sa = np.clip(ppa / BASE_PPP * (1 - MARGIN_K * m), 0.3, 2.0)   # A's scoring rate in each state
    sb = np.clip(ppb / BASE_PPP * (1 + MARGIN_K * m), 0.3, 2.0)
    zero = np.zeros((wa, wb)); zero[0, 0] = 1
    ga, gb = zero.copy(), zero.copy()        # ga/gb[x, y]: chance of landing on target when x, y short
    for _ in range(n):                       # build backward from the target, one possession at a time
        new_a = (1 - sa * SC_TOT) * gb       # A has the ball and doesn't score; B gets it
        new_b = (1 - sb * SC_TOT) * ga
        for p, cum in SC:
            c = cum[-1]
            ta, tb = np.zeros((wa, wb)), np.zeros((wa, wb))
            if c < wa:
                ta[c:, :] = gb[: wa - c, :]
            if c < wb:
                tb[:, c:] = ga[:, : wb - c]
            for cj in cum:                   # landed on the target in the middle of a possession
                if cj < wa:
                    ta[cj, 0] = 1
                if cj < wb:
                    tb[0, cj] = 1
            new_a += p * sa * ta
            new_b += p * sb * tb
        new_a[0, 0] = new_b[0, 0] = 1
        ga, gb = new_a, new_b
    return float((ga[da, db] + gb[da, db]) / 2)  # average over who has the ball first


def chubb(p):
    """0-100 display scale (NOT a probability): log scale from 0.1% up to 50%+."""
    if p <= 0:
        return 0
    x = (math.log(p) - math.log(0.001)) / (math.log(0.5) - math.log(0.001))
    return int(round(100 * min(1, max(0, x))))


def tier(s):
    if s >= 85:
        return "SCOREGASM IMMINENT", "🔥🔥🔥"
    if s >= 65:
        return "HOT", "🔥🔥"
    if s >= 45:
        return "HEATING UP", "🔥"
    if s >= 25:
        return "WARMING UP", "🌡️"
    return "ICE COLD", "🧊"


@st.cache_data(ttl=10)
def get_games():
    data = requests.get(FEED, timeout=10).json()
    games = []
    for e in data.get("events", []):
        c, s = e["competitions"][0], e["status"]
        side = {x["homeAway"]: x for x in c["competitors"]}
        away, home = side["away"]["team"]["abbreviation"], side["home"]["team"]["abbreviation"]
        games.append({
            "label": f'{away} @ {home} ({s["type"]["shortDetail"]})', "away": away, "home": home,
            "away_id": str(side["away"]["team"]["id"]), "home_id": str(side["home"]["team"]["id"]),
            "state": s["type"]["state"], "period": s["period"], "clock": s.get("clock"),
            "a": int(side["away"].get("score") or 0), "b": int(side["home"].get("score") or 0),
        })
    return games


# ---------------------------------------------------------------- team tuning
HOME_EDGE = 0.012          # home team scores ~1.2% more per possession, away ~1.2% less
NEUTRAL = {"off": 1.0, "def": 1.0}


def default_season():
    """ESPN names a season by its ending year. Before October, last season ends this year."""
    t = datetime.date.today()
    return (t.year + 1 if t.month >= 10 else t.year) - 1


def _entries(node):
    if isinstance(node, dict):
        if isinstance(node.get("entries"), list):
            yield from node["entries"]
        for v in node.values():
            yield from _entries(v)
    elif isinstance(node, list):
        for v in node:
            yield from _entries(v)


def parse_ratings(data):
    """ESPN standings JSON -> {team_id: {abbr, name, off, def}} where off/def are points for/against
    per game divided by the league average (off > 1 scores a lot; def > 1 gives up a lot)."""
    raw = {}
    for e in _entries(data):
        team, stats = e.get("team") or {}, {x.get("name"): x.get("value") for x in e.get("stats", [])}
        pf, pa = stats.get("avgPointsFor"), stats.get("avgPointsAgainst")
        if pf is None and stats.get("pointsFor") and stats.get("gamesPlayed"):
            pf, pa = stats["pointsFor"] / stats["gamesPlayed"], stats["pointsAgainst"] / stats["gamesPlayed"]
        if team.get("id") is None or pf is None or pa is None:
            continue
        raw[str(team["id"])] = {"abbr": team.get("abbreviation", "?"), "name": team.get("displayName", "?"),
                                "pf": float(pf), "pa": float(pa)}
    if not raw:
        return {}
    lg = sum(t["pf"] for t in raw.values()) / len(raw)
    return {k: {"abbr": t["abbr"], "name": t["name"], "off": t["pf"] / lg, "def": t["pa"] / lg} for k, t in raw.items()}


@st.cache_data(ttl=6 * 3600)
def load_ratings(season):
    url = f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={season}"
    return parse_ratings(requests.get(url, timeout=10).json())


def team_ppp(id_a, id_b, base, trust, ratings, b_is_home=False):
    """Points per possession for A and B. Each team's scoring = league base x its offense x the
    other team's defense, shrunk toward average by 'trust' (last season only partly carries over)."""
    if not ratings or (id_a not in ratings and id_b not in ratings):
        return base, base, False
    ra, rb = ratings.get(id_a, NEUTRAL), ratings.get(id_b, NEUTRAL)
    shr = lambda x: 1 + trust * (x - 1)
    ppa = base * shr(ra["off"]) * shr(rb["def"])
    ppb = base * shr(rb["off"]) * shr(ra["def"])
    if b_is_home:
        ppa, ppb = ppa * (1 - HOME_EDGE), ppb * (1 + HOME_EDGE)
    return ppa, ppb, True


# ---------------------------------------------------------------- the look
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');
.stApp{background:radial-gradient(900px 520px at 8% -8%,rgba(139,92,255,.28),transparent 60%),radial-gradient(800px 480px at 100% 0%,rgba(255,46,147,.22),transparent 55%),radial-gradient(700px 500px at 50% 110%,rgba(34,228,255,.12),transparent 60%),#07060d;color:#f4f1ff;font-family:'Space Grotesk',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
[data-testid="stSidebar"]{background:#0c0a16;border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stWidgetLabel"] p,label{color:#a79fc9!important;text-transform:uppercase;letter-spacing:.09em;font-size:.7rem!important;font-weight:700!important}
input{font-family:'JetBrains Mono',monospace!important;font-weight:700!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{background:#12101f!important;border-color:rgba(255,255,255,.12)!important;border-radius:12px!important}
.logo{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(64px,15vw,128px);line-height:.88;letter-spacing:.015em;margin:.1em 0 0;background:linear-gradient(95deg,#ff2e93 0%,#8b5cff 52%,#22e4ff 100%);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 26px rgba(255,46,147,.45))}
.tag{color:#a79fc9;letter-spacing:.32em;text-transform:uppercase;font-size:.72rem;margin:.4em 0 1.2em}
.ticker{overflow:hidden;white-space:nowrap;border-top:1px solid rgba(255,255,255,.1);border-bottom:1px solid rgba(255,255,255,.1);padding:9px 0;margin-bottom:22px;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#cfc8ee}
.ticker div{display:inline-block;padding-left:100%;animation:scroll 38s linear infinite}
.ticker b{color:#fff}.ticker i{color:#ff2e93;font-style:normal;margin:0 14px}
@keyframes scroll{to{transform:translateX(-100%)}}
.board{display:flex;align-items:center;justify-content:center;gap:clamp(14px,5vw,40px);margin:6px 0 4px}
.team{text-align:center}
.team .ab{font-size:.75rem;letter-spacing:.25em;color:#a79fc9;font-weight:700}
.team .sc{font-family:'Bebas Neue',sans-serif;font-size:clamp(58px,15vw,96px);line-height:1;color:#fff;text-shadow:0 0 18px rgba(139,92,255,.7)}
.team .need{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#22e4ff}
.dash{font-family:'Bebas Neue',sans-serif;font-size:56px;color:#ff2e93;text-shadow:0 0 16px rgba(255,46,147,.8)}
.clock{text-align:center;font-family:'JetBrains Mono',monospace;color:#cfc8ee;font-size:.85rem;letter-spacing:.12em;margin-bottom:6px}
.card{margin-top:18px;padding:24px 20px 20px;text-align:center;border-radius:24px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.12);box-shadow:0 0 0 1px rgba(139,92,255,.15),0 20px 70px rgba(139,92,255,.25),inset 0 1px 0 rgba(255,255,255,.12);backdrop-filter:blur(14px)}
.card .q{color:#a79fc9;letter-spacing:.22em;text-transform:uppercase;font-size:.72rem;font-weight:700}
.alert{margin:-4px -4px 16px;padding:8px 12px;border-radius:12px;font-weight:700;letter-spacing:.14em;font-size:.8rem;color:#fff;background:linear-gradient(90deg,#ff2e93,#8b5cff);animation:blink 1.1s ease-in-out infinite}
@keyframes blink{50%{opacity:.55;box-shadow:0 0 28px rgba(255,46,147,.9)}}
.duo{display:flex;align-items:center;justify-content:center;gap:clamp(12px,4vw,36px);flex-wrap:wrap;margin-top:6px}
.big{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(72px,19vw,150px);line-height:.95;background:linear-gradient(180deg,#fff 0%,#ff7ac0 35%,#ff2e93 70%,#8b5cff 100%);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 30px rgba(255,46,147,.55));animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{50%{filter:drop-shadow(0 0 48px rgba(255,46,147,.9))}}
.lab{color:#7d76a0;font-size:.68rem;letter-spacing:.2em;font-weight:700}
.ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#22e4ff 0,#8b5cff calc(var(--v)*.5%),#ff2e93 calc(var(--v)*1%),rgba(255,255,255,.09) calc(var(--v)*1%));box-shadow:0 0 34px rgba(139,92,255,.45)}
.ringin{width:122px;height:122px;border-radius:50%;background:#0b0914;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rn{font-family:'Bebas Neue',sans-serif;font-size:62px;line-height:.9;color:#fff;text-shadow:0 0 16px rgba(255,46,147,.7)}
.rl{font-size:.62rem;letter-spacing:.3em;color:#22e4ff;font-weight:700}
.tier{font-family:'Bebas Neue',Impact,sans-serif;font-size:2rem;letter-spacing:.06em;margin-top:14px;background:linear-gradient(90deg,#ff2e93,#8b5cff,#22e4ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.plain{color:#cfc8ee;font-size:.95rem;margin-top:2px}
.meter{height:10px;border-radius:99px;background:rgba(255,255,255,.08);margin:18px 4px 6px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#22e4ff,#8b5cff,#ff2e93);box-shadow:0 0 18px rgba(255,46,147,.7)}
.small{color:#7d76a0;font-size:.68rem;letter-spacing:.08em}
.foot{color:#6a6490;font-size:.72rem;line-height:1.5;margin-top:22px;text-align:center}
</style>
"""

st.set_page_config(page_title="scoregasm", page_icon="🔥")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="logo">SCOREGASM</div><div class="tag">the odds of the perfect tie</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Tuning")
    target = st.number_input("Target tied score", 1, 150, 69)
    use_teams = st.toggle("Auto-tune to team averages", value=True)
    season = st.number_input("Ratings season (ending year, 2026 = 2025-26)", 2015, 2100, default_season(), 1)
    trust = st.slider("Trust in that season's averages", 0, 100, 50, 5) / 100
    base = st.number_input("League-average points/possession", 0.5, 2.0, 1.11, 0.01)
    pace = st.number_input("Pace (possessions per 48 min, per team)", 80, 120, 100)

ratings = {}
if use_teams:
    try:
        ratings = load_ratings(int(season))
    except Exception:
        ratings = {}
    if not ratings:
        st.sidebar.warning("Couldn't load team averages, so odds use league-average teams.")


def board(ab_a, a, ab_b, b, target):
    na, nb = max(target - a, 0), max(target - b, 0)
    return (f'<div class="board"><div class="team"><div class="ab">{ab_a}</div><div class="sc">{a}</div>'
            f'<div class="need">needs {na}</div></div><div class="dash">–</div>'
            f'<div class="team"><div class="ab">{ab_b}</div><div class="sc">{b}</div>'
            f'<div class="need">needs {nb}</div></div></div>')


def result_card(p, target):
    pct = p * 100
    txt = "0%" if pct == 0 else "<0.1%" if pct < 0.1 else f"{pct:.2f}%" if pct < 10 else f"{pct:.1f}%"
    score = chubb(p)
    label, emoji = tier(score)
    if p >= 0.995:
        plain = "It's already there."
    elif p == 0:
        plain = "No path left to this tie."
    elif p < 0.01:
        plain = f"About 1 in {round(1 / p):,} games in this spot get there."
    else:
        plain = f"About 1 in {round(1 / p):,}, or roughly {round(pct)} out of every 100 games in this spot."
    alert = '<div class="alert">🚨 TIE ALERT · THIS ONE IS LIVE</div>' if p >= 0.10 else ""
    return (f'<div class="card">{alert}<div class="q">chance of passing through {target}–{target}</div>'
            f'<div class="duo"><div><div class="big">{txt}</div><div class="lab">CHANCE</div></div>'
            f'<div><div class="ring" style="--v:{score}"><div class="ringin"><div class="rn">{score}</div>'
            f'<div class="rl">CHUBB</div></div></div><div class="lab" style="margin-top:8px">CHUBB METER</div></div></div>'
            f'<div class="tier">{emoji} {label}</div><div class="plain">{plain}</div>'
            f'<div class="meter"><i style="width:{score}%"></i></div>'
            f'<div class="small">CHUBB METER · 0–100 · LOG SCALE · 1% ≈ 37 · 3% ≈ 55 · 10% ≈ 74 · 30% ≈ 92</div></div>')


@st.fragment(run_every=15)  # re-checks the feed every 15 seconds
def live_view():
    try:
        games = get_games()
    except Exception:
        games = []
        st.warning("Couldn't reach the score feed. Use manual entry below.")
    if games:
        items = '<i>◆</i>'.join(f'{g["away"]} <b>{g["a"]}</b> – <b>{g["b"]}</b> {g["home"]}' for g in games)
        st.markdown(f'<div class="ticker"><div>{items}</div></div>', unsafe_allow_html=True)

    choice = st.selectbox("Game", ["Manual entry"] + [g["label"] for g in games])
    g = next((x for x in games if x["label"] == choice), None)

    if g is None:
        c1, c2 = st.columns(2)
        a = c1.number_input("Team A score", 0, 200, 24)
        b = c2.number_input("Team B score", 0, 200, 30)
        q = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
        mm = st.number_input("Minutes left in quarter", 0.0, 12.0, 12.0, 0.5)
        by_name = {t["name"]: k for k, t in ratings.items()}
        opts = ["League average"] + sorted(by_name)
        t1, t2 = st.columns(2)
        pick_a, pick_b = t1.selectbox("Team A", opts), t2.selectbox("Team B", opts)
        id_a, id_b = by_name.get(pick_a), by_name.get(pick_b)
        names = (ratings[id_a]["abbr"] if id_a else "TEAM A", ratings[id_b]["abbr"] if id_b else "TEAM B")
        b_home = False
    else:
        a, b = g["a"], g["b"]
        names = (g["away"], g["home"])
        id_a, id_b, b_home = g["away_id"], g["home_id"], True
        if g["state"] == "post":
            st.markdown(board(names[0], a, names[1], b, target), unsafe_allow_html=True)
            return st.info("Game is over.")
        if g["period"] > 4:
            return st.info("Overtime: this model only covers regulation.")
        q = max(g["period"], 1)
        mm = (g["clock"] / 60) if g["state"] == "in" and g["clock"] is not None else 12.0

    minutes_left = (4 - q) * 12 + mm
    ppa, ppb, tuned = team_ppp(id_a, id_b, base, trust, ratings, b_home)
    note = f'TEAM-TUNED · {names[0]} {ppa:.2f} · {names[1]} {ppb:.2f} PTS/POSS' if tuned else 'LEAGUE-AVERAGE TEAMS'
    st.markdown(board(names[0], a, names[1], b, target), unsafe_allow_html=True)
    st.markdown(f'<div class="clock">Q{q} · {int(mm)}:{int((mm % 1) * 60):02d} LEFT<br>{note}</div>', unsafe_allow_html=True)
    p = tie_probability(a, b, minutes_left, ppa, ppb, pace, target)
    st.markdown(result_card(p, target), unsafe_allow_html=True)


live_view()
st.markdown('<div class="foot">Backtested on 2,663 real NBA games (2018–21): predicted tie rates now match actual '
            'within about 0.1 point overall and across game margins and seasons. The high-odds end (above 15%) may run a bit '
            'high. Team tuning (tested on 2019-20 using 2018-19 averages) moved odds a median 5%, in the right direction, '
            'but the gain was too small to prove. Regulation only. Just for fun, not betting advice.</div>',
            unsafe_allow_html=True)
