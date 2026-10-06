"""SCOREGASM: odds an NBA game passes through an exact tied score (default 69-69).
Run:  pip3 install streamlit requests numpy pandas   then   python3 -m streamlit run scoregasm.py
Keep scoregasm_history.csv in the same folder (it powers the Hall of Scoregasms tab)."""
import csv
import datetime
import functools
import glob
import math
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

FEED = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_LOG = os.path.join(HERE, "scoregasm_live_log.csv")
LOG_COLS = ["date", "season", "game_type", "away", "home", "tie_score", "quarter", "clock", "ot",
            "away_final", "home_final", "game_id", "source"]
PACE = 100  # possessions per 48 minutes, per team (barely matters for this question)

# ================================================================ the math
# One possession = (probability, running score after each step). Free-throw trips and and-ones
# pass through in-between scores, so a team can sit on the target mid-possession.
SEQ = [(.513, []), (.115, [3]), (.003, [3, 4]), (.254, [2]), (.033, [2, 3]), (.060, [1, 2]), (.022, [1])]
SC = SEQ[1:]                          # the scoring outcomes
SC_TOT = sum(p for p, _ in SC)        # chance a possession scores at all (about 0.49)
MARGIN_K, MARGIN_CAP = 0.002, 20      # trailing team scores ~0.2% faster per point behind (up to 20)


def tie_probability(a, b, minutes_left, pace=PACE, target=69):
    """Chance the game is exactly target-target at some point before regulation ends (average teams).
    The trailing team scores slightly faster and the leader slightly slower, a fix found by
    backtesting on 2,663 real games."""
    if a > target or b > target:
        return 0.0
    if a == target and b == target:
        return 1.0
    n = round(minutes_left * 2 * pace / 48)  # possessions left, both teams combined
    da, db = target - a, target - b          # points each team is short of the target
    wa, wb = da + 1, db + 1
    ia, ib = np.meshgrid(np.arange(wa), np.arange(wb), indexing="ij")
    m = np.clip(ib - ia, -MARGIN_CAP, MARGIN_CAP)   # margin (A minus B) at each state = db - da
    sa = np.clip(1 - MARGIN_K * m, 0.3, 2.0)        # A's scoring rate in each state
    sb = np.clip(1 + MARGIN_K * m, 0.3, 2.0)
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


@functools.lru_cache(maxsize=4000)
def tp(a, b, minutes_left, target):
    return tie_probability(a, b, minutes_left, PACE, target)


# ================================================================ the scale
def chubb(p):
    """Odds score 0-100 (NOT a probability): log scale from 0.1% up to 50%+."""
    if p <= 0:
        return 0
    x = (math.log(p) - math.log(0.001)) / (math.log(0.5) - math.log(0.001))
    return int(round(100 * min(1, max(0, x))))


def closeness(a, b, target):
    """How far along the road to target-target the game is: points scored / points both teams need."""
    return min(1.0, (a + b) / (2.0 * target))


def meter(p, a, b, target):
    """The Chubb meter, 0-100 = odds score x road progress. A 10-10 game sits near the bottom
    (it's far from 69) and the meter climbs as the game heads toward the target."""
    if p >= 0.995:
        return 100
    return int(round(chubb(p) * closeness(a, b, target)))


TIERS = [(80, "SCOREGASM", "🎆"), (60, "CLOSE", "🔥"), (40, "HALF CHUB", "🌶️"), (20, "EYES ON IT", "👀"), (0, "LIMP", "🥀")]


def tier(score):
    for lo, label, emoji in TIERS:
        if score >= lo:
            return label, emoji


def fmt_pct(p):
    pct = p * 100
    return "0%" if pct == 0 else "<0.1%" if pct < 0.1 else f"{pct:.2f}%" if pct < 10 else f"{pct:.1f}%"


# ================================================================ live feed + scoregasm log
@st.cache_data(ttl=10)
def get_games():
    data = requests.get(FEED, timeout=10).json()
    games = []
    for e in data.get("events", []):
        c, s = e["competitions"][0], e["status"]
        side = {x["homeAway"]: x for x in c["competitors"]}
        away, home = side["away"]["team"]["abbreviation"], side["home"]["team"]["abbreviation"]
        games.append({
            "id": str(e["id"]), "label": f'{away} @ {home} ({s["type"]["shortDetail"]})', "away": away, "home": home,
            "state": s["type"]["state"], "detail": s["type"]["shortDetail"], "period": s["period"], "clock": s.get("clock"),
            "a": int(side["away"].get("score") or 0), "b": int(side["home"].get("score") or 0),
        })
    return games


def scan_ties(plays):
    """{tied score: (period, clock)} for the first moment each tied score appears (0-0 = tip-off)."""
    seen = {}
    for p in plays:
        try:
            a, h = int(p.get("awayScore")), int(p.get("homeScore"))
        except (TypeError, ValueError):
            continue
        if a == h and a not in seen:
            seen[a] = (int((p.get("period") or {}).get("number") or 0), (p.get("clock") or {}).get("displayValue", ""))
    return seen


_FAILED = set()


def logged_ids():
    if not os.path.exists(LIVE_LOG):
        return set()
    try:
        return set(pd.read_csv(LIVE_LOG, usecols=["game_id"]).game_id.astype(str))
    except Exception:
        return set()


def log_finished(g):
    """When a game ends, read its play-by-play and log every tied score it passed through."""
    gid = g["id"]
    if gid in _FAILED or gid in logged_ids():
        return
    try:
        data = requests.get(SUMMARY.format(gid), timeout=10).json()
        ties = scan_ties(data.get("plays", []))
        if not ties:
            _FAILED.add(gid)
            return
        date = ((data.get("header", {}).get("competitions") or [{}])[0].get("date") or "")[:10] or datetime.date.today().isoformat()
    except Exception:
        _FAILED.add(gid)
        return
    new = not os.path.exists(LIVE_LOG)
    with open(LIVE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        for score, (per, clk) in sorted(ties.items()):
            w.writerow({"date": date, "season": "", "game_type": "", "away": g["away"], "home": g["home"], "tie_score": score,
                        "quarter": per, "clock": clk, "ot": per > 4, "away_final": g["a"], "home_final": g["b"],
                        "game_id": gid, "source": "live"})


@st.cache_data(ttl=300)
def load_log():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "scoregasm_history*.csv")))]
    if os.path.exists(LIVE_LOG):
        frames.append(pd.read_csv(LIVE_LOG))
    if not frames:
        return pd.DataFrame(columns=LOG_COLS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "tie_score"])


def game_state(g, target):
    """Odds and meter for one game from the live feed (pregame games use tip-off odds)."""
    if g["state"] == "post":
        return {"kind": "final"}
    if g["state"] == "in" and g["period"] > 4:
        return {"kind": "ot"}
    a, b = g["a"], g["b"]
    if g["state"] == "pre":
        ml = 48.0
    else:
        ml = (4 - max(g["period"], 1)) * 12 + (g["clock"] / 60 if g["clock"] is not None else 12.0)
    p = tp(a, b, round(ml, 2), target)
    return {"kind": "live" if g["state"] == "in" else "pre", "p": p, "ml": ml, "score": meter(p, a, b, target),
            "close": closeness(a, b, target)}


# ================================================================ the look
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');
.stApp{background:radial-gradient(900px 520px at 8% -8%,rgba(139,92,255,.30),transparent 60%),radial-gradient(800px 480px at 100% 0%,rgba(255,46,147,.24),transparent 55%),radial-gradient(700px 500px at 50% 110%,rgba(34,228,255,.14),transparent 60%),#07060d;color:#f4f1ff;font-family:'Space Grotesk',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none}
#MainMenu,footer{visibility:hidden}
[data-testid="stWidgetLabel"] p,label{color:#a79fc9!important;text-transform:uppercase;letter-spacing:.09em;font-size:.7rem!important;font-weight:700!important}
input{font-family:'JetBrains Mono',monospace!important;font-weight:700!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{background:#12101f!important;border-color:rgba(255,255,255,.14)!important;border-radius:12px!important}
button[data-baseweb="tab"] p{font-weight:700;letter-spacing:.04em;font-size:1rem}
.logo{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(64px,16vw,132px);line-height:.88;letter-spacing:.015em;margin:.1em 0 0;background:linear-gradient(95deg,#ff2e93,#8b5cff,#22e4ff,#ff2e93);background-size:300% 100%;animation:shift 7s linear infinite;-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 26px rgba(255,46,147,.5))}
@keyframes shift{to{background-position:300% 0}}
.tag{color:#a79fc9;letter-spacing:.32em;text-transform:uppercase;font-size:.72rem;margin:.4em 0 1.1em}
.ticker{overflow:hidden;white-space:nowrap;border-top:1px solid rgba(255,255,255,.12);border-bottom:1px solid rgba(255,255,255,.12);padding:9px 0;margin:6px 0 18px;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#cfc8ee}
.ticker div{display:inline-block;padding-left:100%;animation:scroll 38s linear infinite}
.ticker b{color:#fff}.ticker i{color:#ff2e93;font-style:normal;margin:0 14px}
@keyframes scroll{to{transform:translateX(-100%)}}
.pill{display:inline-block;padding:5px 14px;border-radius:99px;background:rgba(255,46,147,.16);border:1px solid #ff2e93;color:#ff9ad0;font-weight:700;letter-spacing:.16em;font-size:.7rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff2e93;margin-right:8px;box-shadow:0 0 10px #ff2e93;animation:blink 1s ease-in-out infinite}
.board{display:flex;align-items:center;justify-content:center;gap:clamp(14px,5vw,40px);margin:8px 0 2px}
.team{text-align:center}
.team .ab{font-size:.75rem;letter-spacing:.25em;color:#a79fc9;font-weight:700}
.team .sc{font-family:'Bebas Neue',sans-serif;font-size:clamp(58px,15vw,96px);line-height:1;color:#fff;text-shadow:0 0 18px rgba(139,92,255,.75)}
.team .need{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#22e4ff}
.dash{font-family:'Bebas Neue',sans-serif;font-size:56px;color:#ff2e93;text-shadow:0 0 16px rgba(255,46,147,.8)}
.clock{text-align:center;font-family:'JetBrains Mono',monospace;color:#cfc8ee;font-size:.85rem;letter-spacing:.12em;margin-bottom:6px}
.party{margin:10px 0;padding:14px;border-radius:16px;text-align:center;font-family:'Bebas Neue',sans-serif;font-size:1.9rem;letter-spacing:.06em;color:#fff;background:linear-gradient(90deg,#ff2e93,#8b5cff,#22e4ff);animation:blink 1s ease-in-out infinite}
.card{margin-top:14px;padding:24px 20px 20px;text-align:center;border-radius:26px;background:linear-gradient(160deg,rgba(255,255,255,.08),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.14);box-shadow:0 0 0 1px rgba(139,92,255,.18),0 20px 70px rgba(139,92,255,.3),inset 0 1px 0 rgba(255,255,255,.14);backdrop-filter:blur(14px)}
.card .q{color:#a79fc9;letter-spacing:.22em;text-transform:uppercase;font-size:.72rem;font-weight:700}
.alert{margin:-4px -4px 16px;padding:8px 12px;border-radius:12px;font-weight:700;letter-spacing:.14em;font-size:.8rem;color:#fff;background:linear-gradient(90deg,#ff2e93,#8b5cff);animation:blink 1.1s ease-in-out infinite}
@keyframes blink{50%{opacity:.55;box-shadow:0 0 28px rgba(255,46,147,.9)}}
.duo{display:flex;align-items:center;justify-content:center;gap:clamp(12px,4vw,36px);flex-wrap:wrap;margin-top:6px}
.big{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(72px,19vw,150px);line-height:.95;background:linear-gradient(180deg,#fff 0%,#ff7ac0 35%,#ff2e93 70%,#8b5cff 100%);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 30px rgba(255,46,147,.55));animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{50%{filter:drop-shadow(0 0 48px rgba(255,46,147,.95))}}
.lab{color:#7d76a0;font-size:.68rem;letter-spacing:.2em;font-weight:700}
.ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#22e4ff 0,#8b5cff calc(var(--v)*.5%),#ff2e93 calc(var(--v)*1%),rgba(255,255,255,.09) calc(var(--v)*1%));box-shadow:0 0 34px rgba(139,92,255,.5)}
.ringin{width:122px;height:122px;border-radius:50%;background:#0b0914;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rn{font-family:'Bebas Neue',sans-serif;font-size:62px;line-height:.9;color:#fff;text-shadow:0 0 16px rgba(255,46,147,.7)}
.rl{font-size:.62rem;letter-spacing:.3em;color:#22e4ff;font-weight:700}
.strip{display:flex;gap:6px;justify-content:center;margin-top:18px}
.seg{flex:1;max-width:96px;padding:7px 2px;border-radius:12px;border:1px solid rgba(255,255,255,.1);opacity:.38}
.seg.on{opacity:1;border-color:#ff2e93;background:rgba(255,46,147,.14);box-shadow:0 0 20px rgba(255,46,147,.65)}
.seg b{display:block;font-size:1.35rem}.seg i{font-style:normal;font-size:.52rem;letter-spacing:.1em;color:#d8d2f2;font-weight:700}
.plain{color:#cfc8ee;font-size:.95rem;margin-top:12px}
.road{height:10px;border-radius:99px;background:rgba(255,255,255,.08);margin:16px 4px 6px;overflow:hidden}
.road i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#22e4ff,#8b5cff,#ff2e93);box-shadow:0 0 18px rgba(255,46,147,.7)}
.small{color:#7d76a0;font-size:.68rem;letter-spacing:.08em}
.foot{color:#6a6490;font-size:.72rem;line-height:1.5;margin-top:22px;text-align:center}
.grow{padding:12px 14px;margin:8px 0;border-radius:16px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.1)}
.grow.hit{border-color:#ff2e93;animation:blink 1s ease-in-out infinite}
.grow.dim{opacity:.5}
.gtop{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.gm{font-family:'JetBrains Mono',monospace;font-size:1rem;color:#cfc8ee}.gm b{color:#fff;font-size:1.15rem}
.gs{color:#7d76a0;font-size:.72rem;letter-spacing:.1em;margin-top:2px}
.gr{text-align:right}.gp{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;line-height:1;color:#ff7ac0}
.gc{font-size:.72rem;letter-spacing:.1em;color:#a79fc9;font-weight:700}
.gbar{height:6px;border-radius:99px;background:rgba(255,255,255,.08);margin-top:9px;overflow:hidden}
.gbar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#22e4ff,#8b5cff,#ff2e93)}
</style>
"""

st.set_page_config(page_title="scoregasm", page_icon="🔥", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="logo">SCOREGASM</div><div class="tag">the odds of the perfect tie</div>', unsafe_allow_html=True)
c_t, c_h = st.columns([1, 2])
target = c_t.number_input("🎯 Target tie score", 1, 150, 69)
c_h.markdown(f'<div class="small" style="padding-top:26px">CHANCE A GAME IS EVER TIED AT EXACTLY {target}–{target}</div>',
             unsafe_allow_html=True)


def get_games_safe():
    try:
        return get_games()
    except Exception:
        st.warning("Couldn't reach the score feed. Use the Check a game tab to type a score in.")
        return []


def scoreboard(ab_a, a, ab_b, b, target):
    na, nb = max(target - a, 0), max(target - b, 0)
    return (f'<div class="board"><div class="team"><div class="ab">{ab_a}</div><div class="sc">{a}</div>'
            f'<div class="need">needs {na}</div></div><div class="dash">–</div>'
            f'<div class="team"><div class="ab">{ab_b}</div><div class="sc">{b}</div>'
            f'<div class="need">needs {nb}</div></div></div>')


def result_card(p, a, b, target):
    score, close = meter(p, a, b, target), closeness(a, b, target)
    label, emoji = tier(score)
    if p >= 0.995:
        plain = "It's already there."
    elif p == 0:
        plain = "No path left to this tie."
    elif p < 0.01:
        plain = f"About 1 in {round(1 / p):,} games in this spot get there."
    else:
        plain = f"About 1 in {round(1 / p):,}, or roughly {round(p * 100)} out of every 100 games in this spot."
    alert = '<div class="alert">🚨 TIE ALERT · THIS ONE IS LIVE</div>' if p >= 0.10 else ""
    strip = "".join(f'<div class="seg{" on" if lab == label else ""}"><b>{em}</b><i>{lab}</i></div>' for _, lab, em in reversed(TIERS))
    na, nb = max(target - a, 0), max(target - b, 0)
    return (f'<div class="card">{alert}<div class="q">chance of passing through {target}–{target}</div>'
            f'<div class="duo"><div><div class="big">{fmt_pct(p)}</div><div class="lab">CHANCE</div></div>'
            f'<div><div class="ring" style="--v:{score}"><div class="ringin"><div class="rn">{score}</div>'
            f'<div class="rl">CHUBB</div></div></div><div class="lab" style="margin-top:8px">CHUBB METER</div></div></div>'
            f'<div class="strip">{strip}</div><div class="plain">{plain}</div>'
            f'<div class="road"><i style="width:{close * 100:.1f}%"></i></div>'
            f'<div class="small">ROAD TO {target}–{target} · {close * 100:.0f}% THERE · NEEDS {na} AND {nb} MORE</div></div>')


def tonight_html(games, st_by_id, target):
    order = {"live": 0, "pre": 1, "ot": 2, "final": 3}
    rows = sorted(games, key=lambda g: (order[st_by_id[g["id"]]["kind"]], -st_by_id[g["id"]].get("score", 0)))
    out = []
    for g in rows:
        s = st_by_id[g["id"]]
        score_line = f'{g["away"]} <b>{g["a"]}</b> – <b>{g["b"]}</b> {g["home"]}'
        if s["kind"] in ("final", "ot"):
            right = '<div class="gc">FINAL</div>' if s["kind"] == "final" else '<div class="gc">OVERTIME · NOT MODELED</div>'
            out.append(f'<div class="grow dim"><div class="gtop"><div><div class="gm">{score_line}</div><div class="gs">{g["detail"]}</div></div>{right}</div></div>')
            continue
        label, emoji = tier(s["score"])
        hit = " hit" if g["state"] == "in" and g["a"] == g["b"] == target else ""
        out.append(f'<div class="grow{hit}"><div class="gtop"><div><div class="gm">{score_line}</div><div class="gs">{g["detail"]}</div></div>'
                   f'<div class="gr"><div class="gp">{fmt_pct(s["p"])}</div><div class="gc">{emoji} {s["score"]} · {label}</div></div></div>'
                   f'<div class="gbar"><i style="width:{s["close"] * 100:.1f}%"></i></div></div>')
    return "".join(out)


@st.fragment(run_every=15)  # re-checks the feed every 15 seconds
def board_view():
    games = get_games_safe()
    if not games:
        return st.info("No games found right now. Try the Check a game tab.")
    for g in games:
        if g["state"] == "post":
            log_finished(g)   # when a game ends, log every tied score it passed through
    st_by_id = {g["id"]: game_state(g, target) for g in games}
    items = '<i>◆</i>'.join(f'{g["away"]} <b>{g["a"]}</b> – <b>{g["b"]}</b> {g["home"]}' for g in games)
    st.markdown(f'<div class="ticker"><div>{items}</div></div>', unsafe_allow_html=True)
    live = [g for g in games if st_by_id[g["id"]]["kind"] == "live"]
    if live:
        for g in live:
            if g["a"] == g["b"] == target:
                st.markdown(f'<div class="party">🎆 SCOREGASM! {g["away"]} AND {g["home"]} ARE TIED {target}–{target} RIGHT NOW 🎆</div>',
                            unsafe_allow_html=True)
        top = max(live, key=lambda g: st_by_id[g["id"]]["score"])
        s = st_by_id[top["id"]]
        st.markdown('<div style="text-align:center"><span class="pill"><span class="dot"></span>HOTTEST GAME RIGHT NOW</span></div>',
                    unsafe_allow_html=True)
        st.markdown(scoreboard(top["away"], top["a"], top["home"], top["b"], target), unsafe_allow_html=True)
        st.markdown(f'<div class="clock">{top["detail"].upper()}</div>', unsafe_allow_html=True)
        st.markdown(result_card(s["p"], top["a"], top["b"], target), unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center"><span class="pill">NO GAMES LIVE · PREGAME ODDS BELOW</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="small" style="margin:22px 0 4px">ALL GAMES · CHANCE · CHUBB METER · BAR = ROAD TO THE TARGET</div>',
                unsafe_allow_html=True)
    st.markdown(tonight_html(games, st_by_id, target), unsafe_allow_html=True)


@st.fragment(run_every=15)
def game_view():
    games = get_games_safe()
    choice = st.selectbox("Pick a live game, or type your own", ["Type my own score"] + [g["label"] for g in games])
    g = next((x for x in games if x["label"] == choice), None)
    if g is None:
        c1, c2 = st.columns(2)
        a, b = c1.number_input("Team A score", 0, 200, 24), c2.number_input("Team B score", 0, 200, 30)
        c3, c4 = st.columns(2)
        q = c3.selectbox("Quarter", [1, 2, 3, 4], index=1)
        mm = c4.number_input("Minutes left in quarter", 0.0, 12.0, 12.0, 0.5)
        names = ("TEAM A", "TEAM B")
    else:
        a, b, names = g["a"], g["b"], (g["away"], g["home"])
        if g["state"] == "post":
            st.markdown(scoreboard(names[0], a, names[1], b, target), unsafe_allow_html=True)
            return st.info("Game is over.")
        if g["period"] > 4:
            return st.info("Overtime: this model only covers regulation.")
        q = max(g["period"], 1)
        mm = (g["clock"] / 60) if g["state"] == "in" and g["clock"] is not None else 12.0
    if a == b == target:
        st.markdown(f'<div class="party">🎆 SCOREGASM! TIED {target}–{target} RIGHT NOW 🎆</div>', unsafe_allow_html=True)
    st.markdown(scoreboard(names[0], a, names[1], b, target), unsafe_allow_html=True)
    st.markdown(f'<div class="clock">Q{q} · {int(mm)}:{int((mm % 1) * 60):02d} LEFT</div>', unsafe_allow_html=True)
    p = tp(a, b, round((4 - q) * 12 + mm, 2), target)
    st.markdown(result_card(p, a, b, target), unsafe_allow_html=True)


@st.fragment(run_every=30)
def log_view():
    h = load_log()
    h = h.assign(season=h.season.fillna("live").replace("", "live"))
    n_games = int((h.tie_score == 0).sum())
    hits = h[h.tie_score == target]
    if n_games == 0:
        return st.info("No log yet. Put scoregasm_history.csv in the same folder as this app.")
    expect = tp(0, 0, 48.0, target)
    c1, c2, c3 = st.columns(3)
    c1.metric("Games logged", f"{n_games:,}")
    c2.metric(f"Scoregasms at {target}–{target}", f"{len(hits):,}")
    c3.metric("Share of games", f"{len(hits) / n_games * 100:.2f}%", f"model at tip-off: {expect * 100:.2f}%", delta_color="off")
    per = h.groupby("season").agg(games=("tie_score", lambda s: int((s == 0).sum())), hits=("tie_score", lambda s: int((s == target).sum())))
    st.caption(" · ".join(f"{k}: {r.hits} of {r.games:,}" for k, r in per.iterrows()))
    if hits.empty:
        return st.info(f"No logged game has hit {target}–{target} yet.")
    t = hits.sort_values(["date", "game_id"], ascending=False)
    when = lambda r: f"{'Q' + str(int(r.quarter)) if int(r.quarter) <= 4 else 'OT' + str(int(r.quarter) - 4)} {r.clock}"
    st.dataframe(pd.DataFrame({"Date": t.date, "Game": t.away + " @ " + t.home, "Hit at": t.apply(when, axis=1),
                               "Final": t.away_final.astype(int).astype(str) + "–" + t.home_final.astype(int).astype(str),
                               "Source": t.source}), hide_index=True, use_container_width=True)


tab_board, tab_game, tab_log = st.tabs(["📺 Tonight", "🧮 Check a game", "🏆 Hall of Scoregasms"])
with tab_board:
    board_view()
with tab_game:
    game_view()
with tab_log:
    log_view()

with st.expander("How does this work, and how accurate is it?"):
    st.markdown(
        "**The odds.** The model plays out the rest of the game one possession at a time, thousands of ways, and counts how "
        "often it lands on exactly the target tie (free-throw trips included, since a team can sit on the number mid-trip). "
        "The team that's behind scores a touch faster and the leader a touch slower, like real games.\n\n"
        "**The Chubb meter** is a display scale, not a probability: the odds on a log scale (1% is about 37, 3% about 55, "
        "10% about 74, 30% about 92), multiplied by how far the game is along the road to the target. That's why 10-10 "
        "sits near the bottom even though the odds look similar, and why the meter climbs as the score heads toward 69. "
        "🥀 Limp 0 · 👀 Eyes on it 20 · 🌶️ Half chub 40 · 🔥 Close 60 · 🎆 Scoregasm 80.\n\n"
        "**Accuracy.** I replayed 2,663 real games (2018–21). At five moments per game the model gave a probability for "
        "every possible tied score using only the score and clock then, and I checked what happened next. Forecasts of "
        "about 3% hit 3.1% of the time (9,272 of 296,350), and 5% forecasts hit 5.3%. Brier score (average squared "
        "error) and log loss agree. A single forecast is never right or wrong, only the pooled record can be judged. "
        "The same game appears in many forecasts, and odds above 15% ran a bit high.")
st.markdown('<div class="foot">Backtested on 2,663 real NBA games (2018–21): predicted tie rates match actual within about '
            '0.1 point overall. Regulation only. Just for fun, not betting advice.</div>', unsafe_allow_html=True)
