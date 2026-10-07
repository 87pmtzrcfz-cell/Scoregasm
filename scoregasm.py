"""SCOREGASM: odds an NBA game passes through an exact tied score (default 69-69).
Run:  pip3 install streamlit requests numpy pandas   then   python3 -m streamlit run scoregasm.py
The Hall of Fame reads scoregasm_archive.csv (built once by build_archive.py) and scoregasm_season.csv (filled daily by update_log.py)."""
import functools
import math
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

FEED = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = "2026-10-01"  # the Hall of Fame counts games from this date on
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
def chub(p):
    """Odds score 0-100 (NOT a probability): log scale from 0.1% up to 50%+."""
    if p <= 0:
        return 0
    x = (math.log(p) - math.log(0.001)) / (math.log(0.5) - math.log(0.001))
    return int(round(100 * min(1, max(0, x))))


def closeness(a, b, target):
    """How far along the road to target-target the game is: points scored / points both teams need."""
    return min(1.0, (a + b) / (2.0 * target))


def meter(p, a, b, target):
    """The Chub meter, 0-100 = odds score x road progress. A 10-10 game sits near the bottom
    (it's far from 69) and the meter climbs as the game heads toward the target."""
    if p >= 0.995:
        return 100
    return int(round(chub(p) * closeness(a, b, target)))


TIERS = [(80, "SCOREGASM", "🎆"), (60, "CLOSE", "🔥"), (40, "HALF CHUB", "🌭"), (20, "PEEPING", "👀"), (0, "LIMP", "🪱")]


TAGLINES = {"SCOREGASM": "IT'S HAPPENING.", "CLOSE": "Almost there, baby.",
            "HALF CHUB": "Hot dog's on the grill. Not ready yet.", "PEEPING": "Sheesh.",
            "LIMP": "Flaccid. Nothing to see here, folks."}


def tier(score):
    for lo, label, emoji in TIERS:
        if score >= lo:
            return label, emoji


def fmt_pct(p):
    pct = p * 100
    return "0%" if pct == 0 else "<0.1%" if pct < 0.1 else f"{pct:.2f}%" if pct < 10 else f"{pct:.1f}%"


# ================================================================ live feed
@st.cache_data(ttl=10)
def fetch_feed():
    """One shared ESPN request, kept for 10 seconds no matter how many people are watching."""
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
    return {"at": time.time(), "games": games}


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
@import url('https://fonts.googleapis.com/css2?family=Monoton&family=Audiowide&family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
:root{--red:#ff2e97;--teal:#19e3d8;--hot:#ffe3f1;--line:rgba(255,46,151,.40);--mute:#8f8aa3;--glow:0 0 4px #ff2e97,0 0 14px #ff2e97,0 0 34px rgba(255,46,151,.7),0 0 60px rgba(25,227,216,.28)}
.stApp{background:radial-gradient(800px 420px at 50% -12%,rgba(255,46,151,.14),transparent 62%),#000;color:#f4effa;font-family:'Space Grotesk',system-ui,sans-serif;font-weight:400}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none}
#MainMenu,footer{visibility:hidden}
[data-testid="stWidgetLabel"] p,label{color:var(--mute)!important;text-transform:uppercase;letter-spacing:.2em;font-size:.62rem!important;font-weight:500!important}
input{font-family:'JetBrains Mono',monospace!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{background:#000!important;border:1px solid var(--line)!important;border-radius:2px!important}
button[data-baseweb="tab"] p{font-weight:500;letter-spacing:.14em;font-size:.72rem;text-transform:uppercase}
.logo{font-family:'Monoton','Audiowide',sans-serif;font-size:clamp(30px,9.4vw,78px);line-height:1.1;letter-spacing:.04em;margin:.1em 0 0;color:var(--hot);text-shadow:var(--glow);animation:flick 7s infinite}
@keyframes flick{0%,18%,22%,62%,64%,100%{opacity:1}20%,63%{opacity:.78}}
.tag{display:inline-block;color:var(--teal);letter-spacing:.42em;text-transform:uppercase;font-size:.62rem;margin:.9em 0 1.3em;padding-top:10px;border-top:2px solid;border-image:linear-gradient(90deg,#ff2e97,#19e3d8) 1;text-shadow:0 0 8px rgba(25,227,216,.5)}
.warn{display:inline-block;margin:-4px 0 14px;padding:4px 10px;border:1px solid var(--line);color:var(--red);font-size:.58rem;letter-spacing:.2em;text-shadow:0 0 8px rgba(255,46,151,.6)}
.ticker{overflow:hidden;white-space:nowrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:9px 0;margin:6px 0 18px;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#d8d2e6}
.ticker div{display:inline-block;padding-left:100%;animation:scroll 38s linear infinite}
.ticker b{color:#fff}.ticker i{color:var(--teal);font-style:normal;margin:0 14px;text-shadow:0 0 8px var(--teal)}
@keyframes scroll{to{transform:translateX(-100%)}}
.pill{display:inline-block;padding:5px 14px;border:1px solid var(--teal);border-radius:2px;color:var(--teal);font-weight:500;letter-spacing:.22em;font-size:.62rem;text-shadow:0 0 8px rgba(25,227,216,.7);box-shadow:0 0 12px rgba(25,227,216,.35),inset 0 0 10px rgba(25,227,216,.1)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--red);margin-right:8px;box-shadow:0 0 10px var(--red);animation:blink 1.4s ease-in-out infinite}
.board{display:flex;align-items:center;justify-content:center;gap:clamp(18px,6vw,48px);margin:14px 0 4px}
.team{text-align:center}
.team .ab{font-size:.66rem;letter-spacing:.34em;color:var(--mute);font-weight:500}
.team .sc{font-family:'Audiowide','Space Grotesk',sans-serif;font-size:clamp(58px,16vw,100px);line-height:1;color:var(--hot);text-shadow:var(--glow)}
.team .need{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--teal)}
.dash{font-size:44px;color:var(--teal);text-shadow:0 0 6px var(--teal),0 0 18px rgba(25,227,216,.7);font-weight:300}
.clock{text-align:center;font-family:'JetBrains Mono',monospace;color:var(--mute);font-size:.8rem;letter-spacing:.2em;margin-bottom:6px}
.party{margin:12px 0;padding:14px;border:1px solid var(--red);border-radius:2px;text-align:center;font-family:'Audiowide',sans-serif;font-size:1.5rem;letter-spacing:.1em;color:var(--hot);text-shadow:var(--glow);box-shadow:0 0 30px rgba(255,46,151,.5),inset 0 0 30px rgba(255,46,151,.18);animation:blink 1s ease-in-out infinite}
.card{margin-top:14px;padding:26px 20px 22px;text-align:center;border:1px solid var(--line);border-radius:2px;background:#040208;box-shadow:0 0 28px rgba(255,46,151,.2),inset 0 0 40px rgba(255,46,151,.06)}
.card .q{color:var(--mute);letter-spacing:.3em;text-transform:uppercase;font-size:.64rem;font-weight:500}
.alert{margin:-4px -4px 16px;padding:8px 12px;border:1px solid var(--red);font-weight:500;letter-spacing:.2em;font-size:.7rem;color:var(--red);text-shadow:0 0 8px rgba(255,46,151,.7);animation:blink 1.1s ease-in-out infinite}
@keyframes blink{50%{opacity:.5}}
.duo{display:flex;align-items:center;justify-content:center;gap:clamp(14px,5vw,40px);flex-wrap:wrap;margin-top:8px}
.big{font-family:'Audiowide','Space Grotesk',sans-serif;font-size:clamp(64px,18vw,140px);line-height:1;color:var(--hot);text-shadow:var(--glow);animation:pulse 3s ease-in-out infinite}
@keyframes pulse{50%{text-shadow:0 0 3px #ff2e97,0 0 10px #ff2e97,0 0 22px rgba(255,46,151,.7)}}
.lab{color:var(--mute);font-size:.62rem;letter-spacing:.3em;font-weight:500}
.ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--red) 0,#b46bff calc(var(--v)*.5%),var(--teal) calc(var(--v)*1%),rgba(255,46,151,.12) calc(var(--v)*1%));box-shadow:0 0 22px rgba(255,46,151,.45)}
.ringin{width:134px;height:134px;border-radius:50%;background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rn{font-family:'Audiowide','Space Grotesk',sans-serif;font-size:60px;line-height:.9;color:var(--hot);text-shadow:var(--glow)}
.rl{font-size:.56rem;letter-spacing:.34em;color:var(--teal);font-weight:500;text-shadow:0 0 8px rgba(25,227,216,.6)}
.strip{display:flex;gap:6px;justify-content:center;margin-top:20px}
.seg{flex:1;max-width:96px;padding:7px 2px;border:1px solid rgba(255,46,151,.18);border-radius:2px;opacity:.35}
.seg.on{opacity:1;border-color:var(--teal);box-shadow:0 0 16px rgba(25,227,216,.5),inset 0 0 12px rgba(25,227,216,.12)}
.seg b{display:block;font-size:1.3rem}.seg i{font-style:normal;font-size:.5rem;letter-spacing:.14em;color:#d8d2e6;font-weight:500}
.plain{color:#d8d2e6;font-size:.92rem;margin-top:14px;font-weight:300}
.road,.gbar{height:3px;background:rgba(255,46,151,.14);margin:18px 4px 8px;overflow:visible}
.gbar{margin:10px 0 0;height:2px}
.road i,.gbar i{display:block;height:100%;background:linear-gradient(90deg,#ff2e97,#b46bff,#19e3d8);box-shadow:0 0 10px rgba(255,46,151,.8),0 0 22px rgba(25,227,216,.45)}
.small{color:var(--mute);font-size:.62rem;letter-spacing:.16em}
.foot{color:#6f6a82;font-size:.68rem;line-height:1.6;margin-top:26px;text-align:center;letter-spacing:.04em}
.grow,.sp,.recap,.blurb{padding:13px 15px;margin:9px 0;border:1px solid rgba(255,46,151,.22);border-radius:2px;background:#030107}
.grow.hit{border-color:var(--teal);box-shadow:0 0 18px rgba(25,227,216,.55);animation:blink 1s ease-in-out infinite}
.grow.dim{opacity:.5}
.gtop{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.gm{font-family:'JetBrains Mono',monospace;font-size:.95rem;color:#d8d2e6}.gm b{color:#fff;font-size:1.1rem}
.gs{color:var(--mute);font-size:.68rem;letter-spacing:.14em;margin-top:2px}
.gr{text-align:right}.gp{font-family:'Audiowide',sans-serif;font-size:1.7rem;line-height:1;color:var(--hot);text-shadow:0 0 10px rgba(255,46,151,.8)}
.gc{font-size:.66rem;letter-spacing:.16em;color:var(--mute);font-weight:500}
.sp.HOF{border-color:var(--red);box-shadow:0 0 22px rgba(255,46,151,.3),0 0 40px rgba(25,227,216,.1)}
.sp.EDGED{border-color:rgba(25,227,216,.5)}
.spt{font-size:.62rem;letter-spacing:.18em;font-weight:500;color:var(--teal);text-shadow:0 0 8px rgba(25,227,216,.5)}
.spg{font-family:'Audiowide','Space Grotesk',sans-serif;font-size:1.45rem;letter-spacing:.05em;color:var(--hot);margin-top:3px}
.sph{color:#d8d2e6;font-size:.88rem;line-height:1.5;margin:5px 0 7px;font-weight:300}
.fresh{text-align:center;color:var(--mute);font-size:.62rem;letter-spacing:.2em;font-weight:500;margin:-4px 0 10px}
.fresh.bad{color:#ffb347}
.recap{color:#e6e0f2;font-size:.88rem}.recap b{color:var(--red);letter-spacing:.2em;font-size:.62rem}
.tagl{text-align:center;font-family:'Audiowide','Space Grotesk',sans-serif;font-size:1.55rem;letter-spacing:.08em;color:var(--hot);margin-top:12px;text-shadow:var(--glow)}
.blurb{border-left:2px solid var(--red);color:#e6e0f2;font-size:.92rem;line-height:1.55}.blurb b{color:#fff}.blurb .hd{font-family:'Audiowide',sans-serif;font-size:1.4rem;letter-spacing:.08em;color:var(--red);margin-bottom:6px}
.ring.xl{width:220px;height:220px;margin:14px auto 0}
.ring.xl .ringin{width:200px;height:200px}
.ring.xl .rn{font-size:112px;line-height:.95}
.ring.xl .rl{font-size:.7rem}
.big.sm{font-size:clamp(34px,9vw,56px);margin-top:6px;animation:none}
.divider{height:1px;margin:20px 12% 16px;background:linear-gradient(90deg,transparent,rgba(255,46,151,.5),rgba(25,227,216,.5),transparent)}

/* ===== retro motion: same colors, just moving ===== */
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9998;background:repeating-linear-gradient(0deg,rgba(0,0,0,.16) 0 1px,transparent 1px 3px);opacity:.55}
.stApp::after{content:"";position:fixed;left:0;right:0;top:-14vh;height:14vh;pointer-events:none;z-index:9998;background:linear-gradient(180deg,transparent,rgba(255,46,151,.06),rgba(25,227,216,.05),transparent);animation:roll 9s linear infinite}
@keyframes roll{to{transform:translateY(130vh)}}
.logo{animation:flick 7s infinite,hue 8s ease-in-out infinite}
@keyframes flick{0%,17%,19%,21%,61%,63%,100%{opacity:1}18%,20%,62%{opacity:.55}}
@keyframes hue{0%,100%{text-shadow:0 0 4px #ff2e97,0 0 14px #ff2e97,0 0 34px rgba(255,46,151,.7),0 0 60px rgba(25,227,216,.28)}50%{text-shadow:0 0 4px #19e3d8,0 0 14px #19e3d8,0 0 34px rgba(25,227,216,.7),0 0 60px rgba(255,46,151,.28)}}
.tag::after{content:"\25AE";margin-left:6px;color:var(--teal);animation:cursor 1s steps(1) infinite}
@keyframes cursor{50%{opacity:0}}
.rn,.team .sc{animation:glitch 6s infinite}
@keyframes glitch{0%,93%,100%{transform:none}94%{transform:translateX(-2px);text-shadow:2px 0 #19e3d8,-2px 0 #ff2e97}96%{transform:translateX(2px);text-shadow:-2px 0 #19e3d8,2px 0 #ff2e97}98%{transform:none}}
.road i,.gbar i{position:relative;overflow:hidden}
.road i::after,.gbar i::after{content:"";position:absolute;top:0;bottom:0;width:36px;left:-40px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.85),transparent);animation:sweep 2.6s ease-in-out infinite}
@keyframes sweep{to{left:100%}}
.seg.on{animation:tier 1.6s steps(2) infinite}
@keyframes tier{50%{box-shadow:0 0 6px rgba(25,227,216,.35),inset 0 0 6px rgba(25,227,216,.08)}}
.alert,.grow.hit{animation:blink 1s steps(2) infinite}
.party{animation:party 1s steps(2) infinite}
@keyframes party{0%{color:var(--hot);border-color:var(--red)}50%{color:#d6fffb;border-color:var(--teal);text-shadow:0 0 4px #19e3d8,0 0 14px #19e3d8,0 0 34px rgba(25,227,216,.7)}}
.dot{animation:blink 1s steps(2) infinite}
.sp,.grow,.card{transition:box-shadow .25s,border-color .25s}
.sp:hover,.grow:hover{border-color:var(--teal);box-shadow:0 0 20px rgba(25,227,216,.35)}
@media (prefers-reduced-motion:reduce){.stApp::after,.stApp::before{display:none}*{animation:none!important;transition:none!important}}
</style>
"""

st.set_page_config(page_title="scoregasm", page_icon="🌴", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="logo">SCOREGASM</div><div class="tag">the odds of finishing together</div>', unsafe_allow_html=True)
st.markdown('<div class="warn">🔞 VIEWER DISCRETION ADVISED · EXCESSIVE TIE-RELATED INNUENDO</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#8f8aa3;font-size:.95rem;letter-spacing:.08em;font-weight:300;margin:2px 0 18px">'
            'Two teams. One number. <b style="color:#fff">69–69.</b> Not 70, not 68.</div>', unsafe_allow_html=True)
target = 69   # Scoregasm is about one number
st.markdown('<div class="small" style="margin:2px 0 6px">CHANCE A GAME IS EVER TIED AT EXACTLY 69–69</div>', unsafe_allow_html=True)


def get_games_safe():
    """Games from ESPN. Keeps the last good copy, so a hiccup shows a warning instead of a blank page."""
    try:
        feed = fetch_feed()
        st.session_state["last_good"], st.session_state["feed_error"] = feed, False
    except Exception:
        feed = st.session_state.get("last_good")
        st.session_state["feed_error"] = True
        if feed is None:
            st.warning("Couldn't reach the score feed. Use the Check a game tab to type a score in.")
            return []
    return feed["games"]


def feed_status_html():
    feed = st.session_state.get("last_good")
    if not feed:
        return ""
    age = max(0, int(time.time() - feed["at"]))
    ago = f"{age}S" if age < 60 else f"{age // 60} MIN"
    if st.session_state.get("feed_error"):
        return f'<div class="fresh bad">⚠️ ESPN IS NOT ANSWERING (EVEN ESPN NEEDS A MINUTE) · SHOWING SCORES FROM {ago} AGO · RETRYING EVERY 15 SECONDS</div>'
    return ""                                                   # healthy feed: no status line, only the error banner above


def recap_html(games):
    """One line: how many games are final and, from the log, how many hit 69-69 or came close."""
    done = [g for g in games if g["state"] == "post"]
    if not done:
        return ""
    waiting = any(g["state"] in ("in", "pre") for g in games)
    head = "TODAY SO FAR" if waiting else "LAST NIGHT"
    log = load_csv("scoregasm_season.csv")
    logged = log[log.game_id.astype(str).isin({g["id"] for g in done})] if log is not None else None
    n = 0 if logged is None else len(logged)
    parts = [f"{len(done)} game{'s' if len(done) != 1 else ''} final"]
    if n:
        hits, near = int((logged.result == "scoregasm").sum()), int((logged.result == "edged").sum())
        parts.append(f"🎆 {hits} scoregasm{'s' if hits != 1 else ''}" if hits else "no scoregasms")
        parts.append(f"🥵 {near} left hanging")
        if n < len(done):
            parts.append(f"{len(done) - n} still being logged")
    else:
        parts.append("scoregasm log updates overnight")
    return f'<div class="recap"><b>{head}</b> · ' + " · ".join(parts) + "</div>"


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
    return (f'<div class="card">{alert}<div class="q">CHUB METER</div>'
            f'<div class="ring xl" style="--v:{score}"><div class="ringin"><div class="rn">{score}</div>'
            f'<div class="rl">CHUB</div></div></div>'
            f'<div class="strip">{strip}</div><div class="tagl">{TAGLINES[label]}</div>'
            f'<div class="divider"></div><div class="q">chance of passing through {target}–{target}</div>'
            f'<div class="big sm">{fmt_pct(p)}</div><div class="plain">{plain}</div>'
            f'<div class="road"><i style="width:{close * 100:.1f}%"></i></div>'
            f'<div class="small">ROAD TO {target}–{target} · {close * 100:.0f}% THERE · NEEDS {na} AND {nb} MORE</div></div>')


def tonight_html(games, st_by_id, target):
    order = {"live": 0, "pre": 1, "ot": 2, "final": 3}
    rows = sorted(games, key=lambda g: (order[st_by_id[g["id"]]["kind"]], -st_by_id[g["id"]].get("score", 0)))
    out = []
    log = load_csv("scoregasm_season.csv")                      # last night's log, plus any 69-69 we watched happen live
    hit_ids = set(st.session_state.get("celebrated", set()))
    if log is not None and len(log):
        hit_ids |= set(log[log.result == "scoregasm"].game_id.astype(str))
    hit_ids = {str(x) for x in hit_ids}
    for g in rows:
        s = st_by_id[g["id"]]
        score_line = f'{g["away"]} <b>{g["a"]}</b> – <b>{g["b"]}</b> {g["home"]}'
        if s["kind"] in ("final", "ot"):
            right = '<div class="gc">FINAL</div>' if s["kind"] == "final" else '<div class="gc">OVERTIME · NOT MODELED</div>'
            smoke = ""
            if s["kind"] == "final" and str(g["id"]) in hit_ids:
                right, smoke = '<div class="gc">🎆 FINAL · WENT 69–69</div>', '<div class="gs" style="margin-top:6px">🚬 Someone light a cigarette.</div>'
            out.append(f'<div class="grow dim"><div class="gtop"><div><div class="gm">{score_line}</div><div class="gs">{g["detail"]}</div></div>{right}</div>{smoke}</div>')
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
    st_by_id = {g["id"]: game_state(g, target) for g in games}
    items = '<i>◆</i>'.join(f'{g["away"]} <b>{g["a"]}</b> – <b>{g["b"]}</b> {g["home"]}' for g in games)
    st.markdown(f'<div class="ticker"><div>{items}</div></div>', unsafe_allow_html=True)
    st.markdown(feed_status_html(), unsafe_allow_html=True)
    live = [g for g in games if st_by_id[g["id"]]["kind"] == "live"]
    if live:
        for g in live:
            if g["a"] == g["b"] == target:
                st.markdown(f'<div class="party">🎆🎆 SCOREGASM! {g["away"]} AND {g["home"]} ARE FINISHING TOGETHER AT {target}–{target}. GET TO THE TV 🎆🎆</div>',
                            unsafe_allow_html=True)
                seen = st.session_state.setdefault("celebrated", set())
                if g["id"] not in seen:          # confetti once per game, not every refresh
                    seen.add(g["id"])
                    st.balloons()
        top = max(live, key=lambda g: st_by_id[g["id"]]["score"])
        s = st_by_id[top["id"]]
        st.markdown('<div style="text-align:center"><span class="pill"><span class="dot"></span>🔥 HOTTEST GAME RIGHT NOW · HANDS OFF THE REMOTE</span></div>',
                    unsafe_allow_html=True)
        st.markdown(scoreboard(top["away"], top["a"], top["home"], top["b"], target), unsafe_allow_html=True)
        st.markdown(f'<div class="clock">{top["detail"].upper()}</div>', unsafe_allow_html=True)
        st.markdown(result_card(s["p"], top["a"], top["b"], target), unsafe_allow_html=True)
    else:
        pre_left = any(st_by_id[g["id"]]["kind"] == "pre" for g in games)
        st.markdown(f'<div style="text-align:center"><span class="pill">NO GAMES LIVE · NO ACTION YET, BACK AT TIP-OFF{" · PREGAME ODDS BELOW" if pre_left else ""}</span></div>',
                    unsafe_allow_html=True)
    hide = st.toggle("Hide finished games (they are already asleep)", key="hide_final")
    shown = [g for g in games if not (hide and st_by_id[g["id"]]["kind"] == "final")]
    if shown:
        st.markdown(tonight_html(shown, st_by_id, target), unsafe_allow_html=True)
    else:
        st.info("Every game on the board is final. Flip the switch to see them.")


@st.fragment(run_every=15)
def game_view():
    games = get_games_safe()
    st.markdown(feed_status_html(), unsafe_allow_html=True)
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
        st.markdown(f'<div class="party">🎆 SCOREGASM! FINISHING TOGETHER AT {target}–{target} 🎆</div>', unsafe_allow_html=True)
    st.markdown(scoreboard(names[0], a, names[1], b, target), unsafe_allow_html=True)
    st.markdown(f'<div class="clock">Q{q} · {int(mm)}:{int((mm % 1) * 60):02d} LEFT</div>', unsafe_allow_html=True)
    p = tp(a, b, round((4 - q) * 12 + mm, 2), target)
    st.markdown(result_card(p, a, b, target), unsafe_allow_html=True)
    st.markdown('<div class="small" style="margin-top:8px">CHUB METER = THE ODDS × HOW CLOSE THE SCORE IS TO 69–69. HIGHER = HOTTER.</div>',
                unsafe_allow_html=True)


GAP_LABEL = {1: "🥵 1 away · BLUE BALLS", 2: "😬 2 away · ALMOST, SWEETHEART", 3: "😮 3 away · THE THOUGHT COUNTS"}
ENDED = {"overshot": "jumped over it", "pulled away": "leader scored again", "end of regulation": "regulation ended"}


@st.cache_data(ttl=300)
def load_csv(name):
    path = os.path.join(HERE, name)
    return pd.read_csv(path, dtype={"game_id": str}) if os.path.exists(path) else None


def when_label(q, clock):
    q = int(q)
    return f"{'Q' + str(q) if q <= 4 else 'OT' + str(q - 4)} {clock}"


def mmss(sec):
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def final(r):
    return f"{int(r.away_final)}–{int(r.home_final)}"


def winner(r):
    return r.away if int(r.away_final) > int(r.home_final) else r.home


def comeback_title(r):
    return f"{r.tied_by} CAME BACK FROM {int(r.max_deficit)} DOWN"


def comeback_detail(r):
    """Second line for the comeback card: where the hole was, and who won."""
    won = winner(r) == r.tied_by
    hi, lo = sorted((int(r.away_final), int(r.home_final)), reverse=True)
    q = getattr(r, "deficit_quarter", None)
    where = f" AT {when_label(q, r.deficit_clock)}" if q is not None and pd.notna(q) and isinstance(r.deficit_clock, str) else ""
    result = f"WON {hi}–{lo}" if won else f"STILL LOST {lo}–{hi}"
    return f"BIGGEST HOLE: {int(r.max_deficit)}{where} · {result}{' IN OT' if bool(r.ot) else ''}"


def when_of(r):
    """Archive games have no date, only a season; live-season games have both."""
    return r.date if isinstance(r.date, str) and r.date else r.season


# (heading, column, biggest-is-best, what to say about the winner)
RECORDS = [
    ("⏰ PREMATURE · EARLIEST SCOREGASM", "elapsed_min", False, lambda r: f"{when_label(r.quarter, r.clock)}, {r.elapsed_min:.1f} minutes in"),
    ("🌙 LATE BLOOMER · LATEST SCOREGASM", "elapsed_min", True, lambda r: f"{when_label(r.quarter, r.clock)}, {r.elapsed_min:.1f} minutes in"),
    ("🧟 BACK FROM THE DEAD (WILD CARD)", "max_deficit", True, comeback_title),
]
MEDAL = ["🥇", "🥈", "🥉"]


def tie_badge(r):
    """How long the 69-69 lasted, with a name for it."""
    t = getattr(r, "tie_sec", None)
    if t is None or pd.isna(t):
        return ""
    t = int(t)
    name = "ONE AND DONE" if t == 0 else "TANTRIC" if t >= 300 else "STAMINA" if t >= 180 else "LINGERED" if t >= 60 else ""
    return f"TIE HELD {mmss(t)}" + (f" · {name}" if name else "")


def record_cards(sp, n=3):
    """Top n games for each record. Equal values: the older game ranks first."""
    out = []
    sp = sp.assign(game_id=sp.game_id.astype(str)).sort_values(["season", "game_id"])
    for title, col, biggest, show in RECORDS:
        out.append(f'<div class="small" style="margin:18px 0 2px">{title}</div>')
        top = sp.sort_values(col, ascending=not biggest, kind="stable").head(n)
        for k, (_, r) in enumerate(top.iterrows()):
            out.append(f'<div class="sp HOF"><div class="spt">{MEDAL[k]} {show(r).upper()}</div>'
                       f'<div class="spg">{r.away} @ {r.home} · {when_of(r)}</div>'
                       f'<div class="sph">{r.how}</div>'
                       f'{"<div class=small>" + comeback_detail(r) + "</div>" if col == "max_deficit" else ""}'
                       f'<div class="small">FINAL {final(r)}{" · " + tie_badge(r) if tie_badge(r) else ""}</div></div>')
    return "".join(out)


def scoregasm_table(sp):
    t = sp.assign(game_id=sp.game_id.astype(str)).sort_values(["season", "game_id"], ascending=False)
    return pd.DataFrame({"When": [when_of(r) for r in t.itertuples()], "Game": t.away + " @ " + t.home,
                         "Hit at": [when_label(q, c) for q, c in zip(t.quarter, t.clock)],
                         "Comeback": t.max_deficit.astype(int), "Tie held": [tie_badge(r).replace("TIE HELD ", "") for r in t.itertuples()], "Story": t.how, "Final": [final(r) for r in t.itertuples()]})


def tease_board(full, min_events):
    """Per team: scoregasms and edged games they were in. Finish rate = scoregasms / (scoregasms + edged)."""
    rows = []
    if not len(full) or "away" not in full:
        return pd.DataFrame(), pd.DataFrame()
    for t in sorted(set(full.away.dropna()) | set(full.home.dropna())):
        mine = full[(full.away == t) | (full.home == t)]
        sg, ed = int((mine.result == "scoregasm").sum()), int((mine.result == "edged").sum())
        if sg + ed:
            rows.append({"Team": f"{t} · {TEAM_NAMES.get(t, t)}", "code": t, "Scoregasms": sg, "Edged": ed, "Finish rate": sg / (sg + ed)})
    df = pd.DataFrame(rows)
    return df, (df[df.Scoregasms + df.Edged >= min_events] if len(df) else df)


def tease_cards(ranked):
    if not len(ranked):
        return ""
    tease, closer = ranked.sort_values("Finish rate").iloc[0], ranked.sort_values("Finish rate", ascending=False).iloc[0]
    card = lambda k, head, line, r: (f'<div class="sp HOF"><div class="spt">{head}</div><div class="spg">{r.Team}</div>'
                                     f'<div class="sph">{line}</div><div class="small">{int(r.Scoregasms)} scoregasms · {int(r.Edged)} edged · finishes {r["Finish rate"] * 100:.0f}% of the time</div></div>')
    return (card(0, "😈 THE LEAGUE\'S BIGGEST TEASE", "Keeps getting right to the edge. Rarely closes.", tease)
            + card(1, "🎯 THE CLOSER", "Gets there and finishes. Reliable.", closer))


def edged_cards_html(ed, target, n=3):
    """The closest calls: smallest gap, then the longest tease."""
    out = []
    for _, r in ed.sort_values(["closest_gap", "wait_sec"], ascending=[True, False]).head(n).iterrows():
        out.append(f'<div class="sp EDGED"><div class="spt">💔 {GAP_LABEL[int(r.closest_gap)].upper()} · SAT ON {target} FOR {mmss(r.wait_sec)}</div>'
                   f'<div class="spg">{r.away} @ {r.home} · {when_of(r)}</div><div class="sph">{r.how}</div>'
                   f'<div class="small">FINAL {final(r)}</div></div>')
    return "".join(out)


def edged_table(ed, target, max_gap=3):
    t = ed[ed.closest_gap <= max_gap].assign(game_id=lambda d: d.game_id.astype(str)).sort_values(["season", "game_id"], ascending=False)
    return pd.DataFrame({
        "When": [when_of(r) for r in t.itertuples()], "Game": t.away + " @ " + t.home, "Close": t.closest_gap.astype(int).map(GAP_LABEL), f"Sat on {target}": t.leader,
        "Other got to": t.closest_score.astype(int), "Waited": t.wait_sec.map(mmss), "Ended": t.ending.map(ENDED), "Story": t.how,
        "Final": [final(r) for r in t.itertuples()]})


CODE_FIX = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
TEAM_NAMES = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls",
              "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
              "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
              "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
              "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
              "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
              "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"}


def filter_games(g, team, season):
    """Keep games involving one team (home or away) and/or from one season."""
    if team and len(g):
        g = g[(g.away == team) | (g.home == team)]
    if season and len(g):
        g = g[g.season.astype(str) == season]
    return g


def filter_bar(full, key):
    """Team and season dropdowns. Returns (team or None, season or None)."""
    c1, c2 = st.columns(2)
    codes = sorted(set(full.away.dropna()) | set(full.home.dropna())) if len(full) and "away" in full else []
    seasons = sorted(full.season.dropna().astype(str).unique(), reverse=True) if len(full) else []
    team = c1.selectbox("Team", ["All teams"] + codes, key=f"{key}_team",
                        format_func=lambda c: c if c == "All teams" else f"{c} · {TEAM_NAMES.get(c, c)}")
    season = c2.selectbox("Season", ["All seasons"] + seasons, key=f"{key}_season")
    return (None if team == "All teams" else team), (None if season == "All seasons" else season)


def games_for(scope):
    """All-time = the archive (1996 on) plus every live-season game. Since launch = live-season games only."""
    season = load_csv("scoregasm_season.csv")
    season = season[season.date >= LAUNCH] if season is not None else None
    if scope == "pre":                                          # preseason games, for testing only (never counted in the records)
        pre = load_csv("scoregasm_preseason.csv")
        if pre is None:
            return pd.DataFrame(columns=["game_id", "result"])
        return pre.assign(**{c: pre[c].replace(CODE_FIX) for c in ("away", "home", "tied_by", "first_to", "leader", "other") if c in pre})
    if scope == "launch":
        if season is None:
            return pd.DataFrame(columns=["game_id", "result"])
        return season.assign(**{c: season[c].replace(CODE_FIX) for c in ("away", "home", "tied_by", "first_to", "leader", "other") if c in season})
    frames = [f for f in (load_csv("scoregasm_archive.csv"), season) if f is not None]
    if not frames:
        return pd.DataFrame(columns=["game_id", "result"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    for c in ("away", "home", "tied_by", "first_to", "leader", "other"):   # one code per team (ESPN, Basketball-Reference and NBA.com differ)
        if c in df:
            df[c] = df[c].replace(CODE_FIX)
    extra = load_csv("deficit_times.csv")                       # when the biggest hole was reached, for older archive games
    if extra is not None:
        if "deficit_quarter" not in df:
            df["deficit_quarter"], df["deficit_clock"] = float("nan"), None
        m = extra.drop_duplicates("game_id").set_index("game_id")
        fill = df.deficit_quarter.isna() & df.game_id.isin(m.index)
        df.loc[fill, "deficit_quarter"] = df.loc[fill, "game_id"].map(m.deficit_quarter)
        df.loc[fill, "deficit_clock"] = df.loc[fill, "game_id"].map(m.deficit_clock)
    return df


def list_view():
    """Every scoregasm ever logged, newest first, 40 at a time."""
    full = games_for("all")
    hits = full[full.result == "scoregasm"] if len(full) else full
    st.markdown(f'<div class="small" style="margin:2px 0 8px">📜 EVERY SCOREGASM, NEWEST FIRST · {len(hits):,} AND COUNTING</div>', unsafe_allow_html=True)
    team, season = filter_bar(hits, "list")
    hits = filter_games(hits, team, season)
    if not len(hits):
        return st.info("🪱 Nothing here yet. Try another team or season.")
    for c in ("date",):
        if c not in hits:
            hits = hits.assign(**{c: ""})
    hits = hits.assign(game_id=hits.game_id.astype(str), date=hits.date.fillna("").astype(str)).sort_values(
        ["season", "date", "game_id"], ascending=False)
    key = f"list_n_{team}_{season}"
    n = st.session_state.get(key, 40)
    cards = []
    for _, r in hits.head(n).iterrows():
        badge = tie_badge(r)
        cards.append(f'<div class="sp"><div class="spt">{when_label(r.quarter, r.clock)} · {r.elapsed_min:.1f} MIN IN{" · " + badge if badge else ""}</div>'
                     f'<div class="spg">{r.away} @ {r.home} · {when_of(r)}</div><div class="sph">{r.how}</div>'
                     f'<div class="small">FINAL {final(r)}</div></div>')
    st.markdown("".join(cards), unsafe_allow_html=True)
    if len(hits) > n:
        st.caption(f"Showing {n} of {len(hits):,}.")
        if st.button("Show 40 more", key=f"more_{key}"):
            st.session_state[key] = n + 40
            st.rerun()


def scope_bar(key, preseason=True):
    """All-time / since launch / preseason switch. Returns 'all', 'launch' or 'pre'."""
    opts = ["🌍 All-time", "🚀 Since launch (2026)"] + (["🧪 Preseason (testing)"] if preseason else [])
    pick = st.radio("Scope", opts, horizontal=True, label_visibility="collapsed", key=f"{key}_scope")
    return "launch" if pick.startswith("🚀") else "pre" if pick.startswith("🧪") else "all"


def scope_notes(scope):
    if scope == "pre":
        st.caption("PRESEASON TEST VIEW: these games are not in the records or the all-time view. Use it to check the logger is working before opening night.")
    elif scope == "all":
        st.caption("All-time = every game in the archive, plus this season. Older games show a season, not a date.")


@st.fragment(run_every=60)
def log_view():
    """Hall of Fame: the records."""
    st.markdown('<div class="small" style="margin:2px 0 8px">🏆 THE HALL OF FAME · WHERE THE LEGENDS GOT TIED</div>', unsafe_allow_html=True)
    scope = scope_bar("hof")
    full = games_for(scope)
    team, season = filter_bar(full, "hof")
    filtered = bool(team or season)
    g = filter_games(full, team, season)
    hits = g[g.result == "scoregasm"]
    ed = g[g.result == "edged"]
    if filtered:                                             # older "no tie" games have no team names, so a game count would be wrong
        c1, c2 = st.columns(2)
        c1.metric("Scoregasms 69–69", f"{len(hits):,}")
        c2.metric("Left hanging", f"{len(ed):,}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Games logged", f"{len(g):,}")
        c2.metric("Scoregasms 69–69", f"{len(hits):,}")
        c3.metric("Share of games", f"{len(hits) / len(g) * 100:.1f}%" if len(g) else "–", f"model at tip-off: {tp(0, 0, 48.0, 69) * 100:.1f}%",
                  delta_color="off")
    scope_notes(scope)
    if not len(hits):
        if filtered:
            return st.info("🪱 Nothing found. Try another team or season.")
        if scope == "pre":
            return st.info("🪱 No preseason games logged yet. Run: python3 update_log.py --preseason --since 2026-10-01")
        return st.info("🪱 Flaccid. Nothing here yet." + (" The first scoregasm of 2026–27 takes every record. Updated daily." if scope == "launch"
                                           else " Build the archive with build_archive.py, then put scoregasm_archive.csv next to this app."))
    st.markdown(record_cards(hits), unsafe_allow_html=True)


@st.fragment(run_every=60)
def hanging_view():
    """Left hanging: games where a team sat on 69 and the tie never came."""
    st.markdown('<div class="small" style="margin:2px 0 8px">🥵 LEFT HANGING · SO CLOSE, NO TIE</div>', unsafe_allow_html=True)
    scope = scope_bar("hang")
    full = games_for(scope)
    team, season = filter_bar(full, "hang")
    ed = filter_games(full, team, season)
    ed = ed[ed.result == "edged"] if len(ed) else ed
    scope_notes(scope)
    st.caption("LEFT HANGING: a team reached exactly 69, the other got within 3 points while it sat there, and the tie never happened.")
    if not len(ed):
        return st.info("Nobody left hanging yet.")
    e1, e2 = st.columns(2)
    e1.metric("Left hanging", f"{len(ed):,}")
    e2.metric("Within 2 points", f"{int((ed.closest_gap <= 2).sum()):,}")
    st.markdown('<div class="small" style="margin:6px 0">CLOSEST CALLS</div>', unsafe_allow_html=True)
    st.markdown(edged_cards_html(ed, 69), unsafe_allow_html=True)
    pick = st.selectbox("How close", ["Within 3 (all)", "Within 2", "Exactly 1 away"], key="hang_close")
    st.dataframe(edged_table(ed, 69, {"Within 3 (all)": 3, "Within 2": 2, "Exactly 1 away": 1}[pick]),
                 hide_index=True, use_container_width=True, column_config={"Story": st.column_config.TextColumn(width="large")})


@st.fragment(run_every=60)
def tease_view():
    """Team tease board: who finishes and who doesn't."""
    st.markdown('<div class="small" style="margin:2px 0 8px">😈 TEAM TEASE BOARD · WHO FINISHES, WHO DOESN\'T</div>', unsafe_allow_html=True)
    scope = scope_bar("tease", preseason=False)
    full = games_for(scope)
    st.caption("Every team ranked by how often their 69-69 chances actually finish. Left hanging = so close, but no tie.")
    allt, ranked = tease_board(full, 8 if scope == "all" else 2)
    if not len(allt):
        return st.info("No teams to rank yet.")
    st.markdown(tease_cards(ranked), unsafe_allow_html=True)
    show = allt.drop(columns="code").sort_values(["Finish rate", "Edged"], ascending=[False, True])
    show["Finish %"] = (show.pop("Finish rate") * 100).round().astype(int)
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={"Finish %": st.column_config.ProgressColumn("Finish %", min_value=0, max_value=100, format="%d%%")})


tab_board, tab_game, tab_log, tab_list, tab_hang, tab_tease = st.tabs(
    ["📺 Tonight", "🧮 Check a game", "🏆 Hall of Fame", "📜 Every scoregasm", "🥵 Left hanging", "😈 Tease board"])
with tab_board:
    board_view()
with tab_game:
    game_view()
with tab_log:
    log_view()
with tab_list:
    list_view()
with tab_hang:
    hanging_view()
with tab_tease:
    tease_view()

with st.expander("Show me the math (be gentle)"):
    st.markdown(
        "**The odds.** The model plays out the rest of the game one possession at a time, thousands of ways, and counts how "
        "often it lands on exactly the target tie (free-throw trips included, since a team can sit on the number mid-trip). "
        "The team that's behind scores a touch faster and the leader a touch slower, like real games.\n\n"
        "**The Chub meter** is a display scale, not a probability: the odds on a log scale (1% is about 37, 3% about 55, "
        "10% about 74, 30% about 92), multiplied by how far the game is along the road to the target. That's why 10-10 "
        "sits near the bottom even though the odds look similar, and why the meter climbs as the score heads toward 69. "
        "🪱 Limp 0 · 👀 Peeping 20 · 🌭 Half chub 40 · 🔥 Close 60 · 🎆 Scoregasm 80.\n\n"
        "**Accuracy.** Two checks on real games, 12,805 of them from 2015-16 through 2024-25.\n\n"
        "*Start of the game:* the model says a game has a 3.6% chance of passing through 69–69. In real life it happened "
        "422 times in 12,805 games, or 3.3%. From 2015 to 2020 it was 3.5%, and from 2020 to 2025 it was 3.1%, a bit "
        "under the model. The gap could be luck (about a 6% chance of that) or a real shift in how the league plays. "
        "We tested several explanations and none held up, so it stays a mystery.\n\n"
        "*During the game:* I replayed games moment by moment, with the model using only the score and clock at that "
        "moment. On 2018-21 games, forecasts of about 3% hit 3.1% of the time (9,272 of 296,350) and 5% forecasts hit "
        "5.3%. Then I tested it on 2015-18, three seasons it had never seen, across 1.78 million forecasts: predicted "
        "2.07% overall, actual 2.02%. Odds from about 2% to 16% were accurate. Very low odds (under 2%) ran a bit high on "
        "those older seasons, and odds above 15% are noisy because there are few of them. A single forecast is never "
        "right or wrong, only the pooled record can be judged. The same game appears in many forecasts.\n\n"
        "*Not in the model:* team strength, injuries, rest, and the point spread. We tested team strength and spread-like "
        "ratings and found no measurable gain.")
st.markdown('<div class="foot">Checked against 12,805 real NBA games (2015–25): 69–69 happens in about 3.3% of games, '
            'the model says 3.6%. Regulation only. Just for fun, not betting advice. Please finish responsibly. '
            'If your tie lasts more than 4 hours, consult a physician. (It won\'t. It\'s one possession.)</div>', unsafe_allow_html=True)
