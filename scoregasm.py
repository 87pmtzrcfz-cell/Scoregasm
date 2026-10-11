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
.tag::after{content:"▮";margin-left:6px;color:var(--teal);animation:cursor 1s steps(1) infinite}
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
/* ===== polish: calmer motion, rounder cards, one consistent spacing system ===== */
.block-container{max-width:780px;padding-top:2rem;padding-bottom:3rem}
.stApp::before{opacity:.18}
.stApp::after{display:none}
.rn,.team .sc{animation:glitch 14s infinite}
div[data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid rgba(255,255,255,.08);margin:4px 0 18px;overflow-x:auto}
div[data-baseweb="tab-border"]{display:none}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--red),var(--teal))!important;height:2px!important}
button[data-baseweb="tab"]{background:transparent!important;padding:10px 14px!important}
button[data-baseweb="tab"] p{color:var(--mute);font-size:.68rem!important;letter-spacing:.16em}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--hot);text-shadow:0 0 10px rgba(255,46,151,.55)}
.grow,.sp,.recap,.blurb,.card{border-radius:14px;border-color:rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.008))}
.sp.HOF{border-color:rgba(255,46,151,.28);box-shadow:0 12px 32px rgba(0,0,0,.45)}
.card{border-color:rgba(255,46,151,.25);box-shadow:0 20px 50px rgba(0,0,0,.5),0 0 40px rgba(255,46,151,.12)}
.seg{border-radius:10px}.pill,.alert,.party,.warn{border-radius:999px}
.ticker{border-color:rgba(255,255,255,.08);border-radius:10px;padding:9px 12px}
.sp:hover,.grow:hover{border-color:rgba(25,227,216,.45);box-shadow:0 0 22px rgba(25,227,216,.18)}
.intro{padding:18px 20px 8px;margin:0 0 18px;border:1px solid rgba(255,255,255,.08);border-radius:16px;background:linear-gradient(180deg,rgba(255,46,151,.06),rgba(255,255,255,.01) 60%)}
.intro .ttl{font-family:'Audiowide',sans-serif;font-size:1.3rem;letter-spacing:.06em;color:var(--hot);text-shadow:0 0 12px rgba(255,46,151,.5)}
.intro .lead{color:#d8d2e6;font-size:.95rem;line-height:1.6;font-weight:300;margin:6px 0 10px}
.intro .lead b,.intro .pt b{color:#fff;font-weight:500}
.intro .pt{padding:10px 0;border-top:1px solid rgba(255,255,255,.06);font-size:.86rem;color:#c9c2d8;line-height:1.55}
.sec{font-size:.62rem;letter-spacing:.3em;color:var(--teal);font-weight:500;margin:26px 0 10px;text-shadow:0 0 8px rgba(25,227,216,.4)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:4px 0 6px}
.kpi{padding:14px 16px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.02)}
.kpi .k{font-size:.56rem;letter-spacing:.24em;color:var(--mute);font-weight:500}
.kpi .v{font-family:'Audiowide',sans-serif;font-size:1.75rem;color:var(--hot);line-height:1.15;margin-top:4px}
.kpi .s{font-size:.74rem;color:#a9a3bb;margin-top:2px;line-height:1.4}
.kpi.good .v{color:#d6fffb;text-shadow:0 0 12px rgba(25,227,216,.6)}
.srow{display:grid;grid-template-columns:62px 1fr 92px;align-items:center;gap:10px;margin:7px 0;font-size:.78rem;color:#d8d2e6}
.srow .yr{font-family:'JetBrains Mono',monospace;color:var(--mute)}
.srow .val{text-align:right;font-family:'JetBrains Mono',monospace}
.track{position:relative;height:10px;border-radius:99px;background:rgba(255,255,255,.05)}
.track i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:linear-gradient(90deg,var(--red),#b46bff);box-shadow:0 0 10px rgba(255,46,151,.5)}
.track u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--teal);box-shadow:0 0 8px var(--teal)}
.legend{font-size:.7rem;color:var(--mute);margin:2px 0 4px}.legend span{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 6px -1px 12px}
.steps{counter-reset:s;margin:0;padding:0}
.step{position:relative;padding:12px 14px 12px 50px;margin:8px 0;border:1px solid rgba(255,255,255,.07);border-radius:14px;background:rgba(255,255,255,.015);font-size:.86rem;line-height:1.55;color:#c9c2d8}
.step::before{counter-increment:s;content:counter(s);position:absolute;left:14px;top:12px;width:24px;height:24px;border-radius:50%;display:grid;place-items:center;font-family:'Audiowide',sans-serif;font-size:.75rem;color:#000;background:linear-gradient(135deg,var(--red),var(--teal))}
.step b{color:#fff;font-weight:500}
.cmp{padding:12px 16px;margin:8px 0;border:1px solid rgba(255,255,255,.07);border-radius:14px;background:rgba(255,255,255,.015)}
.cmp .t{font-size:.84rem;color:#fff;margin-bottom:8px}.cmp .n{font-size:.72rem;color:var(--mute);margin-top:6px;line-height:1.5}
.cmp .r{display:grid;grid-template-columns:70px 1fr 56px;gap:10px;align-items:center;font-size:.72rem;color:var(--mute);margin:4px 0}
.cmp .r b{text-align:right;font-family:'JetBrains Mono',monospace;color:#fff;font-weight:400}
.cmp .r .track i.m{background:linear-gradient(90deg,#19e3d8,#7ef3ec);box-shadow:0 0 10px rgba(25,227,216,.5)}
.prose{font-size:.88rem;line-height:1.65;color:#c9c2d8}.prose b{color:#fff;font-weight:500}
@media (max-width:520px){.srow{grid-template-columns:52px 1fr 80px}.intro{padding:16px 16px 6px}}
@media (prefers-reduced-motion:reduce){.stApp::after,.stApp::before{display:none}*{animation:none!important;transition:none!important}}
</style>
"""

st.set_page_config(page_title="scoregasm", page_icon="🌴", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="logo">SCOREGASM</div><div class="tag">the odds of finishing together</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#8f8aa3;font-size:.95rem;letter-spacing:.08em;font-weight:300;margin:2px 0 18px">'
            'Two teams. One number. <b style="color:#fff">69–69.</b> Not 70, not 68.</div>', unsafe_allow_html=True)
target = 69   # Scoregasm is about one number


def get_games_safe():
    """Games from ESPN. Keeps the last good copy, so a hiccup shows a warning instead of a blank page."""
    try:
        feed = fetch_feed()
        st.session_state["last_good"], st.session_state["feed_error"] = feed, False
    except Exception:
        feed = st.session_state.get("last_good")
        st.session_state["feed_error"] = True
        if feed is None:
            st.warning("Couldn't reach the score feed. Try again in a minute.")
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
        parts.append(f"🥵 {near} near misses")
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


def intro(title, what, points):
    """Intro box at the top of a tab: a title, one plain sentence, then one line per thing to know."""
    pts = "".join(f'<div class="pt">{p}</div>' for p in points)
    st.markdown(f'<div class="intro"><div class="ttl">{title}</div><div class="lead">{what}</div>{pts}</div>', unsafe_allow_html=True)


@st.fragment(run_every=15)  # re-checks the feed every 15 seconds
def board_view():
    intro("📺 TONIGHT", "Every NBA game on tonight, with the odds that it hits <b>exactly 69–69</b> at some point. Updates by itself every few seconds.",
          ["🎆 <b>Chub meter</b> (the big number, 0–100): how worked up the hottest game is. Higher = closer to a 69–69.",
           "📈 <b>The percent</b>: the actual chance this game passes through 69–69, from the score right now.",
           "📋 <b>Game list</b>: every game tonight, hottest first. Finished games show 🎆 if they went 69–69."])
    games = get_games_safe()
    if not games:
        return st.info("No games found right now. Check back closer to tip-off.")
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
    if games:
        st.markdown(tonight_html(games, st_by_id, target), unsafe_allow_html=True)


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
    ("🍑 COMING FROM BEHIND", "max_deficit", True, comeback_title),
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


def tease_board(full, min_chances):
    """Per team: a 'chance' is a game where one side sat on exactly 69 and the other got within 3 points.
    Scoregasm = the chance became a 69-69 tie. Near miss = it didn't. Conversion = scoregasms / chances."""
    if not len(full) or "away" not in full:
        return pd.DataFrame(), pd.DataFrame(), 0.0
    rows = []
    for t in sorted(set(full.away.dropna()) | set(full.home.dropna())):
        mine = full[(full.away == t) | (full.home == t)]
        sg, nm = int((mine.result == "scoregasm").sum()), int((mine.result == "edged").sum())
        if sg + nm:
            rows.append({"Team": f"{t} · {TEAM_NAMES.get(t, t)}", "Chances": sg + nm, "Scoregasms": sg, "Near misses": nm, "Conversion": sg / (sg + nm)})
    df = pd.DataFrame(rows)
    sg_all, nm_all = int((full.result == "scoregasm").sum()), int((full.result == "edged").sum())
    league = sg_all / (sg_all + nm_all) if sg_all + nm_all else 0.0
    return df, (df[df.Chances >= min_chances] if len(df) else df), league


def leaderboard_html(title, sub, rows, flame):
    """rows: list of (team label, big value text, bar fraction 0-1, small note). First row glows."""
    out = [f'<div class="sp HOF" style="margin-top:14px"><div class="spt">{title}</div><div class="small" style="margin:2px 0 10px">{sub}</div>']
    for k, (team, val, frac, note) in enumerate(rows):
        glow = "text-shadow:var(--glow)" if k == 0 else "opacity:.88"
        out.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:9px 0">'
            f'<div style="font-family:Audiowide,sans-serif;font-size:1.1rem;width:26px;color:var(--teal)">{k + 1 if k else flame}</div>'
            f'<div style="flex:1;min-width:0"><div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">'
            f'<span style="font-family:Audiowide,sans-serif;font-size:1rem;letter-spacing:.04em;color:var(--hot);{glow}">{team}</span>'
            f'<span style="font-family:Audiowide,sans-serif;font-size:1.15rem;color:var(--hot);{glow}">{val}</span></div>'
            f'<div style="height:5px;margin:4px 0 3px;background:rgba(255,46,151,.14)"><div style="height:5px;width:{max(frac, .03) * 100:.0f}%;'
            f'background:linear-gradient(90deg,var(--red),var(--teal));box-shadow:0 0 10px rgba(255,46,151,.7)"></div></div>'
            f'<div class="small">{note}</div></div></div>')
    out.append("</div>")
    return "".join(out)


def tease_boards(allt, ranked, n=5):
    """Three leaderboards: most scoregasms, best conversion (closers), worst conversion (teases)."""
    short = lambda t: t.split(" · ")[0]
    most = allt.sort_values(["Scoregasms", "Conversion"], ascending=False).head(n)
    top = max(int(most.Scoregasms.iloc[0]), 1)
    html = leaderboard_html("🎆 MOST SCOREGASMS", "Who has been in the most 69–69s",
                            [(short(r.Team), f"{int(r.Scoregasms)}", r.Scoregasms / top, f"{r.Team.split(' · ')[1]}") for r in most.itertuples()], "🎆")
    if len(ranked) >= 2:
        best = ranked.sort_values(["Conversion", "Chances"], ascending=False).head(n)
        worst = ranked.sort_values(["Conversion", "Chances"], ascending=[True, False]).head(n)
        mk = lambda d: [(short(r.Team), f"{r.Conversion * 100:.0f}%", r.Conversion, f"{int(r.Scoregasms)} of {int(r.Chances)} chances") for r in d.itertuples()]
        html += leaderboard_html("🎯 THE CLOSERS", "Gets close, then finishes", mk(best), "🎯")
        html += leaderboard_html("😈 THE TEASES", "Gets close, then leaves you hanging", mk(worst), "😈")
    return html


CODE_FIX = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
TEAM_NAMES = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls",
              "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
              "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
              "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
              "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
              "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
              "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"}


def games_for(scope="all"):
    """Every game in the archive plus every live-season game."""
    season = load_csv("scoregasm_season.csv")
    season = season[season.date >= LAUNCH] if season is not None else None
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


@st.fragment(run_every=60)
def log_view():
    """Hall of Fame: the records."""
    intro("🏆 HALL OF FAME", "The wildest games that ever went <b>exactly 69–69</b> (since 2015), ranked by record.",
          ["⏰ <b>Earliest</b>: tied at 69 with the least time played. Fastest finish ever.",
           "🌙 <b>Latest</b>: tied at 69 deep into the game. A slow burn.",
           "🍑 <b>Coming from behind</b>: the team that dug out of the biggest hole to get there, and whether they won.",
           "⏱️ <b>Tie held</b> = how long the score stayed 69–69 before someone scored."])
    hits = games_for("all")
    hits = hits[hits.result == "scoregasm"] if len(hits) else hits
    if not len(hits):
        return st.info("🪱 Nothing here yet.")
    st.markdown(record_cards(hits), unsafe_allow_html=True)


@st.fragment(run_every=60)
def tease_view():
    """Leaderboards: who finishes and who just teases."""
    allt, ranked, league = tease_board(games_for("all"), 8)
    if not len(allt):
        return st.info("No teams to rank yet.")
    intro("😈 TEASE BOARD", "Which teams actually finish a 69–69, and which ones just tease. Three leaderboards, top 5 each.",
          ["🔥 <b>A chance</b>: a team reaches exactly 69 points while the other team is within 3 points of it (66, 67 or 68). "
           "Either side could tie it with one more bucket or free throw, so the tie was right there.",
           "🎆 <b>Scoregasm</b>: the chance pays off. The trailing team gets to 69 too and the game is tied 69–69. It finished.",
           "😈 <b>Tease</b>: the chance doesn't pay off. The trailing team never lands on 69 (it jumps past it, or the leader scores again). Left hanging.",
           f"📊 <b>The percent</b>: scoregasms ÷ chances, so 40% means 4 of every 10 chances finished. League average: {league * 100:.0f}%. "
           "A team needs 8+ chances to make the percent boards."])
    st.markdown(tease_boards(allt, ranked), unsafe_allow_html=True)


def kpi(label, value, sub, good=False):
    return f'<div class="kpi{" good" if good else ""}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def compare_html(title, rows, note):
    """Predicted vs actual as two bars on the same scale."""
    top = max(v for _, v, _ in rows) * 1.25
    bars = "".join(f'<div class="r">{lab}<div class="track"><i class="{c}" style="width:{v / top * 100:.0f}%"></i></div><b>{v:.2f}%</b></div>'
                   for lab, v, c in rows)
    return f'<div class="cmp"><div class="t">{title}</div>{bars}<div class="n">{note}</div></div>'


@st.fragment(run_every=300)
def stats_view():
    """How the odds were tested, with the tip-off check recomputed from the data files every time."""
    model = tp(0, 0, 48.0, target) * 100
    full = games_for("all")
    reg = full[full.game_type.fillna("regular") == "regular"] if "game_type" in full else full
    intro("📊 THE STATS", "How we know the odds on this site are honest. The model was graded against "
          "<b>every real NBA game we could get play-by-play for</b>, and the toughest test used seasons it had never seen.",
          ["🎯 <b>Predicted</b>: what the model said would happen, before it saw how the games ended.",
           "🏀 <b>Actual</b>: what really happened, counted from every scoring play of every game.",
           "🔒 <b>Unseen seasons</b>: games kept locked away while the model was built, then used once as a final exam. "
           "If a model only works on games it was tuned on, this is where it gets caught.",
           "🎲 <b>Normal luck</b>: even a perfect model won't match exactly. The range shows how far off pure chance could push it."])
    if not len(reg):
        return st.info("No games to check yet.")
    n, k = len(reg), int((reg.result == "scoregasm").sum())
    real = k / n * 100
    exp, sd = n * model / 100, math.sqrt(n * model / 100 * (1 - model / 100))
    lo, hi = (exp - 2 * sd) / n * 100, (exp + 2 * sd) / n * 100
    inside = lo <= real <= hi
    st.markdown('<div class="sec">THE HEADLINE CHECK · AT TIP-OFF</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpis">'
                + kpi("GAMES CHECKED", f"{n:,}", f"Regular season, {reg.season.min()} to {reg.season.max()}")
                + kpi("MODEL SAYS", f"{model:.1f}%", "Chance a game passes through 69–69, before tip-off")
                + kpi("REALITY", f"{real:.1f}%", f"{k:,} games actually went 69–69")
                + kpi("VERDICT", "PASS" if inside else "CLOSE", f"Normal luck range: {lo:.1f}% to {hi:.1f}%"
                      + ("" if inside else ". Reality sits just outside it; see below."), good=inside)
                + '</div>', unsafe_allow_html=True)

    by = reg.groupby("season").result.agg(n="size", k=lambda r: int((r == "scoregasm").sum())).reset_index()
    by = by[by.n >= 300]
    if len(by):
        scale = max(6.0, (by.k / by.n * 100).max() * 1.15)
        rows = "".join(f'<div class="srow"><div class="yr">{r.season}</div><div class="track"><i style="width:{r.k / r.n * 100 / scale * 100:.1f}%"></i>'
                       f'<u style="left:{model / scale * 100:.1f}%"></u></div><div class="val">{r.k / r.n * 100:.1f}% · {r.k}</div></div>'
                       for r in by.itertuples())
        st.markdown('<div class="sec">SEASON BY SEASON</div><div class="legend"><span style="background:#ff2e97;margin-left:0"></span>'
                    f'share of games that went 69–69<span style="background:#19e3d8"></span>model ({model:.1f}%)</div>' + rows
                    + '<div class="legend" style="margin-top:8px">One season has about 1,230 games, so a single season can swing by about ±1 point from luck alone.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec">THE LIVE CHECK · DURING GAMES</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="prose">The tip-off number is only one forecast per game. The live odds change with every basket, so they were tested '
        'by replaying old games moment by moment. At each moment the model only saw the score and the clock, then we checked whether '
        'that game really went on to hit 69–69.</div>'
        + compare_html("🔒 Final exam: 2015–18 (seasons the model never saw)", [("Predicted", 2.07, "m"), ("Actual", 2.02, "")],
                       "1.78 million live forecasts. Off by 0.05 points overall.")
        + compare_html("When the model said about 3%", [("Predicted", 3.0, "m"), ("Actual", 3.1, "")],
                       "2018–21 games: 9,272 hits out of 296,350 forecasts.")
        + compare_html("When the model said about 5%", [("Predicted", 5.0, "m"), ("Actual", 5.3, "")],
                       "Same games. Forecasts between about 2% and 16% landed close to reality."),
        unsafe_allow_html=True)

    st.markdown('<div class="sec">HOW IT WAS TESTED · STEP BY STEP</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="steps">'
        '<div class="step"><b>Build the model from how basketball scores.</b> Every possession ends in 0, 1, 2 or 3 points '
        '(free throws and and-ones included) at league-average rates. The model plays out the rest of the game one possession at a time '
        'and adds up every path that lands on exactly 69–69. One small tweak, that the team behind scores slightly faster, was tuned on 2018–21 games.</div>'
        '<div class="step"><b>Collect real games.</b> Every scoring play from every regular-season game, 2015–16 through 2024–25, from public '
        'play-by-play records. A script replays each game and notes whether it ever sat at 69–69.</div>'
        '<div class="step"><b>Check the tip-off odds.</b> Compare the model\'s pregame number with how often games really went 69–69 (the headline check above).</div>'
        '<div class="step"><b>Check the live odds on unseen seasons.</b> Replay games moment by moment and compare forecasts with outcomes, '
        'grouped by forecast size. The final exam used 2015–18, which played no part in building the model.</div>'
        '<div class="step"><b>Try to beat it.</b> We tested fancier versions: updating to each season\'s scoring averages, a late-season effect, team '
        'strength, and a pure lookup table built from real games. None did better on unseen seasons, so the simple model stayed.</div>'
        '<div class="step"><b>Write predictions down first.</b> Every new theory gets a written prediction <i>before</i> looking at the data, '
        'so a lucky pattern can\'t sneak in after the fact.</div>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">WHAT IT DOESN\'T KNOW</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="prose">The model treats both teams as league average. It ignores team strength, injuries, rest and the betting spread '
        '(we tested strength and spread-style ratings and found no real gain). Overtime isn\'t modeled, only regulation. '
        'And recent seasons (2020–25) came in a bit under the model, about 3.1% versus 3.6%. That gap has roughly a 6% chance of being '
        'pure luck; none of the explanations we tested held up, so it stays an open question.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">ABOUT THE CHUB METER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="prose">The chub meter is a display scale, not a probability. It puts the odds on a log scale (1% ≈ 37, 3% ≈ 55, '
        '10% ≈ 74, 30% ≈ 92) and multiplies by how far the score has traveled toward 69. That\'s why 10–10 sits low even when the odds '
        'look similar, and why the meter climbs as both teams close in.<br>'
        '🪱 Limp 0 · 👀 Peeping 20 · 🌭 Half chub 40 · 🔥 Close 60 · 🎆 Scoregasm 80</div>', unsafe_allow_html=True)


tab_board, tab_log, tab_tease, tab_stats = st.tabs(["📺 Tonight", "🏆 Hall of Fame", "😈 Tease board", "📊 Stats"])
with tab_board:
    board_view()
with tab_log:
    log_view()
with tab_tease:
    tease_view()
with tab_stats:
    stats_view()

st.markdown('<div class="foot">SCOREGASM · Live scores from ESPN · Regulation only · Just for fun, not betting advice</div>', unsafe_allow_html=True)
