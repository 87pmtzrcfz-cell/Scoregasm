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


TAGLINES = {"SCOREGASM": "IT'S HAPPENING.", "CLOSE": "Just keep it going.",
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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Press+Start+2P&family=VT323&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');
.stApp{background:radial-gradient(900px 520px at 8% -8%,rgba(139,92,255,.30),transparent 60%),radial-gradient(760px 460px at 100% 4%,rgba(255,122,26,.17),transparent 58%),radial-gradient(700px 500px at 50% 112%,rgba(139,92,255,.14),transparent 60%),repeating-linear-gradient(115deg,rgba(255,255,255,.022) 0 14px,transparent 14px 46px),#07060d;color:#f4f1ff;font-family:'Space Grotesk',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none}
#MainMenu,footer{visibility:hidden}
[data-testid="stWidgetLabel"] p,label{color:#a79fc9!important;text-transform:uppercase;letter-spacing:.09em;font-size:.7rem!important;font-weight:700!important}
input{font-family:'JetBrains Mono',monospace!important;font-weight:700!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{background:#12101f!important;border-color:rgba(255,255,255,.14)!important;border-radius:12px!important}
button[data-baseweb="tab"] p{font-weight:700;letter-spacing:.04em;font-size:1rem}
.logo{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(64px,16vw,132px);line-height:.88;letter-spacing:.03em;margin:.1em 0 0;background:linear-gradient(95deg,#8b5cff,#c27bff 35%,#ff7a1a 75%,#ffb27a);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 24px rgba(139,92,255,.5));border-bottom:2px solid rgba(255,122,26,.6);display:inline-block;padding-bottom:6px}
@keyframes shift{to{background-position:300% 0}}
.tag{color:#a79fc9;letter-spacing:.36em;text-transform:uppercase;font-size:.72rem;margin:.7em 0 1.1em}
.ticker{overflow:hidden;white-space:nowrap;border-top:1px solid rgba(255,255,255,.12);border-bottom:1px solid rgba(255,255,255,.12);padding:9px 0;margin:6px 0 18px;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#cfc8ee}
.ticker div{display:inline-block;padding-left:100%;animation:scroll 38s linear infinite}
.ticker b{color:#fff}.ticker i{color:#ff7a1a;font-style:normal;margin:0 14px}
@keyframes scroll{to{transform:translateX(-100%)}}
.pill{display:inline-block;padding:5px 14px;border-radius:99px;background:rgba(255,122,26,.16);border:1px solid #ff7a1a;color:#ffc08a;font-weight:700;letter-spacing:.16em;font-size:.7rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff7a1a;margin-right:8px;box-shadow:0 0 10px #ff7a1a;animation:blink 1s ease-in-out infinite}
.board{display:flex;align-items:center;justify-content:center;gap:clamp(14px,5vw,40px);margin:8px 0 2px}
.team{text-align:center}
.team .ab{font-size:.75rem;letter-spacing:.25em;color:#a79fc9;font-weight:700}
.team .sc{font-family:'Bebas Neue',sans-serif;font-size:clamp(58px,15vw,96px);line-height:1;color:#fff;text-shadow:0 0 16px rgba(139,92,255,.45)}
.team .need{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#ffb27a}
.dash{font-family:'Bebas Neue',sans-serif;font-size:56px;color:#ff7a1a;text-shadow:0 0 16px rgba(255,122,26,.8)}
.clock{text-align:center;font-family:'JetBrains Mono',monospace;color:#cfc8ee;font-size:.85rem;letter-spacing:.12em;margin-bottom:6px}
.party{margin:10px 0;padding:14px;border-radius:12px;text-align:center;font-family:'Bebas Neue',sans-serif;font-size:1.9rem;letter-spacing:.08em;color:#fff;background:linear-gradient(90deg,#ff7a1a,#8b5cff,#ff7a1a);animation:blink 1s ease-in-out infinite}
.card{margin-top:14px;padding:24px 20px 20px;text-align:center;border-radius:14px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.015));border:1px solid rgba(255,255,255,.13);border-top:3px solid #ff7a1a;box-shadow:0 0 0 1px rgba(139,92,255,.18),0 20px 70px rgba(139,92,255,.28),inset 0 1px 0 rgba(255,255,255,.12)}
.card .q{color:#a79fc9;letter-spacing:.22em;text-transform:uppercase;font-size:.72rem;font-weight:700}
.alert{margin:-4px -4px 16px;padding:8px 12px;border-radius:8px;font-weight:700;letter-spacing:.16em;font-size:.8rem;color:#fff;background:linear-gradient(90deg,#ff7a1a,#8b5cff);animation:blink 1.1s ease-in-out infinite}
@keyframes blink{50%{opacity:.55;box-shadow:0 0 28px rgba(255,122,26,.9)}}
.duo{display:flex;align-items:center;justify-content:center;gap:clamp(12px,4vw,36px);flex-wrap:wrap;margin-top:6px}
.big{font-family:'Bebas Neue',Impact,sans-serif;font-size:clamp(72px,19vw,150px);line-height:.95;background:linear-gradient(180deg,#fff 0%,#ffc08a 36%,#ff7a1a 70%,#8b5cff 100%);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 28px rgba(139,92,255,.55));animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{50%{filter:drop-shadow(0 0 48px rgba(255,122,26,.95))}}
.lab{color:#7d76a0;font-size:.68rem;letter-spacing:.2em;font-weight:700}
.ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#ffb27a 0,#8b5cff calc(var(--v)*.5%),#ff7a1a calc(var(--v)*1%),rgba(255,255,255,.09) calc(var(--v)*1%));box-shadow:0 0 34px rgba(139,92,255,.5)}
.ringin{width:122px;height:122px;border-radius:50%;background:#0b0914;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rn{font-family:'Bebas Neue',sans-serif;font-size:62px;line-height:.9;color:#fff;text-shadow:0 0 16px rgba(255,122,26,.7)}
.rl{font-size:.62rem;letter-spacing:.3em;color:#ffb27a;font-weight:700}
.strip{display:flex;gap:6px;justify-content:center;margin-top:18px}
.seg{flex:1;max-width:96px;padding:7px 2px;border-radius:12px;border:1px solid rgba(255,255,255,.1);opacity:.38}
.seg.on{opacity:1;border-color:#ff7a1a;background:rgba(255,122,26,.14);box-shadow:0 0 20px rgba(255,122,26,.65)}
.seg b{display:block;font-size:1.35rem}.seg i{font-style:normal;font-size:.52rem;letter-spacing:.1em;color:#cfc8ee;font-weight:700}
.plain{color:#cfc8ee;font-size:.95rem;margin-top:12px}
.road{height:10px;border-radius:99px;background:rgba(255,255,255,.08);margin:16px 4px 6px;overflow:hidden}
.road i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#8b5cff,#c27bff,#ff7a1a);box-shadow:0 0 18px rgba(255,122,26,.7)}
.small{color:#7d76a0;font-size:.68rem;letter-spacing:.08em}
.foot{color:#6a6490;font-size:.72rem;line-height:1.5;margin-top:22px;text-align:center}
.grow{padding:12px 14px;margin:8px 0;border-radius:12px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.1)}
.grow.hit{border-color:#ff7a1a;animation:blink 1s ease-in-out infinite}
.grow.dim{opacity:.5}
.gtop{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.gm{font-family:'JetBrains Mono',monospace;font-size:1rem;color:#cfc8ee}.gm b{color:#fff;font-size:1.15rem}
.gs{color:#7d76a0;font-size:.72rem;letter-spacing:.1em;margin-top:2px}
.gr{text-align:right}.gp{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;line-height:1;color:#ffa05a}
.gc{font-size:.72rem;letter-spacing:.1em;color:#a79fc9;font-weight:700}
.gbar{height:6px;border-radius:99px;background:rgba(255,255,255,.08);margin-top:9px;overflow:hidden}
.gbar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#8b5cff,#c27bff,#ff7a1a)}
.sp{margin:10px 0;padding:14px 16px;border-radius:12px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.14)}
.sp.LEGENDARY{border-color:#ffd24a;box-shadow:0 0 28px rgba(255,210,74,.45)}
.sp.EPIC{border-color:#b36bff;box-shadow:0 0 22px rgba(179,107,255,.4)}
.sp.RARE{border-color:#4aa8ff;box-shadow:0 0 18px rgba(74,168,255,.3)}
.sp.UNCOMMON{border-color:#4be08a}
.sp.EDGED{border-color:#ff7a1a;box-shadow:0 0 22px rgba(255,122,26,.35)}
.sp.HOF{border-color:#ffd24a;box-shadow:0 0 26px rgba(255,210,74,.4)}
.sp.HOF .spt{color:#ffd24a}
.spt{font-size:.72rem;letter-spacing:.14em;font-weight:700;color:#ffd9a0}
.sp.EPIC .spt{color:#d9b3ff}.sp.RARE .spt{color:#a9d4ff}.sp.UNCOMMON .spt{color:#a8f0c6}.sp.EDGED .spt{color:#ffc08a}
.spg{font-family:'Bebas Neue',sans-serif;font-size:1.55rem;letter-spacing:.04em;color:#fff;margin-top:2px}
.spb{font-size:1.1rem;letter-spacing:.2em;margin:2px 0}
.sph{color:#cfc8ee;font-size:.9rem;line-height:1.45;margin:4px 0 6px}
.fresh{text-align:center;color:#7d76a0;font-size:.7rem;letter-spacing:.12em;font-weight:700;margin:-4px 0 10px}
.fresh.bad{color:#ffb05a}
.recap{margin:4px 0 14px;padding:10px 14px;border-radius:12px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);color:#e4defa;font-size:.9rem}
.recap b{color:#ffa05a;letter-spacing:.12em}
.tagl{text-align:center;font-family:'Bebas Neue',sans-serif;font-size:1.7rem;letter-spacing:.08em;color:#ffa05a;margin-top:10px;text-shadow:0 0 14px rgba(255,122,26,.55)}
.blurb{margin:4px 0 14px;padding:14px 16px;border-radius:12px;border-left:4px solid #ff7a1a;background:rgba(255,255,255,.04);color:#e4defa;font-size:.95rem;line-height:1.5}
.blurb b{color:#fff}.blurb .hd{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;letter-spacing:.06em;color:#ffa05a;line-height:1;margin-bottom:6px}

/* ===== RETRO ARCADE LAYER: pixel type, scanlines, hardwood floor, purple + orange ===== */
.stApp{background:radial-gradient(900px 520px at 8% -8%,rgba(139,92,255,.34),transparent 60%),radial-gradient(760px 460px at 100% 6%,rgba(255,122,26,.14),transparent 58%),#0a0614}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9999;background:repeating-linear-gradient(0deg,rgba(0,0,0,.20) 0 1px,transparent 1px 3px);mix-blend-mode:multiply}
.stApp::after{content:"";position:fixed;left:0;right:0;bottom:0;height:20vh;pointer-events:none;z-index:0;opacity:.55;background:repeating-linear-gradient(90deg,rgba(0,0,0,.35) 0 2px,transparent 2px 46px),repeating-linear-gradient(0deg,#3a1d0c 0 14px,#4a2610 14px 28px);-webkit-mask-image:linear-gradient(to top,#000,transparent);mask-image:linear-gradient(to top,#000,transparent)}
.logo{font-family:'Press Start 2P',monospace;font-size:clamp(26px,8.2vw,68px);line-height:1.15;letter-spacing:.02em;background:none;-webkit-background-clip:border-box;background-clip:border-box;color:#ffa43a;filter:none;border-bottom:0;padding-bottom:0;text-shadow:3px 3px 0 #6d28d9,6px 6px 0 #1b0b3a,0 0 26px rgba(139,92,255,.55)}
.tag{font-family:'Press Start 2P',monospace;font-size:.55rem;letter-spacing:.14em;color:#b9a6ff;margin:1.6em 0 1.3em}
.tag::after{content:"_";color:#ff7a1a;animation:blink 1s steps(1) infinite}
.ticker{font-family:'VT323',monospace;font-size:1.15rem;letter-spacing:.04em;border-top:2px solid #6d28d9;border-bottom:2px solid #6d28d9;background:rgba(0,0,0,.45);color:#d7c9ff}
.ticker b{color:#ffa43a}
.pill{font-family:'Press Start 2P',monospace;font-size:.5rem;letter-spacing:.06em;line-height:1.7;border-radius:0;border:2px solid #ff7a1a;background:#1a0d33;color:#ffb877;box-shadow:3px 3px 0 #6d28d9}
.board{background:#000;border:3px solid #6d28d9;box-shadow:0 0 0 3px #0a0614,0 0 0 5px #ff7a1a,0 0 30px rgba(139,92,255,.35);padding:14px 10px;margin:12px 0 6px}
.team .ab{font-family:'Press Start 2P',monospace;font-size:.55rem;color:#b9a6ff}
.team .sc{font-family:'VT323',monospace;font-size:clamp(72px,19vw,120px);color:#ffa43a;text-shadow:0 0 14px rgba(255,140,40,.75),0 0 2px #fff}
.team .need{font-family:'VT323',monospace;font-size:1.05rem;color:#d7c9ff}
.dash{font-family:'VT323',monospace;color:#8b5cff;text-shadow:0 0 14px rgba(139,92,255,.9)}
.clock{font-family:'VT323',monospace;font-size:1.25rem;color:#ffa43a;letter-spacing:.2em}
.card{border-radius:0;background:linear-gradient(160deg,#17102b,#0e0820);border:3px solid #6d28d9;border-top:3px solid #6d28d9;box-shadow:0 0 0 3px #0a0614,0 0 0 5px #ff7a1a,0 22px 60px rgba(139,92,255,.3)}
.card .q{font-family:'Press Start 2P',monospace;font-size:.5rem;line-height:1.8;letter-spacing:.12em}
.big{font-family:'Press Start 2P',monospace;font-size:clamp(34px,11.5vw,88px);line-height:1.1;background:none;-webkit-background-clip:border-box;background-clip:border-box;color:#ffa43a;filter:none;text-shadow:4px 4px 0 #6d28d9,8px 8px 0 #1b0b3a,0 0 30px rgba(255,140,40,.45);animation:pulse 2.6s steps(2) infinite}
@keyframes pulse{50%{transform:translateY(-3px)}}
.rn{font-family:'VT323',monospace;font-size:78px;line-height:.8;color:#ffa43a;text-shadow:0 0 14px rgba(255,140,40,.7)}
.rl{font-family:'Press Start 2P',monospace;font-size:.45rem;color:#b9a6ff}
.ring{box-shadow:0 0 0 3px #1b0b3a,0 0 30px rgba(139,92,255,.5)}
.seg{border-radius:0;border:2px solid #3b2a66;background:#120a24;opacity:.45}
.seg.on{opacity:1;border-color:#ff7a1a;background:#1f1038;box-shadow:3px 3px 0 #6d28d9,0 0 18px rgba(255,122,26,.5)}
.seg i{font-family:'Press Start 2P',monospace;font-size:.4rem;letter-spacing:.04em;line-height:1.6}
.tagl{font-family:'Press Start 2P',monospace;font-size:.8rem;line-height:1.7;letter-spacing:.04em;color:#ffa43a;text-shadow:3px 3px 0 #6d28d9}
.road,.gbar{border-radius:0;background:#120a24;border:2px solid #3b2a66;height:14px}
.road i,.gbar i{border-radius:0;background:repeating-linear-gradient(90deg,#ff7a1a 0 8px,transparent 8px 11px);box-shadow:none}
.gbar{height:10px}
.party{font-family:'Press Start 2P',monospace;font-size:.8rem;line-height:1.7;border-radius:0;border:3px solid #fff;background:repeating-linear-gradient(90deg,#8b5cff 0 24px,#ff7a1a 24px 48px);text-shadow:2px 2px 0 #000;animation:blink .7s steps(1) infinite}
.alert{font-family:'Press Start 2P',monospace;font-size:.55rem;border-radius:0;background:#ff7a1a;color:#1b0b3a}
.grow,.sp,.recap,.blurb{border-radius:0;background:#120a24;border:2px solid #3b2a66;box-shadow:3px 3px 0 #1b0b3a}
.grow.hit{border-color:#ff7a1a}
.sp.HOF{border-color:#ffa43a;box-shadow:3px 3px 0 #6d28d9,0 0 22px rgba(255,164,58,.3)}
.sp.EDGED{border-color:#8b5cff;box-shadow:3px 3px 0 #1b0b3a,0 0 18px rgba(139,92,255,.4)}
.spt{font-family:'Press Start 2P',monospace;font-size:.5rem;line-height:1.8;letter-spacing:.06em}
.spg{font-family:'VT323',monospace;font-size:1.9rem;letter-spacing:.03em}
.gm,.gp{font-family:'VT323',monospace}.gm{font-size:1.25rem}.gm b{font-size:1.4rem}.gp{font-size:2.3rem;color:#ffa43a}
.blurb .hd{font-family:'Press Start 2P',monospace;font-size:.7rem;line-height:1.6;color:#ffa43a}
.recap b{font-family:'Press Start 2P',monospace;font-size:.5rem}
button[data-baseweb="tab"] p{font-family:'Press Start 2P',monospace;font-size:.55rem!important;letter-spacing:.02em}
[data-testid="stWidgetLabel"] p,label{font-family:'Press Start 2P',monospace;font-size:.5rem!important;letter-spacing:.04em;line-height:1.7}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{border-radius:0!important;border:2px solid #3b2a66!important}
.warn{display:table;margin:-6px auto 14px 0;padding:6px 10px;border:2px solid #ff7a1a;background:#1a0d33;color:#ffb877;font-family:'Press Start 2P',monospace;font-size:.42rem;line-height:1.8;letter-spacing:.05em;box-shadow:3px 3px 0 #6d28d9}
.foot{font-family:'VT323',monospace;font-size:1rem;color:#8d80c4}
</style>
"""

st.set_page_config(page_title="scoregasm", page_icon="🏀", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="logo">SCOREGASM</div><div class="tag">the odds of finishing together</div>', unsafe_allow_html=True)
st.markdown('<div class="warn">🔞 VIEWER DISCRETION ADVISED · EXCESSIVE TIE-RELATED INNUENDO</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#a79fc9;font-size:1rem;letter-spacing:.04em;margin:2px 0 18px">'
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
    return f'<div class="fresh"><span class="dot"></span>LIVE FEED · UPDATED {ago} AGO</div>'


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
    return (f'<div class="card">{alert}<div class="q">chance of passing through {target}–{target}</div>'
            f'<div class="duo"><div><div class="big">{fmt_pct(p)}</div><div class="lab">CHANCE</div></div>'
            f'<div><div class="ring" style="--v:{score}"><div class="ringin"><div class="rn">{score}</div>'
            f'<div class="rl">CHUB</div></div></div><div class="lab" style="margin-top:8px">CHUB METER</div></div></div>'
            f'<div class="strip">{strip}</div><div class="tagl">{TAGLINES[label]}</div><div class="plain">{plain}</div>'
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
    st.markdown(feed_status_html() + recap_html(games), unsafe_allow_html=True)
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
    st.markdown('<div class="small" style="margin:22px 0 4px">ALL GAMES · WHO IS WARMING UP · CHUB METER = THE ODDS × HOW CLOSE THE SCORE IS TO 69–69 '
                '(HIGHER = HOTTER) · BAR = ROAD TO 69</div>', unsafe_allow_html=True)
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


def comeback_line(r):
    won = winner(r) == r.tied_by
    hi, lo = sorted((int(r.away_final), int(r.home_final)), reverse=True)
    score, ot = f"{hi}–{lo}", (" in OT" if bool(r.ot) else "")
    end = f"went on to win {score}{ot}" if won else f"still lost {lo}–{hi}{ot}"
    q = getattr(r, "deficit_quarter", None)
    at = f" ({when_label(q, r.deficit_clock)})" if q is not None and pd.notna(q) and isinstance(r.deficit_clock, str) else ""
    return f"{r.tied_by} was down {int(r.max_deficit)}{at}, tied it at 69–69, and {end}"


def when_of(r):
    """Archive games have no date, only a season; live-season games have both."""
    return r.date if isinstance(r.date, str) and r.date else r.season


# (heading, column, biggest-is-best, what to say about the winner)
RECORDS = [
    ("⏰ PREMATURE · EARLIEST SCOREGASM", "elapsed_min", False, lambda r: f"{when_label(r.quarter, r.clock)}, {r.elapsed_min:.1f} minutes in"),
    ("🌙 LATE BLOOMER · LATEST SCOREGASM", "elapsed_min", True, lambda r: f"{when_label(r.quarter, r.clock)}, {r.elapsed_min:.1f} minutes in"),
    ("🧟 BACK FROM THE DEAD (WILD CARD)", "max_deficit", True, comeback_line),
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
                       f'<div class="sph">{r.how}</div><div class="small">FINAL {final(r)}'
                       f'{" · " + tie_badge(r) if tie_badge(r) else ""}</div></div>')
    return "".join(out)


def scoregasm_table(sp):
    t = sp.assign(game_id=sp.game_id.astype(str)).sort_values(["season", "game_id"], ascending=False)
    return pd.DataFrame({"When": [when_of(r) for r in t.itertuples()], "Game": t.away + " @ " + t.home,
                         "Hit at": [when_label(q, c) for q, c in zip(t.quarter, t.clock)],
                         "Comeback": t.max_deficit.astype(int), "Tie held": [tie_badge(r).replace("TIE HELD ", "") for r in t.itertuples()], "Story": t.how, "Final": [final(r) for r in t.itertuples()]})


def tease_board(full, min_events):
    """Per team: scoregasms and edged games they were in. Finish rate = scoregasms / (scoregasms + edged)."""
    rows = []
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


def filter_games(g, team, text):
    """Keep games involving one team (home or away) and/or whose story, players or teams contain the search text."""
    if team and len(g):
        g = g[(g.away == team) | (g.home == team)]
    t = (text or "").strip().lower()
    if t and len(g):
        blob = (g.away.fillna("") + " " + g.home.fillna("") + " " + g.get("how", pd.Series("", index=g.index)).fillna("") + " "
                + g.get("play_by", pd.Series("", index=g.index)).fillna("") + " " + g.season.fillna("").astype(str) + " "
                + g.get("date", pd.Series("", index=g.index)).fillna("").astype(str)).str.lower()
        g = g[blob.str.contains(t, regex=False)]
    return g


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


@st.fragment(run_every=60)
def log_view():
    st.markdown('<div class="small" style="margin:2px 0 8px">🏆 THE HALL OF FAME · WHERE THE LEGENDS GOT TIED</div>', unsafe_allow_html=True)
    pick_scope = st.radio("Scope", ["🌍 All-time", "🚀 Since launch (2026)", "🧪 Preseason (testing)"], horizontal=True,
                          label_visibility="collapsed")
    scope = "launch" if pick_scope.startswith("🚀") else "pre" if pick_scope.startswith("🧪") else "all"
    full = games_for(scope)
    f1, f2 = st.columns([1, 1])
    codes = sorted(set(full.away.dropna()) | set(full.home.dropna())) if len(full) and "away" in full else []
    team = f1.selectbox("Team", ["All teams"] + codes, format_func=lambda c: c if c == "All teams" else f"{c} · {TEAM_NAMES.get(c, c)}")
    text = f2.text_input("Search", placeholder="Search a player, team or season (like LeBron or 2018-19)")
    team = None if team == "All teams" else team
    filtered = bool(team or text.strip())
    g = filter_games(full, team, text)
    n_games = len(g)
    hits = g[g.result == "scoregasm"]
    ed = g[g.result == "edged"]
    if filtered:                                             # older "no tie" games have no team names, so a game count would be wrong
        c1, c2 = st.columns(2)
        c1.metric("Scoregasms 69–69", f"{len(hits):,}")
        c2.metric("Left hanging", f"{len(ed):,}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Games logged", f"{n_games:,}")
        c2.metric("Scoregasms 69–69", f"{len(hits):,}")
        c3.metric("Share of games", f"{len(hits) / n_games * 100:.1f}%" if n_games else "–", f"model at tip-off: {tp(0, 0, 48.0, 69) * 100:.1f}%",
                  delta_color="off")
    if scope == "pre":
        st.caption("PRESEASON TEST VIEW: these games are not in the records or the all-time view. Use it to check the logger is working before opening night.")
    if scope == "all":
        st.caption("All-time = every game in the archive, plus this season. Older games show a season, not a date.")
    if filtered and not len(hits) and not len(ed):
        st.info("🪱 Nothing found. Try another team or a shorter search.")
    elif not len(hits) and not filtered and scope == "pre":
        st.info("🪱 No preseason games logged yet. Run: python3 update_log.py --preseason --since 2026-10-01")
    elif not len(hits) and not filtered:
        st.info("🪱 Flaccid. Nothing here yet." + (" The first scoregasm of 2026–27 takes every record. Updated daily." if scope == "launch"
                                           else " Build the archive with build_archive.py, then put scoregasm_archive.csv next to this app."))
    view = st.radio("Log", ["🏆 Hall of Fame", "🥵 Left hanging", "😈 Team tease board"], horizontal=True, label_visibility="collapsed")
    if view.startswith("🏆"):
        if len(hits):
            st.markdown(record_cards(hits), unsafe_allow_html=True)
            with st.expander(f"Every scoregasm in this view ({len(hits):,})"):
                st.dataframe(scoregasm_table(hits), hide_index=True, use_container_width=True,
                             column_config={"Story": st.column_config.TextColumn(width="large")})
    elif view.startswith("😈"):
        st.caption("Every team ranked by how often their 69-69 chances actually finish. Edged = so close, but no tie. (Team and search filters don't apply here.)")
        allt, ranked = tease_board(full, 8 if scope == "all" else 2)
        if not len(allt):
            return st.info("No teams to rank yet.")
        st.markdown(tease_cards(ranked), unsafe_allow_html=True)
        show = allt.drop(columns="code").sort_values(["Finish rate", "Edged"], ascending=[False, True])
        show["Finish %"] = (show.pop("Finish rate") * 100).round().astype(int)
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"Finish %": st.column_config.ProgressColumn("Finish %", min_value=0, max_value=100, format="%d%%")})
    else:
        st.caption("EDGED: a team reached exactly 69, the other got within 3 points while it sat there, and the tie never happened.")
        if not len(ed):
            return st.info("Nobody left hanging yet.")
        e1, e2 = st.columns(2)
        e1.metric("Left hanging", f"{len(ed):,}")
        e2.metric("Within 2 points", f"{int((ed.closest_gap <= 2).sum()):,}")
        st.markdown('<div class="small" style="margin:6px 0">CLOSEST CALLS</div>', unsafe_allow_html=True)
        st.markdown(edged_cards_html(ed, 69), unsafe_allow_html=True)
        pick = st.selectbox("How close", ["Within 3 (all)", "Within 2", "Exactly 1 away"])
        st.dataframe(edged_table(ed, 69, {"Within 3 (all)": 3, "Within 2": 2, "Exactly 1 away": 1}[pick]),
                     hide_index=True, use_container_width=True, column_config={"Story": st.column_config.TextColumn(width="large")})


tab_board, tab_game, tab_log = st.tabs(["📺 Tonight", "🧮 Check a game", "🏆 Hall of Fame"])
with tab_board:
    board_view()
with tab_game:
    game_view()
with tab_log:
    log_view()

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
