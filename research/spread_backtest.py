"""Does the betting spread make the 69-69 odds more accurate? Re-runs the test behind the site's spread model.

Run from the project folder:
    pip3 install pandas numpy requests
    python3 research/spread_backtest.py            # full test, 2015-16 to 2024-25 (about 5 minutes after downloads)
    python3 research/spread_backtest.py --quick    # only compares "no spread" vs the site's setting (1.4)

What it downloads (once, into research/downloads/, about 100 MB; all public GitHub files, no keys or logins):
  * Play-by-play, one file per season ........ github.com/shufinskiy/nba_data
  * Closing spreads 2015-16 to 2021-22 ........ github.com/flancast90/sportsbookreview-scraper (SportsbookReview archive)
  * Closing spreads 2022-23 to 2024-25 ........ github.com/kyleskom/NBA-Machine-Learning-Sports-Betting (OddsData.sqlite)
The matched lines are saved to research/lines.csv, so later runs skip the odds downloads.

Quirks of the odds files (found the hard way):
  * SportsbookReview: the spread and over/under columns are mixed up row by row. The number under 60 is the spread,
    the one over 150 is the total, and the moneyline says who was favored.
  * kyleskom: before 2022-23 the spreads have no sign and some seasons have columns swapped, so only 2022-25 is used.
  * Games are matched on season + home + away + final score (the play-by-play files have no dates). About 98% match.

How the test works: every 2 minutes of game clock in every game, the model makes a forecast from the score, the clock
and (optionally) the pregame line. Each forecast is scored against what really happened (log loss: lower is better).
The spread's strength is tuned on some seasons and graded on the others, both ways round."""
import argparse
import io
import json
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import build_archive as B     # noqa: E402  (play-by-play reader)
import update_log as U        # noqa: E402  (scoring events, 69-69 check)

DL = os.path.join(HERE, "downloads")
LINES = os.path.join(HERE, "lines.csv")
SBR_URL = "https://github.com/flancast90/sportsbookreview-scraper/raw/main/data/nba_archive_10Y.json"
KYLE_URL = "https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/raw/master/Data/OddsData.sqlite"
TARGET = 69

# ------------------------------------------------------------------ the model (same math as scoregasm.py)
SEQ = [(.513, []), (.115, [3]), (.003, [3, 4]), (.254, [2]), (.033, [2, 3]), (.060, [1, 2]), (.022, [1])]
SC = SEQ[1:]
SC_TOT = sum(p for p, _ in SC)
PPP = sum(p * c[-1] for p, c in SC)      # points per possession, about 1.11
W = TARGET + 1


def table(line=0.0, scale=1.4, k=0.002, cap=20, n_max=200):
    """P[n, da, db] = chance of passing through 69-69 with the away team da points short, home db short,
    n possessions left. Same recursion as tie_probability() in scoregasm.py, run once for every state at once."""
    edge = scale * line / (2 * PPP * 100)
    ia, ib = np.meshgrid(np.arange(W), np.arange(W), indexing="ij")
    m = np.clip(ib - ia, -cap, cap)
    sa = np.clip((1 - k * m) * (1 - edge), 0.3, 2.0)
    sb = np.clip((1 + k * m) * (1 + edge), 0.3, 2.0)
    zero = np.zeros((W, W)); zero[0, 0] = 1
    ga, gb = zero.copy(), zero.copy()
    P = np.empty((n_max + 1, W, W)); P[0] = zero
    for n in range(1, n_max + 1):
        new_a, new_b = (1 - sa * SC_TOT) * gb, (1 - sb * SC_TOT) * ga
        for p, cum in SC:
            c = cum[-1]
            ta, tb = np.zeros((W, W)), np.zeros((W, W))
            ta[c:, :] = gb[: W - c, :]; tb[:, c:] = ga[:, : W - c]
            for cj in cum:
                ta[cj, 0] = 1; tb[0, cj] = 1
            new_a += p * sa * ta; new_b += p * sb * tb
        new_a[0, 0] = new_b[0, 0] = 1
        ga, gb = new_a, new_b
        P[n] = (ga + gb) / 2
    return P


def check_matches_app():
    """Guard: the copy above must give the same numbers as the live app. Reads only the math section of scoregasm.py."""
    src = open(os.path.join(ROOT, "scoregasm.py")).read()
    ns = {"np": np, "PACE": 100}
    exec(src[src.index("SEQ = "):src.index("@functools.lru_cache")], ns)
    for line in (0, 7.5, -12):
        P = table(line, ns.get("SPREAD_SCALE", 1.4))
        for a, b, mins in ((0, 0, 48), (55, 52, 16), (62, 60, 8)):
            app = ns["tie_probability"](a, b, mins, line=line)
            here = P[round(mins * 2 * 100 / 48), TARGET - a, TARGET - b]
            if abs(app - here) > 1e-9:
                sys.exit(f"The research model no longer matches scoregasm.py (line {line}, {a}-{b}): {app:.5f} vs {here:.5f}. Update table().")
    print("Model check: research copy matches scoregasm.py.\n")


# ------------------------------------------------------------------ data
def download(url, name):
    os.makedirs(DL, exist_ok=True)
    path = os.path.join(DL, name)
    if not os.path.exists(path):
        print("  downloading", name)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return path


def load_games(first, last):
    """Every regular-season game: teams, finals, regulation scoring path, and whether it hit 69-69. Cached."""
    cache = os.path.join(DL, f"games_{first}_{last}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    games = []
    for y in range(first, last + 1):
        df = B.read_tar(B.get_file("nbastats", y, DL))
        df["gid"] = df.GAME_ID.map(lambda x: f"{int(x):010d}")
        for gid, g in df.groupby("gid", sort=False):
            if gid[2] != "2":                       # regular season only
                continue
            got = B.game_rows(g)
            if got is None:
                continue
            rows, teams, final = got
            games.append({"gid": gid, "season": y, "away": teams["away"], "home": teams["home"], "af": final[0], "hf": final[1],
                          "path": [(e["t"], e["a"], e["h"]) for e in U.events(rows) if e["period"] <= 4],
                          "hit": U.analyze(rows, teams)["result"] == "scoregasm"})
        print(f"  {y}-{str(y + 1)[-2:]}: {sum(g['season'] == y for g in games)} games")
    pickle.dump(games, open(cache, "wb"))
    return games


NICK = {"Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA", "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL",
        "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Golden State": "GSW", "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC",
        "Lakers": "LAL", "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP", "Knicks": "NYK",
        "Thunder": "OKC", "Magic": "ORL", "Seventysixers": "PHI", "Suns": "PHX", "Trailblazers": "POR", "Kings": "SAC", "Spurs": "SAS",
        "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS"}
FULL = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"}


def build_lines(games):
    """Match a closing line to each game -> research/lines.csv (gid, line = points HOME is favored by, ou = over/under)."""
    num = lambda s: pd.to_numeric(s, errors="coerce")
    d = pd.DataFrame(json.load(open(download(SBR_URL, "sbr_nba.json"))))
    d = d[num(d.season).between(2015, 2021)].copy()
    for c in ("home_final", "away_final", "home_close_spread", "close_over_under", "home_close_ml", "away_close_ml"):
        d[c] = num(d[c])
    a, b = d.home_close_spread.abs(), d.close_over_under          # unscramble: spread < 60, total > 150
    sp = np.where(a < 60, a, np.where(b < 60, b, np.nan)); sp[sp == 0] = np.nan
    sbr = pd.DataFrame({"season": num(d.season).astype(int), "home": d.home_team.astype(str).map(NICK), "away": d.away_team.astype(str).map(NICK),
                        "hf": d.home_final, "af": d.away_final,
                        "line": np.where(d.home_close_ml < d.away_close_ml, 1, -1) * sp,
                        "ou": np.where(a > 150, a, np.where(b > 150, b, np.nan))})
    con = sqlite3.connect(download(KYLE_URL, "kyleskom_odds.sqlite"))
    k = pd.concat([pd.read_sql(f'select * from "{t}"', con).assign(season=y)
                   for y, t in ((2022, "odds_2022-23_new"), (2023, "2023-24"), (2024, "2024-25"))])
    for c in ("Spread", "OU", "Points", "Win_Margin"):
        k[c] = num(k[c])
    kyle = pd.DataFrame({"season": k.season, "home": k.Home.map(FULL), "away": k.Away.map(FULL), "hf": (k.Points + k.Win_Margin) / 2,
                         "af": (k.Points - k.Win_Margin) / 2, "line": k.Spread, "ou": k.OU})
    lines = pd.concat([sbr, kyle]).dropna(subset=["home", "away", "hf", "af"]).drop_duplicates(["season", "home", "away", "hf", "af"])
    g = pd.DataFrame([{k: v for k, v in x.items() if k != "path"} for x in games])
    g = g.merge(lines, on=["season", "home", "away", "hf", "af"], how="left").drop_duplicates("gid")
    g[["gid", "line", "ou"]].to_csv(LINES, index=False)
    return g[["gid", "line", "ou"]]


def forecast_states(games, lines):
    """One row per forecast: every 2 minutes of game clock, skipping states where 69-69 is already impossible or done."""
    L = lines.dropna(subset=["line"]).set_index("gid").line
    rows = []
    for gi, g in enumerate(games):
        if g["gid"] not in L.index:
            continue
        tie_t = next((t for t, a, h in g["path"] if a == h == TARGET), None)
        for T in range(0, 2880, 120):
            a = h = 0
            for t, x, y in g["path"]:
                if t > T:
                    break
                a, h = x, y
            if a > TARGET or h > TARGET or (tie_t is not None and tie_t <= T):
                continue
            rows.append((gi, g["season"], L[g["gid"]], T, a, h, round((2880 - T) / 60 * 2 * 100 / 48), int(tie_t is not None and tie_t > T)))
    F = pd.DataFrame(rows, columns=["game", "season", "line", "T", "a", "h", "n", "y"])
    F["lr"] = (F.line * 2).round() / 2
    return F


def predict(F, scale):
    p = np.empty(len(F))
    for lr, idx in F.groupby("lr").indices.items():
        P = table(lr, scale)
        s = F.iloc[idx]
        p[idx] = P[s.n.values, TARGET - s.a.values, TARGET - s.h.values]
    return np.clip(p, 1e-6, 1 - 1e-6)


def logloss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2015-2024", help="first-last START year (play-by-play goes to 2024-25)")
    ap.add_argument("--quick", action="store_true", help="only test no-spread vs the site's setting")
    ap.add_argument("--refresh-lines", action="store_true", help="re-download the odds files and rebuild research/lines.csv")
    a = ap.parse_args()
    first, last = (int(x) for x in a.seasons.split("-"))
    check_matches_app()
    print("Loading games (first run downloads ~85 MB of play-by-play)...")
    games = load_games(first, last)
    lines = pd.read_csv(LINES, dtype={"gid": str}) if os.path.exists(LINES) and not a.refresh_lines else build_lines(games)
    F = forecast_states(games, lines)
    tip = F["T"] == 0
    mid = (first + last) // 2
    early, late = (F.season <= mid).values, (F.season > mid).values
    print(f"\n{F.game.nunique():,} games with a closing line, {len(F):,} forecasts, {int(F.y[tip].sum())} games hit 69-69.\n")

    scales = [0, 1.4] if a.quick else [0, 0.6, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
    preds, rows = {}, []
    for c in scales:
        preds[c] = predict(F, c)
        l = logloss(preds[c], F.y.values)
        rows.append({"spread strength": c, f"error {first}-{mid}": l[early].mean() * 1000, f"error {mid + 1}-{last}": l[late].mean() * 1000})
    t = pd.DataFrame(rows).set_index("spread strength")
    print("FORECAST ERROR BY SPREAD STRENGTH (log loss x 1000, lower is better; 0 = ignore the spread)")
    print(t.round(3).to_string(), "\n")
    if not a.quick:
        for tune, grade, lab_t, lab_g in ((early, late, f"{first}-{mid}", f"{mid + 1}-{last}"), (late, early, f"{mid + 1}-{last}", f"{first}-{mid}")):
            best = min(scales[1:], key=lambda c: logloss(preds[c][tune], F.y.values[tune]).mean())
            gain = (logloss(preds[0][grade], F.y.values[grade]).mean() - logloss(preds[best][grade], F.y.values[grade]).mean()) * 1000
            print(f"Tuned on {lab_t}: best strength {best}. On unseen {lab_g} it cuts the error by {gain:.3f} ({'better' if gain > 0 else 'WORSE'}).")
    c = 1.4
    per = pd.DataFrame({"game": F.game, "d": logloss(preds[0], F.y.values) - logloss(preds[c], F.y.values)}).groupby("game").d.sum().values
    rng = np.random.default_rng(0)
    share = np.mean([rng.choice(per, len(per)).sum() > 0 for _ in range(2000)])
    print(f"Luck check (strength {c}): the spread version wins in {share:.1%} of 2,000 resamples of games.\n")

    T0 = F[tip].assign(old=preds[0][tip.values], new=preds[c][tip.values])
    T0["spread"] = pd.cut(T0.line.abs(), [-0.1, 2.5, 5, 7.5, 10, 13, 30], labels=["0-2.5", "3-5", "5.5-7.5", "8-10", "10.5-13", "13.5+"])
    b = T0.groupby("spread", observed=True).agg(games=("y", "size"), actual=("y", "mean"), old_model=("old", "mean"), with_spread=("new", "mean"))
    b[["actual", "old_model", "with_spread"]] *= 100
    print("AT TIP-OFF, BY SIZE OF THE SPREAD (% of games that hit 69-69)")
    print(b.round(2).to_string())
    print(f"All games: actual {T0.y.mean() * 100:.2f}%, old model {T0.old.mean() * 100:.2f}%, with spread {T0.new.mean() * 100:.2f}%\n")
    F["p"] = preds[c]
    F["bucket"] = pd.cut(F.p, [0, .01, .02, .04, .08, .16, 1], labels=["<1%", "1-2%", "2-4%", "4-8%", "8-16%", "16%+"])
    k = F.groupby("bucket", observed=True).agg(forecasts=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    k[["predicted", "actual"]] *= 100
    print("LIVE FORECASTS, GROUPED BY WHAT THE MODEL SAID (%)")
    print(k.round(2).to_string())


if __name__ == "__main__":
    main()
