"""Build the all-time Scoregasm archive (scoregasm_archive.csv) from free NBA play-by-play files.

Source: https://github.com/shufinskiy/nba_data  (stats.nba.com play-by-play, one file per season, 1996-97 on).
Run this ONCE on your own computer (it downloads about 1 GB over a few minutes; safe to stop and re-run, it resumes):

    pip3 install pandas requests
    python3 build_archive.py                      # every season the dataset has (1996-97 to 2024-25)
    python3 build_archive.py --seasons 2016-2024  # only some seasons (the number is the year the season STARTS)

For 2025-26 (not in that dataset) use ESPN instead, writing into the same file:
    python3 update_log.py --since 2025-10-21 --until 2026-06-30 --out scoregasm_archive.csv

The dataset has no game dates, so archive rows show the season (like 2016-17) instead of a date.
"""
import argparse
import csv
import io
import os
import tarfile

import pandas as pd
import requests

import update_log as U

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/{}_{}.tar.xz"
WANT = {"GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
        "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION"}
KINDS = {"2": "regular", "4": "playoff", "5": "play-in"}          # game id 00 K yy nnnnn ; K=1 is preseason (skipped)


def season_label(y):
    return f"{y}-{str(y + 1)[-2:]}"


def get_file(name, year, folder):
    path = os.path.join(folder, f"{name}_{year}.tar.xz")
    if not os.path.exists(path):
        r = requests.get(URL.format(name, year), timeout=300)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def read_tar(path):
    frames = []
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".csv"):
                frames.append(pd.read_csv(io.BytesIO(t.extractfile(m).read()), usecols=lambda c: c in WANT, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else None


def clock_sec(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(float(sec))
    except ValueError:
        return 0


def split_score(s):
    try:
        left, right = str(s).split("-")
        return int(left), int(right)
    except ValueError:
        return None


def game_rows(g):
    """One game's plays -> (rows for update_log.analyze, teams, finals) or None if the data is unusable."""
    g = g.sort_values("EVENTNUM", kind="stable") if "EVENTNUM" in g else g
    prev, pairs, last = (0, 0), [], None
    for sc in g.SCORE.tolist():
        p = split_score(sc) if isinstance(sc, str) else None
        if p is not None:
            last = p
        pairs.append(last if last is not None else (0, 0))
    # which side of "x - y" is the visiting team? Look at who the description belongs to when a score changes.
    votes, prev = 0, (0, 0)
    for p, hd, vd in zip(pairs, g.HOMEDESCRIPTION.tolist(), g.VISITORDESCRIPTION.tolist()):
        if p != prev and (p[0] > prev[0]) != (p[1] > prev[1]):
            left_scored = p[0] > prev[0]
            if isinstance(vd, str) and not isinstance(hd, str):
                votes += 1 if left_scored else -1          # visitor scored and it was the left number
            elif isinstance(hd, str) and not isinstance(vd, str):
                votes += -1 if left_scored else 1
        prev = p
    left_is_away = votes >= 0                               # default: stats.nba.com writes "visitor - home"
    rows, teams, prev = [], {}, (0, 0)
    for p, per, clk, nm, tm in zip(pairs, g.PERIOD, g.PCTIMESTRING, g.PLAYER1_NAME, g.PLAYER1_TEAM_ABBREVIATION):
        a, h = p if left_is_away else (p[1], p[0])
        if (a, h) != prev and (a > prev[0]) != (h > prev[1]) and isinstance(tm, str):
            teams.setdefault("away" if a > prev[0] else "home", tm)
        prev = (a, h)
        rows.append({"period": int(per), "sec": clock_sec(clk), "a": a, "h": h, "text": f"{nm} makes" if isinstance(nm, str) else ""})
    final = (rows[-1]["a"], rows[-1]["h"]) if rows else (0, 0)
    if len(rows) < 100 or max(final) < 50:
        return None
    return rows, {"away": teams.get("away", "AWAY"), "home": teams.get("home", "HOME")}, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="1996-2024", help="first-last START year, e.g. 2016-2024")
    ap.add_argument("--out", default=os.path.join(HERE, "scoregasm_archive.csv"))
    ap.add_argument("--folder", default=os.path.join(HERE, "archive_downloads"), help="where downloaded files are kept")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seasons.split("-"))
    os.makedirs(a.folder, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            done = {r["game_id"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=U.COLS, extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        for year in range(lo, hi + 1):
            added = skipped = hits = 0
            for name in ("nbastats", "nbastats_po"):
                path = get_file(name, year, a.folder)
                if path is None:
                    print(f"{season_label(year)} {name}: not in the dataset")
                    continue
                df = read_tar(path)
                if df is None:
                    continue
                df["gid"] = df.GAME_ID.map(lambda x: f"{int(x):010d}")
                for gid, g in df.groupby("gid", sort=False):
                    kind = KINDS.get(gid[2])
                    if kind is None or gid in done:
                        continue
                    got = game_rows(g)
                    if got is None:
                        skipped += 1
                        continue
                    rows, teams, final = got
                    res = U.analyze(rows, teams)
                    base = {"game_id": gid, "date": "", "season": season_label(year), "game_type": kind,
                            "away": teams["away"], "home": teams["home"], "away_final": final[0], "home_final": final[1]}
                    # games with no scoregasm / near miss only need a short row (keeps the file small)
                    w.writerow({**base, **res} if res["result"] != "none" else {k: base[k] for k in ("game_id", "season", "game_type")} | {"result": "none"})
                    done.add(gid); added += 1; hits += res["result"] == "scoregasm"
                f.flush()
            print(f"{season_label(year)}: {added} games, {hits} scoregasms" + (f", {skipped} skipped (bad data)" if skipped else ""), flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
