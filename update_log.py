"""Scoregasm season tracker: reads finished NBA games from ESPN and logs one row per game to scoregasm_season.csv.

Usage:
    python3 update_log.py                         # last 3 days (what the daily GitHub job runs)
    python3 update_log.py --since 2026-10-01      # catch up from a date (use this once at launch)
    python3 update_log.py --since 2026-10-20 --limit 5   # try a few games first
Safe to re-run: games already in the file are skipped. Needs: pip3 install requests

Each row says whether the game was a scoregasm (tied exactly 69-69 in regulation), edged (a team sat on 69 while the other
got within 3 points but the tie never came), or neither, plus the details the Hall of Fame needs."""
import argparse
import csv
import datetime
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scoregasm_season.csv")
TARGET = 69
BOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}&limit=100"
PLAYS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/plays?limit=1000&page={1}"
REG_END = 2880  # seconds in regulation
COLS = ["game_id", "date", "season", "game_type", "away", "home", "away_final", "home_final", "ot", "result",
        "quarter", "clock", "elapsed_min", "first_to", "tied_by", "play_type", "play_by", "wait_sec", "behind", "tie_sec",
        "max_deficit", "leader", "other", "leader_quarter", "leader_clock", "closest_gap", "closest_score", "ending", "how"]


# ---------------------------------------------------------------- reading a game
def elapsed(period, sec):
    return (period - 1) * 720 + (720 - sec) if period <= 4 else REG_END + (period - 5) * 300 + (300 - sec)


def mmss(sec):
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def play_type(delta):
    return {1: "free throw", 3: "three-pointer"}.get(delta, "two-pointer")


def player(text):
    return text.split(" makes ")[0].strip() if isinstance(text, str) and " makes " in text else ""


def clock_seconds(clock):
    """ESPN clock object -> seconds left in the period."""
    clock = clock or {}
    if isinstance(clock.get("value"), (int, float)):
        return int(clock["value"])
    s = str(clock.get("displayValue") or "0")
    try:
        return int(s.split(":")[0]) * 60 + int(float(s.split(":")[1])) if ":" in s else int(float(s))
    except ValueError:
        return 0


def rows_from_espn(items):
    """ESPN core 'plays' items -> [{period, sec, a, h, text}] (skips anything without usable scores)."""
    out = []
    for p in items:
        try:
            out.append({"period": int((p.get("period") or {}).get("number")), "sec": clock_seconds(p.get("clock")),
                        "a": int(p["awayScore"]), "h": int(p["homeScore"]), "text": p.get("text") or ""})
        except (TypeError, ValueError, KeyError):
            continue
    return out


def events(rows):
    """Scoring events in order (each score change once, scores never going down)."""
    out, prev = [], (0, 0)
    for r in rows:
        a, h = r["a"], r["h"]
        if (a, h) == prev or a < prev[0] or h < prev[1]:
            continue
        side = "away" if a > prev[0] else "home"
        out.append({"period": r["period"], "sec": r["sec"], "t": elapsed(r["period"], r["sec"]), "a": a, "h": h, "side": side,
                    "delta": (a - prev[0]) if side == "away" else (h - prev[1]), "text": r["text"]})
        prev = (a, h)
    return out


def analyze(rows, teams, target=TARGET):
    """-> dict of result columns. result is 'scoregasm', 'edged' or 'none' (regulation only)."""
    ev = events(rows)
    reg = [e for e in ev if e["period"] <= 4]
    score = lambda e, side: e["a"] if side == "away" else e["h"]
    other = lambda side: "home" if side == "away" else "away"
    res = {"result": "none", "ot": any(r["period"] > 4 for r in rows)}
    i = next((k for k, e in enumerate(reg) if e["a"] == target and e["h"] == target), None)
    j = next((k for k, e in enumerate(reg) if max(e["a"], e["h"]) >= target), None)
    if j is None:
        return res
    if i is not None:                                           # ---- scoregasm
        e, first = reg[i], other(reg[i]["side"])
        nxt = reg[i + 1]["t"] if i + 1 < len(reg) else REG_END
        res.update(result="scoregasm", quarter=e["period"], clock=mmss(e["sec"]), elapsed_min=round(e["t"] / 60, 2),
                   first_to=teams[first], tied_by=teams[e["side"]], play_type=play_type(e["delta"]), play_by=player(e["text"]),
                   wait_sec=max(0, e["t"] - reg[j]["t"]), behind=target - score(reg[j], e["side"]), tie_sec=max(0, nxt - e["t"]),
                   max_deficit=max(0, max(score(x, first) - score(x, e["side"]) for x in reg[: i + 1])))
        tail = "the very next play broke it." if res["tie_sec"] == 0 else f"the score stayed {target}-{target} for {mmss(res['tie_sec'])}."
        when = "right away" if res["wait_sec"] == 0 else f"{mmss(res['wait_sec'])} later"
        res["how"] = (f"{res['tied_by']} trailed {target}-{target - res['behind']} when {res['first_to']} got to {target}. "
                      f"{res['tied_by']} tied it {when} on a {res['play_type']}{' by ' + res['play_by'] if res['play_by'] else ''}, and {tail}")
        if res["max_deficit"] >= 9:
            res["how"] += f" {res['tied_by']} had been down as many as {res['max_deficit']} earlier."
        return res
    x = reg[j]
    lead = "away" if x["a"] >= x["h"] else "home"
    if score(x, lead) != target:
        return res                                              # jumped straight over the target
    best = target - score(x, other(lead)); best_score = score(x, other(lead)); ending, end = "end of regulation", None
    for y in reg[j + 1:]:
        if y["side"] == lead:
            ending, end = "pulled away", y
            break
        if score(y, other(lead)) >= target:
            ending, end = "overshot", y
            break
        if target - score(y, other(lead)) < best:
            best, best_score = target - score(y, other(lead)), score(y, other(lead))
    if best > 3:
        return res
    wait = max(0, (end["t"] if end else REG_END) - x["t"])
    sat = "a split second" if wait == 0 else mmss(wait)
    how = f"{teams[lead]} sat on {target} for {sat} while {teams[other(lead)]} got as close as {target - best}."
    if end is None:
        how += " Then regulation ran out."
    else:
        who, before = (teams[lead], target) if ending == "pulled away" else (teams[other(lead)], best_score)
        verb = "scored again" if ending == "pulled away" else "jumped right over it"
        how += f" Then {who} {verb} ({before}→{before + end['delta']}) on a {play_type(end['delta'])}{' by ' + player(end['text']) if player(end['text']) else ''}"
        how += ", and the tie was gone." if ending == "pulled away" else "."
    res.update(result="edged", leader=teams[lead], other=teams[other(lead)], leader_quarter=x["period"], leader_clock=mmss(x["sec"]),
               closest_gap=best, closest_score=target - best, wait_sec=wait, ending=ending, how=how)
    return res


# ---------------------------------------------------------------- talking to ESPN
def get_json(url, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=25)
            if r.ok:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (k + 1))
    return None


def fetch_plays(event_id):
    """All plays for a game (follows ESPN's paging). Returns [] if anything goes wrong."""
    items, page = [], 1
    while True:
        data = get_json(PLAYS.format(event_id, page))
        if not data:
            return []
        items += data.get("items", [])
        if page >= int(data.get("pageCount") or 1):
            return items
        page += 1


def season_label(day):
    y = day.year if day.month >= 10 else day.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def process_event(ev, day):
    """One scoreboard event -> a row dict, or None (unfinished, preseason, or data didn't check out)."""
    comp = ev["competitions"][0]
    if (ev.get("status") or comp.get("status") or {}).get("type", {}).get("state") != "post":
        return None
    kind = (ev.get("season") or {}).get("type")
    if kind not in (2, 3):                                      # skip preseason and anything odd
        return None
    side = {c["homeAway"]: c for c in comp["competitors"]}
    teams = {k: side[k]["team"]["abbreviation"] for k in ("away", "home")}
    finals = {k: int(side[k].get("score") or 0) for k in ("away", "home")}
    rows = rows_from_espn(fetch_plays(ev["id"]))
    if len(rows) < 100 or (rows[-1]["a"], rows[-1]["h"]) != (finals["away"], finals["home"]):
        print(f"  skipped {teams['away']}@{teams['home']} {ev['id']}: play-by-play incomplete or doesn't match the final score (will retry)")
        return None
    base = {"game_id": str(ev["id"]), "date": day.isoformat(), "season": season_label(day), "game_type": "regular" if kind == 2 else "playoff",
            "away": teams["away"], "home": teams["home"], "away_final": finals["away"], "home_final": finals["home"]}
    return {**base, **analyze(rows, teams)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="first date to check, YYYY-MM-DD (default: 3 days ago)")
    ap.add_argument("--until", help="last date to check (default: today)")
    ap.add_argument("--limit", type=int, help="stop after logging this many new games")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    today = datetime.datetime.now(datetime.timezone.utc).date()
    day = datetime.date.fromisoformat(a.since) if a.since else today - datetime.timedelta(days=3)
    end = datetime.date.fromisoformat(a.until) if a.until else today
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as f:
            done = {r["game_id"] for r in csv.DictReader(f)}
    new_file, added = not os.path.exists(a.out), 0
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        while day <= end:
            board = get_json(BOARD.format(day.strftime("%Y%m%d")))
            if board is None:
                print(day, "scoreboard unavailable, skipping")
            for ev in (board or {}).get("events", []):
                if str(ev["id"]) in done:
                    continue
                row = process_event(ev, day)
                if row is None:
                    continue
                w.writerow(row); f.flush(); done.add(row["game_id"]); added += 1
                print(day, row["away"], "@", row["home"], {"scoregasm": "🎆 SCOREGASM", "edged": "edged"}.get(row["result"], ""))
                if a.limit and added >= a.limit:
                    print("limit reached:", added, "games ->", a.out); return
                time.sleep(0.3)
            day += datetime.timedelta(days=1)
    print(f"done: {added} new games -> {a.out}")


if __name__ == "__main__":
    main()
