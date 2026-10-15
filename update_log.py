"""Scoregasm season tracker: reads every finished NBA game from ESPN and keeps one row per game in scoregasm_season.csv.

Usage:
    python3 update_log.py                         # normal nightly run: picks up where the file left off
    python3 update_log.py --since 2026-10-21      # re-scan from a date (safe: games already logged are skipped)
    python3 update_log.py --since 2025-10-21 --until 2026-06-30 --out scoregasm_archive.csv   # add a past season to the archive
Needs: pip3 install requests.  Uses ESPN's public scoreboard and play-by-play API (no key, no login).

How it makes sure every game gets documented:
  * It starts from the last date already in the file (minus 2 days), so a missed night, or a week, gets filled next run.
  * Every finished game gets a row. If ESPN's play-by-play is broken, the game is saved as "unchecked" and retried
    on every later run until the data checks out.
  * Each row also stores the game's full scoring path (column "scores"), so any future stat can be worked out
    from the file alone, without asking ESPN again.
  * It saves ESPN's betting line too (column "line" = points the home team is favored by, plus "over_under"), when ESPN has one.
  * Re-running is always safe: the file is rewritten in date order with no duplicates.

result is 'scoregasm' (tied exactly 69-69 in regulation), 'edged' (a team sat on 69 while the other got within 3 but no tie),
'none', or 'unchecked' (play-by-play not usable yet)."""
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
        "max_deficit", "deficit_quarter", "deficit_clock", "leader", "other", "leader_quarter", "leader_clock", "closest_gap", "closest_score", "ending", "how", "line", "over_under", "scores"]
GAME_TYPE = {2: "regular", 3: "playoff", 5: "play-in"}   # ESPN season types we log (1 = preseason, skipped)
MAX_LOOKBACK = 45   # days; a bigger gap needs --since, so a summer break doesn't trigger a 4-month scan


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
        if res["max_deficit"] > 0:                              # when the biggest hole was first reached
            d = next(x for x in reg[: i + 1] if score(x, first) - score(x, e["side"]) == res["max_deficit"])
            res.update(deficit_quarter=d["period"], deficit_clock=mmss(d["sec"]))
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


def read_odds(comp, away, home):
    """ESPN scoreboard odds -> (line, over_under). line = points the HOME team is favored by. Blank if ESPN has none.
    Reads the "BOS -5.5" text first (no sign confusion), then the favorite flags."""
    line = ou = ""
    try:
        odds = (comp.get("odds") or [{}])[0]
        det = str(odds.get("details") or "").strip().upper()
        parts = det.split()
        if det in ("EVEN", "PK", "PICK", "PICK'EM"):
            line = 0.0
        elif len(parts) == 2 and parts[0] in (away.upper(), home.upper()):
            line = abs(float(parts[1])) * (1 if parts[0] == home.upper() else -1)
        elif odds.get("spread") is not None and (odds.get("homeTeamOdds") or {}).get("favorite") is not None:
            line = abs(float(odds["spread"])) * (1 if odds["homeTeamOdds"]["favorite"] else -1)
        if line != "" and abs(line) > 30:
            line = ""
        if odds.get("overUnder") is not None and 150 < float(odds["overUnder"]) < 300:
            ou = float(odds["overUnder"])
    except (TypeError, ValueError, AttributeError, IndexError):
        pass
    return line, ou


def score_path(rows):
    """Every scoring change as 'period,seconds_left,away,home' joined by ';'. About 1 KB per game."""
    return ";".join(f"{e['period']},{e['sec']},{e['a']},{e['h']}" for e in events(rows))


def process_event(ev, day, today):
    """One scoreboard event -> a row dict, or None (not finished yet, preseason, or postponed).
    If play-by-play doesn't check out, returns an 'unchecked' row once the game is 2+ days old (retried on later runs)."""
    comp = ev["competitions"][0]
    status = (ev.get("status") or comp.get("status") or {}).get("type", {})
    if status.get("state") != "post" or status.get("completed") is False:
        return None
    kind = (ev.get("season") or {}).get("type")
    if kind not in GAME_TYPE:
        return None
    side = {c["homeAway"]: c for c in comp["competitors"]}
    teams = {k: side[k]["team"]["abbreviation"] for k in ("away", "home")}
    finals = {k: int(side[k].get("score") or 0) for k in ("away", "home")}
    base = {"game_id": str(ev["id"]), "date": day.isoformat(), "season": season_label(day), "game_type": GAME_TYPE[kind],
            "away": teams["away"], "home": teams["home"], "away_final": finals["away"], "home_final": finals["home"]}
    base["line"], base["over_under"] = read_odds(comp, teams["away"], teams["home"])
    rows = rows_from_espn(fetch_plays(ev["id"]))
    if len(rows) < 100 or (rows[-1]["a"], rows[-1]["h"]) != (finals["away"], finals["home"]):
        print(f"  {teams['away']}@{teams['home']} {ev['id']}: play-by-play incomplete or doesn't match the final score")
        return {**base, "result": "unchecked"} if (today - day).days >= 2 else None
    return {**base, **analyze(rows, teams), "scores": score_path(rows)}


def load(path):
    """Existing file -> (header, {game_id: row})."""
    if not os.path.exists(path):
        return list(COLS), {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or COLS), {row["game_id"]: row for row in r}


def save(path, header, games):
    """Rewrite the whole file in date order (written to a temp file first, so a crash can't leave half a file)."""
    cols = header + [c for c in COLS if c not in header]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        for row in sorted(games.values(), key=lambda r: (str(r.get("season") or ""), str(r.get("date") or ""), str(r["game_id"]))):
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="first date to check, YYYY-MM-DD (default: last date in the file minus 2 days)")
    ap.add_argument("--until", help="last date to check (default: today)")
    ap.add_argument("--limit", type=int, help="stop after logging this many new games")
    ap.add_argument("--out", default=OUT, help="file to update (default: scoregasm_season.csv)")
    a = ap.parse_args()
    today = datetime.datetime.now(datetime.timezone.utc).date()
    header, games = load(a.out)
    end = datetime.date.fromisoformat(a.until) if a.until else today
    if a.since:
        start = datetime.date.fromisoformat(a.since)
    else:
        dates = [datetime.date.fromisoformat(r["date"]) for r in games.values() if r.get("date")]
        start = min(max(dates) - datetime.timedelta(days=2), today - datetime.timedelta(days=3)) if dates else today - datetime.timedelta(days=3)
        if start < today - datetime.timedelta(days=MAX_LOOKBACK):
            start = today - datetime.timedelta(days=MAX_LOOKBACK)
            print(f"note: last logged game is {max(dates)}; only checking the last {MAX_LOOKBACK} days. Use --since to go further back.")
    days = {start + datetime.timedelta(days=k) for k in range((end - start).days + 1)}
    days |= {datetime.date.fromisoformat(r["date"]) for r in games.values() if r.get("result") == "unchecked" and r.get("date")}
    done = {g for g, r in games.items() if r.get("result") != "unchecked"}
    added = fixed = 0
    try:
        for day in sorted(days):
            board = get_json(BOARD.format(day.strftime("%Y%m%d")))
            if board is None:
                print(day, "scoreboard unavailable, will try again next run")
                continue
            for ev in board.get("events", []):
                if str(ev["id"]) in done:
                    continue
                row = process_event(ev, day, today)
                if row is None:
                    continue
                was_unchecked = row["game_id"] in games
                games[row["game_id"]] = row
                if row["result"] != "unchecked":
                    done.add(row["game_id"])
                    fixed += was_unchecked
                added += not was_unchecked
                print(day, row["away"], "@", row["home"], {"scoregasm": "🎆 SCOREGASM", "edged": "edged", "unchecked": "(unchecked)"}.get(row["result"], ""))
                if (added + fixed) % 25 == 0:
                    save(a.out, header, games)            # save as we go on long catch-ups
                if a.limit and added >= a.limit:
                    raise StopIteration
                time.sleep(0.3)
    except StopIteration:
        print("limit reached")
    save(a.out, header, games)
    left = sum(r.get("result") == "unchecked" for r in games.values())
    print(f"done: {added} new games, {fixed} fixed, {left} still unchecked, {len(games)} total -> {a.out}")


if __name__ == "__main__":
    main()
