# Scoregasm: notes for Claude

A Streamlit site giving the live chance that an NBA game passes through **exactly 69–69** in regulation, plus a
Hall of Fame, a team Tease board and a Stats page showing how the model was tested. A fun side project, and a way for
the owner to learn to build software with AI.

## How the owner likes to work (follow these)
- **Say when an API or outside service will be used** (ESPN, GitHub downloads, anything on the network), before using it.
- **Keep it simple.** Few files, plain code, short comments in plain English. Don't add files, frameworks or
  dependencies unless there's a clear reason, and say so first.
- **Readable output.** No raw pandas dumps. Round numbers, label columns in plain words, explain results in a sentence or two.
- **Minimal "access stuff."** Ask before anything that needs logins, tokens or permissions. Prefer steps they can run themselves.
- **Privacy.** The owner's name must not appear on the site or in commits' content. The GitHub repo is private.
- **Test theories honestly.** New ideas go in `experiments.md` with a written prediction *before* looking at data,
  tuned on some seasons and graded on others. Ship a model change only if it wins on unseen seasons by more than luck.
- The owner knows a little Python (used Jupyter notebooks), with a strong analytical background.

## Files
| File | What it does |
|---|---|
| `scoregasm.py` | The whole site: model, ESPN live feed, styling, the four tabs. |
| `update_log.py` | Nightly logger: every finished game from ESPN -> one row in `scoregasm_season.csv`. |
| `.github/workflows/update-log.yml` | Runs `update_log.py` at 09:17 and 16:17 UTC, commits the log. Can be run by hand with a start date. |
| `scoregasm_archive.csv` | 2015-16 to 2024-25, 12,805 games (2015-20 from Basketball-Reference-style files, 2020-25 from shufinskiy/nba_data). 2020-25 rows have no dates or player names. |
| `scoregasm_season.csv` | This season's games, written by the logger. The site counts games from `LAUNCH = 2026-10-01`. |
| `deficit_times.csv` | When the biggest comeback hole happened, for 2015-20 archive games. |
| `requirements.txt` | streamlit, requests, numpy, pandas. |
| `experiments.md` | The owner's theory log (done / open / ideas). |
| `research/build_archive.py` | Rebuilds archive seasons from shufinskiy/nba_data play-by-play. |
| `research/spread_backtest.py` | Re-runs the betting-spread test (see below). Downloads go to `research/downloads/` (don't commit them). |
| `research/lines.csv` | Closing spread + over/under matched to 11,670 games (by NBA.com game id). |

## Running things
```
pip3 install -r requirements.txt
python3 -m streamlit run scoregasm.py                 # the site, locally (uses ESPN's live scoreboard)
python3 update_log.py                                 # log finished games (picks up where the file left off)
python3 update_log.py --since 2025-10-21 --until 2026-06-30 --out scoregasm_archive.csv   # add 2025-26 to the archive (STILL TO DO)
python3 research/spread_backtest.py --quick           # re-check the spread model
```
Outside services: ESPN's public scoreboard and play-by-play API (no key), Google Fonts (CSS), GitHub raw downloads in `research/`.

## The model (scoregasm.py, "the math")
- Possession-by-possession backward recursion. Possession outcomes in `SEQ` (0/1/2/3 points, free-throw trips pass
  through in-between scores). 100 possessions per team per 48 minutes; regulation only, overtime not modeled.
- Catch-up: trailing team scores `MARGIN_K = 0.002` faster per point behind (cap 20).
- **Betting spread** (added 2026-10-13): `line` = points the HOME team is favored by. Home scores (1+e) as often, away (1-e),
  `e = SPREAD_SCALE * line / (2 * points-per-possession * 100)`, `SPREAD_SCALE = 1.4` (fitted; plain math says ~1.2
  because the catch-up effect shrinks leads). Lines come from ESPN's scoreboard odds (`read_line`: reads "BOS -5.5"
  text first, then favorite flags; anything odd -> None -> even matchup). Pregame lines are remembered once a game goes live.
- Not modeled: injuries, rest, lineups (the line covers most of that), over/under (weak effect), overtime.
- Accuracy: tip-off averaged over real matchups 3.21% vs 3.28% actual (old even-matchup model said 3.57%).
  Live forecasts land within a few tenths of reality from 1% to 16%. Results are hard-coded on the Stats tab
  (`AVG_WITH_LINES`, `LINE_CHECK`, the live-check cards); update them if the model changes.
- `research/spread_backtest.py` contains a copy of the model and checks it still matches `scoregasm.py` before running.

## The site's language (keep consistent)
- **Chub meter** 0–100 = odds on a log scale x progress toward 69. Not a probability.
  Tiers: 🪱 LIMP 0 · 👀 PEEPING 20 · 🌭 HALF CHUB 40 · 🔥 CLOSE 60 · 🎆 SCOREGASM 80.
  Taglines: "Flaccid. Nothing to see here, folks." / "Sheesh." / "Hot dog's on the grill. Not ready yet." / "Almost there, baby." / "IT'S HAPPENING."
- **Scoregasm** = tied exactly 69–69 in regulation. Finals that did show "🚬 Someone light a cigarette."
- **Chance** = one team reaches exactly 69 while the other is within 3 points (66–68). **Tease / near miss** = a chance that
  never became 69–69 (logged as `edged`). Conversion = scoregasms ÷ chances; Tease board needs 8+ chances.
- Hall of Fame records: ⏰ PREMATURE (earliest), 🌙 LATE BLOOMER (latest), 🍑 COMING FROM BEHIND (biggest comeback,
  with where the hole was and who won). Tie badges: ONE AND DONE / LINGERED / STAMINA / TANTRIC.
- Look: black background, Miami Vice neon (pink `#ff2e97`, teal `#19e3d8`), Monoton logo, Audiowide numbers,
  calm retro motion, rounded cards, an intro box at the top of every tab.
- Removed on purpose (don't bring back unless asked): Check-a-game tab, Left hanging tab, team/season filters,
  preseason view, tables (leaderboards only), the "hide finished games" toggle.

## Log file columns (update_log.py)
One row per game: ids, date, season, game_type (regular/playoff/play-in), teams, finals, `result`
(scoregasm / edged / none / unchecked), details of the tie or near miss, `line`, `over_under`, and `scores`
(every scoring change as `period,seconds_left,away,home;...`, about 1 KB per game). `unchecked` = ESPN play-by-play
didn't add up; retried on every later run.

## Open items
- Run the 2025-26 catch-up into the archive (command above), then check the season's hit rate against experiments.md #6.
- After opening night: confirm the Tonight list shows lines ("LINE BOS −5.5"). ESPN's odds format couldn't be checked
  when this was written; if lines are missing, fix `read_line` / `read_odds`.
- Ideas logged in experiments.md: does over/under add anything after the spread; pick'em games run hot (4.3% actual vs 3.5%);
  smarter catch-up (depends on time left too); late-game fouling.
- Possible features discussed, not built: text/phone alerts (ntfy, run from a computer during games), a free-to-play group
  pick game with bar nights. **No real-money features** (entry fees, cash prizes, sweepstakes coins): gambling law.
