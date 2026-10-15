# Scoregasm

A Streamlit site showing the live chance that an NBA game passes through **exactly 69–69** in regulation,
plus a Hall of Fame, a team Tease board and a Stats page showing how the model was tested.

## Run it

```
pip3 install -r requirements.txt
python3 -m streamlit run scoregasm.py
```

The site reads ESPN's public scoreboard (no key needed) and opens at http://localhost:8501.
In a Codespace it installs and starts by itself.

## What's here

- `scoregasm.py` is the whole site: the model, the ESPN live feed and the four tabs.
- `update_log.py` logs every finished game to `scoregasm_season.csv`. GitHub Actions runs it every night
  (`.github/workflows/update-log.yml`).
- `scoregasm_archive.csv` holds past seasons (2015-16 to 2024-25) for the Hall of Fame and Stats.
- `deficit_times.csv` says when the biggest comeback hole happened, for older archive games.
- `research/` has the scripts that rebuild the archive and re-test the betting-spread model.
- `experiments.md` is the log of theories tested, with predictions written before looking at data.
- `CLAUDE.md` has fuller notes on the model, the site's language and open items.
