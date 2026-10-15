# Scoregasm experiments log

A theory only counts once it's written down BEFORE you look at the data, with a prediction that could be wrong.

## How to run an experiment (the recipe)
1. **Theory** in one sentence. ("Late-season games are more likely to hit 69-69.")
2. **Prediction**, with a number and what would prove it wrong. Write it first.
3. **Data**: which file, which seasons. Check it is the file you think it is (row count, first rows).
4. **Test on games the idea was not built from.** Fit on some seasons, check on others.
5. **Verdict** in plain words: supported / not supported / can't tell. Say how likely luck is.
6. **Next step**: what would make you change the site? If nothing would, say so.

Rule of thumb: 10 seasons is a small sample. A pattern with a 1-in-20 chance of being luck is a hint, not a fact.

---

## Done

| # | Theory | Result | Verdict |
|---|---|---|---|
| 1 | The live odds are well calibrated | 2015-18 (unseen): predicted 2.07% vs actual 2.02%. Mid-range odds accurate; very low odds (<2%) ran high | Supported |
| 2 | Tip-off odds (3.57%) match real games | 2015-20: 3.46%. 2020-25: 3.13% (about 28 fewer than expected, ~6% chance of luck) | Mostly; recent dip unexplained |
| 3 | Updating to each season's league averages fixes the dip | Adjusted odds stayed 3.5-3.7% every season; did not explain the dip | Not supported |
| 4 | Late-season (April) games behave differently | Last 10% of season 3.18% vs 3.29% rest (p=0.93). April alone 2.47% on 365 games (noise) | Not supported |
| 5 | Stronger-vs-weaker matchups hit less often (proxy: last season's margin) | Slope about zero (p=0.76), no gain on held-out seasons | Not supported with this proxy |
| 7 | A real betting spread predicts 69-69 better than my proxy | 11,670 games 2015-25 with closing lines (SportsbookReview archive 2015-22, kyleskom odds database 2022-25). Pick'em games 4.3% vs 10+ pt favorites 2.3%. Spread built into the possession model (scale 1.4, theory 1.2); tuned on 2015-19 it beat the old model on 2020-24, and the reverse; better in 99.8% of game resamples. Also explains most of #2's gap: averaged over real matchups the model now says 3.21% vs 3.28% actual | Supported, shipped 2026-10-13 |

## Open (written prediction, waiting on data)

| # | Theory | Prediction (written 2026-10-05) | Needs |
|---|---|---|---|
| 6 | More free-throw attempts per game means more 69-69s (correlation 0.61 across 10 seasons) | 2025-26 had 23.5 FTA/game, same as 2022-23. Expect a hit rate of 3.4% or higher. If it is below 3.0%, the idea is probably wrong | Run the 2025-26 ESPN catch-up command and read the result |
| 8 | Very low odds (<2%) run too high on older seasons only | Should look fine on 2020-25 live forecasts | 2022-24 raw play-by-play (nbastats files) |

## Ideas not yet tried
- Does the over/under add anything once the spread is in? (Raw look: only a weak effect, 3.4% low totals vs 2.9% very high.)
- Pick'em games still run hot (4.3% actual vs 3.5% predicted, about 2 standard errors). Is the catch-up effect stronger in even games?
- Do rest days or back-to-backs change the odds?
- Are home or away teams more often the one to tie it?
- Do overtime-bound games (close at the end) visit 69-69 more?
- Does the tying play type (free throw vs three) change with the era?
