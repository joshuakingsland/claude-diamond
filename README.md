# claude-diamond

MLB moneyline, run line, and totals modelling with an auditable market
baseline, point-in-time features, and weather.

This repository does not place wagers and does not claim an edge. It exists
to answer one question honestly: **does a model beat the price?** Everything
is arranged so that the answer can be no.

## Why this exists

A sibling project models UFC fights. Its problem is not the model, it is the
sample: 3,227 fights since 2019, a forward test that produces roughly two
wagers a month, and a live gate that is years away on current throughput.
Every methodological question there — is the edge rule right, is the extreme
bucket real, do sharp books lead — is unanswerable because the evidence
arrives too slowly.

MLB plays 2,430 regular-season games a year. Five seasons is 12,400 games,
nearly four times the entire UFC record, and it can be pulled in under a
minute from a free API. The point is not that baseball is a better bet. It
is that baseball can *answer questions*, and the methodology is what is
actually being tested.

## What is established, and what is not

| Question | Status |
| --- | --- |
| Does the model predict baseball? | **Yes.** Calibration, log loss, Brier against 12,400 real games |
| Does the model beat a price? | **No.** 186 days of 2025 prices; the close wins on the moneyline and run line with intervals excluding zero |
| Should anyone stake money? | **No.** The live gate reads `research_only` and there is no path to `live` in this code |

A well-calibrated model that loses money is the *normal* outcome in a liquid
market, and that is what happened here. The two claims are reported
separately — `validate.py` imports the market verdict from `market.py` rather
than inferring it from accuracy — because conflating them is the most common
way this kind of project fools its author.

## Data, all free and keyless

| Source | Used for | Notes |
| --- | --- | --- |
| MLB StatsAPI | schedules, results, linescores, probable pitchers, venues | one request per season; no key |
| Open-Meteo | first-pitch weather, archive and forecast | reanalysis and forecast share units and variables |
| The Odds API | prices for all three markets | needs `ODDS_API_KEY`; the only paid input |

## Three decisions worth knowing about

**One distribution, three markets.** Moneyline, run line, and total are read
off a single joint distribution over (home runs, away runs) in `runs.py`.
Pricing them with three separate models is how you end up quoting a total
that contradicts your own moneyline and calling the contradiction an edge.
Runs are modelled as negative binomial rather than Poisson because team runs
per game have roughly twice the variance a Poisson allows.

**Ties are resolved, not deleted.** Baseball has no draws. Zeroing the
diagonal of the joint distribution would be the easy fix and would bias every
total downward, because extra innings add runs. Tied mass is moved into extra
innings using a home edge and run bump measured from the games themselves.

**Weather comes from one source for training and serving.** StatsAPI reports
observed conditions, but only once a game is under way. Training on that and
serving on a forecast would fit the model to information the live path never
has. Open-Meteo supplies both, so it is the single source of truth, and
StatsAPI weather is used only to check the join. The same logic governs
roofs: 1,764 games in 2021-2025 report "Roof Closed", but that is known
after the fact, so the model sees the park's roof *category* and learns the
average attenuation instead.

## Point-in-time discipline

`features.py` walks games in date order, emits each feature row from current
state, and only then folds that game's result into state. A feature
therefore cannot see its own outcome by construction rather than by care.

This is enforced by a test that rewrites one game into a 30-0 blowout,
rebuilds, and asserts every feature row up to and including that game is
byte-identical — and that later rows *did* move, so the test cannot pass on a
builder that ignores results entirely.

## Layout

```
config.py     markets, regions, gates, staking policy
results.py    StatsAPI schedule and result ingestion
parks.py      venue coordinates, elevation, orientation, roof category
weather.py    Open-Meteo archive/forecast, wind resolved onto park axes
features.py   point-in-time feature construction
runs.py       joint run distribution; the three markets are read off it
models.py     expected-runs estimators and the pricing layer
validate.py   walk-forward validation and the honest report
odds.py       three-market odds capture with paired-book de-vig
historical_odds.py  capped, resumable historical snapshots
market.py     joins prices to predictions; asks whether the model beats them
```

## Automation

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `tests.yml` | push, PR | Unit tests, plus a check that odds capture exits clean with no key |
| `odds.yml` | hourly 15:00-03:00 UTC in season | Captures the upcoming card; commits lines and quotes |
| `backfill-odds.yml` | manual | Capped historical capture; dry run by default |
| `revalidate.yml` | weekly, manual | Rebuilds features, re-runs the comparison and validation |

`odds.yml` needs an `ODDS_API_KEY` repository secret and runs
`odds.py --require-key`, so a missing secret fails loudly rather than
reporting green while capturing nothing.

Capture costs one credit per region per market, so 6 per run and about 3,100 a
month on the schedule above; it does not scale with the size of the card. The
historical endpoint is far more expensive at 30 credits per snapshot, which is
why the backfill is manual, capped, and resumable from its manifest.

Scheduled workflows only run on the default branch, and GitHub disables them
after 60 days without repository activity. A gap in `data/market_quotes/` is a
data problem, not only an ops one, so check the files rather than trusting the
cron.

## Running it

```bash
python -m pip install -r requirements.txt
python parks.py --refresh
python results.py --seasons 2021-2025
python weather.py            # resumable; rerun to fill any failed venue
python features.py
python market.py             # before validate.py, which imports its verdict
python validate.py
PYTHONPATH=. python -m unittest discover -s tests -t .
```

Or `./run_pipeline.sh`, which runs the same chain in the same order.

`weather.py` is resumable by design: a venue that fails a TLS handshake is
reported and skipped, and rerunning fills only what is missing.

## First result

Walk-forward over 2022-2025, training only on prior seasons. 9,917 games.

| Market | Model log loss | Baseline | Calibration error |
| --- | ---: | ---: | ---: |
| Moneyline | **0.67937** | 0.69153 (home-field constant) | 0.021 |
| Total 8.5 | 0.68945 | — | 0.024 |
| Run line -1.5 | 0.64371 | — | 0.052 |

The moneyline interval is [0.6757, 0.6817] season-clustered, wholly below the
baseline. The model predicts baseball.

## Second result: it does not beat the price

186 daily snapshots across the 2025 season, 2,395 priced events, joined to
games on the team pair and the scheduled start. Model minus market log loss,
so **positive means the market is better**, with a 90% interval resampled over
slates rather than games.

| Market | Games | Δ log loss vs close | 90% interval | Verdict |
| --- | ---: | ---: | :---: | --- |
| Moneyline | 2,289 | +0.00542 | [0.0013, 0.0097] | Market better |
| Run line -1.5 | 1,272 | +0.00846 | [0.0033, 0.0135] | Market better |
| Total 8.5 | 695 | +0.00191 | [-0.0046, 0.0086] | Undecided |

This is the answer the repository was built to be able to receive. A model
that beats a home-field constant by 0.012 of log loss loses to the closing
price by 0.005, on the same games, in the same season. Nothing here is
evidence of edge.

The totals row is the one worth staring at. Against the *entry* price the
model is ahead by 0.00007 of log loss, which sets `model_beats_market: true`
and means nothing whatsoever — its interval is [-0.0075, 0.0073]. A boolean
comparison of two point estimates is exactly the trap the rest of this
project is arranged to avoid, so `market.py` now reports the interval and a
verdict beside the flag.

### The join that was hiding a tenth of the card

The first comparison ran on 3,601 game-markets. It should have been 4,314.
Two mechanical failures were dropping 9.5% of priced events, and neither was
visible in the output, because a dropped game leaves a smaller clean sample
rather than an error:

- **The Athletics rename.** StatsAPI dropped the city for 2025 while the books
  carried "Oakland Athletics" all season, so every A's game — 129 keys — fell
  out of the join.
- **UTC date rollover.** The join keyed on the UTC calendar date of first
  pitch, but a 19:10 Pacific start is 02:10 UTC the *next* day. That removed
  68 more, and removed them non-randomly: late West Coast games only.

Matching on the team pair plus the closest scheduled start fixes both, and
separates the two halves of a doubleheader as a side effect. Unmatched events
fell from 201 to 21, and the corrected sample moved the moneyline gap *against*
the model, from +0.0038 to +0.0054.

The lesson rhymes with the dispersion bug below: the dangerous errors are the
ones that leave the output looking reasonable.

### The bug that made this look impossible

The first run scored 0.6954 on the moneyline — worse than a constant. The
estimator was fine; the *width* of the distribution was measured wrong.

`fit_dispersion` was being handed the model's own training predictions. A
gradient boosted estimator's in-sample residuals are far too small, so the
method-of-moments fit concluded the runs were not over-dispersed at all,
pinned the negative binomial size at its clamp ceiling of 50, and produced a
run distribution far tighter than baseball. Win probabilities then ran from
0.05 to 0.97 on games that are close to coin flips, and the calibration table
showed the damage plainly: games priced at 0.845 came in at 0.593.

Measuring dispersion on a held-out fold instead moved it to 2.4-4.3 and the
moneyline from 0.6954 to 0.68737 for the same estimator.

The lesson generalises past this repository: an in-sample estimate of
*uncertainty* is far more dangerous than an in-sample estimate of the mean,
because it does not look wrong. The point predictions were unbiased the whole
time — mean predicted runs 4.44 against 4.43 actual.

### Why the linear model wins

The Poisson GLM beats the gradient booster on every market. That is the
expected result for a low signal-to-noise problem: a single MLB game is close
to a coin flip, the honest spread of win probability is roughly 0.35-0.65, and
a flexible learner spends its capacity fitting noise it cannot distinguish
from signal. `--kind glm` is the default for that reason.
