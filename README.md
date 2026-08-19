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

MLB plays 2,430 regular-season games a year. Six seasons is 14,580 games,
more than four times the entire UFC record, and it can be pulled in under a
minute from a free API. The point is not that baseball is a better bet. It
is that baseball can *answer questions*, and the methodology is what is
actually being tested.

## What is established, and what is not

| Question | Status |
| --- | --- |
| Does the model predict baseball? | **Yes.** Calibration, log loss, Brier against 13,857 completed games |
| Does the standalone model beat a price? | **No.** On the coherent main-line comparison the close wins all three markets with date-clustered intervals excluding zero |
| Is there a credible route to an edge? | **Yes, but not from the model.** Taking the best price when one book sits 0.25 points off the panel's own fair consensus is worth **+0.31 points of CLV against Pinnacle's close** [+0.07, +0.49], robust to dropping any single book. The model's own 24-hour movement signal is worth **0.4-0.6 points** on the moneyline and run line, but live CLV is **-0.52 points** at a 0% fill rate |
| Should anyone stake money? | **No.** 70 shops over 13 dates is not a sample, and the study cannot see whether the outlier book would accept the bet. The promotion gate requires 500 independent games, 95% accepted fills, and positive sharp-close CLV; there is no real-money path in this code |

A well-calibrated model that loses money is the *normal* outcome in a liquid
market, and that is what happened here. The two claims are reported
separately — `validate.py` imports the market verdict from `market.py` rather
than inferring it from accuracy — because conflating them is the most common
way this kind of project fools its author.

## Data, all free and keyless

| Source | Used for | Notes |
| --- | --- | --- |
| MLB StatsAPI | schedules, results, linescores, probable pitchers, venues | one request per season; no key |
| Open-Meteo | first-pitch historical forecast and live forecast | training uses the forecast available pregame, not realised reanalysis |
| The Odds API | full-game prices plus an isolated first-inning totals audit | needs `ODDS_API_KEY`; the only paid input |

## Three decisions worth knowing about

**One distribution, three markets.** Moneyline, run line, and total are read
off a single joint distribution over (home runs, away runs) in `runs.py`.
Pricing them with three separate models is how you end up quoting a total
that contradicts your own moneyline and calling the contradiction an edge.
Runs are built from innings rather than from games: an inning is scoreless
about 74.7% of the time, and a game is nine of them convolved. The older
game-level negative binomial is still in `runs.py` behind one argument, and
the reason it was replaced is that matching the mean and the variance uses up
both its parameters and leaves the shape wrong — it puts a fifth too little
mass on a shutout.

**Ties are resolved, not deleted.** Baseball has no draws. Zeroing the
diagonal of the joint distribution would be the easy fix and would bias every
total downward, because extra innings add runs. Tied mass is moved into extra
innings using a home edge and run bump measured from the games themselves.

**Weather comes from the same information set for training and serving.**
StatsAPI reports observed conditions, but only once a game is under way.
Training on realised weather and serving on a forecast fits information the
live path never has. Historical games therefore use Open-Meteo's archived
operational forecast and live games use its current forecast endpoint with
the same variables and units. StatsAPI weather only checks the join. The same logic governs
roofs: 2,034 games in 2021-2026 report "Roof Closed", but that is known
after the fact, so the model sees the park's roof *category* and learns the
average attenuation instead.

## Point-in-time discipline

`features.py` walks games in actual first-pitch order, emits each feature row from current
state, and only then folds that game's result into state. A feature
therefore cannot see its own outcome by construction rather than by care.

`schedule_snapshots.py` and `lineup_snapshots.py` add the other half of that
contract: probable starters, status, and submitted batting orders are captured
append-only before pricing. The live builder selects only versions available
at decision time, so a late scratch cannot rewrite an earlier card, and the
paper policy rejects a row until both batting orders are confirmed. The
repository does not pretend to reconstruct historical starter or lineup
snapshots that were never captured.

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
schedule_snapshots.py  append-only pregame probable-starter provenance
lineup_snapshots.py  append-only confirmed batting orders
features.py   point-in-time feature construction
runs.py       joint run distribution; the three markets are read off it
models.py     expected-runs estimators and the pricing layer
validate.py   walk-forward validation and the honest report
odds.py       three-market odds capture with paired-book de-vig
historical_odds.py  capped, resumable historical snapshots
full_game_event_odds.py  per-event full-game closes, sharded by season
full_game_close_evaluation.py  frozen walk-forward model-versus-close report
full_game_movement_evaluation.py  sealed 24-hour-to-close movement study
movement_forecast.py  frozen 24-hour movement artifact and serving gate
first_inning_odds.py  capped historical YRFI/NRFI market-coverage audit
first_inning_results.py  exact first-inning result labels from MLB linescores
first_inning_report.py  market-only data-integrity report; not a prediction model
first_inning_model_evaluation.py  frozen 2023/2024/2025 market-anchored YRFI test
first_inning_open_odds.py  resumable, fixed-ladder YRFI/NRFI opening archive
first_inning_open_evaluation.py  frozen open-to-close movement and CLV test
market.py     joins prices to predictions; asks whether the model beats them
market_offset.py  constrained market-logit residual and price-movement fit
forward_evidence.py  accepted fills, independent games, sharp-close CLV gate
signal_ledger.py  append-only, non-wager forward probe of predicted movement
extremes.py   does a large disagreement pay? no, and here is why it looks like it does
devig.py      four ways to strip the margin, and whether the benchmark book is right
line_shopping.py  does the best price beat the close? the one arm that says yes
alerts.py     live shop alerts, an append-only log, and an honest expiry clock
alert_evidence.py  scores the alerts that actually fired; cannot be re-selected
stationarity.py  does the run environment drift enough to reweight the seasons?
mean_calibration.py  the predicted means are over-spread; correcting them makes pricing worse
statcast.py   Baseball Savant pitch data, aggregated per player-game
umpires.py    home-plate umpire assignments, one per game
umpire_effect.py  permutation tests for an umpire main effect
leverage.py   win expectancy, for leverage-weighting bullpen workload
predict_upcoming.py  prices the live card; the only forward-looking path
ledger.py     the paper forward test, and where the staking policy is applied
model_card.py generates the public page at docs/index.html
merge_data.py union/replace rules for two runs committing at once
commit_data.sh  commit and push generated data, merging on collision
```

## The public page

`model_card.py` renders `docs/index.html` from files already in the repository
— the board, the ledger, the rejections, the market comparison, the validation
report. Nothing is recomputed, so if a number on the page disagrees with the
repo the page is stale rather than right, which is why the header carries the
timestamp of the data rather than of the render. It regenerates on every
capture.

The one exception to "nothing is recomputed" is the shop panel at the top, and
it is an exception on purpose. Those alerts decay in about ninety seconds while
the page is written hourly, so no freshness judgement is baked in at render
time: each row carries its capture stamp and the measured survival curve, and
the browser computes the age against the reader's own clock, re-ticking every
fifteen seconds. A static page that stamped a row "live" would be wrong within
two minutes of being written and would stay wrong for an hour.

The layout deliberately mirrors the sibling UFC project's page so the two read
the same way. The content does not, because the results are not the same:
the verdict — that the closing price beats this model on the moneyline — sits
*above* tonight's card rather than below it. A model card that
opens with picks and buries the measurement is advertising.

The module is `model_card.py` and not `site.py` because Python imports a
stdlib module called `site` at interpreter startup; a file of that name in the
repository root is a trap.

Fonts are served from `docs/fonts` rather than fetched from Google: static
files committed once, while the page itself is rewritten every capture.
Embedding them as data URIs would add about 190KB of base64 to an hourly
commit, and the latin subsets are all this needs.

A locked wager is displayed as it was struck, not as the board reads now. The
market moves after a lock — one taken at five books was showing the single
book still quoting the line, which reads as a wager that breached the
book-count gate. The recorded quote is also the one it settles against.

To serve it: **Settings → Pages → Source: Deploy from a branch → `main` /docs**.

## The live path

`odds.py` captures the board, `predict_upcoming.py` prices the same games, and
`ledger.py` decides which of them the staking policy would have taken.

```bash
python odds.py --require-key      # tonight's board
python schedule_snapshots.py      # probable starters known at decision time
python lineup_snapshots.py        # submitted batting orders
python predict_upcoming.py        # model probabilities beside it
python signal_ledger.py           # freeze the CLV hypothesis before the close
python ledger.py                  # screen, record, settle
python forward_evidence.py        # independent-game and sharp-close gate
```

The standalone model is no longer treated as the fair-price prior. The card
starts from the paired-book, de-vigged market logit and retains only the
constrained fraction of the model-market residual supported in expanding-date
outcome tests. Currently that fraction is zero in all three markets:
market-only. The completed per-game archive separately confirms that 24-hour
market microstructure predicts the 20-minute close in all three markets, with
the largest result on totals. `movement_forecast.py` serves that frozen model
only between 23 and 25 hours before first pitch and feeds a non-wager CLV
probe. It is never extrapolated into the late lock window. The raw model gap
remains on the card as a diagnostic.

Three properties keep the live path honest. It builds features with the *same*
builder as training, which emits a row for an unplayed game and folds nothing
into state. It takes forecast weather from Open-Meteo, matching the
historical-forecast training contract, and keeps it in a separate file. It drops games
already under way, because the odds feed keeps quoting them and those prices
reflect a score the model cannot see.

Only one executable main point per market reaches the card: the point quoted
by the broadest set of priced books. Alternate lines remain in the raw quote
log for research. This prevents book-subset changes from producing both sides
of a run line or several correlated totals. Historical evaluation reconstructs
the stored joint distribution at the exact main point that was quoted instead
of validating fixed -1.5 and 8.5 lines against a different universe.

Whole-number lines push, and a book's two-way price has that mass removed
already, so the model probability is renormalised onto the same basis before
the two are compared. A live board quotes totals from 7 to 10; leaving this
alone understates every whole-number line on it.

### The staking policy is now applied

`config.py` declared an edge rule, a stake, a day cap, lock timing and
execution limits, and **not one of them was referenced by any code**. A control
that is written down and never enforced reads to a later auditor as a control
that was in force. `ledger.py` applies them, and writes every rejection down
with the gate that caused it:

| Gate | Constant |
| --- | --- |
| Disagreement too small | `EDGE_RULE` |
| Market too thin | `MIN_MARKET_BOOKS` |
| Batting orders not confirmed | `REQUIRE_CONFIRMED_LINEUPS` |
| Outside the lock window | `MIN/MAX_LOCK_LEAD_MINUTES` |
| Quote too old | `MAX_ODDS_AGE_MINUTES` |
| Best book's own update too old | `MAX_BOOK_QUOTE_AGE_MINUTES` |
| Best price too far from consensus | `MAX_EXECUTION_DEVIATION` |
| Expected profit at executable price too small | `MIN_EXPECTED_VALUE` |
| Correlated game bucket already used | `GAME_RISK_BUCKET_STAKE_CAP` |
| Day's stake exhausted | `GAME_DAY_STAKE_CAP`, `MAX_STAKE` |

Books disagreeing among themselves past `MARKET_DISAGREEMENT_WARNING` are
flagged rather than dropped, because config calls it a warning and a forward
test that silently discards its awkward rows is not measuring anything.

The day cap is counted against the ledger, not against the run. The capture
workflow screens the same card every hour, so a cap computed per call would
hand out a fresh allowance each time — thirteen runs against a three-unit cap
is thirty-nine units on a day the policy limits to three. That is not
hypothetical: it happened on the first live day, six units before it was
caught, which is why the paper ledger starts from the day the cap began
holding across runs rather than from the first capture.

**No money moves.** Wagers are recorded at a price that was on the board and
settled against real results. The expected outcome is a loss, and the ledger is
worth running because it is the measurement that would show that honestly.
The ledger records the git revision, ordered feature-schema hash, trained
through date, offset version, actual book update time, and policy version, so
materially different decisions cannot collapse under one static model label.

Because outcome evidence currently supports no wagers, `signal_ledger.py`
separately freezes price-movement predictions. The confirmed 24-hour model is
recorded as `paper_clv_probe`; the older lock-window diagnostic remains a
`paper_quote`. At most one main-line observation per game risk bucket and
target is kept, and neither is an accepted fill. This lets later sharp-close
CLV test the only signal that survived confirmation without loosening the
wagering gates to manufacture a sample.

## Automation

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `tests.yml` | push, PR | Unit tests, plus a check that odds capture exits clean with no key |
| `odds.yml` | hourly 15:00-03:00 UTC in season | Refreshes inputs and starter snapshots; captures the main board, prices, screens, alerts on shop opportunities, and refreshes forward evidence |
| `odds-burst.yml` | daily 21:50 UTC in season, manual | 90-second polling across the evening card, screening each poll for shop alerts; this is the only cadence that catches them |
| `backfill-odds.yml` | manual | Capped historical capture; dry run by default |
| `first-inning-audit.yml` | manual | One-market, one-region historical YRFI/NRFI coverage audit; dry run by default |
| `first-inning-study.yml` | manual | Stratified, capped historical first-inning sample plus official outcome labels; dry run by default |
| `first-inning-labels.yml` | manual | Free refresh of StatsAPI labels and the market-only baseline; no odds request |
| `first-inning-open-ladder.yml` | manual | Reuses the settled cohort to collect a frozen opening-price ladder; dry run by default and no betting |
| `automated-full-game-early-backfill.yml` | daily, manual | Capped 24-hour snapshots and sealed entry-to-close evaluation; no betting |
| `continue-full-game-early-backfill.yml` | successful early batch | Launches one non-overlapping successor until provider coverage is complete; stops on failure |
| `revalidate.yml` | weekly, manual | Builds predictions once, then market comparison, offset fit, forward evidence and final reports in provenance-safe order |

`odds.yml` needs an `ODDS_API_KEY` repository secret and runs
`odds.py --require-key`, so a missing secret fails loudly rather than
reporting green while capturing nothing.

Capture costs one credit per region per market, so 6 per run and about 3,100 a
month on the schedule above; it does not scale with the size of the card. The
historical endpoint is far more expensive at 30 credits per snapshot, which is
why the backfill is manual, capped, and resumable from its manifest.

`odds.py` records the credit balance the API reports on every run and fails
below `--min-credits`. That quota is shared with anything else using the same
key, so it can drop for reasons this repository cannot see, and the symptom of
running dry is a capture that quietly stops returning data.

Scheduled workflows only run on the default branch, and GitHub disables them
after 60 days without repository activity. A gap in `data/market_quotes/` is a
data problem, not only an ops one, so check the files rather than trusting the
cron.

### First-inning totals: research track, not a pick feed

`totals_1st_1_innings` is the correct two-way market for an NRFI/YRFI-style
first-inning over/under. It is kept out of `MARKETS`, the full-game model, and
the paper ledger: first-inning pricing needs separately frozen pitcher,
lineup, and outcome data before it can be evaluated honestly.

Start with exactly one historical event, ten minutes before first pitch:

```bash
python first_inning_odds.py --date 2026-08-10 --max-events 1 --dry-run
# Remove --dry-run only after checking the stated ceiling.
python first_inning_odds.py --date 2026-08-10 --max-events 1 --lead-minutes 10
```

The paid run writes `data/first_inning_audit.csv` and
`data/first_inning_quotes.csv`.  `offered` means the feed returned at least
one paired Over/Under price; `no_offer` is equally useful evidence that the
book/region/date cannot support the study.  Every audit id is keyed by event,
market, region, and requested timestamp, so it will not rebuy an already
checked snapshot.  The GitHub Actions equivalent is the manual
`first-inning-audit.yml` workflow, whose default is also dry-run.

After coverage is confirmed, `first-inning-study.yml` takes a deterministic,
evenly date-stratified sample across the available seasons. Its initial
configuration makes at most 500 event-odds calls at one pregame timestamp:
about 5,000 credits plus the small event-discovery overhead. The script then
joins each offered event to the exact first-inning MLB linescore and writes a
market-only Brier/log-loss baseline. It does **not** build a first-inning
model, treat a historical starting lineup as a pregame snapshot, or display a
YRFI/NRFI pick. Those would be later, separately validated stages.

The completed 5,000-attempt close study still found no confirmed YRFI edge, so
the public website intentionally has no first-inning pick feed. The next frozen
question is whether the earliest broadly available quote predicts its own
ten-minute close. `YRFI_OPEN_RESEARCH_PLAN.md` fixes the protocol before any
new data is purchased: reuse the same event IDs and sample 1,440, 720, 360,
180, and 60 minutes before first pitch. Only settled, multi-book-close games
can enter the analysis. That makes the 2023/2024 development stage 11,240
event-odds calls (about 112,400 credits). It stops there if selection fails;
only a locked candidate unlocks the 5,215-call 2025 confirmation. Maximum
useful spend is therefore about 164,550 credits, with no new event discovery
spend and no retrospective spend on excluded 2026 outcomes.

Estimate a batch without spending credits:

```bash
python first_inning_open_odds.py --max-calls 1000 --seasons 2023,2024 --dry-run
```

The matching `first-inning-open-ladder.yml` workflow is manual, shares the
paid first-inning concurrency lock, checkpoints every 100 calls by default,
and cannot write a pick or a wager. A real run must be explicitly launched
with `dry_run` unchecked. `first_inning_open_evaluation.py` keeps 2025 sealed
until development is complete and a candidate is locked; the paid collector
enforces the same gate. It measures predictable close movement and best-price
closing-line value first, and treats historical win/loss ROI as secondary
evidence.

### The two days the page was empty

Worth writing down, because every workflow was green throughout.

`revalidate.yml` passed `results.py --seasons 2021-2025`. That default was
correct when it was typed and became wrong on 1 January 2026, and `results.py`
*replaced* the game table with whatever it was told to fetch — so the weekly
run deleted the entire 2026 season, 2,430 rows. Nothing errored. The capture
kept buying odds every hour and committing them. But no game on the board
existed in the table any more, so every event failed to match, the card was
written with a header and no rows, and the public page showed nothing at all.

Three separate things had to be true for it to be silent, and all three are
now fixed:

- **A stale year in a default.** `results.py` and `parks.py` now derive the
  range from the current date. An end year written into a default is a bug
  with a delayed fuse.
- **A write that could delete.** `results.py` now replaces only the seasons a
  fetch actually returned games for, and keeps every other season. Correcting
  the year would have fixed the instance; this makes the class impossible. An
  empty or failed fetch now erases nothing.
- **An empty card that looked like an ordinary night.** `predict_upcoming.py`
  now separates the two cases. No future events on the board is fine — an off
  day, or every game already started. Future events of which *none* can be
  priced is a failure and exits non-zero. The odds are already on disk and the
  commit step runs on `always()`, so failing costs no data; it just turns a
  silent nothing into a red run.

Two adjacent gaps surfaced while fixing it. `boxscores.py` and `umpires.py`
were in **no** workflow and had only ever been run by hand, which decays
quietly rather than loudly — a starter whose recent outings were never
ingested keeps his old rating and then falls back to a league prior, and the
model goes on pricing with no error anywhere. And results were refreshed
weekly, so a card could be priced on team form up to seven days stale. All
four ingesters now run on every capture; each is resumable and keyless, so the
steady state is about fifteen games a day.

## Running it

```bash
python -m pip install -r requirements.txt
python parks.py --refresh --seasons 2021-2026
python results.py --seasons 2021-2026
python weather.py --refresh-source  # historical operational forecasts
python features.py
python validate.py --kind glm --predictions data/predictions_glm.csv
python market.py
python market_offset.py
python forward_evidence.py
python validate.py --kind glm --reuse-predictions \
  --predictions data/predictions_glm.csv
PYTHONPATH=. python -m unittest discover -s tests -t .
```

Or `./run_pipeline.sh`, which runs the same chain in the same order.

`weather.py` is resumable by design: a venue that fails a TLS handshake is
reported and skipped, and rerunning fills only what is missing.

### Completed per-game close archive

The 2020-2024 per-event archive is stored as yearly parts under
`data/full_game_event_quotes/`; no part approaches GitHub's 100 MB file limit.
Run the frozen comparison with:

```bash
python full_game_close_report.py
python full_game_close_evaluation.py
```

The completed 2022-2024 comparison contains 6,914 games with walk-forward
predictions and qualified closes. The closing market beats the standalone GBM,
GLM, and equal-weight ensemble on all three markets, including the 2024
confirmation block. A development-selected market-anchored offset also finds
no confirmed incremental signal. See `FULL_GAME_CLOSE_EVALUATION.md` and
`full_game_close_evaluation.json`. Nothing in that result authorises a wager.

### Frozen entry-to-close movement study

The next experiment treats the market as the baseline rather than trying to
replace it. The completed per-game archive supplies the 20-minute close; a
separate resumable collector adds one 24-hour snapshot for 2022--24. New early
rows are written to `quotes_early_YYYY.csv` parts so no season file approaches
the host's large-file limit.

The protocol is frozen before those rows arrive: fit candidates on 2022,
select exactly one on 2023, and keep 2024 sealed until every provider event has
an attempted early snapshot. The primary target is moneyline close-logit
movement versus a no-move baseline. Run lines and totals are secondary and
enter only when their main point is unchanged. A date-clustered 95% interval
must clear zero before the result can be called a research signal; even then it
feeds only the existing paper CLV ledger and never authorises a wager.

```bash
python full_game_event_odds.py --start 2022-01-01 --end 2024-09-29 \
  --max-events 100 --lead-minutes 1440 --snapshot-role early --dry-run
python full_game_movement_evaluation.py
```

The automated workflow is capped at 1,000 paid event attempts per day. Full
2022--24 coverage is 7,358 attempts, roughly 220,740 odds credits. The
completed close manifest is reused as the event catalog, eliminating recurring
historical-discovery calls. The last completed close row reported more than 4.58
million credits remaining, but the audit still records the live balance on
every paid response.

## First result

Walk-forward over 2022-2026, training only on prior seasons. 11,428 games.

| Market | Model log loss | Baseline | Calibration error |
| --- | ---: | ---: | ---: |
| Moneyline | **0.68102** | 0.69151 (home-field constant) | 0.022 |
| Total 8.5 | 0.68784 | — | 0.019 |
| Run line -1.5 | 0.64603 | — | 0.051 |

The moneyline interval is [0.6781, 0.6843] season-clustered, wholly below the
baseline. The model predicts baseball.

## Second result: it does not beat the price

186 daily snapshots across the 2025 season, 2,395 priced events, joined to
games on the team pair and the scheduled start. Model minus market log loss,
so **positive means the market is better**, with a 90% interval resampled over
slates rather than games.

| Market | Games | Δ log loss vs close | 90% interval | Verdict |
| --- | ---: | ---: | :---: | --- |
| Moneyline | 2,230 | +0.00562 | [0.0022, 0.0090] | Market better |
| Run line -1.5 | 1,245 | +0.00361 | [-0.0014, 0.0086] | Undecided |
| Total 8.5 | 680 | +0.00363 | [-0.0027, 0.0098] | Undecided |

This is the answer the repository was built to be able to receive. A model
that beats a home-field constant by 0.010 of log loss loses to the closing
price by 0.006, on the same games, in the same season. Nothing here is
evidence of edge.

The totals row is the one worth staring at. Against the *entry* price the
model has at times come out marginally ahead — by 0.00007 of log loss on one
earlier build — which sets `model_beats_market: true` and means nothing
whatsoever, because the interval spans zero either way. A boolean comparison
of two point estimates is exactly the trap the rest of this project is
arranged to avoid, so `market.py` reports the interval and a verdict beside
the flag.

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
fell from 201 to 39, and the corrected sample moved the moneyline gap *against*
the model, from +0.0038 to +0.0059.

The lesson rhymes with the dispersion bug below: the dangerous errors are the
ones that leave the output looking reasonable.

### Does leverage-weighting rescue the bullpen feature? No

`features.py` carries raw bullpen outs over three days and earns +0.001 for
it — the right sign and nearly nothing. The hypothesis worth testing was that
the blunt measure is the problem rather than the idea: an out recorded in a
tied ninth should cost a reliever more than an out in a six-run game, and the
market adjusts for innings rather than for what those innings were worth.

`leverage.py` walks 1,045,047 plate appearances across all 13,857 games and
weights each by how far the game could still swing from that state. Four
measures of yesterday's bullpen, declared before looking, each as the gap
between the two sides and tested against the model's own residual:

| Measure | Correlation | Median split |
| --- | ---: | ---: |
| Leverage-weighted work | +0.0066 | +0.19 SE |
| High-leverage batters faced | +0.0131 | +1.35 SE |
| Raw outs — the blunt one | +0.0018 | +0.21 SE |
| Warm-up debt proxy | -0.0056 | +0.08 SE |

Every sign points the way the hypothesis predicts: a taxed away bullpen means
the home side beats the model. None of them clears noise, and the refined
measure is no better than the blunt one it was meant to replace — 0.19 SE
against 0.21.

The strongest, high-leverage batters faced, holds up no better in the tail the
hypothesis actually describes. Games where the away pen faced three or more
extra high-leverage batters yesterday run +0.016, on a date-clustered interval
of [-0.0001, 0.0328]; at six or more it is +0.011 on [-0.0094, 0.0324]. Both
touch zero, and the second is *weaker* than the first, which is the wrong
direction for a dose-response.

The warm-up debt proxy — late high-leverage batters the starter absorbed, so
the pen was up and throwing without entering — is the flattest of the four.
The observable half of that idea is not there. The unobservable half needs
bullpen camera footage, which is a different project.

### Do umpires move the moneyline? Not measurably

The plate umpire is the one input assigned to every game rather than a
fraction of them, which is why it was worth ingesting all 13,857 of them
before building anything on top. An interaction between an umpire's zone and a
pitcher's arsenal is necessarily *smaller* than the umpire main effect, so the
main effect is the precursor: if it is not there, nothing built on it can be.

Each test compares the spread of per-umpire means against the spread from
shuffling assignments within season. Umpires work different schedules, so some
spread is guaranteed by sample size alone; reporting the widest umpire instead
would be the mistake `extremes.py` documents.

| Measure | Implied between-umpire SD | p |
| --- | ---: | ---: |
| Strikeouts per game | 0.18 K | 0.052 |
| Residual total runs | 0.21 runs | 0.066 |
| **Residual home win probability** | **0.014** | **0.28** |

The moneyline — the market this would be bet into — shows nothing at all, and
no individual season comes below p = 0.30. Strikeouts and total runs are
marginal pooled and consistent in no season. One cell of sixteen season-level
tests came in at p = 0.002, which the null produces somewhere about 3% of the
time, and it sits in 2024 with neither adjacent season showing anything.

Taking the point estimates at face value anyway: one standard deviation of
umpire is worth 1.4 points of win probability, so an extreme umpire is worth
under three. The arsenal interaction is a fraction of that, and it would have
to be extracted from pitch-level data the repository does not have.

**What this cannot say** is anything about the challenge system. Teams now
carry two challenges a game and keep them when correct, which should erode
exactly this kind of bias, and the test was built to measure that. It cannot:
2026 shows no signal, but neither does 2025, so there is no established
pre-challenge baseline to have lost. The honest statement is that the effect is
too small to detect in either era, not that challenges removed it.

### Built anyway, and the walk-forward said no

A permutation test is an argument. The walk-forward is a verdict, and it is
cheap, so the umpire features were built and put in front of it rather than
argued about: `ump_run_rate` and `ump_k_rate`, each an umpire's own history
shrunk toward the contemporaneous league with a 60-game prior and expressed as
a *deviation* from it. Centring matters — strikeouts per game rose across these
seasons, so the raw rate is mostly a clock and only incidentally an official.

The prior is heavy on purpose. Before building the features, the one test that
decides whether a trait is buildable at all is whether it persists:

| Measure | Umpire-seasons paired | Year-over-year correlation | p |
| --- | ---: | ---: | ---: |
| Strikeouts per game | 268 | +0.091 | 0.14 |
| Residual total runs | 268 | **−0.087** | 0.16 |
| Residual home win | 268 | −0.031 | 0.61 |

An umpire's run tendency *anti-predicts* his next season. Whatever the spread
of per-umpire means contains, it is not a trait that carries forward, and a
feature can only exploit what carries forward.

The walk-forward agreed, and made it worse on all three markets with intervals
excluding zero:

| Market | Δ log loss | 90% interval |
| --- | ---: | --- |
| Moneyline | +0.00001 | [+0.00000, +0.00003] |
| Run line | +0.00001 | [+0.00000, +0.00003] |
| Total | +0.00011 | [+0.00001, +0.00022] |

The GLM had already reached the same conclusion on its own: it assigned
`ump_run_rate` +0.002 and `ump_k_rate` −0.005, against +0.018 for the park
factor and +0.082 for away defence. The model recognises there is nothing
there; the small, consistent degradation is the price of fitting two
parameters to noise.

So both columns are still built and still written to `data/features.csv`, and
neither is in `FEATURE_COLUMNS`. Re-enabling them is adding two strings, and
the reason not to is recorded next to them rather than left to be rediscovered.

Two bugs surfaced on the way, both worth naming because they are the same bug:
league strikeouts were double-counted (`× 2` for two teams, on a total that
already summed both), and a game with no boxscore was folded in as *zero*
strikeouts, which taught an umpire he called no strikes and got further from
the truth the more games he worked. Absence is now a `None` sentinel with its
own denominator, on the umpire and on the league both. A test caught the second
one by asserting the feature's spread — an inert feature has an sd near zero,
and 2.28 is not near zero.

### The first change that worked: the home ninth inning

The home team bats the bottom of the ninth only when it is not already ahead,
and stops the moment it goes ahead. The model gave both sides a full nine
innings, and the error showed up exactly where it should: across 11,428 games
it put 14.2% on the home side losing by one against 11.1% observed, and 15.2%
on winning by one against 17.1%. **The run line is decided at that boundary**,
which is why it was the worst calibrated of the three markets.

The correction costs nothing. A negative binomial with mean `mu` and size `d`
shares its `p` with the pieces `NB(8mu/9, 8d/9)` and `NB(mu/9, d/9)`, so the
eight innings and the ninth sum back to exactly the distribution already
fitted — no new parameter, only a rearrangement of when the ninth counts. A
walk-off then ends on the go-ahead run, and the winning margin is measured
from the games rather than assumed: 87% by one run, the rest by more.

(The splitting property is how this worked at the time. The inning model
below made the split literal rather than algebraic — the eight-inning piece is
now eight innings convolved — which removed the one approximation left in it,
that the ninth carries exactly a ninth of the mean.)

| | Before | After |
| --- | ---: | ---: |
| Run line log loss | 0.64545 | **0.63904** |
| Run line calibration error | 0.05653 | **0.01667** |
| Paired change, 90% interval | | **[-0.00676, -0.00616]** |
| Gap to the closing price | +0.00955 | **+0.00342** |
| Verdict | market better | **undecided** |

The interval excludes zero, and it is narrow because this is a structural
correction rather than a noisy estimate: given the same expected runs it moves
the same mass every time. The moneyline is untouched to four decimal places,
which is the right result — censoring redistributes home wins across margins
without creating or destroying any.

That is the whole of it. The run line was one of two markets where the closing
price demonstrably beat the model; it is now undecided. This did not come from
new information but from removing a wrong assumption, which on the evidence of
this repository is the better-paying kind of work.

### The second change that worked: an inning is mostly a zero

Prompted by asking which assumptions the model actually rests on. The answer
that turned out to be checkable in an afternoon was *runs are negative
binomial*, and it is wrong.

A negative binomial has two parameters. Matching the mean and the variance
uses up both of them, and there is nothing left to control the shape — so the
shape is whatever falls out, and it can be checked:

| | Observed | NB predicted | Miss |
| --- | ---: | ---: | ---: |
| P(away shutout) | 0.0719 | 0.0588 | **+22% relative** |
| P(home shutout) | 0.0609 | 0.0478 | **+28% relative** |
| Away variance | 10.601 | 10.491 | +1.0% |
| Home variance | 9.621 | 9.596 | +0.3% |

The variance is right, which says the out-of-sample dispersion fit is doing
its job. The shutouts are not. The first question is whether that is the
*family* being wrong or the model's means being too tightly bunched — both
produce excess zeros. They are distinguishable, because heterogeneity in the
mean would pile the excess at the low end:

| Predicted mean | Games | Observed P(0) | NB P(0) | Excess |
| --- | ---: | ---: | ---: | ---: |
| 2.31–3.78 | 2286 | 0.0958 | 0.0870 | +0.0088 |
| 3.78–4.17 | 2285 | 0.0845 | 0.0674 | +0.0171 |
| 4.17–4.52 | 2286 | 0.0735 | 0.0575 | +0.0160 |
| 4.52–4.97 | 2285 | 0.0586 | 0.0482 | +0.0105 |
| 4.97–9.77 | 2286 | 0.0472 | 0.0341 | +0.0132 |

Present at every level, and *growing* in relative terms — +39% in the top
bucket. That is the family.

**The fix is to change the unit.** A game is not a natural place to model run
scoring; an inning is, because an inning is mostly a zero. About 74.7% of them
score nothing, so P(shutout) is roughly `0.747⁹`, which is enormously
sensitive to a quantity the game level cannot see at all. `inning_pmf` is
scoreless with probability `scoreless`, otherwise a zero-truncated negative
binomial whose mean is *pinned* by the inning mean — so the shape cannot
smuggle in a different expected-runs estimate. A game is nine of them
convolved.

This also makes the ninth-inning censoring exact rather than nearly so. The
old code leaned on the NB's splitting property and had to assume the ninth
carried exactly a ninth of the mean; the eight-inning piece is now literally
eight innings convolved.

Two parameters, two moments, fitted out of sample on the away side only — the
away team bats nine innings whatever the score, so its distribution is run
scoring with nothing else mixed in, where the home side's has the censoring in
it and would be corrected twice. Fitting by likelihood was tried first and
rejected: the surface in `tail` is nearly flat and a search returned 2.50,
8.00, 7.75 and 4.25 on four consecutive seasons with one pinned at the edge of
the grid. That is a parameter fitted to noise, which this repository has paid
for once already. Two moments have no surface to get lost on.

`scoreless` lands between 0.745 and 0.750 on every season and at every value
of `tail`; `tail` itself is weakly identified and wanders along a ridge where
both moments stay matched. A sensitivity sweep pinning `tail` from 1.5 to 12
moved the moneyline gain only between −0.00076 and −0.00054, so the conclusion
does not rest on it. `SCORELESS_BOUNDS` is tight and `TAIL_BOUNDS` is wide for
exactly that reason.

Walk-forward, all three markets improved on log loss *and* calibration:

| Market | Log loss before | after | Δ | Calibration before | after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Moneyline | 0.68012 | 0.67955 | −0.00057 | 0.01625 | **0.00938** |
| Run line | 0.63894 | 0.63870 | −0.00024 | 0.00931 | **0.00569** |
| Total | 0.68848 | 0.68798 | −0.00050 | 0.02353 | **0.02008** |

The calibration movement is the larger story — moneyline calibration error
nearly halved. A held-out check with the estimator frozen and only the pricing
family swapped put the moneyline delta at −0.00064 with a date-clustered 90%
interval of [−0.00122, −0.00006] and the total at −0.00057, [−0.00091,
−0.00025], both excluding zero.

Per season the moneyline improved in four of five, and the one that did not
was 2025 — which is most of the market-comparison sample. So the gap to the
close moved the *wrong* way there, from +0.00542 to +0.00562, while the model
got better overall. Both facts are reported because reporting only the first
would be picking the sample that flatters the change. The run line and total
gaps narrowed slightly, to +0.00361 and +0.00363; all three verdicts are
unchanged.

An order of magnitude smaller than the censoring fix above, and the same kind
of thing: an assumption removed, not information added. The tally is now five
information sources that did nothing and two wrong assumptions that paid.

**What this does not fix** is the predicted means themselves, which are too
widely spread. That was the next thing looked at, and the answer was
surprising enough to get its own section.

### Do the Rockies score less on the road than the model thinks? Yes, and no

Prompted by exactly that question, which is the right instinct: the model
carries **one** offence number per team and leans on `park_factor` for the
rest, and Coors is extreme enough to strain that.

The residual says the concern is well founded. Out of sample, by team and
side, Colorado is the worst road cell in the league by a distance:

| | Home residual | Road residual |
| --- | ---: | ---: |
| Colorado Rockies | −0.090 | **−0.614** |
| next worst road | | −0.326 |

**A real double-count, found.** `expected_home_runs_prior` multiplies the
offence rating by the park factor, and the model carries `park_factor` as a
feature besides — so a rating built from raw runs applies the park once inside
itself and again outside. The two do not cancel, because a team plays only
half its games at home. Coors sits at 1.135 on the point-in-time factor, so
Colorado's rating carries about (1.135−1)/2 of that inflation into every road
game: **+0.30 runs** of predicted over-scoring, same sign and order as the
0.61 observed.

Nobody else shows it, and that is the part worth understanding rather than
explaining away. The same arithmetic gives the next most extreme park a bias
of 0.18 runs against a per-team noise floor of 3.1/√324 = 0.17. The effect is
league-wide and only clears the noise at the one park extreme enough — which
is why `TeamState` now stores run rates **park-adjusted** for every team, not
a Rockies special case.

**What the fix bought, and what it did not.** On a walk-forward over the same
11,495 games:

| Market | Raw ratings | Park-adjusted | Δ | 90% interval |
| --- | ---: | ---: | ---: | --- |
| Moneyline | 0.67956 | 0.67967 | +0.00011 | [−0.00002, +0.00024] |
| Run line | 0.63883 | 0.63890 | +0.00007 | [−0.00007, +0.00021] |
| **Total** | 0.68797 | **0.68767** | **−0.00030** | **[−0.00047, −0.00014]** |

Kept for the total, and for being right. But it does **not** fix the case that
prompted it: Colorado's road residual goes from −0.445 to −0.440. Nothing. The
double-count is real in the feature, and the estimator was already absorbing
most of it through its own coefficients, so removing it at source mostly
redistributes weight. A defect being real in the construction does not mean
the model was suffering from it — which is the second time in this file that
a correctly identified flaw turned out not to be what was hurting.

**And the remainder is not worth chasing.** The per-team home/road residual
split has a year-over-year correlation of **+0.008** across 119 team-seasons.
Colorado's own runs +1.47, +0.17, −0.05, +0.48, **−0.39** — and in 2026 they
are scoring *more* on the road than predicted, not less. The pooled −0.44 is
carried by 2022 and 2025. There is no persistent trait to model, so a per-team
home/road split would be fitting five seasons of noise.

### Auditing the scoreboard: is the de-vig right? Is the benchmark?

Two of the ten assumptions this project rests on live in the *measurement*
rather than the model, and if either is wrong the headline verdict is partly
arithmetic. Both are now tested, and the answer is that the scoreboard is
sound — which is worth knowing before spending more effort on the model.

**The de-vig.** `odds.py` strips the bookmaker's margin proportionally, which
is the convenient choice rather than a justified one. `devig.py` scores four
methods against what actually happened, over 43,060 settled quotes:

| Method | Per quote | Consensus | Δ vs current | 90% interval |
| --- | ---: | ---: | ---: | --- |
| **Proportional** (current) | 0.69547 | **0.68540** | — | — |
| Additive | 0.69609 | 0.68554 | +0.00015 | [−0.00013, +0.00044] |
| Power | 0.69644 | 0.68565 | +0.00025 | [−0.00016, +0.00071] |
| Shin | 0.69609 | 0.68554 | +0.00015 | [−0.00013, +0.00043] |

At consensus level all four are indistinguishable and the current one is
nominally best. It wins clearly on the moneyline (0.70786 against 0.70899 to
0.70962), where prices are most lopsided and the methods most disagree; power
edges it on spreads by 0.0008.

There was never much room for it to matter. The median overround is **4.64%**,
and the four methods differ by a mean of **0.26 to 0.39 probability points** —
with 78% of quotes in the 0.40–0.60 band where they differ by 0.15 to 0.28p. A
quarter of a point cannot produce a 0.00562 log-loss gap. The de-vig is not why
the market wins.

**The benchmark.** `market.py` scores the model against the median of US
books, while Pinnacle — the sharp reference — is captured under `eu` and
deliberately not priced. On the 1,644 captures where both exist, Pinnacle is
the better price: 0.68671 against 0.68781, and better on all three markets.

That runs the *wrong* way for the model. If the benchmark is soft, the model's
deficit against the market that actually exists is understated, not
overstated. But those 1,644 rows span only **5 distinct dates**, and the
interval is date-clustered, so it rests on five clusters and is not settled.
`devig.py` says so in its own output rather than leaving it to be noticed. The
sample grows on its own; promoting a region is a model change, not a config
flip, and it is not being made on five days of data.

### Is 2022 baseball the same game as 2026 baseball?

The model trains on every prior season with equal weight and no era term,
across the pitch clock and shift ban (2023) and the challenge system (2026).
Stationarity is a wrong assumption in plain sight, and wrong assumptions are
the category that has paid here.

The run environment does move — and not much:

| Season | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runs per team per game | 4.531 | 4.283 | **4.616** | 4.393 | 4.447 | 4.485 |

The 2023 jump is visible and is the rule changes. But the whole range is 0.332
runs against a single game's own standard deviation of 3.180 — **10% of the
noise in one game**.

`stationarity.py` walks forward under seven schemes. The result splits cleanly:

- **Discarding old seasons is definitively bad.** Training on the last season
  only costs +0.00126 on the moneyline, +0.00110 on the run line and +0.00286
  on the total, all three intervals excluding zero. The model wants the data
  more than it minds the drift.
- **Gentle recency weighting is at the chance rate.** A two-season half-life
  helps the run line by 0.00025 with an interval excluding zero, and is inside
  noise on the other two. But 7 schemes × 3 markets is 21 comparisons at 90%,
  so ~1.05 should exclude zero each way by chance — and 2 did. That is the
  `extremes.py` correction applied to my own result, and it says this is not
  evidence.

So the model is unchanged. The honest statement is not that seasons are
interchangeable — they visibly are not — but that the drift is small next to
game noise and far smaller than the cost of throwing data away.

### Statcast: expected outcomes, and the seventh null

Bought nothing, and the way it failed is the most informative version of this
result so far.

`statcast.py` ingests Baseball Savant — free, keyless, and referenced nowhere
in this repository until now — aggregating pitches on the way in to 404,785
player-game rows across 14,017 of 14,020 completed games. The case for it was
specific. `PitcherState` is a proxy built from *team* runs allowed in games a
pitcher started, carrying the bullpen that followed him and the defence behind
him; expected wOBA is the same quantity with those removed, scoring a batted
ball by how hard and at what angle it left the bat rather than by whether it
found grass. That is why it settles in dozens of batted balls where runs
allowed takes a season. The team columns were also the first thing in this
model to know batters exist at all.

The raw correlations were even mildly encouraging: starter expected wOBA
predicts opposing runs at +0.106 against +0.098 for the existing runs-allowed
proxy. Team batting was already worse — +0.042 against +0.076 for team runs.

Walk-forward over 11,589 games, identical baseline, eight new columns:

| Variant | Market | Baseline | With | Δ | 90% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Starters only | Moneyline | 0.67984 | 0.67983 | −0.00001 | [−0.00002, +0.00001] |
| Starters only | Run line | 0.63918 | 0.63917 | −0.00001 | [−0.00007, +0.00005] |
| Starters only | **Total** | 0.68780 | 0.68817 | **+0.00038** | **[+0.00010, +0.00065]** |
| All eight | Moneyline | 0.67984 | 0.67983 | −0.00001 | [−0.00004, +0.00003] |
| All eight | Run line | 0.63918 | 0.63917 | −0.00001 | [−0.00011, +0.00009] |
| All eight | Total | 0.68780 | 0.68809 | +0.00029 | [−0.00015, +0.00072] |

Six comparisons, **none helped**, and the one interval excluding zero points
the wrong way.

**Why, and this is the part worth keeping.** The GLM shrinks all eight to an
order of magnitude below the established features — the largest is
`away_off_barrel` at +0.0115 against `home_off` at +0.0657 and `away_def` at
+0.0506 — and two come back with the *wrong sign*: `away_sp_xwoba` at −0.0064,
when a worse opposing starter must raise home scoring, and `home_off_xwoba` at
−0.0026. Wrong signs at negligible magnitude are what a collinear duplicate
looks like after shrinkage.

So the lesson is not "expected statistics are no good." It is that being
*better measured* buys nothing once the noisier original is already in the
model and has had 14,000 games to settle. `home_off` and `away_def` are crude
and slow, and across six seasons they have converged on the same information
xwOBA delivers faster. Speed of stabilisation is worth something in April and
almost nothing in a walk-forward that trains on whole prior seasons.

That reframes what a genuinely new input would have to be. Not a cleaner
measurement of team quality — that is saturated — but something the scoreboard
cannot eventually reveal, which points at the lineup actually posted tonight
rather than the team's season-long profile.

The columns stay in `data/features.csv`, unlisted in `FEATURE_COLUMNS`, exactly
as the umpire features do. The tally is now three wrong assumptions removed
that paid, **seven** information sources that did nothing, and three correctly
identified defects that should be left alone.

### The first thing that beat a price, and it was not the model

Every study above asks whether the model beats the market. This one takes the
model out entirely and asks whether the **market's disagreement with itself**
beats the market. `line_shopping.py`, against 261,890 quotes from 284 bursts.

Start with the number that sets the bar. Across the eleven US books that appear
in at least half of all captures, the two-way overround at the **median** price
is 1.0357 — 1.79 points of vig per side. At the **best** price on the same
panel it is 1.0186, or 0.93 points. Shopping halves the house edge. It does not
remove it, and that is not an empirical claim but an arithmetic one: sum the
closing line value of both sides of one game and the consensus cancels, leaving
`1 - best_overround` whatever the close does. A study that bets both sides of
everything is measuring the overround and nothing else. The code says so in the
docstring and the printout, because it would otherwise read like a result.

The real question is what happens when one book is *badly* off. The rule is
decidable at bet time: at each capture, compare the best raw price on a side
against the de-vigged median of the same fixed panel, and take the first
capture in the lock window where the gap clears a threshold. Nothing looks
forward. Scored against the panel's own close:

| Deviation at entry | Bets | Dates | CLV vs panel close | CLV vs Pinnacle close |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 pt | 70 | 13 | **+0.437** [+0.256, +0.569] | **+0.312** [+0.072, +0.486] |
| 0.50 pt | 35 | 10 | **+0.663** [+0.431, +0.833] | **+0.486** [+0.194, +0.733] |
| 1.00 pt | 11 | 6 | **+1.147** [+0.881, +1.489] | **+0.863** [+0.417, +1.278] |

Intervals are 90%, bootstrapped over dates. This is the first positive result
in the repository, and it is worth being precise about why it is not obviously
a mirage.

**It is not just the entry condition read back.** Selecting on "beats the
consensus by a point" and then scoring against a *later* consensus is close to
tautological — if the consensus is a martingale, CLV equals the entry deviation
in expectation. So the number that carries the content is the **decay**: of the
1.397 points claimed at entry, 1.147 survive to the close. The market takes back
**18%**. The outlier book was not early; it was wrong, and the other ten never
came to it.

**It is not one book.** Outlier prices concentrate — BetRivers supplies 38 of
the 70 bets at the 0.25-point threshold, which is exactly the fingerprint of a
persistently stale book rather than a market phenomenon. So the study is rebuilt
eleven times, each time dropping one book from the panel, the consensus and the
close together. CLV ranges **+0.270 to +0.508** across all eleven exclusions and
never touches zero. Dropping BetRivers costs the most and the signal survives it.

**It is not scored against itself.** The panel median is built from the same
quotes the bet deviated from, so the study is also run against Pinnacle, which
sits outside the shopping panel and covers 280 of 284 captures. The result
attenuates by roughly a third at every threshold and stays positive. That
attenuation is the honest measure of how much of the panel-median figure was
self-reference. The Pinnacle column is the one to believe.

Writing that third check turned up a real defect in the second. A bet taken at
the *last* capture before first pitch was being scored against the consensus at
its own capture — so its CLV was identical to the deviation it was selected on,
by construction. Fourteen of 84 bets, 17%, were pure tautology. `settle` now
drops any bet with no strictly later close, which is why the tables above read
70 rather than 84.

**What it does not establish.** Thirteen dates is two weeks of burst capture,
and the 1-point row is eleven bets — the sweep is shown precisely so that no
single threshold carries the verdict. Positive CLV against a close is a
necessary condition for an edge and never a sufficient one: the study cannot
see whether BetRivers would have accepted the bet, or at what size, and the book
posting the outlier is reliably the book that limits fastest. That limitation
is not visible anywhere in quote history and no amount of further capture will
reveal it.

The tally, restated. Three wrong assumptions removed that paid, seven
information sources that did nothing, three correctly identified defects that
should be left alone — and one edge that was never in the run distribution at
all. If there is money here it is in execution, not in modelling baseball.

### Telling you when to bet, and why email is the wrong instrument

An edge that exists is not the same as an edge you can be told about in time.
Before building any alerting, the question worth answering is how long a
qualifying price actually stays on the screen. Measured on the burst captures,
where consecutive polls sit about 90 seconds apart:

| Deviation | Gone by the next poll | Alive at 5 min | Alive at 15 min | Alive at 1 h |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 pt | 75% | 19% | 15% | 3% |
| 0.50 pt | 67% | 21% | 15% | 3% |
| 1.00 pt | 70% | 10% | **0%** | 0% |

Median lifetime is under one poll. **The bigger the mispricing the faster it
dies**, which is the opposite of convenient: the prices worth the most are the
ones with no chance of surviving a round trip through a mailbox.

That table is a design constraint, not a caveat, and it kills the obvious
version of what was asked for. A page that says LIVE cannot be right, because
it is written hourly and read whenever. An email that says "bet this" is
usually describing something that no longer exists. So:

**The page does the arithmetic in the reader's browser.** `model_card.py` emits
each alert with its capture stamp and the measured survival curve, and the page
computes the age against your clock, re-ticking every fifteen seconds. An alert
reads "go now · 1 min ago · ~25% still up" or "gone · 5.0 h ago · ~3% still up".
Nothing is styled to look urgent that is not; a row past fifteen minutes fades
and stays only as a record of cadence. Every share on that curve is observed —
the only judgement is where the buckets break.

**Latency, not sampling interval, is what an alert has to beat.** The 3%
one-hour figure is often misread as "hourly alerting is 3% useful". It is not:
the alert fires against the snapshot just taken, so what reaches the inbox is
about two minutes stale, not sixty. Expect roughly **one in five** to still be
takeable. What the hourly path loses is not freshness but *coverage* — it only
ever sees the opportunities that happen to exist at the top of the hour.

**Coverage needs a poller, so the alerting lives in the burst.**
`odds_burst.py --alert` screens every 90-second poll in flight, which is the
only cadence matched to a 90-second half-life. The book panel is fixed once
before the first poll: inside a single capture every book has 100% coverage, so
a panel derived there would grow on thin nights and invent alerts — the same
min-of-N trap the study is built to avoid.

**A stale alert costs nothing but attention.** The price is either on the screen
or it is not; there is no losing bet in a missed one. That asymmetry is why the
honest response is to send anyway and label the odds, rather than suppress
anything unlikely to survive.

`data/shop_alerts.csv` is append-only and never revised. It is the forward test:
the study looked backwards over 13 dates, and this is the record that will
eventually say whether the rule holds forwards on prices flagged live. Nothing
in it is a wager, and mail settings are optional throughout — with no SMTP
secrets the detector still runs and still logs, exactly as a fork with no odds
key still runs.

### The burst was pointed at the wrong hours

The alerting is only as good as the hours it is awake for, and the original
burst schedule — one three-hour window from 21:50 UTC, chosen because "most
first pitches land between 23:00 and 02:00" — turned out to be aimed badly.

The right thing to count is not first pitches but **lock-window game-minutes**:
how many games sit inside their own 20–240 minute betting window at each minute
of the day. That curve is bimodal, with a matinee block peaking near 16:00 UTC
and the main block running 19:00–23:00. Against it, the old schedule covered
**28%**, and a third of all games got no burst coverage whatsoever. Given that
three quarters of opportunities die within one poll, an uncovered window is not
a late detection. It is a missed one.

Two 5.5-hour bursts at 14:00 and 19:30 cover **96%**. The pair was found by
searching the demand curve rather than by reasoning about when games "usually"
start — the same mistake, made twice, would have been easy. Cost is not the
binding constraint at ~2,640 credits a day against a balance near four million;
GitHub's six-hour job ceiling is, which is why each window is 5.5 hours.

### Scoring the alerts that actually fired

`alert_evidence.py` scores `data/shop_alerts.csv` against the close. It exists
because every study in this repository that ever looked good looked good
*backwards*, and most stopped looking good the moment something independent was
asked of them. The alerts are the independent version: the rule, threshold and
panel were fixed when each row was written, so there is no report in which a
better rule is tried on the same alerts. A disappointing number can only be
reported, never tuned away.

The gate is deliberately hard. It requires 70 scored alerts over 13 dates — the
same weight the backward study carried — **and** a positive interval against
Pinnacle, never against the panel median, which is built from the very quotes
each alert deviated from and would flatter the record.

Writing it turned up a trap worth naming. The first version computed a
date-clustered bootstrap regardless of how many dates existed, and on the first
real day it returned a 90% interval of `[+0.130, +0.130]` — resampling a single
date gives the same mean every draw, so the interval collapses onto the point
estimate and any positivity test on it passes automatically. That is precisely
the kind of number that reads as evidence and is not. Below three dates the
function now returns no interval at all and says why.

### A real defect that must not be fixed

The means are over-spread, and unambiguously so. Regress observed runs on
predicted across the walk-forward and the slope is **0.712**, not 1. Split the
predictions into quintiles and the pattern is sharper than a slope:

| Predicted runs | Games | Observed − predicted | Seasons with that sign |
| --- | ---: | ---: | :---: |
| 2.31–3.85 | 4,571 | **+0.236** | 5/5 |
| 3.85–4.23 | 4,571 | +0.103 | 4/5 |
| 4.23–4.57 | 4,571 | −0.032 | 1/5 |
| 4.57–5.01 | 4,571 | +0.009 | 2/5 |
| 5.01–9.77 | 4,572 | **−0.374** | 0/5 |

The middle three quintiles are within 0.10 and their per-season gaps change
sign. The extremes are off in the same direction every single season. This is
regression to the mean — the ordinary consequence of conditioning on a noisy
estimate — and it is exactly the kind of measured, persistent defect that the
rest of this repository says is worth fixing.

Fixing it makes the model worse. Both standard corrections, fitted out of
sample, with the dispersion and inning shape refitted on the corrected
predictions so the comparison is fair:

| Market | Uncorrected | Linear Δ | Isotonic Δ |
| --- | ---: | ---: | ---: |
| Moneyline | 0.68078 | +0.00076 [−0.0003, +0.0018] | +0.00176 [+0.0004, +0.0032] |
| Run line | 0.63896 | +0.00156 [+0.0005, +0.0026] | +0.00265 [+0.0014, +0.0041] |
| Total | 0.68701 | +0.00246 [+0.0014, +0.0035] | +0.00274 [+0.0013, +0.0042] |

Five of six intervals exclude zero, on the wrong side.

**Why.** The run-scoring mean and the price are not the same quantity. The
between-game spread of predicted means is 0.743 runs against a residual spread
of 3.143 — the signal is under a quarter of the per-game noise — and the
dispersion is fitted from residuals around the *unshrunk* mean, so the priced
distribution has already absorbed the attenuation. Correcting the mean on top
charges for the same uncertainty twice, and it shows up precisely where that
account predicts:

| | Calibration slope | Calibration error | sd(predicted probability) |
| --- | ---: | ---: | ---: |
| Uncorrected | 0.876 | 0.01341 | 0.0837 |
| Linear | **1.286** | 0.02109 | 0.0557 |
| Isotonic | **1.100** | 0.01354 | 0.0652 |

The correction does not move the model toward calibrated, it carries it
through 1.0 and out the other side into under-confidence. Isotonic is the
instructive case: calibration error essentially unchanged, log loss clearly
worse, and the spread of predicted probabilities down 22%. The loss is not in
calibration, it is in discrimination — and a model that answers 0.5 to
everything is perfectly calibrated and worth nothing.

The conclusion is *not* that the means are fine. An estimator that fixed the
attenuation at source, by being less noisy, would be a genuine improvement.
What does not work is repairing the symptom downstream of a width that already
accounts for it. `mean_calibration.py` reproduces all of it.

This is the counterexample to the pattern the rest of this file reports. Two
wrong assumptions removed paid; five information sources did nothing; and here
a real, persistent, correctly measured defect turns out to be one that should
be left alone. "Removing wrong assumptions pays" is a summary of what has
happened, not a rule that can be applied without checking.

### Three more wrong assumptions

Fixing the ninth inning exposed the next layer. All three are defects rather
than hypotheses, and each is verifiable directly rather than through a metric.

**The home mean was censored twice.** The estimator is trained on observed
home scores, and those are already censored — the home side did not bat the
ninth in 43% of them. Its output is a censored mean, and the new distribution
censored it again. The priced expectation came out at 4.34 against an
estimator saying 4.51: the home side biased 0.14 runs low, on the largest
market. The fix inverts the transform, solving for the full-length mean whose
*priced* expectation matches what the estimator learned. A single global
factor would not do — censoring bites harder on a home favourite, which leads
after eight more often, so the inflation runs from 1.02 for an underdog to
1.06 for a favourite. Priced E[home] now equals the estimator exactly, and
P(home win) moves from 0.520 to 0.534 against 0.538 observed.

**Extra innings were resolved two runs apart.** `calibrate_extra_innings`
measured a mean margin of 1.58 and rounded it, so every one of the 8.8% of
games that go past regulation was resolved by two — while 69% of them end one
apart. The same mistake the first walk-off attempt made, and it lands on the
same boundary. Now measured as a distribution.

**121 games were seven innings, priced as nine.** The 2021 doubleheader rule.
`scheduled_innings` was already in the data and the joint ignored it, inflating
both sides' run expectation by a fifth. A seven-inning total now prices at
0.29 where nine prices at 0.49, and the moneyline barely moves, which is the
right shape.

Together they are not detectable in log loss — the paired intervals all span
zero. Calibration error is another matter, improving on all three markets at
once: moneyline 0.0184 to 0.0163, run line 0.0167 to **0.0093**, total 0.0251
to 0.0235. The moneyline gap to the close narrows from +0.00641 to +0.00542,
still excluding zero.

That is the honest reading: three real errors removed, one metric clearly
better, the metric that decides still unmoved.

### Does new information help? Not detectably

`boxscores.py` had been ingesting per-start pitcher lines since 2021 and
nothing read them. Its own docstring names the reason to care: everything in
`features.py` was derived from runs scored and allowed, which is what the
market has already priced, while a starter's own strikeout, walk and home-run
rates are a different input. That is the most plausible place for edge to
live, so it was wired in — with bullpen quality and three-day workload
alongside — and the run distribution was given a dispersion per side, since
home scoring is censored and away is not.

The coefficients behave. Strikeout rate carries **-0.056** on the away
starter against **0.085** for team defence, the strongest established feature,
and five of the six new terms take the sign baseball says they should.
Calibration error on the moneyline fell from 0.0223 to 0.0183.

None of it beat noise.

| | Before | After |
| --- | ---: | ---: |
| Moneyline log loss | 0.68102 | 0.68022 |
| Paired change | | **-0.00080** |
| 90% interval, season-clustered | | **[-0.00205, +0.00042]** |
| Gap to the closing price | +0.00594 | +0.00641 |

The interval spans zero, so the improvement is not established. Per season the
change is +0.0016, -0.0030, -0.0020, +0.0004, -0.0011 — three better, two
worse, none of them large. The gap to the close widened rather than narrowed,
and that is inside its own interval too.

The reading worth taking is not that starting pitching does not matter. It is
that starting pitching is the single most discussed input in this market, so a
crude component model of it is the last place a price is likely to be soft.
Adding genuinely new information moved nothing measurable, which is evidence
about the market rather than about the feature.

Both changes are kept. They are principled — the censoring is real and the
variance measurement was unambiguous — and the pitcher history will carry more
weight as seasons accumulate than it does over five. But nothing here is a
result, and the live gate is unchanged.

### Are the big disagreements worth betting?

The natural question, and the one the UFC project cannot answer at two wagers
a month. A season of baseball fills the tail, so `extremes.py` asks it
properly. Positive ROI means backing the model's side pays at the market's own
de-vigged price; every figure is **before vig**, and the measured overround is
4.55%, about 2.3 points a side.

| Disagreement | Games | Model says | Market says | Actually won | No-vig ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-3 pts | 1,719 | 50.2% | 48.7% | 48.1% | -0.9% |
| 3-5 | 890 | 50.1% | 46.2% | 46.6% | +0.8% |
| 5-8 | 884 | 51.1% | 44.8% | 44.6% | -1.8% |
| 8-10 | 318 | 51.7% | 42.7% | 42.5% | +0.7% |
| 10-12 | 179 | 52.5% | 41.6% | 43.0% | +0.5% |
| 12-13 | 51 | 52.4% | 39.9% | 43.1% | +4.6% |
| **13-14** | 44 | 55.4% | 42.1% | **27.3%** | **-37.8%** |
| **14-15** | 36 | 53.7% | 39.3% | **52.8%** | **+35.1%** |
| 15-17 | 20 | 56.6% | 40.8% | 50.0% | +18.2% |
| 17+ | 14 | 55.1% | 36.5% | 64.3% | +76.2% |

Cut the table at 14 points and the tail returns +38.5% over 70 games with a
bootstrap interval excluding zero. It is not real, for three reasons.

**The shape.** The bands are disjoint, and 13-14 loses 37.8% over 44 games
while 14-15 wins 35.1% over 36. Nothing makes a 13.5-point disagreement lose
badly and a 14.5-point one win big. The edge at 14+ exists because the cut
excludes the bad band; a cumulative threshold hides exactly this.

**The search.** Thresholds from 8 to 20 were tried. Under a null where the
market price is simply correct, the best cut a searcher finds still averages
**+33%** and clears +30% in **43%** of simulated seasons. After pricing that
search the observed result gives p = 0.12. Finding a spectacular bucket is what
searching produces.

**The direction.** In every band the market's number is the accurate one. At
10-15 points the market implied 41.6-42.1% and 43.0% came in; the model said
52.5%. The model is not finding value where it disagrees most — it is most
wrong there, which is what a large disagreement with a sharper counterparty
should be expected to mean.

Across all 4,155 rows the trend of return against disagreement is Spearman
rho = +0.03 (p = 0.08). There is no gradient to ride.

### Four more of the same kind

Ingesting the current season turned over a row of defects that had all been
sitting in plain sight, each producing output that looked fine.

**Every season had too many games.** A postponed game is returned twice by the
schedule endpoint under one `game_pk`, and both copies survived. `features.py`
folds each row's result into team state as it walks the table, so 274 games
counted twice toward Elo, run rates and park factors. Deduplicating brings
every season to exactly the 2,430 a regular season contains — 2021 had been
reporting 2,512.

**Elevation never reached the model.** Read through pandas a venue id arrives
as `3313.0`, because two games in the schedule carry no venue and that types
the whole column as float. No park lookup ever matched, so `elevation_km` was
zero on all 14,580 rows and Coors Field's mile of altitude — the largest park
effect in baseball — was absent.

**And it was in the wrong units anyway.** StatsAPI reports elevation in feet
under a field this code calls `elevation_m`: Coors comes back as 5190, which
is its height in feet and 1,582 metres. Fixing the lookup alone would have
handed the model an altitude scale inflated by 3.28, which is worse than the
zero it replaced because it looks plausible.

**Tonight's games arrived already scored.** StatsAPI opens a linescore at 0-0
before first pitch, and everything downstream reads `home_score.notna()` as
"this game happened". Ten scheduled games were being trained on as genuine
shutouts.

Nothing in the previous section's numbers was safe from these, which is why
the table above is regenerated rather than edited.

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
