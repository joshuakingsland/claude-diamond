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
| Does the model beat a price? | **No.** 186 days of 2025 prices. The close still wins the moneyline with an interval excluding zero; the run line and total are undecided |
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
roofs: 2,034 games in 2021-2026 report "Roof Closed", but that is known
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
extremes.py   does a large disagreement pay? no, and here is why it looks like it does
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
python predict_upcoming.py        # model probabilities beside it
python ledger.py                  # screen, record, settle
```

The card reports a `disagreement` column, never an `edge`. The market
comparison above says the closing price beats this model, so a gap between the
two is a disagreement the model is more likely to be wrong about — naming it
edge would be the whole failure this project exists to avoid.

Three properties keep the live path honest. It builds features with the *same*
builder as training, which emits a row for an unplayed game and folds nothing
into state. It takes forecast weather from Open-Meteo, the same source as the
training archive, and keeps it in a separate file so a forecast can never
overwrite reanalysis for a game that has since been played. And it drops games
already under way, because the odds feed keeps quoting them and those prices
reflect a score the model cannot see.

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
| Outside the lock window | `MIN/MAX_LOCK_LEAD_MINUTES` |
| Quote too old | `MAX_ODDS_AGE_MINUTES` |
| Best price too far from consensus | `MAX_EXECUTION_DEVIATION` |
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
`BOOTSTRAP_MODELS` and `WEATHER_SOURCE` remain unreferenced; they are
documentation, not controls.

## Automation

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `tests.yml` | push, PR | Unit tests, plus a check that odds capture exits clean with no key |
| `odds.yml` | hourly 15:00-03:00 UTC in season | Captures the board, prices the card, screens the ledger |
| `backfill-odds.yml` | manual | Capped historical capture; dry run by default |
| `revalidate.yml` | weekly, manual | Rebuilds features, re-runs the comparison and validation, settles the ledger |

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

## Running it

```bash
python -m pip install -r requirements.txt
python parks.py --refresh --seasons 2021-2026
python results.py --seasons 2021-2026
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
| Moneyline | 2,230 | +0.00542 | [0.0020, 0.0088] | Market better |
| Run line -1.5 | 1,245 | +0.00391 | [-0.0012, 0.0088] | Undecided |
| Total 8.5 | 680 | +0.00386 | [-0.0031, 0.0107] | Undecided |

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

### The one change that worked: the home ninth inning

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
