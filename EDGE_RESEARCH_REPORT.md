# Edge research report

Status: **one confirmed price-movement signal; no confirmed outcome edge and
no wager promotion**.

## What the completed archives say

### Full-game outcomes

The closing consensus remains better than the standalone GBM, GLM, and their
equal-weight ensemble on moneylines, run lines, and totals. Market anchoring
collapses the useful model weight to almost zero. The current model should not
replace a closing-market probability.

### First-inning YRFI

The frozen YRFI study required at least two books, collapsed duplicate provider
events to one MLB game, and found zero feature joins or post-start snapshot
violations. It trained on 2023, selected on 2024, and opened 2025 once.

On 1,043 confirmation games:

| Forecast | Log loss | Delta vs raw market | Date-clustered 95% interval |
| --- | ---: | ---: | :---: |
| Raw market | 0.685819 | — | — |
| Market recalibration | 0.685408 | -0.000411 | [-0.002296, 0.001494] |
| Best baseball candidate | 0.687373 | +0.001554 | [-0.002186, 0.005454] |

The recalibration gain is noise, and the baseball candidate is worse on the
point estimate. There is no confirmed YRFI edge. Existing 2026 rows are not
scored by this version; prospective evidence begins after the model lock.

### Twenty-four-hour price movement

The separate movement protocol fit 2022, selected candidates on 2023, and
opened 2024 only after that season's early-snapshot audit was complete. All
three candidates improved close-movement MSE with date-clustered 95% intervals
above zero:

| Market | 2024 rows | Direction accuracy on nontrivial moves | Relative MSE reduction | Improvement interval |
| --- | ---: | ---: | ---: | :---: |
| Moneyline | 1,438 | 59.28% | 4.82% | [0.000254, 0.000671] |
| Run line | 929 | 60.14% | 6.40% | [0.000236, 0.000846] |
| Total | 611 | 73.14% | 38.11% | [0.001773, 0.002700] |

### Correction: most of the totals figure is an artifact

Protocol v2 adds candidate families without the raw entry price, because v1
selected a totals fit dominated by `entry_logit` at -0.038, the largest
coefficient anywhere in the three markets. That is mean reversion on the entry
price, and for totals the entry price is nearly all noise: sd(entry_logit) is
0.057 against sd(move) 0.072, so the movement is larger than the whole spread
of entry prices. A total's two-way price sits near even money because the book
moves the *line* rather than the price.

`move = close - entry`, so measurement error in `entry` reappears in `move`
with a negative sign whether or not the market reverts. On the 2023 selection
year:

| Market | With entry price | Without | Gap |
| --- | ---: | ---: | --- |
| Moneyline | 3.92% | 3.29% | negligible |
| Run line | 5.23% | 5.03% | negligible |
| **Total** | **35.57%** | **7.66%** | 78% of it was the entry price |

The moneyline and run line are unaffected, which is what separates a signal
from an artifact. Both also select a family that *includes* the baseball model,
and `model_gap_logit` is the largest standardised coefficient on the moneyline
and the second largest on the run line — the price does drift toward the model.
Totals selects microstructure only; the model is not in it at all.

Note what this means about method: the artifact reproduced perfectly in a
sealed 2024 window. A held-out confirmation season protects against
overfitting, not against a target that is mechanically correlated with a
feature. `tests/test_movement_artifact.py` builds a world with no market
force whatever and shows the same family scoring above 50% out of sample.

### What the reduction is worth as a price

A relative MSE reduction is not a return and reads far too much like one, so
`movement_metrics` now reports the implied CLV directly. Betting the predicted
direction on every 2024 row earns:

| Market | Relative MSE reduction | Implied CLV |
| --- | ---: | ---: |
| Moneyline | 4.82% | **0.445 probability points** |
| Run line | 6.40% | **0.550 points** |
| Total (entry-price-free) | 7.66% | **0.277 points** |
| | vig per side at the median book | **2.31 points** |
| | measured live CLV, 44 sharp-close matches | **-0.522 points** |

So the honest figure across all three markets is 0.3 to 0.6 probability points
of price improvement against 2.31 points of margin — roughly a quarter of what
it costs to place the bet. It predicts the direction and size of a later
consensus price; it does **not** prove that an available wager beats vig,
survives limits, or produces positive returns, and the live probe is currently
negative on the only metric that matters.

One further reason for caution on tradeability: `entry_prob` is the *median*
across seven or eight books. A median is a statistic, not an offer. A single
book posting an off-market total would be a real dislocation worth taking;
sampling noise in a cross-book median is not something anyone can bet.

## What changes operationally

`movement_forecast.py` freezes the selected configurations into
`movement_model.json`, fitted only on 2022-2023 rows. The live card applies it
only 23-25 hours before first pitch—the horizon present in the archive. An
eligible prediction is appended by `signal_ledger.py` as a `paper_clv_probe`.
It does not require a submitted lineup because it is not a wager, and it never
enters the staking ledger.

The next hurdle is prospective sharp-close CLV. Until at least 500 independent
games establish positive date-clustered CLV at executable prices, the signal
remains research-only. No money moves.
