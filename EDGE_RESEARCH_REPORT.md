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

This is the only confirmed research signal in the repository. It predicts the
direction and size of a later consensus price; it does **not** prove that an
available wager beats vig, survives limits, or produces positive returns.

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
