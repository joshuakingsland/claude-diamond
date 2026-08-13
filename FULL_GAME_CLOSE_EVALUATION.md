# Completed full-game close evaluation

Status: **research only; no model or betting-rule promotion**.

The provider archive contains 10,354 attempted events and 10,233 offered
events from 2020-2024. The model comparison uses 6,914 games in 2022-2024,
the seasons for which stored walk-forward GBM and GLM predictions overlap the
close archive. Every stored prediction was trained through a date before the
game it scores; the audit found zero violations.

## Headline result

The devigged closing market beats the standalone GBM, GLM, and equal-weight
ensemble on moneylines, run lines, and totals. The date-clustered 90% interval
excludes zero in the market's favor for every candidate in both the 2022-2023
development block and the 2024 confirmation block.

The GLM was the closest standalone candidate in 2024:

| Market | Games | GLM log loss | Close log loss | GLM minus close | 90% interval |
| --- | ---: | ---: | ---: | ---: | :---: |
| Moneyline | 2,408 | 0.677425 | 0.673788 | +0.003637 | [0.00029, 0.00698] |
| Run line | 2,408 | 0.678699 | 0.674355 | +0.004343 | [0.00078, 0.00767] |
| Total | 2,306 | 0.697403 | 0.692973 | +0.004430 | [0.00105, 0.00810] |

Positive delta means the closing market is better.

## Does the model add anything after anchoring to the market?

A second-stage incremental test selected a convex weight on 2022-2023 using
`close + weight * (model - close)` and applied that weight unchanged to 2024.
Moneyline and run-line weights collapsed to 0-2%. Totals retained 8-9%, but
the 2024 improvement was only 0.000109 of log loss and its interval crossed
zero. None of the nine market/model candidates demonstrated incremental
confirmation signal.

## Interpretation

The present Diamond probabilities should not replace closing-market
probabilities and do not establish a betting edge. The defensible next model
architecture is market-anchored: start from a devigged market probability and
allow point-in-time baseball information to make small, prevalidated offsets.
Any such offset remains experimental until it improves a later sealed window
and forward CLV.

## Coverage limitations

- The comparison uses the same matched games for model and market, so missing
  games cannot favor one forecast within a scored row.
- Schedule matching is weakest in 2020-2022. The 2024 confirmation join is
  strongest, with only 13 unmatched odds events.
- 2024 is a confirmation window for this execution, not a pristine unseen
  trial, because the repository was developed around historical MLB data.
- Results measure a close requested 20 minutes before scheduled first pitch.
  This archive contains no early price, so it cannot by itself measure
  early-to-close CLV.

Machine-readable evidence is in `full_game_close_evaluation.json`; the exact
joined game-market rows are in `data/research/full_game_close_rows.csv`.
