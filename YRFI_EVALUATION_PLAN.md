# YRFI market-anchored evaluation plan

This protocol was frozen before inspecting any candidate-feature performance.
It is research-only and cannot place or authorize a wager.

1. Use only regular-season games with a settled first-inning result, a 0.5-run
   first-inning market, at least two quoted books, and a snapshot returned
   before scheduled first pitch.
2. Collapse duplicate provider events to one row per MLB game before fitting or
   scoring. Join only the repository's point-in-time features, which are emitted
   before each game's result is folded into feature state.
3. Train on 2023. Select a market-recalibration strength and one predeclared
   baseball feature family on 2024. Refit those two locked candidates on
   2023-2024 and evaluate them once on 2025.
4. Do not evaluate the already collected 2026 rows in this version. They are
   excluded, not represented as a pristine holdout. The prospective forward
   window starts with predictions frozen after 2026-08-14.
5. The primary comparison is log loss versus the devigged multi-book market
   consensus. Confidence intervals resample whole game dates. Brier score and
   calibration error are secondary diagnostics.
6. A baseball signal must beat both the raw market and the separately fitted
   market-only recalibration on 2025, with both date-clustered 95% intervals
   below zero. A point estimate, accuracy, or ROI alone is not evidence.
7. No thresholds, teams, parks, months, or price bands are searched. No bets are
   placed. Any confirmed result remains paper-only until it survives the
   post-lock forward window and executable-price/CLV validation.
