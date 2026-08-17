# YRFI opening-price research plan

This protocol is frozen before any historical opening first-inning quote is
requested. It is research-only and cannot place or authorize a wager.

## Data contract

1. The existing `data/first_inning_audit.csv` archive is the closing snapshot:
   every request was scheduled ten minutes before first pitch. It remains
   immutable.
2. Opening discovery reuses provider event IDs from the same closing cohort
   and requests a fixed ladder at 1,440, 720, 360, 180, and 60 minutes before
   first pitch. Reusing event IDs avoids another paid event-discovery call and
   ensures opening availability is measured against the closing cohort.
3. An executable opening observation is the earliest rung with a paired 0.5
   YRFI/NRFI total from at least two books. Failed and no-offer requests still
   count toward archive completion; they are never silently removed.
4. The close is the existing multi-book, de-vigged consensus returned before
   first pitch. An event without both a qualified open and close is a coverage
   miss, not a zero movement.

Collection is sequential by evidence stage and restricted in advance to
settled regular-season games with a multi-book close; other events could never
enter the analysis. The 524 eligible games from 2023 and 1,724 from 2024
require 11,240 event-snapshot calls, at most approximately 112,400 credits. If
selection rejects every development candidate, collection stops there. Only
the locked-candidate status can unlock the 1,043-game 2025 confirmation,
another 5,215 calls or approximately 52,150 credits. The 128 already-observed
2026 outcomes remain excluded and receive no retrospective opening calls.
Maximum useful spend is therefore 16,455 calls and about 164,550 credits. The
collector is resumable and commits in bounded checkpoints. No paid run occurs
merely by merging this protocol or its code.

## Frozen evaluation

1. Fit candidate open-to-close movement models on 2023 and choose one model and
   one expected-value threshold on 2024. The collector refuses paid 2025 calls
   unless the complete development archive produces a locked candidate. The
   2025 open-to-close target is not inspected until every predeclared opening
   request for that season has been attempted.
2. The primary baseline is no movement: the opening consensus is assumed to be
   the closing consensus. The primary model metric is close-logit MSE
   improvement with confidence intervals resampled over whole game dates.
3. Candidate features are limited to opening-market microstructure and
   information safely knowable the prior day: team run context, rest, bullpen
   workload, park, and prior expected runs. Starting-pitcher identity/rates,
   umpire, weather, and confirmed-lineup fields are excluded because their
   historical values may not have been known at the opening quote. The nine
   book identities already present in the closing archive are frozen before
   opening collection; their deviation, presence, and quote staleness can test
   the specific hypothesis that a leading book predicts lagging-book movement.
4. A paper entry uses the best captured price on exactly one side and must pass
   the threshold selected on 2024. Confirmation requires positive 2025
   consensus-close expected value with a date-clustered 95% interval above
   zero, at least 150 entries overall, and positive close value in at least two
   independently named books with 30 or more entries each. A positive
   point-estimate ROI is also required. Historical ROI is secondary because
   the outcomes were already inspected by the earlier closing-price study.
5. The 2026 rows already in the repository are excluded from historical model
   selection. Only quotes frozen prospectively after 2026-08-15 can supply new
   outcome confirmation.
6. No team, month, park, book, side, or price-band subgroup is searched. No bet
   is placed. A confirmed historical opening signal remains paper-only until
   prospective executable fills establish positive closing-line value.
