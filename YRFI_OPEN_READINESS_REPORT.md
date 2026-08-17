# YRFI opening-edge readiness report

## Decision

The existing first-inning archive cannot test an opening-price strategy. All
5,000 historical requests targeted exactly ten minutes before first pitch, and
no event has more than one returned snapshot. The honest next target is not a
second outcome model at the same close; it is predictable **open-to-close line
movement** and executable closing-line value.

The new collector and evaluator are ready, but no historical opening call has
been made. Launching the paid workflow requires a separate explicit decision.
Nothing in this work places a wager or writes a YRFI/NRFI pick to the website.

## Existing evidence

- Close attempts: 5,000 (4,950 offered, 46 failed, 4 no-offer).
- Raw close quotes: 21,721 across 4,950 events and nine books.
- Official result rows: 4,950, including 4,941 final labels.
- Qualified multi-book, feature-matched games: 3,419.
- Requested lead: exactly 10 minutes for every attempt. Provider-returned
  snapshots range from 10.32 to 14.38 minutes before first pitch.
- Measured prior spend: 49,500 event-odds credits and 4,684 discovery credits.
  Last recorded balance: 4,071,430 credits.

The frozen 2025 confirmation had 1,043 games. Raw market log loss was 0.685819.
Market-only recalibration reached 0.685408, an improvement of 0.000411 whose
date-clustered 95% interval was [-0.002296, 0.001494]. The selected baseball
candidate was worse at 0.687373. Neither result confirmed a YRFI edge.

An exploratory best-price diagnostic explains why raw ROI is not enough. On
2025 close quotes, choosing the side with at least 2% consensus-implied value
returned +12.3%, but only across 60 games; the broader non-negative-value rule
made 183 entries and returned -3.0%. In 2023 the same non-negative rule found
only 12 entries and returned -50.6%. These post-outcome, unstable, small-cell
results are hypotheses—not evidence for deployment.

## Frozen opening study

The fixed ladder is 1,440, 720, 360, 180, and 60 minutes before first pitch.
The opening observation is the earliest rung with a paired 0.5-run total from
at least two books. The model predicts the ten-minute closing consensus from:

- opening probability, vig, dispersion, quote freshness, and availability;
- predeclared deviations for the nine book keys already in the close archive;
- prior-day-safe team, rest, bullpen, park, and run-context features.

Starter, umpire, weather, and confirmed-lineup fields are excluded because a
retrospective value does not prove it was known at the opening timestamp.

The primary test is reduction in close-logit squared error versus “no movement.”
The paper execution test uses the best captured opening price. Confirmation
requires all of the following on untouched 2025 prices: at least 150 entries,
positive movement improvement and closing-value intervals, positive close
value at two books with at least 30 entries apiece, and positive historical ROI.
Even a pass remains paper-only pending prospective fills.

## Cost and stop rule

Stage 1 reuses the 2,248 settled, multi-book-close 2023/2024 games: 11,240
calls, at most about 112,400 credits (2.8% of the last recorded balance), with
zero discovery spend. If 2024 selection rejects the development candidates,
the study stops.

Only a locked development candidate can unlock Stage 2: 1,043 games from
2025, 5,215 calls, or about 52,150 additional credits. Maximum useful spend is
therefore about 164,550 credits (4.0% of the last balance). The 128 excluded
2026 games are not purchased retrospectively.

## Website status

There is intentionally no first-inning pick on the public site. The current
research says “no confirmed YRFI edge,” and the opening archive does not yet
exist. A research-status panel would be honest; a prop recommendation would
not be. Promotion to a website signal requires the locked historical gates and
then prospective executable closing-line value.
