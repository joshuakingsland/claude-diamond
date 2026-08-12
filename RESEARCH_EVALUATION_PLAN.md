# Full-game close evaluation plan

This plan is frozen before evaluating the completed archive. It is research-only.

1. Validate every event snapshot is pregame and report coverage, books, provider gaps, and duplicates.
2. Join only predictions and features available before each event's requested snapshot.
3. Use rolling-origin development windows. Keep the latest calendar block sealed until feature/model choices are frozen.
4. Primary outcomes: log loss and Brier score versus the devigged closing consensus. Secondary: calibration and early-to-close CLV where an eligible early snapshot exists.
5. Do not select thresholds, segments, or paper recommendations from ROI alone. A candidate needs consistent development gains and confirmation on the sealed holdout.
6. This repository does not place wagers.
