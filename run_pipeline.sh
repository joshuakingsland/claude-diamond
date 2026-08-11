#!/usr/bin/env bash
# Finish weather (resumable), build features, compare against the market, and
# validate. Mirrors .github/workflows/revalidate.yml so a local run and a
# scheduled run produce the same artifacts.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
fi

echo "=== WEATHER ==="
# Resumable: a venue that fails a TLS handshake is reported and skipped, so
# rerunning fills only what is missing.
for attempt in 1 2 3 4; do
    "$PYTHON" weather.py --games data/games.csv --out data/weather.csv \
      --refresh-source && break
  echo "weather attempt $attempt incomplete; resuming"
  sleep $((attempt * 5))
done

"$PYTHON" - <<'PY'
import pandas as pd
weather = pd.read_csv("data/weather.csv")
games = pd.read_csv("data/games.csv")
print("weather coverage %.1f%%"
      % (100 * weather.game_pk.nunique() / games.game_pk.nunique()))
print(weather.roof_category.value_counts().to_dict())
PY

echo "=== FEATURES ==="
"$PYTHON" features.py

echo "=== WALK-FORWARD PREDICTIONS (glm) ==="
"$PYTHON" validate.py --kind glm --report /tmp/validation_glm_pre_market.json \
  --predictions data/predictions_glm.csv

echo "=== WALK-FORWARD PREDICTIONS (gbm) ==="
"$PYTHON" validate.py --kind gbm --report /tmp/validation_gbm_pre_market.json \
  --predictions data/predictions_gbm.csv

echo "=== MARKET COMPARISON ==="
"$PYTHON" market.py

echo "=== MARKET OFFSET ==="
"$PYTHON" market_offset.py

echo "=== SETTLE PAPER LEDGER ==="
"$PYTHON" ledger.py --settle-only

echo "=== FORWARD EVIDENCE ==="
"$PYTHON" forward_evidence.py

echo "=== FINAL REPORTS ==="
"$PYTHON" validate.py --kind glm --reuse-predictions \
  --report validation_glm.json --predictions data/predictions_glm.csv
"$PYTHON" validate.py --kind gbm --reuse-predictions \
  --report validation_gbm.json --predictions data/predictions_gbm.csv

echo "=== PIPELINE COMPLETE ==="
