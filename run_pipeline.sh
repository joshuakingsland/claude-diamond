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
  "$PYTHON" weather.py --games data/games.csv --out data/weather.csv && break
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

# Before validation, not after: validate.py imports this verdict rather than
# recomputing it, so running it later would leave the reports a run behind.
echo "=== MARKET COMPARISON ==="
"$PYTHON" market.py

echo "=== VALIDATE (glm) ==="
"$PYTHON" validate.py --kind glm --report validation_glm.json \
  --predictions data/predictions_glm.csv

echo "=== VALIDATE (gbm) ==="
"$PYTHON" validate.py --kind gbm --report validation_gbm.json \
  --predictions data/predictions_gbm.csv

echo "=== PIPELINE COMPLETE ==="
