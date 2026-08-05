#!/usr/bin/env bash
# Finish weather (resumable), then build features and validate.
set -u
cd /home/user/claude-diamond
while pgrep -f "python weather.py" >/dev/null; do sleep 10; done
for attempt in 1 2 3 4; do
  .venv/bin/python weather.py --games data/games.csv --out data/weather.csv \
    >> /tmp/claude-0/-home-user-claude-fights/2ec83d9b-d981-5a68-b84b-b4eccb6c2c56/scratchpad/weather.log 2>&1 && break
  echo "weather attempt $attempt incomplete; resuming"
  sleep $((attempt * 5))
done
echo "=== WEATHER DONE ==="
.venv/bin/python -c "
import pandas as pd
w=pd.read_csv('data/weather.csv'); g=pd.read_csv('data/games.csv')
print('weather coverage %.1f%%' % (100*w.game_pk.nunique()/g.game_pk.nunique()))
print(w.roof_category.value_counts().to_dict())
"
echo "=== FEATURES ==="
.venv/bin/python features.py
echo "=== VALIDATE (gbm) ==="
.venv/bin/python validate.py --kind gbm --report validation_gbm.json \
  --predictions data/predictions_gbm.csv
echo "=== VALIDATE (glm) ==="
.venv/bin/python validate.py --kind glm --report validation_glm.json \
  --predictions data/predictions_glm.csv
echo "=== PIPELINE COMPLETE ==="
