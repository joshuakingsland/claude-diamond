"""Stable configuration shared by production and research entry points.

Production code imports this module, never a research script with its own
experimental settings.
"""

# --------------------------------------------------------------- markets
# Every market this project prices. The moneyline, the run line, and the
# total are all read off one joint run distribution, so a model that moves
# one of them moves all three consistently.
MARKETS = ("h2h", "spreads", "totals")

MODEL_VERSION = "diamond-v0"
ODDS_CONSENSUS_VERSION = "paired-book-devig-v1"

# --------------------------------------------------------------- odds feed
SPORT_KEY = "baseball_mlb"

# Regions requested from the odds API, each as its own request. The API bills
# one credit per region per market, so cost is cadence x regions x markets and
# does not scale with the number of games on the card.
ODDS_REGIONS = ("us", "eu")

# Regions whose books may reach the model consensus and the executable price.
# Capturing a region is not the same as trusting it: `eu` exists so Pinnacle,
# the reference sharp baseball market, accumulates history before any decision
# to price it. Promoting a region is a model change, not a config flip.
PRICED_ODDS_REGIONS = ("us",)

# Books that set the baseball market rather than follow it. Research column
# only; nothing here feeds the model, the consensus, or the edge rule.
LEADER_BOOK_KEYS = ("pinnacle", "betonlineag", "lowvig", "circasports")

# --------------------------------------------------------------- quote gates
MIN_MARKET_BOOKS = 3
MAX_ODDS_AGE_MINUTES = 180
MARKET_DISAGREEMENT_WARNING = 0.04

# A single book can publish a stale or mis-mapped price far better than the
# paired-book consensus. Line shopping in a liquid market is worth a point or
# two; past this gap it is a broken quote, not an edge.
MAX_EXECUTION_DEVIATION = 0.08

# --------------------------------------------------------------- policy
# Nothing here allocates real money. The forward ledger is paper until the
# promotion gates in `validate.py` pass, and this repository never places a
# wager.
EDGE_RULE = 0.03
MAX_STAKE = 1
GAME_DAY_STAKE_CAP = 3
STAKING_POLICY_VERSION = "paper-flat-1u-daycap3-v1"

BOOTSTRAP_MODELS = 30

# --------------------------------------------------------------- timing
# Baseball's decisive information arrives late: lineups post roughly three
# hours before first pitch and the bullpen picture only settles once the
# previous day's games are final. A snapshot taken earlier than this is
# recorded for research but is not eligible to lock a wager.
MIN_LOCK_LEAD_MINUTES = 20
MAX_LOCK_LEAD_MINUTES = 240

# --------------------------------------------------------------- weather
# Train and serve from the SAME weather source. StatsAPI reports observed
# conditions only once a game is under way or complete, so training on it and
# serving on a forecast would build a model on information the live path never
# has. Open-Meteo supplies both a reanalysis archive and a forecast, so it is
# the single source of truth for model inputs; StatsAPI weather is kept purely
# as an independent check on that join.
WEATHER_SOURCE = "open-meteo"
WEATHER_FIELDS = (
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
)
