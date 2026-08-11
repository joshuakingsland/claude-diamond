"""Stable configuration shared by production and research entry points.

Production code imports this module, never a research script with its own
experimental settings.
"""

# --------------------------------------------------------------- markets
# Every market this project prices. The moneyline, the run line, and the
# total are all read off one joint run distribution, so a model that moves
# one of them moves all three consistently.
MARKETS = ("h2h", "spreads", "totals")

# Human-readable model family.  The complete version written to cards and the
# ledger also carries the git revision and feature-schema hash; see
# ``provenance.py``.  A static label by itself made materially different
# models indistinguishable in the forward ledger.
MODEL_FAMILY = "diamond-v1"
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
# The API response can be fresh while the best book inside it is stale.  Both
# ages are gated separately by ``ledger.py``.
MAX_ODDS_AGE_MINUTES = 15
MAX_BOOK_QUOTE_AGE_MINUTES = 15
MARKET_DISAGREEMENT_WARNING = 0.04

# A single book can publish a stale or mis-mapped price far better than the
# paired-book consensus. Line shopping in a liquid market is worth a point or
# two; past this gap it is a broken quote, not an edge.
MAX_EXECUTION_DEVIATION = 0.03

# --------------------------------------------------------------- policy
# Nothing here allocates real money. The forward ledger is paper until the
# promotion gates in `validate.py` pass, and this repository never places a
# wager.
EDGE_RULE = 0.03
# Minimum expected profit per unit at the actual quoted price.  A probability
# disagreement is not sufficient when vig and payout differ across rows.
MIN_EXPECTED_VALUE = 0.015
# Research-only price-movement observations. This threshold never authorises a
# wager; it only fixes which predictions enter the append-only CLV probe.
MIN_CLV_SIGNAL = 0.001
MAX_STAKE = 1
GAME_DAY_STAKE_CAP = 3
GAME_RISK_BUCKET_STAKE_CAP = 1
REQUIRE_CONFIRMED_LINEUPS = True
STAKING_POLICY_VERSION = "paper-ev-mainline-riskbucket-v3"

BOOTSTRAP_MODELS = 30

# A real-money path remains deliberately absent.  These are promotion gates
# for the forward-evidence report, not switches that place wagers.
MIN_FORWARD_INDEPENDENT_GAMES = 500
MIN_FORWARD_ACCEPTED_FILL_RATE = 0.95

# --------------------------------------------------------------- timing
# Baseball's decisive information arrives late: lineups post roughly three
# hours before first pitch and the bullpen picture only settles once the
# previous day's games are final. A snapshot taken earlier than this is
# recorded for research but is not eligible to lock a wager.
MIN_LOCK_LEAD_MINUTES = 20
MAX_LOCK_LEAD_MINUTES = 240

# --------------------------------------------------------------- weather
# Train and serve from the SAME kind of weather information. StatsAPI reports
# observed conditions only once a game is under way or complete, so training
# on it would leak realised weather. Open-Meteo's historical-forecast archive
# supplies the operational forecast that was available before each game; the
# live path uses its current forecast endpoint with the same fields and units.
# StatsAPI weather is kept only as an independent join diagnostic.
WEATHER_SOURCE = "open-meteo"
WEATHER_FIELDS = (
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
)
