"""Generate the public model card at docs/index.html.

A sibling page exists for the UFC project and this one deliberately mirrors
its layout, so the two read the same way. The content does not mirror it,
because the results are not the same. That project reports no verified edge;
this one has measured a specific negative — the closing price beats the
standalone model on all three main-line markets with intervals excluding zero — and a page
that buried that under a list of tonight's picks would be advertising, not a
model card. The verdict sits above the card for that reason.

Everything on the page comes from files already in the repository:

    data/predictions_upcoming.csv   tonight's board, priced
    data/clv_signals.csv            frozen, non-wager price-movement probes
    data/paper_ledger.csv           legacy paper entries and their results
    data/paper_rejections.csv       what it declined, and which gate fired
    market_comparison.json          does the model beat the price
    validation_glm.json             does the model predict baseball
    data/credit_log.csv             what the capture has spent
    forward_evidence.json           whether the CLV probe can be promoted

Nothing is recomputed here. If a number on the page disagrees with the repo,
the page is stale, not right — which is why the header carries the timestamp
of the data rather than of the render.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from config import MAX_LOCK_LEAD_MINUTES, MIN_LOCK_LEAD_MINUTES

# Sides are named for the bettor, not for the frame the model prices in. The
# model always reports a home-oriented probability, and on a total "home" is
# the Over; printing that raw would put "home" beside a number about run
# scoring.
SIDE_LABELS = {
    ("h2h", "home"): "{home} ML",
    ("h2h", "away"): "{away} ML",
    ("spreads", "home"): "{home} {point:+g}",
    ("spreads", "away"): "{away} {away_point:+g}",
    ("totals", "home"): "Over {point:g}",
    ("totals", "away"): "Under {point:g}",
}

MARKET_LABELS = {"h2h": "moneyline", "spreads": "run line", "totals": "total"}

GATE_LABELS = {
    "below_edge_rule": "below rule",
    "too_few_books": "thin market",
    "lineups_unconfirmed": "lineups unconfirmed",
    "outside_lock_window": "outside lock window",
    "stale_quote": "stale quote",
    "stale_book_quote": "stale book quote",
    "execution_deviation": "broken quote",
    "below_expected_value": "negative/low EV",
    "risk_bucket_already_locked": "game exposure used",
    "risk_bucket_dominated": "better correlated position",
    "day_cap": "day cap",
    "already_locked": "already locked",
    "no_executable_price": "no price",
}


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_key(value):
    """One spelling of a line point, for matching rows across three files.

    The card, the ledger and the rejection log each record an absent point
    differently — "", NaN, the string "nan" — and they are matched on it. Any
    disagreement here is silent and total: every moneyline row stops matching,
    so no moneyline wager is ever marked on the page and no moneyline
    rejection ever shows the gate that stopped it.
    """
    number = _float(value)
    return "" if number is None else f"{number:g}"


def _float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def side_label(market, side, home, away, point):
    template = SIDE_LABELS.get((market, side), "{home}")
    return template.format(home=home, away=away, point=point or 0,
                           away_point=-(point or 0))


def market_label(market, point):
    """Name the market without restating a point the pick already carries.

    The run line is quoted home-oriented, so an away pick at -1.5 sits beside
    a market point of +1.5. Both are right and together they read as a
    contradiction, so the signed number is left to the pick, which states it
    from the side actually being backed.
    """
    if market == "h2h":
        return "moneyline"
    if market == "spreads":
        return "run line"
    return f"total {point:g}" if point is not None else "total"


def _timing(lead):
    """Explain the lock window independently of whichever policy gate fired."""
    if lead > MAX_LOCK_LEAD_MINUTES:
        return f"eligible window in {lead - MAX_LOCK_LEAD_MINUTES} min"
    if lead < MIN_LOCK_LEAD_MINUTES:
        return "timing window closed"
    return "timing window open"


def build_board(card, ledger, rejections, signals=None):
    """One row per market, separating model, market and projected close.

    The deployable fair price currently equals the market because the outcome
    residual failed its forward interval.  Collapsing those two values into a
    tick labelled ``model`` made the public page look broken and, at an exact
    zero, arbitrarily labelled the away/under side as a pick.  The standalone
    model remains a diagnostic; the only supported live signal is movement
    toward a projected later price.
    """
    locked = {}
    for wager in ledger:
        key = (str(wager["game_pk"]), wager["market"],
               point_key(wager["point"]))
        locked[key] = wager
    declined = {}
    for rejection in rejections:
        key = (str(rejection["game_pk"]), rejection["market"],
               point_key(rejection["point"]))
        # Keep the last gate seen: screening runs hourly and the reason a row
        # is not a wager changes as the lock window closes.
        declined[key] = rejection["gate"]
    frozen_signals = {}
    for signal in signals or []:
        key = (str(signal["game_pk"]), signal["market"],
               point_key(signal.get("point")))
        frozen_signals[key] = signal

    board = []
    for row in card:
        point = _float(row["point"])
        standalone_home = _float(row.get("model_prob_home"))
        market_home = _float(row["market_prob_home"])
        fair_home = _float(row.get("fair_prob_home"), standalone_home)
        close_home = _float(row.get("predicted_close_prob_home"), market_home)
        if standalone_home is None or market_home is None or fair_home is None:
            continue
        standalone_gap_home = standalone_home - market_home
        fair_gap_home = fair_home - market_home
        predicted_clv_home = _float(row.get("predicted_clv"),
                                    close_home - market_home)
        movement_weight = _float(row.get("movement_weight"), 0)
        movement_supported = movement_weight > 0 and abs(predicted_clv_home) > 0
        if movement_supported:
            side = "home" if predicted_clv_home > 0 else "away"
            signal_kind = "projected close move"
        elif abs(standalone_gap_home) > 1e-12:
            side = "home" if standalone_gap_home > 0 else "away"
            signal_kind = "standalone diagnostic"
        else:
            side = None
            signal_kind = "no directional signal"

        def orient(value):
            if side is None:
                return value
            return value if side == "home" else 1 - value

        standalone = orient(standalone_home)
        market = orient(market_home)
        fair = orient(fair_home)
        projected_close = orient(close_home)
        price = row.get(f"best_price_{side}") if side else None
        book = row.get(f"best_book_{side}") if side else None
        books = int(_float(row["market_books"], 0))
        spread = _float(row["market_spread"], 0)
        lead = int(_float(row["lead_minutes"], 0))
        key = (str(row["game_pk"]), row["market"], point_key(row["point"]))
        wager = locked.get(key)
        probe = frozen_signals.get(key)
        frozen_move = None
        if probe is not None and wager is None:
            probe_side = probe.get("side")
            if probe_side in ("home", "away") and probe_side != side:
                side = probe_side
                standalone = orient(standalone_home)
                market = orient(market_home)
                fair = orient(fair_home)
                projected_close = orient(close_home)
            price = _float(probe.get("price"), price)
            book = probe.get("book") or book
            frozen_move = _float(probe.get("predicted_clv"), 0)
            signal_kind = "frozen projected close move"
        if wager is not None:
            # A locked wager is shown as it was struck, not as the board reads
            # now. The market moves after a lock — one taken at five books was
            # displaying the one book still quoting it, which makes a wager
            # that cleared the gate look like it breached it. The recorded
            # quote is also the one it will be settled against.
            # .get throughout: the ledger is append-only, so it can still hold
            # rows written before a column existed. A page regenerated every
            # hour must not fail on its own history.
            side = wager.get("side") or side
            standalone = _float(wager.get("model_prob"), standalone)
            market = _float(wager.get("market_prob"), market)
            fair = standalone
            projected_close = market
            price = _float(wager.get("price"), price)
            book = wager.get("book") or book
            books = int(_float(wager.get("market_books"), books))
            spread = _float(wager.get("market_spread"), spread)
            lead = int(_float(wager.get("lead_minutes"), lead))
            signal_kind = "legacy paper entry"
            movement_supported = False
        raw_gap = abs(standalone - market)
        fair_gap = abs(fair - market)
        projected_move = (frozen_move if frozen_move is not None
                          else projected_close - market)
        if side is None:
            display_name = "No directional signal"
        else:
            display_name = side_label(row["market"], side, row["home_team"],
                                      row["away_team"], point)
        board.append({
            "game": f"{row['away_team']} @ {row['home_team']}",
            "home": row["home_team"], "away": row["away_team"],
            "market": row["market"],
            "market_label": market_label(row["market"], point),
            "pick": display_name,
            "signal_kind": signal_kind,
            "standalone": round(standalone * 100, 1),
            # Keep the old key for downstream consumers of the generated JSON.
            "model": round(standalone * 100, 1),
            "consensus": round(market * 100, 1),
            "fair": round(fair * 100, 1),
            "projected_close": round(projected_close * 100, 1),
            "raw_gap": round(raw_gap * 100, 1),
            "gap": round(fair_gap * 100, 1),
            "projected_move": round(max(0, projected_move) * 100, 2),
            "movement_supported": movement_supported,
            "probe": probe is not None,
            "price": _float(price),
            "book": book,
            "books": books,
            "spread": round(spread * 100, 1),
            "lead": lead,
            "timing": _timing(lead),
            "commence": row["commence_time"],
            "date": row["official_date"],
            "runs": f"{_float(row['expected_away_runs'], 0):.2f}-"
                    f"{_float(row['expected_home_runs'], 0):.2f}",
            "bet": wager is not None,
            "stake": _float(wager["stake"], 0) if wager else 0,
            "reason": ("paper quote captured" if probe is not None else
                       GATE_LABELS.get(declined.get(key),
                                       "" if wager else "outcome disabled")),
        })
    board.sort(key=lambda item: (-item["probe"], -item["movement_supported"],
                                 -item["projected_move"], -item["raw_gap"]))
    return board


def build_settled(ledger):
    rows = []
    for wager in ledger:
        outcome = (wager.get("outcome") or "").strip()
        if not outcome or outcome == "nan":
            continue
        point = _float(wager["point"])
        rows.append({
            "date": wager["official_date"],
            "pick": side_label(wager["market"], wager["side"],
                               wager["home_team"], wager["away_team"], point),
            "market": market_label(wager["market"], point),
            "model": round(_float(wager["model_prob"], 0) * 100, 1),
            "gap": round(_float(wager["disagreement"], 0) * 100, 1),
            "price": _float(wager["price"]),
            "outcome": outcome,
            "profit": round(_float(wager["profit"], 0), 3),
        })
    rows.sort(key=lambda item: item["date"], reverse=True)
    return rows


def build_verdict(comparison):
    """The market comparison, which is the finding the page exists to report."""
    rows = []
    for market, block in (comparison.get("close_prob") or {}).items():
        interval = block.get("delta_ci90_date_clustered")
        rows.append({
            "market": MARKET_LABELS.get(market, market),
            "games": block.get("games"),
            "delta": block.get("delta"),
            "interval": interval,
            "verdict": block.get("verdict", ""),
            "beaten": bool(interval and interval[1] < 0),
        })
    return rows


def build_chips(validation, comparison, ledger, credits, board, signals=None,
                evidence=None):
    moneyline = validation.get("moneyline") or {}
    settled = [w for w in ledger
               if (w.get("outcome") or "").strip() not in ("", "nan")]
    staked = sum(_float(w["stake"], 0) for w in settled)
    profit = sum(_float(w["profit"], 0) for w in settled)
    beaten = len([r for r in build_verdict(comparison) if r["beaten"]])
    remaining = credits[-1]["credits_remaining"] if credits else None
    chips = [
        {"label": "games", "value": f"{validation.get('games', 0):,}",
         "gold": False},
        {"label": "moneyline log loss",
         "value": f"{moneyline.get('log_loss', 0):.5f}", "gold": False},
        {"label": "vs constant",
         "value": f"{moneyline.get('log_loss_home_field_baseline', 0):.5f}",
         "gold": False},
        {"label": "markets beating close", "value": f"{beaten} of 3",
         "gold": False},
        {"label": "board", "value": f"{len(board)} quoted", "gold": False},
        {"label": "forward probes", "value": f"{len(signals or [])}",
         "gold": bool(signals)},
        {"label": "promotion",
         "value": (evidence or {}).get("promotion_status", "research_only"),
         "gold": False},
        {"label": "legacy paper",
         "value": f"{len(ledger)} ({len(settled)} settled)", "gold": False},
    ]
    if settled:
        chips.append({"label": "legacy P/L (descriptive)",
                      "value": f"{profit:+.2f}u on {staked:.0f}u", "gold": False})
    if remaining:
        chips.append({"label": "credits", "value": f"{int(remaining):,}",
                      "gold": False})
    return chips


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Diamond Ledger | Model Card Read</title>
<style>
  /* Fonts are served from this repository rather than fetched from Google.
     They are static files under docs/fonts, committed once, while this page is
     rewritten every capture — embedding them as data URIs would add ~190KB of
     base64 to an hourly commit. Latin subsets only: team names, prices and
     probabilities are all ASCII. */
  @font-face{font-family:'Anton';font-style:normal;font-weight:400;font-display:swap;
    src:url(fonts/anton-400.woff2) format('woff2')}
  @font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:400;font-display:swap;
    src:url(fonts/plexmono-400.woff2) format('woff2')}
  @font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:600;font-display:swap;
    src:url(fonts/plexmono-600.woff2) format('woff2')}
  /* Inter ships as a variable font, so one file covers every weight used. */
  @font-face{font-family:'Inter';font-style:normal;font-weight:100 900;font-display:swap;
    src:url(fonts/inter-var.woff2) format('woff2')}
  :root{--ink:#0b0e14;--surface:#141926;--line:#232b40;--text:#e8eaf0;--muted:#7a8299;
    --faint:#4a5268;--market:#8b93a7;--gold:#e8b54d;--gold-dim:#8a6f35;--close:#63c3b4;--warn:#c9705b}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--ink);color:var(--text);font-family:'Inter',system-ui,sans-serif;
    font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
  .wrap{max-width:940px;margin:0 auto;padding:0 20px 96px}
  header{padding:52px 0 8px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;
    color:var(--gold);text-transform:uppercase}
  h1{font-family:'Anton',sans-serif;font-size:clamp(44px,8vw,76px);line-height:.98;
    text-transform:uppercase;margin:10px 0 8px;letter-spacing:0}
  .sub{color:var(--muted);max-width:62ch}.sub b{color:var(--text);font-weight:600}
  .record{display:flex;flex-wrap:wrap;gap:10px;margin:26px 0 8px;
    font-family:'IBM Plex Mono',monospace;font-size:12px}
  .chip{border:1px solid var(--line);border-radius:4px;padding:7px 11px;color:var(--muted)}
  .chip b{color:var(--text);font-weight:600}.chip.gold b{color:var(--gold)}
  .legend{display:flex;flex-wrap:wrap;gap:22px;align-items:center;margin:30px 0 10px;
    font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)}
  .key{display:flex;gap:8px;align-items:center}.tick{width:2px;height:14px;background:var(--market)}
  .tick.model{background:var(--gold)}.tick.close{background:var(--close)}
  .band{width:22px;height:8px;background:var(--gold-dim);opacity:.7;border-radius:2px}
  .freshness{margin-top:18px;padding:10px 12px;border-left:3px solid var(--faint);
    color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px}
  .freshness.current{border-color:var(--gold)}
  .verdictbox{margin-top:30px;border:1px solid var(--warn);border-radius:6px;padding:18px 20px}
  .verdictbox h2{font-family:'Anton',sans-serif;font-size:20px;letter-spacing:.06em;
    text-transform:uppercase;font-weight:400;color:var(--warn)}
  .verdictbox p{color:var(--muted);font-size:13.5px;margin-top:8px;max-width:76ch}
  /* The verdict and ledger tables are the only fixed-width things on the
     page; on a phone they must scroll inside their own box rather than push
     the whole document sideways. */
  .scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
  table.verdict-table{width:100%;min-width:520px;border-collapse:collapse;margin-top:14px;
    font-family:'IBM Plex Mono',monospace;font-size:12.5px}
  .verdict-table th{color:var(--faint);font-weight:500;text-align:left;padding:8px;
    border-bottom:1px solid var(--line);font-size:11px}
  .verdict-table td{padding:8px;border-bottom:1px solid var(--line);color:var(--muted)}
  .verdict-table td.m{color:var(--text)}.verdict-table td.bad{color:var(--warn)}
  .sect{margin-top:34px}.sect-head{display:flex;justify-content:space-between;align-items:baseline;
    gap:16px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:4px}
  .sect-head h2{font-family:'Anton',sans-serif;font-size:20px;letter-spacing:.06em;text-transform:uppercase;font-weight:400}
  .sect-head span{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);text-align:right}
  .fight{display:grid;grid-template-columns:minmax(220px,1.15fr) 1.7fr minmax(135px,auto);gap:20px;
    align-items:center;padding:20px 14px;border-bottom:1px solid var(--line)}
  .fight.bet{background:linear-gradient(90deg,rgba(232,181,77,.05),transparent 55%);
    border-left:2px solid var(--gold);padding-left:16px}
  .who .name{font-weight:600;font-size:15px}.who .name.pick{color:var(--gold)}
  .who .vs{color:var(--faint);font-size:12px;margin:1px 0}
  .who .meta{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:6px}
  .who .quote{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--faint);margin-top:4px;line-height:1.5}
  .gauge{position:relative;height:62px}.rail{position:absolute;top:22px;left:0;right:0;height:2px;background:var(--line)}
  .mid{position:absolute;top:16px;left:50%;width:1px;height:14px;background:var(--faint);opacity:.6}
  .edgeband{position:absolute;top:19px;height:8px;border-radius:2px;background:var(--gold-dim);opacity:.65}
  .mark{position:absolute;top:12px;width:2px;height:22px;background:var(--market)}
  .mark.model{background:var(--gold)}.mark.close{background:var(--close);top:16px;height:14px}
  .glabel{position:absolute;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
    transform:translateX(-50%);white-space:nowrap}.glabel.top{top:-4px}.glabel.bot{top:36px}
  .glabel .v{color:var(--text)}.glabel.gold .v{color:var(--gold)}
  .triplet{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);margin-top:40px;text-align:center}
  .triplet b{color:var(--text);font-weight:500}.triplet .closev{color:var(--close)}
  .verdict{text-align:right;min-width:135px}.verdict .edge{font-family:'IBM Plex Mono',monospace;font-size:17px;font-weight:600}
  .bet .verdict .edge{color:var(--gold)}.tag{display:inline-block;margin-top:5px;font-family:'IBM Plex Mono',monospace;
    font-size:10.5px;letter-spacing:.1em;padding:3px 8px;border-radius:3px}
  .tag.yes{background:var(--gold);color:#141310;font-weight:600}.tag.no{border:1px solid var(--line);color:var(--faint)}
  .thresholds{margin-top:7px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--muted);line-height:1.7}
  table.ledger{width:100%;min-width:620px;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12.5px}
  .ledger th{color:var(--faint);font-weight:500;text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);font-size:11px}
  .ledger td{padding:9px 8px;border-bottom:1px solid var(--line);color:var(--muted)}.ledger td.pick{color:var(--text)}
  .ledger .w{color:var(--gold)}.ledger .l{color:var(--warn)}
  .empty{padding:22px 14px;color:var(--faint);font-family:'IBM Plex Mono',monospace;font-size:12px}
  .method{margin-top:52px;border-top:1px solid var(--line);padding-top:26px;color:var(--muted);font-size:13.5px}
  .method p{margin-bottom:10px;max-width:76ch}.disclaimer{margin-top:18px;padding:14px 16px;border:1px solid var(--line);
    border-radius:6px;font-size:13px;color:var(--muted)}.disclaimer b{color:var(--text)}
  @media(max-width:700px){.fight{grid-template-columns:1fr;gap:14px}.verdict{text-align:left}.sect-head{align-items:flex-end}}
  @media(prefers-reduced-motion:no-preference){.mark.model{transition:left .9s cubic-bezier(.2,.8,.2,1)}
    .edgeband{transition:all .9s cubic-bezier(.2,.8,.2,1)}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">Diamond Ledger | walk-forward __KIND__ | updated __UPDATED__</div>
  <h1>Model<br>Card Read</h1>
  <p class="sub">Every quoted line on tonight's card, priced by a model trained on
  <b>__GAMES__ completed MLB games</b> and benchmarked against captured closing prices.
  The standalone model is shown as a diagnostic, but it does not beat the closing market.
  Outcome fair value is therefore market-only. The supported research target is the much smaller
  projected move toward a later price; a frozen probe is <b>not a wager</b>.</p>
  <div class="record" id="chips"></div>
  <div class="legend"><div class="key"><span class="tick model"></span> standalone diagnostic</div>
    <div class="key"><span class="tick"></span> current market (de-vig)</div>
    <div class="key"><span class="tick close"></span> projected later price</div></div>
  <div class="freshness current">Board captured <b>__CAPTURED__</b> | results through <b>__RESULTS__</b></div>
</header>

<div class="verdictbox">
  <h2>The model does not beat the price</h2>
  <p>Measured over __PRICED__ priced game-markets from __EVENTS__ captured events. Delta is model
  minus market log loss, so <b>positive means the market is better</b>, with a 90% interval
  resampled over slates rather than games. A well-calibrated model that still loses to the close
  is the normal outcome in a liquid market.</p>
  <div class="scroll"><table class="verdict-table"><thead><tr><th>MARKET</th><th>GAMES</th>
    <th>DELTA VS CLOSE</th><th>90% INTERVAL</th><th>VERDICT</th></tr></thead>
    <tbody id="verdict"></tbody></table></div>
</div>

<section class="sect"><div class="sect-head"><h2>Frozen forward CLV probes</h2><span id="nprobes"></span></div><div id="probes"></div></section>
<section class="sect"><div class="sect-head"><h2>Research board</h2><span id="nrest"></span></div><div id="rest"></div></section>
<section class="sect"><div class="sect-head"><h2>Legacy paper history</h2><span id="nsettled"></span></div><div id="settled"></div></section>

<div class="method">
  <p><b>How a line gets here.</b> Prices are captured hourly across the card from every book in
  the priced region, paired two-sided per book and de-vigged individually, then taken as a median
  — never a de-vig of aggregated prices. The model prices moneyline, run line and total off a
  single joint distribution over (home runs, away runs), so the three cannot contradict one
  another. Whole-number lines can push, and the model probability is renormalised onto the
  market's push-excluded basis before the two are compared.</p>
  <p><b>What the policy does.</b> Outcome deployment is currently market-only: the standalone
  disagreement cannot authorise an entry. A separate forward probe freezes at most one supported
  moneyline or run-line movement signal per game risk bucket, only with confirmed lineups, at
  least three paired books, a fresh quote and 20–240 minutes before first pitch.</p>
  <p><b>What this page is not.</b> It is not a tip sheet. The verdict above is the finding; the
  board below is what a stated policy would have done given that finding, recorded so that it can
  be checked later against results rather than argued about now.</p>
  <div class="disclaimer"><b>Research only.</b> A CLV probe is a timestamped quote, not an accepted
  fill or wager. The legacy P/L above predates the current promotion policy and is descriptive,
  not evidence of an edge. Nothing here is betting advice. If you gamble, gamble only what
  you can afford to lose.</div>
</div>
</div>
<script>
const BOARD=__BOARD__, SETTLED=__SETTLED__, VERDICT=__VERDICT__, CHIPS=__CHIPS__;
const pct=v=>v.toFixed(1)+'%';
const money=v=>v==null?'--':(v>0?'+':'')+v.toFixed(0);
// The gauge spans the honest range of a baseball win probability rather than
// 0-100. A single game is close to a coin flip, so a full-width axis would
// compress every mark into the middle and show nothing.
const LO=20,HI=80,span=HI-LO;
const at=v=>Math.max(0,Math.min(100,(v-LO)/span*100));

document.getElementById('chips').innerHTML=CHIPS.map(c=>
  `<div class="chip${c.gold?' gold':''}">${c.label} <b>${c.value}</b></div>`).join('');

document.getElementById('verdict').innerHTML=VERDICT.map(v=>{
  const iv=v.interval?`[${v.interval[0].toFixed(4)}, ${v.interval[1].toFixed(4)}]`:'--';
  return `<tr><td class="m">${v.market}</td><td>${v.games}</td>`+
    `<td class="${v.beaten?'m':'bad'}">${v.delta>0?'+':''}${v.delta.toFixed(5)}</td>`+
    `<td>${iv}</td><td class="${v.beaten?'m':'bad'}">${v.verdict}</td></tr>`;
}).join('');

function row(g){
  const a=at(g.consensus),b=at(g.standalone),c=at(g.projected_close);
  return `<div class="fight${g.probe?' bet':''}">
    <div class="who"><div class="name${g.probe?' pick':''}">${g.pick}</div>
      <div class="vs">${g.game}</div>
      <div class="meta">${g.signal_kind} | ${g.market_label} | ${g.books} books | exp ${g.runs}</div>
      <div class="quote">${g.book||'--'} ${money(g.price)} | first pitch in ${g.lead} min | book range ${g.spread}pp</div>
      <div class="quote">${g.timing} | ${g.reason}</div></div>
    <div class="gauge"><div class="rail"></div><div class="mid"></div>
      <div class="mark" style="left:${a}%"></div>
      <div class="mark model" style="left:${b}%"></div>
      <div class="mark close" style="left:${c}%"></div>
      <div class="triplet">standalone <b>${pct(g.standalone)}</b> · market <b>${pct(g.consensus)}</b> · projected close <b class="closev">${pct(g.projected_close)}</b></div></div>
    <div class="verdict"><div class="edge">${g.movement_supported?g.projected_move.toFixed(2)+' pp move':g.raw_gap.toFixed(1)+' pp raw gap'}</div>
      <span class="tag ${g.probe?'yes':'no'}">${g.probe?'FROZEN CLV PROBE':g.movement_supported?'CLV RESEARCH':'DIAGNOSTIC ONLY'}</span>
      <div class="thresholds">outcome fair ${pct(g.fair)} · ${g.date}</div></div></div>`;
}

const probes=BOARD.filter(g=>g.probe), rest=BOARD.filter(g=>!g.probe);
document.getElementById('probes').innerHTML=probes.length?probes.map(row).join(''):
  '<div class="empty">No forward CLV observation has frozen on this board yet.</div>';
document.getElementById('nprobes').textContent=probes.length+' frozen, non-wager quote'+(probes.length===1?'':'s');
document.getElementById('rest').innerHTML=rest.length?rest.map(row).join(''):
  '<div class="empty">Nothing else quoted.</div>';
document.getElementById('nrest').textContent=rest.length+' quoted lines · outcome deployment disabled';

document.getElementById('settled').innerHTML=SETTLED.length?
  `<div class="scroll"><table class="ledger"><thead><tr><th>DATE</th><th>PICK</th><th>MARKET</th><th>MODEL %</th>
   <th>GAP</th><th>PRICE</th><th>RESULT</th><th>P/L</th></tr></thead><tbody>`+
  SETTLED.map(s=>`<tr><td>${s.date}</td><td class="pick">${s.pick}</td><td>${s.market}</td>
    <td>${pct(s.model)}</td><td>${s.gap.toFixed(1)}</td><td>${money(s.price)}</td>
    <td class="${s.outcome==='win'?'w':s.outcome==='loss'?'l':''}">${s.outcome}</td>
    <td class="${s.profit>0?'w':s.profit<0?'l':''}">${s.profit>0?'+':''}${s.profit.toFixed(2)}u</td></tr>`).join('')+
  '</tbody></table></div>':
  '<div class="empty">No legacy paper entry has settled.</div>';
document.getElementById('nsettled').textContent=SETTLED.length+' historical entries · descriptive only';
</script>
</body>
</html>
"""


def render(card, ledger, rejections, comparison, validation, credits,
           signals=None, evidence=None):
    board = build_board(card, ledger, rejections, signals)
    settled = build_settled(ledger)
    verdict = build_verdict(comparison)
    chips = build_chips(validation, comparison, ledger, credits, board,
                        signals, evidence)
    coverage = comparison.get("coverage") or {}

    captured = card[0]["odds_fetched_at"] if card else "no board captured"
    results = max((w["official_date"] for w in ledger
                   if (w.get("outcome") or "").strip() not in ("", "nan")),
                  default="nothing settled")
    # The data's timestamp, not the render's: a page that stamps itself with
    # the moment it ran looks fresh even when the inputs are days old.
    updated = card[0]["priced_at"] if card else (
        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}")

    html = TEMPLATE
    for token, value in (
        ("__KIND__", (card[0]["model_kind"] if card else "glm")),
        ("__UPDATED__", updated),
        ("__GAMES__", f"{validation.get('games', 0):,}"),
        ("__CAPTURED__", captured),
        ("__RESULTS__", results),
        ("__PRICED__", f"{coverage.get('priced_game_markets', 0):,}"),
        ("__EVENTS__", f"{coverage.get('odds_events', 0):,}"),
        ("__BOARD__", json.dumps(board)),
        ("__SETTLED__", json.dumps(settled)),
        ("__VERDICT__", json.dumps(verdict)),
        ("__CHIPS__", json.dumps(chips)),
    ):
        html = html.replace(token, value)
    return html


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="data/predictions_upcoming.csv")
    parser.add_argument("--ledger", default="data/paper_ledger.csv")
    parser.add_argument("--rejections", default="data/paper_rejections.csv")
    parser.add_argument("--comparison", default="market_comparison.json")
    parser.add_argument("--validation", default="validation_glm.json")
    parser.add_argument("--credits", default="data/credit_log.csv")
    parser.add_argument("--signals", default="data/clv_signals.csv")
    parser.add_argument("--evidence", default="forward_evidence.json")
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args(argv)

    html = render(
        _rows(args.card), _rows(args.ledger), _rows(args.rejections),
        _json(args.comparison), _json(args.validation), _rows(args.credits),
        _rows(args.signals), _json(args.evidence),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
