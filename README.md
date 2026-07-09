# Indian Stock Signal Telegram Bot

A personal-use Telegram bot for NSE/BSE stocks: swing and intraday technical analysis, a 100-point intraday scoring engine, an automated setup scanner with alert loops, tip capture from Telegram channels and forwarded messages, and an honest outcome-tracking / stats pipeline that scores every call the bot (or a tip channel) actually made.

## ⚠️ Important Disclaimer

This is an **educational tool**. It is **not** SEBI-registered investment advice. Technical indicators are lagging — they can and do generate false signals. These signals are **not demonstrated to be profitable** (see "Evaluating performance honestly" below). Never trade real money based purely on these outputs. Consult a SEBI-registered investment advisor before making any actual investment decision.

## Features

- **Swing analysis** (`/swing`) — daily candles, 4-indicator majority vote (SMA, RSI, MACD, Bollinger)
- **Intraday analysis** (`/intraday`, `/score`) — 5-min candles; 100-point score across gap, relative volume, VWAP, EMA, ORB, volume breakout
- **Setup scanner** (`/scan`) — ORB / VWAP / range setups from `STRATEGY.md`, with universal filters and quota-aware batch slicing
- **Alert loops** — intraday auto-scan every 5 min (`/scan_alerts`), end-of-day swing calls at 15:45 IST (`/swing_alerts`), daily outcome report at 16:20 IST (`/eod_report`)
- **Narrative reports** (`/report`) — deterministic institutional-style write-up, templated over engine output; no LLM/API involved
- **Tip capture** — forward a message or paste a screenshot (OCR via Tesseract) and the bot extracts symbol/action/entry/SL/targets, logs it immutably, and re-analyzes it; a Telethon listener can also monitor public channels automatically
- **Outcome tracking** (`/stats`, `/today`) — every actionable alert is resolved against its own published levels; realized win rate, P&L, Sharpe/Sortino, t-stat, 95% CI
- **Data**: Angel One SmartAPI (realtime) when configured, yfinance (~15 min delayed) fallback otherwise
- **Access control** — allowlist via `AUTHORIZED_USERS`; without it the bot is open to anyone who finds it (startup warns — this is a SEBI-exposure and resource-abuse risk)

## Setup

### 1. Get a Telegram Bot Token
- Open Telegram and search `@BotFather`
- Send `/newbot`, follow the prompts
- Copy the token it gives you

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
For tip-screenshot OCR the **Tesseract binary** must also be installed on the host (Windows: UB-Mannheim build; Linux: `sudo apt install tesseract-ocr`).

### 3. Configure

Create a `.env` file in the project root. `TELEGRAM_BOT_TOKEN` is the only required key; everything else enables optional features.

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | **Required.** Bot token from @BotFather |
| `AUTHORIZED_USERS` | Comma-separated numeric Telegram user IDs allowed to use the bot. **Strongly recommended** — empty = open to everyone |
| `ANGEL_API_KEY` / `ANGEL_CLIENT_CODE` / `ANGEL_PASSWORD` / `ANGEL_TOTP_SECRET` | Angel One SmartAPI credentials → realtime data. Unset = yfinance fallback (~15 min delayed) |
| `TELETHON_API_ID` / `TELETHON_API_HASH` / `TELETHON_PHONE` | Telethon user-account API creds (my.telegram.org) for the channel listener |
| `TELETHON_CHANNELS` | Comma-separated public channel usernames to monitor for tips |
| `TELETHON_NOTIFY_USER_ID` | Your Telegram user ID — receives detected-tip notifications (auto-added to the allowlist) |
| `MARKETAUX_API_TOKEN` | Marketaux key for news snippets on `/score` (optional, rate-limited free tier) |
| `WEBHOOK_URL` / `WEBHOOK_SECRET` / `PORT` | Run in webhook mode instead of polling (leave unset for polling) |
| `TESSERACT_CMD` | Path to the Tesseract binary if not on `PATH` |
| `LOG_LEVEL` / `LOG_FILE` | Logging config (defaults: INFO, stderr) |
| `DISABLE_SSL_VERIFY` / `ANGEL_DISABLE_SSL` | Local-dev only, for corp SSL-inspecting proxies. **Never in production** |

### 4. (Optional) Authenticate the Telethon channel listener
```bash
python auth_once.py
```
Run once per machine — enter the OTP Telegram sends you; a `telethon_session.session` file is created. The session file is device-bound: don't copy it between machines, re-run `auth_once.py` on each.

### 5. Run
```bash
python bot.py
```
Polling mode by default; set `WEBHOOK_URL` for webhook mode. Only one poller per token can run at a time — stop local runs before starting the VM instance (and vice versa).

Open Telegram, find your bot, and send `/start`. Send `/guide` for an interactive walkthrough.

## Commands

### Analysis
| Command | Description |
|---|---|
| `/swing SYMBOL` (alias `/analyze`) | Full swing analysis — daily candles, all indicators |
| `/intraday SYMBOL` | Full intraday analysis — 5-min candles |
| `/quick SYMBOL` / `/quickintra SYMBOL` | One-line swing / intraday signal |
| `/score SYMBOL [SYMBOL…]` | 100-point intraday scorecard (max 5) |
| `/report SYMBOL [swing]` | Institutional-style narrative read (deterministic, no LLM) |
| Send a ticker directly | Same as `/swing` |

### Scanner & alerts
| Command | Description |
|---|---|
| `/scan [SYMBOL…]` | Scan tier-1 watchlist (or given symbols, max 5) for STRATEGY.md setups |
| `/scan_alerts on\|off` | Intraday auto-scan every 5 min during market hours, ping on setups |
| `/swing_alerts on\|off` | End-of-day BUY/SELL calls at 15:45 IST on your watchlist |
| `/eod_report on\|off` | Daily alert-outcome summary at 16:20 IST |
| `/today` | On-demand EOD report right now |
| `/stats` | Realized performance from resolved alert outcomes |

### Watchlist & market
| Command | Description |
|---|---|
| `/watch` / `/unwatch SYMBOL` | Manage watchlist |
| `/mywatch` | Swing-analyze entire watchlist |
| `/index` | NIFTY / Bank NIFTY / SENSEX snapshot |
| `/universe` | Show the stock universe scanned by alert loops |

### Diagnostics & onboarding
| Command | Description |
|---|---|
| `/start` / `/help` | Command reference |
| `/guide` | Interactive walkthrough with examples |
| `/angel_status` / `/angel_login` | Data-provider session status / force fresh login |
| `/tg_status` | Telethon channel-listener status |

Tickers default to NSE (`.NS` added automatically). For BSE append `.BO` (e.g. `RELIANCE.BO`).

## Tip capture

Three ingestion paths, all funneled into the same parser (`tip_parser.py`) and logged immutably at receipt:

1. **Forward a message** to the bot in private chat (text or photo caption)
2. **Paste a screenshot** — OCR'd with Tesseract
3. **Group chats** — add the bot to a group (privacy mode disabled in BotFather); it stays silent unless a tip is detected
4. **Telethon listener** — monitors the public channels in `TELETHON_CHANNELS` via your user account and notifies `TELETHON_NOTIFY_USER_ID`

Extracted fields: symbol (validated against the NSE universe), action, entry, stop-loss, targets. Channel tips are later scored on the channel's *own* posted levels (`channel_call` in `/stats`).

## How signals are generated

**Swing** (`/swing`, `/quick`, swing alerts) — each of four indicators casts a vote on daily candles:

1. **SMA crossover (20/50-day)** — golden cross = bullish, death cross = bearish
2. **RSI (14, Wilder)** — <30 oversold (buy), >70 overbought (sell)
3. **MACD (12,26,9)** — line vs signal crossovers + histogram momentum
4. **Bollinger Bands (20, 2σ)** — touches of upper/lower band

Final call = majority vote; confidence = % of indicators agreeing; at least 2 same-way votes needed for a non-HOLD call. The still-forming candle is dropped before analysis so live signals don't repaint.

**Intraday** (`/intraday`) uses the same vote on 5-min candles; `/score` uses the separate 100-point engine (gap 20, relative volume 25, VWAP/EMA/ORB/volume-breakout components — see `intraday_score.py`).

**Scanner** (`/scan`, scan alerts) detects the ORB / VWAP-reclaim / range setups defined in `STRATEGY.md`, behind universal filters (liquidity, spread, market trend), with quota-aware batch slicing and a circuit breaker for Angel One rate limits.

## Evaluating performance honestly

These signals are **not** demonstrated to be profitable, and short-horizon price
moves are close to noise. Treat the tooling as measurement, not a profit oracle:

- **Out-of-sample is the only number that counts.** A good in-sample backtest
  proves nothing. Use the walk-forward harness, which optimises parameters on a
  training window and reports only the unseen test folds:
  ```bash
  python backtest.py --watchlist --score --walkforward
  ```
- **Win rate alone can't tell skill from luck.** The backtest and `/stats`
  report Sharpe/Sortino, a t-stat, and a 95% CI on the mean return. If the CI
  straddles 0, there is no demonstrable edge yet — regardless of win rate.
- **Survivorship bias** is present until dated universe snapshots accumulate
  (the bot writes one per day; see `universe.py`). Backtests print a warning
  while biased.
- **Channel tips** are scored on the channel's *own* posted levels
  (`channel_call` in `/stats`), captured immutably at receipt — so you can see
  whether a source is actually any good, not whether the bot's re-analysis was.
- All P&L is hypothetical, gross of slippage, and optimistic on fills. Realised,
  cost-inclusive results will be worse.

## File structure

```
stockagent/
├── bot.py                 # Telegram handlers, auth gate, background loops
├── config.py              # Central strategy config (all tunable thresholds)
├── engines/               # Independent analysis engines (structured evidence,
│   ├── base.py            #   never bare BUY/SELL): EngineResult contract
│   ├── gap.py             #   opening-gap measurement + points
│   ├── volume.py          #   RVOL / volume-breakout measurement + points
│   ├── vwap.py            #   VWAP confirmation
│   ├── orb.py             #   opening-range breakout (dual window, spent-move cap)
│   ├── market_regime.py   #   day-type / volatility / breadth / expiry labels
│   ├── liquidity.py       #   turnover, spread-proxy, slippage floors (proxies)
│   ├── risk.py            #   fixed-fractional sizing + daily R budget
│   ├── execution.py       #   itemized statutory Indian cost model
│   ├── context_daily.py   #   continuous indicator features (votes → numbers)
│   ├── volatility.py      #   ATR percentile, NR7, inside bar, squeeze state
│   ├── price_action.py    #   swing structure, HH/HL/LH/LL, BOS/CHoCH, S/R
│   └── relative_strength.py #  return vs index over windows, beta, RS rank
├── feature_eval.py        # Feature evaluation harness (the referee — Phase 3.1)
├── analyzer.py            # Swing/intraday multi-indicator vote engine
├── intraday_score.py      # 100-point intraday scorecard (composes engines/)
├── scanner.py             # Intraday setup scanner (STRATEGY.md setups)
├── scanner_filters.py     # Universal pre-trade filters
├── scanner_indicators.py  # Scanner-specific indicator math
├── scanner_setups.py      # ORB / VWAP / range setup detectors
├── data_provider.py       # Angel One SmartAPI primary, yfinance fallback
├── data_archive.py        # Local 5-min candle archive + paged backfill CLI
├── indices.py             # NIFTY / Bank NIFTY / SENSEX definitions
├── universe.py            # NSE index constituents + daily dated snapshots
├── market_calendar.py     # NSE holiday calendar (nse_holidays.txt)
├── market_context.py      # Delivery %, earnings dates, Marketaux news
├── subscriptions.py       # SQLite state: watchlists, alert opt-ins, alert log
├── features.py            # Decision-time feature snapshots (training data)
├── evidence.py            # TradeEvidence structured-output contract
├── eod_report.py          # Daily outcome resolution vs published levels
├── stats.py               # Realized-performance stats from resolved outcomes
├── riskmetrics.py         # Sharpe/Sortino/t-stat/CI (stdlib-only)
├── narrative.py           # Deterministic institutional write-ups (no LLM)
├── backtest.py            # Walk-forward backtest harness
├── tip_parser.py          # Tip extraction from text/OCR
├── telethon_listener.py   # Public-channel tip monitor (user account)
├── auth_once.py           # One-time Telethon OTP authentication
├── ssl_dev.py             # Local-dev TLS workaround (corp proxies)
├── constants.py           # Shared constants (IST timezone, swing gates, disclaimer)
├── tests/                 # pytest suite (indicators, resolver, repaint, auth, …)
├── deploy/                # VM deploy script + notes (see DEPLOY.md)
├── STRATEGY.md            # Setup definitions, filters, exit rules
└── DEPLOY.md              # GCP VM / systemd deployment guide
```

## Backtesting

```bash
python backtest.py RELIANCE          # single symbol, scanner setups
python backtest.py --watchlist      # tier-1 watchlist
python backtest.py --watchlist --score --walkforward   # honest OOS numbers
python backtest.py --watchlist --local                 # replay from the archive
```
Replays 5-min history candle-by-candle through the **same** filters and detectors used live, simulates §7 partial-exit rules, and reports cost-adjusted stats. See `backtest.py --help` for all flags.

## Candle archive & research data

Provider history is capped (~60 days of 5-min via yfinance), so the bot **archives every fetched candle** to a local `market_data.db` automatically (disable with `ARCHIVE_DISABLED=true`). Seed deeper history and inspect coverage with:

```bash
python data_archive.py backfill --days 365 --universe tier1   # paged Angel fetch
python data_archive.py coverage
```

`backtest.py --local` then replays from this frozen store — reproducible inputs, no provider cap.

Alongside candles, every logged alert stores a **decision-time feature snapshot** (`alerts_log.features` JSON via `features.py`) and every resolved outcome stores **MFE/MAE and duration**. `subscriptions.get_training_rows()` exports the joined (features, outcome) pairs — the dataset any future probability model will train on. Strategy thresholds live in `config.py` (overridable via `STOCKAGENT_CONFIG=overrides.json`), pinned by golden-parity tests so a config typo can't silently change live signals.

## Engines, risk layer & cost models

Analysis is organized as **independent engines** (`engines/`) that each return structured evidence (`EngineResult`), never a bare BUY/SELL — the 100-point scorer now composes them, golden-parity-tested to identical output. Auto-scan alerts additionally log **market-regime features** (day-type, volatility state via India VIX, universe breadth, expiry flag) so regime-conditioned performance becomes measurable. All of the following are **off by default** and activate only via config:

- **Position sizing & daily risk budget** — set `risk.capital` (₹) in a `STOCKAGENT_CONFIG` overrides file to get fixed-fractional position sizes on alerts and a hypothetical-exposure section in the EOD report; set `risk.max_daily_risk_r` to stop signaling after that many R lost in a day.
- **Statutory cost model** — `costs.model: "statutory"` (or `backtest.py --cost-model statutory`) replaces the flat 0.13%/round-trip estimate with itemized brokerage, STT, exchange, SEBI, stamp duty, GST and a separate slippage estimate; swing outcomes use delivery rates (higher STT/stamp), which the flat estimate understates.
- **Probabilities are still never printed** — `TradeEvidence.probability_*` stays `None` until a calibrated, walk-forward-validated model exists (Phase 4 gate).

## Feature engines & the evaluation harness

Beyond the scorer, the platform computes a wide **feature set** on every fired alert (logged to `alerts_log.features`): gap classification (tiny/normal/breakaway/runaway/exhaustion + fill candidate), the VWAP state machine (hold/reclaim/failure, time-above, ATR-normalized distance), volume microstructure (curve-relative RVOL, acceleration, an honestly-labeled institutional proxy), volatility regime (ATR percentile, NR7, inside bar, squeeze), price-action structure (swing points, HH/HL/LH/LL, break-of-structure vs change-of-character, S/R), and relative strength vs the index. These are **features, not signals** — none of them gates a live trade.

The discipline is enforced by `feature_eval.py`, the **evaluation harness**: it reads the `(features, resolved outcome)` pairs the bot collects and reports each feature's Spearman information coefficient, quintile outcome ladder, top-vs-bottom spread, first/second-half stability, and cross-feature correlation — all stamped with sample size, and flagged **underpowered** below a threshold so a lucky correlation on tiny data can't masquerade as edge.
```bash
python feature_eval.py                 # over all resolved+featured trades
python feature_eval.py --outcome R     # rank features by R-multiple instead of %
```
No feature is promoted to a decision until it demonstrably separates outcomes here on real, out-of-sample data.

## Deployment

Runs as a systemd service on a GCP VM — see `DEPLOY.md` and `deploy/`. Two operational gotchas:

- `.env` is gitignored: new config keys must be set on the VM by hand after `git pull`.
- Telethon session files are device-bound: run `auth_once.py` on the VM itself (over SSH), never copy a session file from another machine.

## Tests

```bash
pytest tests/
```
Covers indicator math (Wilder RSI/ATR), the outcome resolver, repaint guards, the auth gate, holiday calendar, tip parsing, walk-forward integrity, and a CI guard that fails if any `.session`/`.env` credential file is ever git-tracked.

## Going live (broker order placement)

The bot **signals** trades; it does not place orders. Angel One SmartAPI is integrated for *data* only. If you extend it to actually place orders:
- You need a broker account with API access and must handle order/position state
- SEBI rules apply — automated trading on someone else's account without registration is illegal
- Always start in paper-trading mode
