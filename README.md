# Indian Stock Signal Telegram Bot

A Telegram bot that analyzes NSE/BSE stocks using technical indicators (RSI, MACD, SMA crossovers, Bollinger Bands) and returns BUY/SELL/HOLD signals.

## ⚠️ Important Disclaimer

This is an **educational tool**. It is **not** SEBI-registered investment advice. Technical indicators are lagging — they can and do generate false signals. Never trade real money based purely on these outputs. Consult a SEBI-registered investment advisor before making any actual investment decision.

## Setup

### 1. Get a Telegram Bot Token
- Open Telegram and search `@BotFather`
- Send `/newbot`, follow the prompts
- Copy the token it gives you

### 2. Install dependencies
```bash
cd stock_bot
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
```
Open `.env` and paste your bot token.

### 4. Run
```bash
python bot.py
```

Open Telegram, find your bot, and send `/start`.

## Commands

| Command | Description |
|---|---|
| `/start` or `/help` | Show help |
| `/analyze RELIANCE` | Full analysis with all indicators |
| `/quick TCS` | Just the signal (one-liner) |
| `/watch INFY` | Add to watchlist |
| `/unwatch INFY` | Remove from watchlist |
| `/mywatch` | Analyze entire watchlist |
| Send a ticker directly | Same as `/analyze` |

Tickers default to NSE. For BSE append `.BO` (e.g. `RELIANCE.BO`).

## How signals are generated

Each of these four indicators casts a vote:

1. **SMA crossover (20/50-day)** — golden cross = bullish, death cross = bearish
2. **RSI (14)** — <30 oversold (buy), >70 overbought (sell)
3. **MACD (12,26,9)** — line vs signal crossovers + histogram momentum
4. **Bollinger Bands (20, 2σ)** — touches of upper/lower band

Final call = majority vote with confidence = % of indicators agreeing. Need at least 2 votes the same way for a non-HOLD call.

## Extending it

- **Persist watchlists** — swap the in-memory dict in `bot.py` for SQLite/Postgres
- **Scheduled alerts** — use `app.job_queue.run_repeating()` to check watchlists hourly
- **Add fundamentals** — pull P/E, ROE from `yf.Ticker.info` and weight them
- **Better intelligence** — feed the indicator output to Claude/GPT API for natural-language explanations
- **Backtesting** — replay the signal logic over historical data to *characterise* past behaviour (hit rate, P&L distribution, drawdown). Past behaviour is not a forecast — see the honest-evaluation note below.

## Evaluating performance honestly

These signals are **not** demonstrated to be profitable, and short-horizon price
moves are close to noise. Treat the tooling as measurement, not a profit oracle:

- **Out-of-sample is the only number that counts.** A good in-sample backtest
  proves nothing. Use the walk-forward harness, which optimises parameters on a
  training window and reports only the unseen test folds:
  ```bash
  python backtest.py --watchlist --score --walkforward
  ```
- **Win rate alone can't tell skill from luck.** The backtest and `/stats` now
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

## Going live (broker integration)

If you want the bot to actually *place* orders (not just signal them), you'd integrate with a broker API like Zerodha Kite Connect, Upstox, or Angel One SmartAPI. Note that:
- You need a real broker account and API subscription (Kite Connect is ₹2000/mo)
- SEBI rules apply — automated trading on someone else's account without registration is illegal
- Always start in paper-trading mode

## File structure

```
stock_bot/
├── analyzer.py       # Technical analysis logic
├── bot.py            # Telegram bot handlers
├── requirements.txt
├── .env.example
└── README.md
```
