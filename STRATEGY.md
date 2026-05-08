# Indian Intraday Strategy — 1–2% Per Trade

> **Disclaimer:** Educational framework, not financial advice. Backtest on at least 6 months of 5-min data and forward-test on paper for 4–6 weeks before risking real capital. Intraday trading can lose money rapidly.

---

## 1. Core Philosophy

Take **2–4 high-conviction trades per day**, not 15 mediocre ones. Each trade must satisfy at least **3 confluence checks** before entry. If three don't line up, skip — the next setup will come.

**Targets:** 1–2% per trade.
**Stop loss:** 0.5–0.8%.
**Min blended exit RR:** 1.5:1 (per-leg may be lower if T2 compensates; portfolio average must clear 1.5).

---

## 2. Risk Management & Daily Circuits

Risk is the gate that opens or closes the trading day **before** any setup is considered.

| Rule | Value |
|---|---|
| Risk per trade | 0.5% of capital. Non-negotiable. |
| Position size | `Qty = (Capital × 0.005) / (Entry − Stop)` |
| Max gross exposure | 3× capital across all open positions (regardless of position-size formula) |
| Max concurrent positions | 2 |
| Max 1 position per sector | Yes — no concurrent HDFCBANK + ICICIBANK |
| Daily loss limit | 1.5% of capital → **stop trading**, no exceptions |
| Daily profit target | 2.5% → consider closing terminal |
| Pyramiding | **Never.** No adding to winners or losers. |
| Averaging down | **Never.** Stop loss is sacred. |
| Position size after a loss | **Never increased.** Same 0.5% risk; revenge sizing is the #1 account killer. |

These rules run in priority order. If the daily loss circuit fires, no setup matters — done for the day.

---

## 3. Stock Universe (Eligibility)

Only trade names that pass **all** of these checks. Refresh the universe monthly.

| Filter | Threshold | Data source |
|---|---|---|
| Index membership | Nifty 50, Nifty Next 50, or Bank Nifty constituent | NSE public list |
| Avg daily volume (20d) | ≥ 1 crore shares OR ≥ ₹500 Cr turnover | yfinance / Angel |
| F&O eligibility | Listed in NSE F&O segment | NSE F&O list, refresh nightly |
| Delivery % (prev day) | ≥ 35% | NSE bhavcopy CSV (must scrape; no API equivalent) |
| Price | ₹100 – ₹5,000 (avoid penny + ultra-high priced) | yfinance / Angel |

**Tier-1 starting watchlist:** RELIANCE, HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, INFY, TCS, LT, BAJFINANCE, MARUTI, TATAMOTORS, ITC, HINDUNILVR.

---

## 4. Pre-Market Routine (8:45 – 9:15 AM)

Run in this order — earlier steps inform later ones.

1. **Global cues** — SGX Nifty / GIFT Nifty + Asian markets → set directional bias.
2. **Gap-up / gap-down stocks** — gaps ≥ 0.7%, identify catalysts (results, news, sector moves).
3. **Key levels** — note previous day's High, Low, Close, VWAP, day's range, for each watchlist stock.
4. **Option chain** — top 10 stocks: significant OI buildup at nearest strikes signals likely intraday range. (Skip on quiet days.)
5. **Stock-specific event blackout** — exclude any stock with results announcement post-market or intraday today.
6. **Build a 5–8 stock focus list** — derived from the eligible universe + today's catalysts. **No trading outside this list.**

**Expiry-day adjustment:** On Nifty (Tuesday) and Bank Nifty (Wednesday) expiry, halve position size or skip the affected index components — IV is higher and moves are violent.

---

## 5. Universal Filters Applied to Every Trade

Every setup must additionally pass **all** of these gates. Filters are applied in the watchlist scan, **before** entry decisions are made.

| Gate | Rule |
|---|---|
| **Live spread** (entry-time) | Bid-ask < 0.05% of LTP. Skip if wider — slippage will eat the trade. *(Requires Angel SmartAPI; yfinance has no quote feed.)* |
| **First-30-min relative volume** | Today's volume ≥ 1.5× of 20-day avg. Confirms genuine participation. *(Requires Angel SmartAPI for sub-15-min data.)* |
| **Trigger candle volume** | Volume on the entry candle ≥ 1.5× of last 12 candles' 5-min avg. |
| **ATR(14) on 5-min** | **0.4% ≤ ATR ≤ 1.5%** of price. Below = no movement to target; above = chop / news event. |
| **VWAP slope** | Directional, not flat. Flat VWAP = chop = no trade. Slope determined by linear-regression of last 30 min of VWAP. |
| **Round-number proximity** | Stock not within 0.3% of a major round number going against your direction (psychological resistance). |
| **Time-of-day window** | **No new entries 12:00 – 1:30 PM** (lunch chop). Exits/management continue normally. |
| **Time-of-day window** | **No new entries after 2:30 PM** (square-off pressure distorts moves). |
| **News/event status** | No pending intraday catalyst (RBI policy, results expected today, sector regulator action). |

If **any** gate fails, the trade is skipped — full stop, regardless of how good the setup looks.

---

## 6. The Three Setups

Pick **one** per stock per day. Don't re-enter the same name after a stop-out. Each setup requires **≥ 3 confluences** in addition to passing all universal filters in §5.

### Setup A — Opening Range Breakout (ORB)

- **Range:** First 15-min candle (9:15 – 9:30) high/low.
- **Buy trigger:** 5-min candle **closes above** ORB high.
- **Sell/short trigger:** 5-min candle **closes below** ORB low.
- **Confluences (≥ 3 required):**
  1. Price above (long) / below (short) **VWAP**
  2. **EMA 9 > EMA 20** (long) / **EMA 9 < EMA 20** (short) on 5-min
  3. **RSI(14)** between 55–70 for longs, 30–45 for shorts (avoid extremes)
- **Skip if:** ORB range > 1.2% of stock price (already extended; lower remaining edge).

### Setup B — VWAP Pullback Continuation

- **Setup context:** After a strong morning trend, price retraces to VWAP and bounces.
- **Buy trigger:** Bullish reversal candle (hammer / engulfing) at VWAP, followed by next candle taking out the wick high.
- **Sell trigger:** Mirror image — bearish reversal at VWAP from above, followed by breakdown candle.
- **Confluences (≥ 3 required):**
  1. Price above EMA 20 on 5-min, EMA 20 above EMA 50 (long); inverse for short
  2. VWAP sloping in trade direction
  3. Recent **swing low holds** above prior structure (long) / swing high holds below (short)
- **Best window:** 10:00 – 11:30 AM and 1:30 – 2:30 PM.

### Setup C — Range Reversal at Key Level

- **Setup context:** Stock rejecting **previous day's high/low** or a clean intraday S/R zone.
- **Buy trigger:** Double-bottom or higher-low at support.
- **Sell trigger:** Double-top or lower-high at resistance.
- **Confluences (≥ 3 required):**
  1. Min 2 prior touches of the level (clean, not noisy)
  2. **RSI bullish/bearish divergence** on the second touch
  3. Volume spike on the rejection / bounce candle

---

## 7. Entry, Stop, Target & Exit Rules

### Entry

| Component | Rule |
|---|---|
| Trigger | Close of trigger candle, OR limit order at re-test of breakout level (preferred — better RR) |
| Initial SL | Below most recent **swing low** (long) / above swing high (short), **or** `1.5 × ATR(14)` on 5-min — whichever is **tighter**. "Swing low" = lowest low of the last 5 candles before entry. |

### Targets and trailing

| Target | Rule |
|---|---|
| **T1** | 1% move → exit 50% of position |
| **T2** | 2% move OR previous day H/L → exit remaining 50% |
| **Trailing SL** | After T1: move SL to entry. After 1.5%: trail using EMA 9 on 5-min |

### Exits (all triggers)

Position must exit on **any** of these, whichever comes first:

1. **T2 hit** → full exit
2. **SL hit** → full exit
3. **Trailing stop hit** → full exit
4. **Time stop** = `min(45 min from entry, 3:15 PM)` — momentum has died or square-off pressure is starting
5. **Mandatory square-off** at 3:15 PM — no exceptions
6. **Spread widens > 0.1%** of LTP → exit immediately (slippage risk too high)
7. **Circuit limit pending** — if stock approaches upper/lower circuit and SL would become unfillable, exit pre-emptively

---

## 8. Daily Trade Journal

Track every trade. After 30 trades, review: which setup has the best win rate? Cut the worst.

```
Date | Stock | Setup (A/B/C) | Entry | SL | T1 | T2 | Exit | RR | Net P&L | Notes
```

**Net P&L** must include round-trip costs:
- Brokerage + STT + exchange + GST + SEBI + stamp ≈ **0.05% per side** (tier-1 names on a discount broker)
- Slippage: 0.05% on entry, 0.03% on exit (assumed)
- **Total round-trip cost ≈ 0.13% on a 1% gross trade** → net edge ≈ 0.87% per leg

Backtest must apply these costs. Ignoring them inflates win rates by ~5–7 percentage points.

---

## 9. Common Mistakes (True Warnings Only)

The hard rules have been moved into §2, §5, and §7. What remains here is judgment-level guidance:

1. **Trading the lunch chop window** is enforced as a filter, but the deeper mistake is **forcing a trade because nothing setup-able is happening**. Sit out — boredom is not a setup.
2. **Cherry-picking confluences after entry.** If you have to "explain" the trade, you didn't have 3 confluences before entering.
3. **Treating gap-fills as automatic setups.** Gaps with weak news / no catalyst often fill, but the entry must still pass §5 + a setup in §6.
4. **Ignoring sector correlation.** §2 caps it at 1 position per sector; the deeper trap is **assuming today's strong sector will stay strong** — sector rotation is hourly during volatility.
5. **Over-trusting backtest results.** Backtests can't model spread widening during news events. Paper-trade is mandatory, not optional.

---

## 10. Suggested Toolchain

- **Charts & alerts:** TradingView (Pine Script for ORB + VWAP + EMA stack alerts).
- **Scanning:** Chartink for Indian-market screeners (volume spike, ORB breakout, RSI filters).
- **Execution:** Zerodha Kite API, Upstox API, Fyers API, or Angel SmartAPI.
- **Data feed:** Broker websocket for real-time tick data; NSE bhavcopy for EOD validation; NSE F&O list for eligibility.
- **Automation stack:** Python (`pandas`, `kiteconnect`, `ta-lib`) or Node.js (`kite-connect-js`).

A scanner skeleton lives in `scanner.py` — it screens your watchlist for setup-A/B/C confluence and prints alerts. Wire it to Telegram or your broker's order endpoint.

---

## 11. Going Live Checklist

- [ ] Backtested on 6+ months of 5-min data, costs applied (0.05% per side + slippage)
- [ ] Paper-traded for 4 weeks, win rate ≥ 50%, blended RR ≥ 1.5
- [ ] Capital allocated is money you can afford to lose
- [ ] Broker API tested with small orders
- [ ] Daily loss circuit + max-exposure cap coded into the bot
- [ ] Circuit-limit detection and pre-emptive exit tested
- [ ] Single-stock results-day exclusion implemented in pre-market routine
- [ ] Expiry-day position-size adjustment automated (or manually flagged)
- [ ] Trade journal automated, costs auto-deducted from gross P&L
- [ ] First live week traded at half size as a final sanity check
