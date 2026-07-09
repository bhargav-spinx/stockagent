"""
Independent analysis engines (Phase 2 architecture).

Each module contributes structured statistical evidence (EngineResult),
never a BUY/SELL decision:

    base.py            EngineResult contract
    gap.py             opening-gap measurement + points   (extracted from scorer)
    volume.py          RVOL / volume-breakout measurement (extracted from scorer)
    vwap.py            VWAP confirmation                  (extracted from scorer)
    orb.py             opening-range breakout             (extracted from scorer)
    market_regime.py   day-type / volatility / breadth / expiry labels
    liquidity.py       turnover, spread-proxy, slippage floors
    risk.py            position sizing + daily risk budget
    execution.py       statutory Indian cost model
    context_daily.py   continuous indicator features (votes → numbers)

The legacy 100-point scorer (intraday_score.py) now composes gap/volume/
vwap/orb — golden-parity-tested to produce byte-identical output.
"""
from engines.base import EngineResult

__all__ = ["EngineResult"]
