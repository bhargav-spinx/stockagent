"""
Shared engine contract (Phase 2).

Every engine returns an EngineResult instead of a decision: `values` is a
flat, JSON-safe dict that merges directly into the alerts_log feature
snapshot (features.py), `ok`/`reject_reason` carry hard gates (e.g. the
liquidity floor), and `diagnostics` carries human-readable context lines.

Engines must be TOTAL over well-formed input: unknown ⇒ a None value, never
an exception into the signal path — the same fail-open-with-a-note discipline
the scorer already uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def tier_points(value: float | None, tiers) -> int:
    """First (threshold, points) tier the value clears, scanning descending
    thresholds; 0 below the lowest tier or when the value is unknown.
    Shared by the gap/volume engines — one tier semantic everywhere."""
    if value is None:
        return 0
    for threshold, points in tiers:
        if value >= threshold:
            return points
    return 0


@dataclass
class EngineResult:
    engine: str                         # e.g. "market_regime"
    values: dict = field(default_factory=dict)   # flat, JSON-safe scalars
    ok: bool = True                     # False = hard reject (see reason)
    reject_reason: str = ""
    diagnostics: list[str] = field(default_factory=list)

    def feature_dict(self, prefix: str | None = None) -> dict:
        """values keyed for a feature snapshot: '<prefix>_<key>' → value."""
        p = prefix if prefix is not None else self.engine
        return {f"{p}_{k}": v for k, v in self.values.items()}
