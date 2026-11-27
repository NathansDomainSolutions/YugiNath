"""Small statistics helpers for win rates."""

from __future__ import annotations

import math


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate (draws may be counted as 0.5)."""
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
