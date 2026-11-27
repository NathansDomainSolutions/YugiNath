"""A stand-in backend for testing the loop's plumbing without the rules engine.

It does NOT play Yu-Gi-Oh. Each deck gets a hidden "strength" from a hash of
its cards, and each game is a seeded coin flip weighted by the two strengths
plus a small first-player edge. Use it only to check that mutation, scoring,
logging and champion updates work end to end.
"""

from __future__ import annotations

import hashlib
import math
import random

from yuginath.duel import GameResult, Matchup


def _strength(fingerprint: str, spread: float) -> float:
    h = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (h - 0.5) * 2 * spread


class FakeBackend:
    name = "fake"
    is_real = False

    def __init__(self, spread: float = 0.6, first_player_edge: float = 0.1):
        self.spread = spread
        self.edge = first_player_edge

    def play(self, matchups: list[Matchup], checkpoint: str | None = None) -> list[GameResult]:
        out = []
        for m in matchups:
            f1, f2 = m.deck1.fingerprint(), m.deck2.fingerprint()
            logit = _strength(f1, self.spread) - _strength(f2, self.spread) + self.edge
            p1 = 1 / (1 + math.exp(-logit))
            rng = random.Random(f"{m.seed}:{f1}:{f2}")
            for i in range(m.games):
                out.append(
                    GameResult(
                        matchup=m.id,
                        deck1=f1,
                        deck2=f2,
                        winner_seat=0 if rng.random() < p1 else 1,
                        length=rng.randint(40, 300),
                        win_reason=-1,
                        seed=m.seed,
                        game_index=i,
                    )
                )
        return out
