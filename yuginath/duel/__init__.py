"""Duel backends: something that can play deck A against deck B many times.

Every backend takes a list of Matchups and returns one GameResult per game.
In a Matchup, deck1 sits in seat 0, which goes first; deck2 sits in seat 1.
The evaluator schedules both orders so each deck goes first half the time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from yuginath.deck import Deck


@dataclass
class Matchup:
    id: str
    deck1: Deck
    deck2: Deck
    games: int
    seed: int


@dataclass
class GameResult:
    matchup: str
    deck1: str  # deck fingerprint in seat 0 (goes first)
    deck2: str  # deck fingerprint in seat 1
    winner_seat: int  # 0, 1, or -1 for a draw
    length: int  # decisions in the game
    win_reason: int  # engine's reason code, -1 if unknown
    seed: int  # seed of the batch this game was played in
    game_index: int

    def as_dict(self) -> dict:
        return asdict(self)


class DuelBackend(Protocol):
    name: str
    is_real: bool  # False for stand-ins that do not run the rules engine

    def play(self, matchups: list[Matchup], checkpoint: str | None) -> list[GameResult]: ...


def get_backend(name: str, settings: dict) -> DuelBackend:
    if name == "ygoenv":
        from yuginath.duel.ygoenv_backend import YgoenvBackend

        return YgoenvBackend(**settings)
    if name == "fake":
        from yuginath.duel.fake import FakeBackend

        return FakeBackend(**settings)
    raise ValueError(f"unknown duel backend {name!r} (use 'ygoenv' or 'fake')")
