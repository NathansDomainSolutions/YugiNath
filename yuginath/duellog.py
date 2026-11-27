"""Every duel goes into a SQLite file under runs/ (kept out of git)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from yuginath.deck import Deck, write_ydk
from yuginath.duel import GameResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    run_id TEXT, round INTEGER, stage TEXT, matchup TEXT,
    deck1 TEXT, deck2 TEXT, winner_seat INTEGER, length INTEGER,
    win_reason INTEGER, seed INTEGER, game_index INTEGER,
    backend TEXT, checkpoint TEXT, logged_at REAL
);
CREATE TABLE IF NOT EXISTS decks (
    fingerprint TEXT PRIMARY KEY, main TEXT, extra TEXT, side TEXT
);
"""


class DuelLog:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def add_deck(self, deck: Deck) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO decks VALUES (?,?,?,?)",
            (deck.fingerprint(), ",".join(map(str, deck.main)), ",".join(map(str, deck.extra)), ",".join(map(str, deck.side))),
        )
        self.db.commit()

    def add_games(self, run_id: str, round_no: int, stage: str, results: list[GameResult], backend: str, checkpoint: str | None):
        now = time.time()
        self.db.executemany(
            "INSERT INTO games (run_id, round, stage, matchup, deck1, deck2, winner_seat, length, win_reason, seed,"
            " game_index, backend, checkpoint, logged_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (run_id, round_no, stage, r.matchup, r.deck1, r.deck2, r.winner_seat, r.length, r.win_reason, r.seed,
                 r.game_index, backend, checkpoint, now)
                for r in results
            ],
        )
        self.db.commit()

    def close(self):
        self.db.close()
