import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from yuginath.deck import Deck
from yuginath.duel import GameResult
from yuginath.duel.fake import FakeBackend
from yuginath.evaluate import matchups_for, play_match, score, successive_halving
from yuginath.loop import run
from yuginath.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]


class Scoring(unittest.TestCase):
    def test_seats_are_balanced(self):
        a, b = Deck([1] * 40), Deck([2] * 40)
        ms = matchups_for(a, b, 9, 0, "t")
        self.assertEqual([m.games for m in ms], [5, 4])
        self.assertIs(ms[0].deck1, a)
        self.assertIs(ms[1].deck1, b)

    def test_score_counts_wins_by_seat(self):
        a, b = "A", "B"
        res = [
            GameResult("m", a, b, 0, 10, 1, 0, 0),  # A first, A wins
            GameResult("m", a, b, 1, 10, 1, 0, 1),  # A first, B wins
            GameResult("m", b, a, 1, 10, 1, 0, 2),  # A second, A wins
            GameResult("m", b, a, -1, 10, 1, 0, 3),  # draw
        ]
        s = score(res, a, b)
        self.assertEqual((s.games, s.wins, s.losses, s.draws), (4, 2, 1, 1))
        self.assertEqual((s.wins_first, s.games_first, s.wins_second, s.games_second), (1, 2, 1, 2))
        self.assertAlmostEqual(s.win_rate, 0.625)

    def test_wilson(self):
        lo, hi = wilson_interval(220, 400)
        self.assertLess(lo, 0.55)
        self.assertGreater(hi, 0.55)
        self.assertAlmostEqual((lo + hi) / 2, 0.55, delta=0.01)

    def test_halving_budget(self):
        champ = Deck(list(range(40)))
        cands = [Deck(list(range(i, 40 + i))) for i in range(1, 9)]
        best, stages = successive_halving(FakeBackend(), champ, cands, 10, 0, None)
        self.assertEqual([s.games_per_candidate for s in stages], [10, 20, 40, 80])
        self.assertEqual([len(s.kept) for s in stages], [4, 2, 1, 1])
        self.assertIn(best, cands)

    def test_play_match_is_deterministic(self):
        a, b = Deck(list(range(40))), Deck(list(range(1, 41)))
        s1, _ = play_match(FakeBackend(), a, b, 50, 7, None)
        s2, _ = play_match(FakeBackend(), a, b, 50, 7, None)
        self.assertEqual(s1.summary(), s2.summary())


class WholeLoop(unittest.TestCase):
    def test_fake_loop_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "c.toml"
            cfg.write_text(
                f"""
[run]
name = "t"
rounds = 3
champion = "{d}/champion.ydk"
start_deck = "{ROOT}/decks/champion.ydk"
history_dir = "{d}/history"
results_dir = "{d}/results"
runs_dir = "{d}/runs"
[rules]
banlist = "{ROOT}/banlists/lflist.conf"
[pool]
file = "{ROOT}/pools/blue_eyes.txt"
[mutation]
candidates = 4
[scoring]
halving_start_games = 10
accept_games = 60
[duel]
backend = "fake"
[train]
backend = "none"
"""
            )
            summaries = run(str(cfg))
            self.assertEqual(len(summaries), 3)
            rows = list(csv.DictReader(open(Path(d) / "results/history.csv")))
            self.assertEqual(len(rows), 3)
            accepted = [s for s in summaries if s["accepted"]]
            self.assertEqual(len(list((Path(d) / "history").glob("*.ydk"))), len(accepted))
            for s in summaries:
                self.assertEqual(s["acceptance"]["games"], 60)
                self.assertFalse(s["real_duels"])
            state = json.loads((Path(d) / "runs/state.json").read_text())
            self.assertEqual(state["rounds_done"], 3)
            import sqlite3

            n = sqlite3.connect(Path(d) / "runs/duels.sqlite").execute("select count(*) from games").fetchone()[0]
            # per round: 4x10 + 2x20 + 1x40 halving games + 60 acceptance games
            self.assertEqual(n, 3 * (40 + 40 + 40 + 60))


if __name__ == "__main__":
    unittest.main()
