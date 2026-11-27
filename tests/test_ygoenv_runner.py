"""Run the real ygoenv backend + runner against stub ygoenv/ygoai packages."""

import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy  # noqa: F401

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy needed for the runner")
class RunnerWithStubs(unittest.TestCase):
    def test_runner_seats_and_winners(self):
        from yuginath.deck import Deck
        from yuginath.duel.ygoenv_backend import YgoenvBackend
        from yuginath.evaluate import play_match

        with tempfile.TemporaryDirectory() as d:
            fake_root = Path(d) / "ygo-agent"
            (fake_root / "assets/locale/en").mkdir(parents=True)
            (fake_root / "assets/locale/en/cards.cdb").write_bytes(b"")
            (fake_root / "scripts").mkdir()
            (fake_root / "scripts/code_list.txt").write_text("")
            old = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = str(ROOT / "tests/stubs")
            try:
                backend = YgoenvBackend(ygo_agent_dir=str(fake_root), max_envs=4, work_dir=str(Path(d) / "jobs"))
                strong, weak = Deck(list(range(40))), Deck(list(range(1, 41)))
                # The stub makes the deck registered under a name ending in "_w" win.
                # Patch fingerprints so `strong` gets that suffix.
                strong.fingerprint = lambda: "strong_w"
                weak.fingerprint = lambda: "weak"
                s, results = play_match(backend, strong, weak, 10, 3, None)
            finally:
                if old is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old
        self.assertEqual(s.games, 10)
        self.assertEqual((s.games_first, s.games_second), (5, 5))
        self.assertEqual(s.wins, 10, s.summary())  # wins from both seats
        self.assertEqual(sorted(r.game_index for r in results if r.matchup.endswith("A-first")), list(range(5)))


if __name__ == "__main__":
    unittest.main()
