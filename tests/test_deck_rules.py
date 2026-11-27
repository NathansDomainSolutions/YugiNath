import random
import tempfile
import unittest
from pathlib import Path

from yuginath.banlist import Banlist, load_banlist, parse_lflist
from yuginath.deck import Deck, read_ydk, write_ydk
from yuginath.mutate import MutationSettings, candidates, mutate
from yuginath.pool import load_pool
from yuginath.rules import DeckRules, is_legal, problems, repair

ROOT = Path(__file__).resolve().parents[1]
RULES = DeckRules()


def tcg():
    return load_banlist(ROOT / "banlists/lflist.conf", "2026.10 TCG")


class DeckFiles(unittest.TestCase):
    def test_ydk_round_trip(self):
        deck = read_ydk(ROOT / "decks/champion.ydk")
        self.assertEqual((len(deck.main), len(deck.extra)), (40, 13))
        with tempfile.TemporaryDirectory() as d:
            again = read_ydk(write_ydk(deck, Path(d) / "x.ydk"))
        self.assertEqual((again.main, again.extra, again.side), (deck.main, deck.extra, deck.side))

    def test_fingerprint_ignores_order_and_side(self):
        a = Deck([1, 2, 3], [9], [5])
        b = Deck([3, 1, 2], [9], [])
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), Deck([1, 2, 4], [9]).fingerprint())


class Banlists(unittest.TestCase):
    def test_parse_multiple_lists(self):
        lists = parse_lflist("!A\n#forbidden\n100 0 --x\n200 1\n!B\n100 2\n")
        self.assertEqual(lists["A"].allowed(100), 0)
        self.assertEqual(lists["A"].allowed(200), 1)
        self.assertEqual(lists["A"].allowed(300), 3)
        self.assertEqual(lists["B"].allowed(100), 2)

    def test_shipped_lists(self):
        self.assertEqual(tcg().allowed(23434538), 0)  # Maxx "C" forbidden in TCG
        ocg = load_banlist(ROOT / "banlists/lflist.conf", "2026.10 OCG")
        self.assertEqual(ocg.allowed(23434538), 1)


class Rules(unittest.TestCase):
    def setUp(self):
        self.pool = load_pool(ROOT / "pools/blue_eyes.txt")

    def test_start_deck_breaks_tcg_list_and_champion_is_legal(self):
        start = read_ydk(ROOT / "decks/start/BlueEyes_ygo-agent.ydk")
        issues = problems(start, tcg(), RULES, self.pool)
        self.assertEqual(len(issues), 3, issues)
        self.assertTrue(is_legal(read_ydk(ROOT / "decks/champion.ydk"), tcg(), RULES, self.pool))

    def test_size_and_copy_rules(self):
        deck = Deck([1] * 4 + list(range(100, 136)), [])
        issues = problems(deck, Banlist(), RULES)
        self.assertTrue(any("4x" in i for i in issues))
        self.assertTrue(problems(Deck(list(range(100, 139))), Banlist(), RULES))  # 39 cards

    def test_repair_makes_legal(self):
        start = read_ydk(ROOT / "decks/start/BlueEyes_ygo-agent.ydk")
        fixed, notes = repair(start, tcg(), RULES, self.pool)
        self.assertTrue(is_legal(fixed, tcg(), RULES, self.pool), problems(fixed, tcg(), RULES, self.pool))
        self.assertTrue(notes)


class Mutations(unittest.TestCase):
    def setUp(self):
        self.pool = load_pool(ROOT / "pools/blue_eyes.txt")
        self.champ = read_ydk(ROOT / "decks/champion.ydk")

    def test_many_mutations_are_legal_and_keep_sizes(self):
        rng = random.Random(1)
        ban = tcg()
        for _ in range(500):
            m = mutate(self.champ, self.pool, ban, RULES, rng)
            self.assertTrue(is_legal(m, ban, RULES, self.pool), problems(m, ban, RULES, self.pool))
            self.assertEqual(len(m.main), len(self.champ.main))
            self.assertEqual(len(m.extra), len(self.champ.extra))
            self.assertFalse(m.same_cards(self.champ))
            added = (m.playing_counts() - self.champ.playing_counts())
            self.assertTrue(set(added) <= set(self.pool.main) | set(self.pool.extra))
            self.assertLessEqual(sum(added.values()), 3)

    def test_extra_cards_only_go_to_extra(self):
        rng = random.Random(2)
        for _ in range(300):
            m = mutate(self.champ, self.pool, tcg(), RULES, rng, MutationSettings(extra_prob=0.9))
            self.assertTrue(set(m.extra) <= set(self.pool.extra) | set(self.champ.extra))
            self.assertFalse(set(m.main) & set(self.pool.extra))

    def test_candidates_are_distinct(self):
        cands = candidates(self.champ, 8, self.pool, tcg(), RULES, random.Random(3))
        self.assertEqual(len({c.fingerprint() for c in cands}), 8)


if __name__ == "__main__":
    unittest.main()
