"""Command line: `python -m yuginath <command>`.

  run       the whole loop (train -> mutate -> score), as many rounds as configured
  check     list the rules a deck breaks under the configured banlist
  repair    write a legal copy of a deck
  mutate    write N legal card-swap variants of a deck
  evaluate  play one deck against others and print a win-rate table
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from yuginath.banlist import load_banlist
from yuginath.config import load_config
from yuginath.deck import read_ydk, write_ydk
from yuginath.duel import get_backend
from yuginath.evaluate import play_match
from yuginath.loop import duel_settings, rules_from, run
from yuginath.mutate import MutationSettings, candidates, describe_change
from yuginath.pool import load_pool
from yuginath.rules import problems, repair


def _common(cfg_path):
    cfg = load_config(cfg_path)
    banlist = load_banlist(cfg["rules"]["banlist"], cfg["rules"]["banlist_name"])
    return cfg, banlist, load_pool(cfg["pool"]["file"]), rules_from(cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="yuginath", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/default.toml", help="TOML settings file")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the train -> mutate -> score loop")
    r.add_argument("--rounds", type=int, help="override run.rounds")

    c = sub.add_parser("check", help="check a deck against the rules and banlist")
    c.add_argument("deck")

    rp = sub.add_parser("repair", help="write a legal copy of a deck")
    rp.add_argument("deck")
    rp.add_argument("out")

    m = sub.add_parser("mutate", help="write N card-swap variants of a deck")
    m.add_argument("deck")
    m.add_argument("--n", type=int, default=4)
    m.add_argument("--out-dir", default="runs/variants")
    m.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("evaluate", help="play DECK against each OTHER deck and print win rates")
    e.add_argument("deck")
    e.add_argument("others", nargs="+")
    e.add_argument("--games", type=int, default=200, help="games per pairing, half going first")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--checkpoint", help="player checkpoint (.flax_model); random moves if omitted")

    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a.config, a.rounds)
        return 0

    cfg, banlist, pool, rules = _common(a.config)
    if a.cmd == "check":
        issues = problems(read_ydk(a.deck), banlist, rules, pool)
        print(f"{a.deck}: " + ("legal on " + banlist.name if not issues else "NOT legal"))
        for i in issues:
            print("  " + i)
        return 1 if issues else 0
    if a.cmd == "repair":
        fixed, notes = repair(read_ydk(a.deck), banlist, rules, pool)
        write_ydk(fixed, a.out, f"repaired for {banlist.name}")
        print("\n".join(notes) or "already legal")
        return 0
    if a.cmd == "mutate":
        deck = read_ydk(a.deck)
        ms = MutationSettings(tuple(cfg["mutation"]["k_weights"]), cfg["mutation"]["extra_prob"])
        for v in candidates(deck, a.n, pool, banlist, rules, random.Random(a.seed), ms):
            path = write_ydk(v, Path(a.out_dir) / f"{v.name}.ydk", describe_change(deck, v, pool))
            print(f"{path}: {describe_change(deck, v, pool)}")
        return 0
    if a.cmd == "evaluate":
        name, settings = duel_settings(cfg)
        backend = get_backend(name, settings)
        deck = read_ydk(a.deck)
        print(f"{'opponent':40s} {'games':>6s} {'win%':>7s} {'95% CI':>15s} {'1st':>7s} {'2nd':>7s}")
        for i, other_path in enumerate(a.others):
            other = read_ydk(other_path)
            s, _ = play_match(backend, deck, other, a.games, a.seed + 10 * i, a.checkpoint, tag=f"eval{i}")
            lo, hi = s.interval()
            f1 = s.wins_first / s.games_first if s.games_first else 0
            f2 = s.wins_second / s.games_second if s.games_second else 0
            print(f"{Path(other_path).name:40s} {s.games:6d} {s.win_rate:7.1%} {lo:7.1%}-{hi:6.1%} {f1:7.1%} {f2:7.1%}")
        if not backend.is_real:
            print("(fake backend: these numbers are not real duels)")
        return 0
    return 2
