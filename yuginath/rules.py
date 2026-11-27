"""Deck-building rules every candidate must pass before it plays a duel."""

from __future__ import annotations

from dataclasses import dataclass

from yuginath.banlist import Banlist
from yuginath.deck import Deck
from yuginath.pool import CardPool


@dataclass
class DeckRules:
    main_min: int = 40
    main_max: int = 60
    extra_max: int = 15
    max_copies: int = 3

    def allowed(self, code: int, banlist: Banlist, pool: CardPool | None = None) -> int:
        n = min(self.max_copies, banlist.allowed(code))
        if pool is not None and code in pool.caps:
            n = min(n, pool.caps[code])
        return n


def problems(deck: Deck, banlist: Banlist, rules: DeckRules, pool: CardPool | None = None) -> list[str]:
    """Every rule the deck breaks, as readable lines. Empty means legal.

    The pool is only used for per-card caps here; cards outside the pool are
    allowed in a deck you supply yourself (the search just never adds them).
    """
    out = []
    if not rules.main_min <= len(deck.main) <= rules.main_max:
        out.append(f"main deck has {len(deck.main)} cards (allowed {rules.main_min}-{rules.main_max})")
    if len(deck.extra) > rules.extra_max:
        out.append(f"extra deck has {len(deck.extra)} cards (max {rules.extra_max})")
    name = (lambda c: pool.name(c)) if pool else str
    for code, n in sorted(deck.counts().items()):
        limit = rules.allowed(code, banlist, pool)
        if n > limit:
            status = {0: "Forbidden", 1: "Limited", 2: "Semi-Limited"}.get(banlist.allowed(code), "max copies")
            out.append(f"{n}x {name(code)} ({code}) but {limit} allowed ({status} on {banlist.name})")
    return out


def is_legal(deck: Deck, banlist: Banlist, rules: DeckRules, pool: CardPool | None = None) -> bool:
    return not problems(deck, banlist, rules, pool)


def repair(deck: Deck, banlist: Banlist, rules: DeckRules, pool: CardPool) -> tuple[Deck, list[str]]:
    """Make a deck legal with as few changes as possible.

    Excess copies are removed from main/extra (then side). If the main deck
    drops below the minimum, it is refilled first from the deck's own side
    deck, then from the pool in pool order. Returns the new deck and a list of
    what changed.
    """
    new = deck.copy()
    notes: list[str] = []

    def excess(code: int) -> int:
        return new.counts()[code] - rules.allowed(code, banlist, pool)

    for code in sorted(set(new.counts())):
        while excess(code) > 0:
            for section, label in ((new.main, "main"), (new.extra, "extra"), (new.side, "side")):
                if code in section:
                    section.remove(code)
                    notes.append(f"removed 1x {pool.name(code)} from {label}")
                    break

    while len(new.extra) > rules.extra_max:
        code = new.extra.pop()
        notes.append(f"removed 1x {pool.name(code)} from extra (too many cards)")
    while len(new.main) > rules.main_max:
        code = new.main.pop()
        notes.append(f"removed 1x {pool.name(code)} from main (too many cards)")

    def fill_from(candidates: list[int], source: str) -> None:
        for code in candidates:
            if len(new.main) >= rules.main_min:
                return
            if pool.section_of(code) == "extra":
                continue
            if new.counts()[code] - (1 if source == "side" else 0) + 1 > rules.allowed(code, banlist, pool):
                continue
            if source == "side":
                new.side.remove(code)
            new.main.append(code)
            notes.append(f"added 1x {pool.name(code)} to main (from {source})")

    fill_from(list(new.side), "side")
    fill_from(list(pool.main), "pool")
    if len(new.main) < rules.main_min:
        raise ValueError("could not refill the main deck from the side deck and pool")
    return new, notes
