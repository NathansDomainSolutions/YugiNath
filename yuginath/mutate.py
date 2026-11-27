"""Card swaps: the mutation step of the deck search.

A mutation removes k cards (k is usually 1, sometimes 2-3) and adds k cards
from the pool into the same section, so main and extra keep their size. A
swap may add another copy of a card already in the deck, which also covers
"change copy counts" (for example 2 -> 3 of a starter).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from yuginath.banlist import Banlist
from yuginath.deck import Deck, diff
from yuginath.pool import CardPool
from yuginath.rules import DeckRules, is_legal


@dataclass
class MutationSettings:
    # Probability of swapping 1, 2 or 3 cards.
    k_weights: tuple[float, float, float] = (0.7, 0.2, 0.1)
    # Probability a single swap happens in the extra deck (if it has cards).
    extra_prob: float = 0.25
    max_tries: int = 200


def _swap_once(deck: Deck, section: str, pool: CardPool, banlist: Banlist, rules: DeckRules, rng: random.Random) -> bool:
    cards = deck.main if section == "main" else deck.extra
    choices = pool.main if section == "main" else pool.extra
    if not cards or not choices:
        return False
    out_idx = rng.randrange(len(cards))
    out_code = cards[out_idx]
    counts = deck.counts()
    counts[out_code] -= 1
    legal_in = [c for c in choices if c != out_code and counts[c] + 1 <= rules.allowed(c, banlist, pool)]
    if not legal_in:
        return False
    cards[out_idx] = rng.choice(legal_in)
    return True


def mutate(
    deck: Deck,
    pool: CardPool,
    banlist: Banlist,
    rules: DeckRules,
    rng: random.Random,
    settings: MutationSettings | None = None,
) -> Deck:
    """Return a legal variant of `deck` that differs from it, or raise ValueError."""
    settings = settings or MutationSettings()
    for _ in range(settings.max_tries):
        k = rng.choices((1, 2, 3), weights=settings.k_weights)[0]
        new = deck.copy()
        for _ in range(k):
            section = "extra" if new.extra and pool.extra and rng.random() < settings.extra_prob else "main"
            _swap_once(new, section, pool, banlist, rules, rng)
        if not new.same_cards(deck) and is_legal(new, banlist, rules, pool):
            new.name = f"cand-{new.fingerprint()}"
            return new
    raise ValueError("could not find a legal mutation; is the pool too small or the deck illegal?")


def candidates(
    deck: Deck,
    n: int,
    pool: CardPool,
    banlist: Banlist,
    rules: DeckRules,
    rng: random.Random,
    settings: MutationSettings | None = None,
) -> list[Deck]:
    """n distinct legal variants of `deck`."""
    seen = {deck.fingerprint()}
    out: list[Deck] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cand = mutate(deck, pool, banlist, rules, rng, settings)
        if cand.fingerprint() not in seen:
            seen.add(cand.fingerprint())
            out.append(cand)
    if len(out) < n:
        raise ValueError(f"only found {len(out)} distinct candidates out of {n}")
    return out


def describe_change(old: Deck, new: Deck, pool: CardPool) -> str:
    d = diff(old, new)
    return f"-[{pool.describe(d['removed'])}] +[{pool.describe(d['added'])}]"
