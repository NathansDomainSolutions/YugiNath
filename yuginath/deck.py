"""Read and write .ydk decklists (the YGOPro / EDOPro / ygo-agent format)."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Deck:
    """A decklist. Card codes are the numeric passcodes used by ygopro-core."""

    main: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    side: list[int] = field(default_factory=list)
    name: str = "deck"

    def copy(self, name: str | None = None) -> "Deck":
        return Deck(list(self.main), list(self.extra), list(self.side), name or self.name)

    def counts(self) -> Counter:
        """Copies of each card across main, extra and side (the 3-copy rule spans all three)."""
        return Counter(self.main + self.extra + self.side)

    def playing_counts(self) -> Counter:
        """Copies of each card in main + extra (what is actually dueled with)."""
        return Counter(self.main + self.extra)

    def canonical(self) -> "Deck":
        return Deck(sorted(self.main), sorted(self.extra), sorted(self.side), self.name)

    def fingerprint(self) -> str:
        """Short hash of the sorted main + extra deck, ignoring order and the side deck."""
        c = self.canonical()
        payload = "main:" + ",".join(map(str, c.main)) + "|extra:" + ",".join(map(str, c.extra))
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def same_cards(self, other: "Deck") -> bool:
        return self.fingerprint() == other.fingerprint()


def read_ydk(path: str | Path) -> Deck:
    path = Path(path)
    deck = Deck(name=path.stem)
    section = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("#main"):
            section = deck.main
        elif low.startswith("#extra"):
            section = deck.extra
        elif low.startswith("!side"):
            section = deck.side
        elif line.startswith("#") or line.startswith("!"):
            continue  # comments such as "#created by ..."
        elif section is not None:
            token = line.split()[0]
            if token.isdigit():
                section.append(int(token))
    return deck


def write_ydk(deck: Deck, path: str | Path, comment: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#created by yuginath{(' - ' + comment) if comment else ''}", "#main"]
    lines += [str(c) for c in deck.main]
    lines.append("#extra")
    lines += [str(c) for c in deck.extra]
    lines.append("!side")
    lines += [str(c) for c in deck.side]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def diff(old: Deck, new: Deck) -> dict[str, dict[int, int]]:
    """Cards removed and added (main + extra), as {code: copies}."""
    a, b = old.playing_counts(), new.playing_counts()
    return {"removed": dict(a - b), "added": dict(b - a)}
