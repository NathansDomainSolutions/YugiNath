"""The card pool: which cards the deck search may add, and to which section.

Pool file format, one card per line:

    <code>  <main|extra>  [max copies]  # name and notes

Only cards the duel engine handles reliably belong here. The shipped pool is
built from the decks ygo-agent marks as tested (see pools/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CardPool:
    main: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    caps: dict[int, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)

    def section_of(self, code: int) -> str | None:
        if code in self.main:
            return "main"
        if code in self.extra:
            return "extra"
        return None

    def name(self, code: int) -> str:
        return self.names.get(code, str(code))

    def describe(self, counts: dict[int, int]) -> str:
        return ", ".join(f"{n}x {self.name(c)}" for c, n in sorted(counts.items(), key=lambda kv: self.name(kv[0])))


def load_pool(path: str | Path) -> CardPool:
    pool = CardPool()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, comment = raw.partition("#")
        parts = body.split()
        if not parts:
            continue
        code = int(parts[0])
        section = parts[1].lower() if len(parts) > 1 else "main"
        if section not in ("main", "extra"):
            raise ValueError(f"{path}: bad section {section!r} for {code}")
        (pool.main if section == "main" else pool.extra).append(code)
        if len(parts) > 2:
            pool.caps[code] = int(parts[2])
        if comment.strip():
            pool.names[code] = comment.split(";")[0].strip()
    return pool
