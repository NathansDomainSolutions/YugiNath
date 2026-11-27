"""Forbidden & Limited lists in the lflist.conf format used by YGOPro and EDOPro.

A file can hold several lists. Each starts with a line "!<name>", followed by
lines "<code> <allowed copies> --comment" (0 = Forbidden, 1 = Limited,
2 = Semi-Limited). Cards not listed may be played at 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Banlist:
    name: str = "unlimited"
    limits: dict[int, int] = field(default_factory=dict)

    def allowed(self, code: int) -> int:
        return self.limits.get(code, 3)


def parse_lflist(text: str) -> dict[str, Banlist]:
    lists: dict[str, Banlist] = {}
    current: Banlist | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            current = Banlist(name=line[1:].strip())
            lists[current.name] = current
            continue
        if current is None:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].lstrip("-").isdigit():
            current.limits[int(parts[0])] = max(0, int(parts[1]))
    return lists


def load_banlist(path: str | Path | None, name: str | None = None) -> Banlist:
    """Load one list from a file. With no name, the first list in the file is used."""
    if not path:
        return Banlist()
    lists = parse_lflist(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not lists:
        raise ValueError(f"no lists found in {path}")
    if name is None:
        return next(iter(lists.values()))
    if name not in lists:
        raise KeyError(f"list {name!r} not in {path}; available: {', '.join(lists)}")
    return lists[name]
