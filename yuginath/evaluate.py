"""Scoring: judge candidate decks against the champion with real duels.

1. play_match: deck A vs deck B, half the games with A going first.
2. successive_halving: every candidate plays the champion; the better half
   survives and the game budget per candidate doubles, until one is left.
3. acceptance: the survivor plays a fresh, larger match against the champion
   and replaces it only at a clear margin (default 55% over 400 games).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from yuginath.deck import Deck
from yuginath.duel import DuelBackend, GameResult, Matchup
from yuginath.stats import wilson_interval


@dataclass
class MatchScore:
    deck: str  # fingerprint of the deck being scored
    opponent: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    wins_first: int = 0
    games_first: int = 0
    wins_second: int = 0
    games_second: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.wins + 0.5 * self.draws, self.games)

    def add(self, other: "MatchScore") -> "MatchScore":
        out = MatchScore(self.deck, self.opponent)
        for f in ("games", "wins", "losses", "draws", "wins_first", "games_first", "wins_second", "games_second"):
            setattr(out, f, getattr(self, f) + getattr(other, f))
        return out

    def summary(self) -> dict:
        lo, hi = self.interval()
        d = asdict(self)
        d.update(win_rate=round(self.win_rate, 4), ci95=[round(lo, 4), round(hi, 4)])
        return d


def score(results: list[GameResult], deck: str, opponent: str) -> MatchScore:
    s = MatchScore(deck, opponent)
    for r in results:
        if {r.deck1, r.deck2} != {deck, opponent} and not (r.deck1 == r.deck2 == deck == opponent):
            continue
        first = r.deck1 == deck
        my_seat = 0 if first else 1
        s.games += 1
        if first:
            s.games_first += 1
        else:
            s.games_second += 1
        if r.winner_seat == -1:
            s.draws += 1
        elif r.winner_seat == my_seat:
            s.wins += 1
            if first:
                s.wins_first += 1
            else:
                s.wins_second += 1
        else:
            s.losses += 1
    return s


def matchups_for(deck: Deck, opponent: Deck, games: int, seed: int, tag: str) -> list[Matchup]:
    """Split `games` into deck-first and opponent-first halves."""
    first = (games + 1) // 2
    return [
        Matchup(f"{tag}:A-first", deck, opponent, first, seed),
        Matchup(f"{tag}:B-first", opponent, deck, games - first, seed + 1),
    ]


def play_match(
    backend: DuelBackend, deck: Deck, opponent: Deck, games: int, seed: int, checkpoint: str | None, tag: str = "match"
) -> tuple[MatchScore, list[GameResult]]:
    ms = [m for m in matchups_for(deck, opponent, games, seed, tag) if m.games > 0]
    results = backend.play(ms, checkpoint)
    return score(results, deck.fingerprint(), opponent.fingerprint()), results


@dataclass
class HalvingStage:
    games_per_candidate: int
    scores: dict[str, dict] = field(default_factory=dict)  # fingerprint -> cumulative summary
    kept: list[str] = field(default_factory=list)


def successive_halving(
    backend: DuelBackend,
    champion: Deck,
    candidates: list[Deck],
    start_games: int,
    seed: int,
    checkpoint: str | None,
    on_results=None,
) -> tuple[Deck, list[HalvingStage]]:
    """Return the best candidate and a record of every stage.

    Scores are cumulative across stages, so later stages refine earlier
    estimates rather than throwing them away.
    """
    alive = list(candidates)
    totals: dict[str, MatchScore] = {}
    stages: list[HalvingStage] = []
    games = start_games
    stage_no = 0
    while True:
        stage = HalvingStage(games_per_candidate=games)
        matchups: list[Matchup] = []
        for i, cand in enumerate(alive):
            matchups += [
                m
                for m in matchups_for(cand, champion, games, seed + 1000 * stage_no + 10 * i, f"halving{stage_no}:{cand.fingerprint()}")
                if m.games > 0
            ]
        results = backend.play(matchups, checkpoint)
        if on_results:
            on_results(f"halving{stage_no}", results)
        for cand in alive:
            fp = cand.fingerprint()
            s = score(results, fp, champion.fingerprint())
            totals[fp] = totals[fp].add(s) if fp in totals else s
            stage.scores[fp] = totals[fp].summary()
        alive.sort(key=lambda d: totals[d.fingerprint()].win_rate, reverse=True)
        if len(alive) == 1:
            stage.kept = [alive[0].fingerprint()]
            stages.append(stage)
            return alive[0], stages
        alive = alive[: max(1, len(alive) // 2)]
        stage.kept = [d.fingerprint() for d in alive]
        stages.append(stage)
        games *= 2
        stage_no += 1
