"""The one-command loop: train -> mutate -> score -> keep or discard, per round."""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

from yuginath.banlist import load_banlist
from yuginath.config import load_config
from yuginath.deck import read_ydk, write_ydk
from yuginath.duel import get_backend
from yuginath.duellog import DuelLog
from yuginath.evaluate import play_match, successive_halving
from yuginath.mutate import MutationSettings, candidates, describe_change
from yuginath.pool import load_pool
from yuginath.rules import DeckRules, problems, repair
from yuginath.train import get_trainer

HISTORY_FIELDS = [
    "run_id", "round", "finished_at", "duel_backend", "real_duels", "train_backend", "checkpoint",
    "champion_before", "candidate", "change", "halving_win_rate", "accept_games", "accept_win_rate",
    "ci95_low", "ci95_high", "threshold", "accepted", "champion_after",
]


def rules_from(cfg: dict) -> DeckRules:
    r = cfg["rules"]
    return DeckRules(r["main_min"], r["main_max"], r["extra_max"], r["max_copies"])


def duel_settings(cfg: dict) -> tuple[str, dict]:
    d = dict(cfg["duel"])
    backend = d.pop("backend")
    if backend == "fake":
        return backend, {k: d[k] for k in ("spread", "first_player_edge") if k in d}
    if backend == "ygoenv":
        d["work_dir"] = str(Path(cfg["run"]["runs_dir"]) / "duel_jobs")
        d["xla_device"] = d.get("xla_device") or None
        d["python"] = str(Path(d["python"]).resolve()) if d.get("python") and "/" in d["python"] else d.get("python", "")
    return backend, d


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(config_path: str | None, rounds: int | None = None) -> list[dict]:
    cfg = load_config(config_path)
    R = cfg["run"]
    seed = int(R["seed"])
    run_id = f"{R['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    runs_dir = Path(R["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(R["results_dir"])
    (results_dir / "rounds").mkdir(parents=True, exist_ok=True)
    history_dir = Path(R["history_dir"])

    banlist = load_banlist(cfg["rules"]["banlist"], cfg["rules"]["banlist_name"])
    pool = load_pool(cfg["pool"]["file"])
    rules = rules_from(cfg)
    msettings = MutationSettings(tuple(cfg["mutation"]["k_weights"]), cfg["mutation"]["extra_prob"])

    champion_path = Path(R["champion"])
    if not champion_path.exists():
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(R["start_deck"], champion_path)
        log(f"started champion from {R['start_deck']}")
    champion = read_ydk(champion_path)
    issues = problems(champion, banlist, rules, pool)
    if issues:
        if not R["repair_champion"]:
            raise SystemExit("champion deck is not legal:\n  " + "\n  ".join(issues))
        champion, notes = repair(champion, banlist, rules, pool)
        write_ydk(champion, champion_path, f"repaired for {banlist.name}")
        log(f"champion broke {banlist.name}; repaired: " + "; ".join(notes))

    backend_name, dsettings = duel_settings(cfg)
    backend = get_backend(backend_name, dsettings)
    trainer = get_trainer(cfg["train"])
    if not backend.is_real:
        log("WARNING: duel backend 'fake' does not play Yu-Gi-Oh; results only test the plumbing")

    state_path = runs_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    checkpoint = state.get("checkpoint") or cfg["train"]["initial_checkpoint"] or None
    done_before = int(state.get("rounds_done", 0))  # so a later run uses fresh seeds
    rng = random.Random(seed * 1_000_003 + done_before)
    duel_log = DuelLog(runs_dir / "duels.sqlite")
    S = cfg["scoring"]
    summaries = []
    n_rounds = rounds if rounds is not None else int(R["rounds"])

    for rnd in range(1, n_rounds + 1):
        round_seed = seed * 100_000 + (done_before + rnd) * 1000
        log(f"round {rnd}/{n_rounds}: champion {champion.fingerprint()}")

        # 1. Train the player on the champion deck.
        checkpoint = trainer.train(champion_path, checkpoint, runs_dir / "checkpoints" / run_id, round_seed)
        state["checkpoint"] = checkpoint
        state_path.write_text(json.dumps(state, indent=2))
        log(f"player: {checkpoint or 'random moves (no checkpoint)'}")

        # 2. Mutate.
        cands = candidates(champion, int(cfg["mutation"]["candidates"]), pool, banlist, rules, rng, msettings)
        for d in [champion, *cands]:
            duel_log.add_deck(d)
        for c in cands:
            log(f"  candidate {c.fingerprint()}: {describe_change(champion, c, pool)}")

        # 3. Successive halving against the champion.
        def on_results(stage, results):
            duel_log.add_games(run_id, rnd, stage, results, backend.name, checkpoint)

        best, stages = successive_halving(
            backend, champion, cands, int(S["halving_start_games"]), round_seed, checkpoint, on_results
        )
        halving_wr = stages[-1].scores[best.fingerprint()]["win_rate"]
        log(f"  best candidate {best.fingerprint()} ({halving_wr:.1%} in halving)")

        # 4. Acceptance match: fresh games, clear margin required.
        score, results = play_match(
            backend, best, champion, int(S["accept_games"]), round_seed + 777, checkpoint, tag=f"accept:{best.fingerprint()}"
        )
        duel_log.add_games(run_id, rnd, "accept", results, backend.name, checkpoint)
        accepted = score.win_rate >= float(S["accept_threshold"])
        lo, hi = score.interval()
        log(
            f"  acceptance: {score.win_rate:.1%} over {score.games} games (95% CI {lo:.1%}-{hi:.1%}) -> "
            + ("NEW CHAMPION" if accepted else "kept old champion")
        )

        # 5. Record.
        before = champion
        if accepted:
            champion = best.copy(name="champion")
            write_ydk(champion, champion_path, f"{run_id} round {rnd}")
            write_ydk(champion, history_dir / f"{run_id}-r{rnd:02d}-{champion.fingerprint()}.ydk", f"{run_id} round {rnd}")
        summary = {
            "run_id": run_id,
            "round": rnd,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duel_backend": backend.name,
            "real_duels": backend.is_real,
            "train_backend": trainer.name,
            "checkpoint": Path(checkpoint).name if checkpoint else None,
            "banlist": banlist.name,
            "champion_before": before.fingerprint(),
            "candidates": [
                {"fingerprint": c.fingerprint(), "change": describe_change(before, c, pool)} for c in cands
            ],
            "halving": [
                {"games_per_candidate": s.games_per_candidate, "scores": s.scores, "kept": s.kept} for s in stages
            ],
            "acceptance": score.summary(),
            "threshold": float(S["accept_threshold"]),
            "accepted": accepted,
            "champion_after": champion.fingerprint(),
            "champion_after_main": champion.main,
            "champion_after_extra": champion.extra,
        }
        (results_dir / "rounds" / f"{run_id}-r{rnd:02d}.json").write_text(json.dumps(summary, indent=2))
        hist = results_dir / "history.csv"
        new_file = not hist.exists()
        with hist.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(
                {
                    **{k: summary[k] for k in HISTORY_FIELDS if k in summary},
                    "candidate": best.fingerprint(),
                    "change": describe_change(before, best, pool),
                    "halving_win_rate": halving_wr,
                    "accept_games": score.games,
                    "accept_win_rate": round(score.win_rate, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                }
            )
        summaries.append(summary)
        state["rounds_done"] = done_before + rnd
        state_path.write_text(json.dumps(state, indent=2))

    duel_log.close()
    log(f"done. champion: {champion_path} ({champion.fingerprint()}); results in {results_dir}")
    return summaries
