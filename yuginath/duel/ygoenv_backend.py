"""Real duels through ygo-agent's ygoenv (ygopro-core underneath).

ygoenv needs Python 3.10-3.12, JAX and a compiled extension, so it usually
lives in its own virtual environment. This backend therefore runs
`ygoenv_runner.py` (next to this file) as a subprocess with that
environment's Python, hands it a JSON job, and reads back one JSON line per
game. The runner imports only the standard library and ygo-agent's packages.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from yuginath.deck import write_ydk
from yuginath.duel import GameResult, Matchup

RUNNER = Path(__file__).with_name("ygoenv_runner.py")


class YgoenvBackend:
    name = "ygoenv"
    is_real = True

    def __init__(
        self,
        ygo_agent_dir: str = "external/ygo-agent",
        python: str = "",
        env_id: str = "YGOPro-v1",
        max_envs: int = 64,
        env_threads: int | None = None,
        xla_device: str | None = None,
        locale: str = "en",
        code_list_file: str | None = None,
        num_embeddings: int | None = None,
        work_dir: str = "runs/duel_jobs",
        timeout: int | None = None,
    ):
        self.ygo_agent_dir = Path(ygo_agent_dir).resolve()
        # YUGINATH_YGO_PYTHON overrides the config, then fall back to this interpreter.
        self.python = os.environ.get("YUGINATH_YGO_PYTHON") or python or sys.executable
        self.env_id = env_id
        self.max_envs = max_envs
        self.env_threads = env_threads
        self.xla_device = xla_device
        self.locale = locale
        self.code_list_file = code_list_file
        self.num_embeddings = num_embeddings
        self.work_dir = Path(work_dir)
        self.timeout = timeout

    def play(self, matchups: list[Matchup], checkpoint: str | None) -> list[GameResult]:
        job_dir = self.work_dir / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{id(matchups) % 10000}"
        deck_dir = job_dir / "decks"
        decks: dict[str, str] = {}
        rows = []
        for m in matchups:
            names = []
            for d in (m.deck1, m.deck2):
                name = f"d_{d.fingerprint()}"
                if name not in decks:
                    decks[name] = str(write_ydk(d, deck_dir / f"{name}.ydk").resolve())
                names.append(name)
            rows.append({"id": m.id, "deck1": names[0], "deck2": names[1], "games": m.games, "seed": m.seed})
        job = {
            "ygo_agent_dir": str(self.ygo_agent_dir),
            "env_id": self.env_id,
            "locale": self.locale,
            "code_list_file": self.code_list_file,
            "decks": decks,
            "matchups": rows,
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
            "num_embeddings": self.num_embeddings,
            "max_envs": self.max_envs,
            "env_threads": self.env_threads,
            "xla_device": self.xla_device,
        }
        job_path = job_dir / "job.json"
        out_path = job_dir / "games.jsonl"
        job_path.write_text(json.dumps(job, indent=2))
        cmd = [self.python, str(RUNNER), str(job_path.resolve()), str(out_path.resolve())]
        proc = subprocess.run(cmd, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"ygoenv runner failed (exit {proc.returncode}); job kept at {job_dir}")
        by_name = {f"d_{d.fingerprint()}": d.fingerprint() for m in matchups for d in (m.deck1, m.deck2)}
        results = []
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            r["deck1"], r["deck2"] = by_name[r["deck1"]], by_name[r["deck2"]]
            results.append(GameResult(**r))
        return results
