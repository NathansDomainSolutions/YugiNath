"""Player training between deck-search rounds.

"ygo-agent" runs ygo-agent's scripts/cleanba.py (self-play PPO with an LSTM,
JAX) on the champion deck, continuing from the previous checkpoint, and
returns the checkpoint it saved. "none" skips training and keeps whatever
checkpoint you already have (or plays random moves if there is none).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


class NoTrainer:
    name = "none"

    def train(self, deck_path: Path, checkpoint: str | None, out_dir: Path, seed: int) -> str | None:
        return checkpoint


class YgoAgentTrainer:
    name = "ygo-agent"

    def __init__(self, s: dict):
        self.s = s
        self.root = Path(s["ygo_agent_dir"]).resolve()
        python = os.environ.get("YUGINATH_YGO_PYTHON") or s.get("python") or "python"
        self.python = str(Path(python).resolve()) if os.sep in python else python

    def command(self, deck_path: Path, checkpoint: str | None, out_dir: Path, seed: int) -> list[str]:
        s = self.s
        iters = int(s["iterations"])
        batch = int(s["local_num_envs"]) * int(s["num_steps"]) * int(s["num_actor_threads"]) * len(s["actor_device_ids"])
        cmd = [
            self.python, "-u", "cleanba.py",
            "--deck", str(deck_path.resolve()),
            "--code-list-file", str(self.root / "scripts" / "code_list.txt"),
            "--actor-device-ids", *map(str, s["actor_device_ids"]),
            "--learner-device-ids", *map(str, s["learner_device_ids"]),
            "--local-num-envs", str(s["local_num_envs"]),
            "--num-actor-threads", str(s["num_actor_threads"]),
            "--num-steps", str(s["num_steps"]),
            "--num-minibatches", str(s["num_minibatches"]),
            "--learning-rate", str(s["learning_rate"]),
            "--vloss-clip", "1.0",
            "--total-timesteps", str(iters * batch),
            # cleanba only saves when the update count is a multiple of save_interval,
            # so save (and evaluate against random play) exactly at the end.
            "--save-interval", str(iters),
            "--eval-interval", str(iters),
            "--local-eval-episodes", str(s["eval_episodes"]),
            "--seed", str(seed),
            "--tb-dir", "None",
            "--ckpt-dir", str(out_dir.resolve()),
        ]
        if checkpoint:
            cmd += ["--checkpoint", str(Path(checkpoint).resolve())]
        cmd += [str(a) for a in s.get("extra_args", [])]
        return cmd

    def train(self, deck_path: Path, checkpoint: str | None, out_dir: Path, seed: int) -> str | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        before = set(out_dir.glob("*.flax_model"))
        env = dict(os.environ)
        if self.s.get("xla_device"):
            env["JAX_PLATFORMS"] = self.s["xla_device"]
        cmd = self.command(deck_path, checkpoint, out_dir, seed)
        print(f"[train] {' '.join(cmd)}", flush=True)
        start = time.time()
        proc = subprocess.run(cmd, cwd=self.root / "scripts", env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"cleanba.py failed with exit code {proc.returncode}")
        new = sorted(set(out_dir.glob("*.flax_model")) - before, key=lambda p: p.stat().st_mtime)
        if not new:
            raise RuntimeError(f"training finished but saved no checkpoint in {out_dir}")
        print(f"[train] saved {new[-1].name} after {time.time() - start:.0f}s", flush=True)
        return str(new[-1])


def get_trainer(s: dict):
    if s["backend"] == "none":
        return NoTrainer()
    if s["backend"] == "ygo-agent":
        return YgoAgentTrainer(s)
    raise ValueError(f"unknown train backend {s['backend']!r} (use 'ygo-agent' or 'none')")
