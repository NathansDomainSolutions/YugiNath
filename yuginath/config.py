"""Loop settings, read from a TOML file and merged over these defaults."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "run": {
        "name": "blue-eyes",
        "rounds": 1,
        "seed": 0,
        "champion": "decks/champion.ydk",  # current best deck, updated when a candidate wins
        "start_deck": "decks/start/BlueEyes_ygo-agent.ydk",  # copied to `champion` if that file is missing
        "repair_champion": True,  # make an illegal champion legal before round 1
        "history_dir": "decks/history",
        "results_dir": "results",
        "runs_dir": "runs",  # checkpoints, duel logs, temp files (not in git)
    },
    "rules": {
        "banlist": "banlists/lflist.conf",
        "banlist_name": "2026.10 TCG",
        "main_min": 40,
        "main_max": 60,
        "extra_max": 15,
        "max_copies": 3,
    },
    "pool": {"file": "pools/blue_eyes.txt"},
    "mutation": {"candidates": 8, "k_weights": [0.7, 0.2, 0.1], "extra_prob": 0.25},
    "scoring": {"halving_start_games": 50, "accept_games": 400, "accept_threshold": 0.55},
    "duel": {
        "backend": "ygoenv",
        "ygo_agent_dir": "external/ygo-agent",
        "python": "external/venv/bin/python",
        "env_id": "YGOPro-v1",
        "max_envs": 64,
        "xla_device": "",
    },
    "train": {
        "backend": "ygo-agent",  # "ygo-agent" runs its cleanba.py PPO trainer; "none" skips training
        "ygo_agent_dir": "external/ygo-agent",
        "python": "external/venv/bin/python",
        "initial_checkpoint": "",
        "iterations": 50,  # PPO updates per round; ygo-agent's demo beats random play after ~50
        "local_num_envs": 16,
        "num_actor_threads": 2,
        "num_steps": 128,
        "num_minibatches": 8,
        "learning_rate": 1e-4,
        "actor_device_ids": [0],
        "learner_device_ids": [0],
        "eval_episodes": 32,
        "xla_device": "",
        "extra_args": [],
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:
        return _merge(DEFAULTS, tomllib.load(f))
