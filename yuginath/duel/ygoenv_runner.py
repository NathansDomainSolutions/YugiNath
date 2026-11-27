"""Play deck-vs-deck duels in ygo-agent's ygoenv. Run by YgoenvBackend.

Usage: python ygoenv_runner.py job.json games.jsonl

Uses only the standard library plus ygo-agent's own dependencies (numpy,
ygoenv, ygoai and, with a checkpoint, jax/flax). Both seats are played by
the same frozen policy: the checkpoint if one is given, otherwise uniformly
random legal moves. The game loop follows ygo-agent's scripts/battle.py.
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


def build_policy(job, obs_space, seed):
    import numpy as np

    if not job.get("checkpoint"):
        rng = np.random.default_rng(seed)

        class RandomPolicy:
            def init_state(self, n):
                return None, None

            def act(self, state, obs, infos, main, dones):
                return state, rng.integers(np.maximum(infos["num_options"], 1))

        return RandomPolicy()

    import flax
    import jax
    import jax.numpy as jnp
    from ygoai.rl.jax.agent import ModelArgs, RNNAgent

    def create():
        return RNNAgent(**asdict(ModelArgs()), embedding_shape=job.get("num_embeddings"))

    agent = create()
    key = jax.random.PRNGKey(seed)
    sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    params = jax.jit(agent.init)(key, sample_obs, agent.init_rnn_state(1))
    with open(job["checkpoint"], "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())
    params = jax.device_put(params)

    @jax.jit
    def step(params, rstate1, rstate2, obs, main, done):
        next1, logits1 = create().apply(params, obs, rstate1)[:2]
        next2, logits2 = create().apply(params, obs, rstate2)[:2]
        probs = jnp.where(main[:, None], jax.nn.softmax(logits1, -1), jax.nn.softmax(logits2, -1))
        rstate1 = jax.tree.map(lambda a, b: jnp.where(main[:, None], a, b), next1, rstate1)
        rstate2 = jax.tree.map(lambda a, b: jnp.where(main[:, None], b, a), next2, rstate2)
        rstate1, rstate2 = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), (rstate1, rstate2))
        return rstate1, rstate2, probs

    class ModelPolicy:
        def init_state(self, n):
            return agent.init_rnn_state(n), agent.init_rnn_state(n)

        def act(self, state, obs, infos, main, dones):
            r1, r2, probs = step(params, state[0], state[1], obs, main, dones)
            return (r1, r2), np.array(probs).argmax(axis=1)

    return ModelPolicy()


def play_batch(job, deck1, deck2, n, seed, policy_cache, out, matchup_id, offset):
    import numpy as np
    import ygoenv
    from ygoai.rl.utils import EnvPreprocess, RecordEpisodeStatistics

    threads = min(job.get("env_threads") or n, n)
    envs = ygoenv.make(
        task_id=job["env_id"],
        env_type="gymnasium",
        num_envs=n,
        num_threads=threads,
        seed=seed,
        player=-1,
        max_options=24,
        n_history_actions=32,
        play_mode="self",
        async_reset=False,
        verbose=False,
        record=False,
        deck1=deck1,
        deck2=deck2,
        oppo_info=False,
    )
    envs.num_envs = n
    envs = EnvPreprocess(envs, skip_mask=True)
    obs_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    if "policy" not in policy_cache:
        policy_cache["policy"] = build_policy(job, obs_space, seed)
    policy = policy_cache["policy"]

    obs, infos = envs.reset()
    next_to_play = infos["to_play"]
    dones = np.zeros(n, dtype=np.bool_)
    collected = np.zeros(n, dtype=np.bool_)
    state = policy.init_state(n)
    # Seat 0 always goes first; the policy keeps separate memory per seat.
    while not collected.all():
        main = next_to_play == 0
        state, actions = policy.act(state, obs, infos, main, dones)
        to_play = next_to_play
        obs, rewards, dones, infos = envs.step(actions)
        next_to_play = infos["to_play"]
        for idx, d in enumerate(dones):
            if not d or collected[idx]:
                continue
            collected[idx] = True
            r = float(infos["r"][idx])  # reward of the player who just moved
            if r == 0:
                winner = -1
            else:
                winner = 0 if (to_play[idx] == 0) == (r > 0) else 1
            out.write(
                json.dumps(
                    {
                        "matchup": matchup_id,
                        "deck1": deck1,
                        "deck2": deck2,
                        "winner_seat": winner,
                        "length": int(infos["l"][idx]),
                        "win_reason": int(infos["win_reason"][idx]),
                        "seed": seed,
                        "game_index": offset + idx,
                    }
                )
                + "\n"
            )
    out.flush()
    if hasattr(envs, "close"):
        envs.close()


def main(job_path, out_path):
    job = json.loads(Path(job_path).read_text())
    if job.get("xla_device"):
        os.environ.setdefault("JAX_PLATFORMS", job["xla_device"])
    root = Path(job["ygo_agent_dir"])
    db = root / "assets" / "locale" / job.get("locale", "en") / "cards.cdb"
    code_list = job.get("code_list_file") or str(root / "scripts" / "code_list.txt")
    if not db.exists():
        sys.exit(f"card database not found at {db}; run `make` in ygo-agent first")
    env_id = job["env_id"]
    if env_id == "YGOPro-v1":
        from ygoenv.ygopro import init_module
    elif env_id == "YGOPro-v0":
        from ygoenv.ygopro0 import init_module
    elif env_id.startswith("EDOPro"):
        from ygoenv.edopro import init_module
    else:
        sys.exit(f"unknown env_id {env_id}")
    init_module(str(db), code_list, job["decks"])

    policy_cache = {}
    total = sum(m["games"] for m in job["matchups"])
    done = 0
    with open(out_path, "w") as out:
        for m in job["matchups"]:
            remaining, chunk = m["games"], 0
            while remaining > 0:
                n = min(remaining, job.get("max_envs") or 64)
                seed = m["seed"] + chunk * 7919
                play_batch(job, m["deck1"], m["deck2"], n, seed, policy_cache, out, m["id"], m["games"] - remaining)
                remaining -= n
                chunk += 1
                done += n
                print(f"[ygoenv] {done}/{total} games", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
