import numpy as np

REGISTERED = {}


class _Space:
    def sample(self):
        return np.zeros(1)


class StubEnv:
    """Each env alternates turns; the game lasts 3-9 steps and seat `winner` wins."""

    def __init__(self, num_envs, seed, deck1, deck2, **kw):
        assert deck1 in REGISTERED and deck2 in REGISTERED, "decks must be registered first"
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.winner = 0 if deck1.endswith("_w") else (1 if deck2.endswith("_w") else -1)
        self.observation_space = _Space()

    def _new(self, i):
        self.left[i] = self.rng.integers(3, 10)
        self.steps[i] = 0
        self.to_play[i] = 0

    def reset(self):
        self.left = np.zeros(self.n, int)
        self.steps = np.zeros(self.n, int)
        self.to_play = np.zeros(self.n, int)
        for i in range(self.n):
            self._new(i)
        return np.zeros(self.n), {"to_play": self.to_play.copy(), "num_options": np.full(self.n, 3)}

    def step(self, actions):
        assert len(actions) == self.n
        mover = self.to_play.copy()
        dones = np.zeros(self.n, bool)
        r = np.zeros(self.n)
        l = np.zeros(self.n, int)
        for i in range(self.n):
            self.steps[i] += 1
            self.left[i] -= 1
            if self.left[i] <= 0:
                dones[i] = True
                l[i] = self.steps[i]
                r[i] = 0 if self.winner == -1 else (1 if mover[i] == self.winner else -1)
                self._new(i)
            else:
                # sometimes the same player acts twice in a row, like chains in a real duel
                self.to_play[i] = mover[i] if self.rng.random() < 0.3 else 1 - mover[i]
        infos = {"to_play": self.to_play.copy(), "num_options": np.full(self.n, 3), "r": r, "l": l,
                 "win_reason": np.ones(self.n, int)}
        return np.zeros(self.n), r, dones, infos


def make(task_id, num_envs, seed, deck1, deck2, **kw):
    return StubEnv(num_envs, seed, deck1, deck2, **kw)
