class EnvPreprocess:
    def __init__(self, env, skip_mask=True):
        self.env = env
        self.observation_space = env.observation_space

    def __getattr__(self, k):
        return getattr(self.env, k)


RecordEpisodeStatistics = EnvPreprocess
