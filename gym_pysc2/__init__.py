from gym.envs.registration import register

register(
    id='pysc2-v0',
    entry_point='pysc2gym.envs:DefaultEnv',
)
register(
    id='pysc2-dummyEnv-v0',
    entry_point='pysc2gym.envs:DummyEnv',
)