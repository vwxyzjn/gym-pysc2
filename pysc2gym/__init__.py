from gym.envs.registration import register

register(
    id='Pysc2gym-v0',
    entry_point='pysc2gym.envs:DefaultEnv',
)
register(
    id='Pysc2gym-DummyEnv-v0',
    entry_point='pysc2gym.envs:DummyEnv',
)