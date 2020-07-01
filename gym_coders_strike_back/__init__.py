from gym.envs.registration import register

register(
    id='CodersStrikeBack-v0',
    entry_point='gym_coders_strike_back.envs:CodersStrikeBackSingle',
    max_episode_steps=10000,
)

register(
    id='CodersStrikeBackFull-v0',
    entry_point='gym_coders_strike_back.envs:CodersStrikeBackSingleFull',
    max_episode_steps=10000,
)