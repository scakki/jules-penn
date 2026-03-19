import gymnasium as gym
from gymnasium.envs.registration import register
from src.envs.bolt_run import BoltEnvRun
from src.envs.bolt_walk import BoltEnvWalk
from src.envs.bolt_simplified import BoltEnvRunSimplified


gym.register(
    id="Bolt-v3",
    entry_point="src.envs.bolt_simplified:BoltEnvRunSimplified",
    max_episode_steps=2000,
)

gym.register(
    id="Bolt-v4",
    entry_point="src.envs.bolt_simplified_lunar:BoltEnvRunSimplifiedlunar",
    max_episode_steps=2000,
)

gym.register(
    id="Bolt-v5",
    entry_point="src.envs.bolt_simplified_lunar2:BoltEnvRunSimplifiedlunar2",
    max_episode_steps=2000,
)