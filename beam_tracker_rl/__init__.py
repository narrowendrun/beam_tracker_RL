from __future__ import annotations

from gymnasium.envs.registration import register, registry

from beam_tracker_rl.env import BeamTrackingEnv
from beam_tracker_rl.sim import (
    ChannelConfig,
    ENV_ID,
    MovementConfig,
    Obstacle,
    RewardConfig,
    ScenarioConfig,
)


def register_envs() -> None:
    if ENV_ID in registry:
        return
    register(
        id=ENV_ID,
        entry_point="beam_tracker_rl.env:BeamTrackingEnv",
        kwargs={"scenario_name": "single_occluder"},
    )


register_envs()

__all__ = [
    "BeamTrackingEnv",
    "ChannelConfig",
    "ENV_ID",
    "MovementConfig",
    "Obstacle",
    "RewardConfig",
    "ScenarioConfig",
    "register_envs",
]
