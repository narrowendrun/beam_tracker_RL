from __future__ import annotations

import csv
import json

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import beam_tracker_rl
from beam_tracker_rl import BeamTrackingEnv, ENV_ID


def test_gym_registration_and_checker() -> None:
    env = gym.make(ENV_ID, scenario_name="single_occluder", max_steps=4)
    try:
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert env.action_space.n == 13
        assert info["selected_beam_idx"] == 1
    finally:
        env.close()

    check_env(
        beam_tracker_rl.BeamTrackingEnv(scenario_name="single_occluder", max_steps=4),
        skip_render_check=True,
    )


def test_reset_step_log_and_observation_contract() -> None:
    env = BeamTrackingEnv(scenario_name="los_straight", max_steps=4)
    obs, info = env.reset()

    assert obs.shape == (11,)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)
    assert info["ue_xy"] == (128.0, 380.0)

    truncated = False
    steps = 0
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(env.current_action)
        assert isinstance(reward, float)
        assert not terminated
        assert env.observation_space.contains(obs)
        steps += 1

    assert steps == 4
    assert len(env.get_episode_log()) == 4


def test_occlusion_and_beam_switch_events() -> None:
    env = BeamTrackingEnv(scenario_name="single_occluder")
    env.reset()

    switched_action = (env.current_action + 1) % env.action_space.n
    _, _, _, _, info = env.step(switched_action)
    assert info["events"]["beam_switched"]

    saw_start = False
    saw_end = False
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(env.current_action)
        saw_start = saw_start or info["events"]["occlusion_started"]
        saw_end = saw_end or info["events"]["occlusion_ended"]

    assert saw_start
    assert saw_end


def test_static_world_export() -> None:
    env = BeamTrackingEnv(scenario_name="single_occluder")
    static_world = env.get_static_world()

    assert static_world["width"] == 1024
    assert static_world["height"] == 512
    assert static_world["bs_xy"] == (512.0, 32.0)
    assert static_world["scenario_name"] == "single_occluder"
    assert static_world["obstacles"] == [
        {"x": 470.0, "y": 220.0, "w": 84.0, "h": 140.0}
    ]


def test_episode_recording_writes_visualization_data(tmp_path) -> None:
    env = BeamTrackingEnv(
        scenario_name="los_straight",
        max_steps=3,
        data_dir=tmp_path,
        run_name="pytest_recording",
    )
    try:
        env.reset()
        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step(env.current_action)
    finally:
        env.close()

    (run_dir,) = tmp_path.iterdir()
    episode_dir = run_dir / "episode_0000"
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    rows = list(csv.DictReader((episode_dir / "steps.csv").open()))

    assert metadata["world"]["scenario_name"] == "los_straight"
    assert metadata["movement"]["model"] == "constant_velocity"
    assert len(rows) == 4
    assert rows[0]["t"] == "0"
    assert rows[-1]["t"] == "3"
    assert "ue_x" in rows[0]
    assert "ue_vx" in rows[0]
    assert "snr_db" in rows[0]
    assert "selected_beam_idx" in rows[0]
    assert "optimal_beam_idx" in rows[0]
