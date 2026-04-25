from __future__ import annotations

import numpy as np

from beam_tracker_rl.sim import (
    ChannelConfig,
    FeedbackHistory,
    Obstacle,
    RewardConfig,
    advance_ue_state,
    angle_error_deg,
    build_observation,
    compute_reward,
    compute_snr,
    euclidean_distance,
    is_blocked,
    make_codebook,
    make_initial_ue_state,
    make_scenario_config,
    nearest_beam_index,
    normalize_angle_deg,
    true_angle_deg,
)


def test_scenario_geometry_and_motion() -> None:
    scenario = make_scenario_config("single_occluder")
    state = make_initial_ue_state(scenario)

    assert state.x == 128.0
    assert state.y == 380.0
    assert advance_ue_state(state, scenario).x == 132.0
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert true_angle_deg(scenario.bs_xy, (512.0, 380.0)) == 0.0
    assert normalize_angle_deg(181.0) == -179.0
    assert is_blocked(scenario.bs_xy, (512.0, 380.0), scenario.obstacles)
    assert not is_blocked(scenario.bs_xy, (512.0, 380.0), ())


def test_codebook_channel_reward_and_observation() -> None:
    codebook = make_codebook()
    np.testing.assert_allclose(codebook, np.arange(-60.0, 61.0, 10.0, dtype=np.float32))
    assert nearest_beam_index(-47.8, codebook) == 1
    assert angle_error_deg(179.0, -179.0) == -2.0

    clear = compute_snr(100.0, 0.0, False, ChannelConfig())
    blocked = compute_snr(100.0, 0.0, True, ChannelConfig())
    assert clear["snr_db"] - blocked["snr_db"] == 22.0

    reward, terms = compute_reward(4.0, action=2, prev_action=1, reward=RewardConfig())
    assert reward == terms["reward"]
    assert terms["outage_penalty"] == 1.0
    assert terms["switch_penalty"] == 0.02

    history = FeedbackHistory(hist_len=5)
    history.reset(snr_db=10.0, action=6)
    obs = build_observation(history, distance=50.0, max_distance=100.0, num_beams=13)
    assert obs.shape == (11,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)


def test_custom_obstacle_tuple_is_supported() -> None:
    scenario = make_scenario_config(
        "los_straight",
        obstacles=(Obstacle(x=470.0, y=220.0, w=84.0, h=140.0),),
    )
    assert is_blocked(scenario.bs_xy, (512.0, 380.0), scenario.obstacles)
