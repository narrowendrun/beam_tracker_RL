from __future__ import annotations

import numpy as np

from beam_tracker_rl.sim import (
    ChannelConfig,
    FeedbackHistory,
    MovementConfig,
    Obstacle,
    RewardConfig,
    UEState,
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


def test_stochastic_motion_is_seed_reproducible_and_bounded() -> None:
    scenario = make_scenario_config("single_occluder")
    state = make_initial_ue_state(scenario)
    movement = MovementConfig(model="stochastic", heading_std_deg=25.0, speed_std=1.5)

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    rng_c = np.random.default_rng(456)

    path_a = []
    path_b = []
    path_c = []
    state_a = state
    state_b = state
    state_c = state
    for _ in range(12):
        state_a = advance_ue_state(state_a, scenario, movement, rng_a)
        state_b = advance_ue_state(state_b, scenario, movement, rng_b)
        state_c = advance_ue_state(state_c, scenario, movement, rng_c)
        path_a.append((state_a.x, state_a.y, state_a.vx, state_a.vy))
        path_b.append((state_b.x, state_b.y, state_b.vx, state_b.vy))
        path_c.append((state_c.x, state_c.y, state_c.vx, state_c.vy))

    np.testing.assert_allclose(path_a, path_b)
    assert not np.allclose(path_a, path_c)
    x_min, x_max = scenario.x_bounds
    y_min, y_max = scenario.y_bounds
    assert all(x_min <= x <= x_max and y_min <= y <= y_max for x, y, _, _ in path_a)


def test_stochastic_motion_reflects_at_bounds() -> None:
    scenario = make_scenario_config("los_straight")
    state = UEState(
        x=scenario.x_bounds[1] - 0.1,
        y=380.0,
        vx=6.0,
        vy=0.0,
    )
    movement = MovementConfig(
        model="stochastic",
        speed_mean=6.0,
        speed_std=0.0,
        heading_std_deg=0.0,
        velocity_damping=1.0,
    )

    next_state = advance_ue_state(
        state,
        scenario,
        movement,
        np.random.default_rng(123),
    )

    assert next_state.x <= scenario.x_bounds[1]
    assert next_state.vx < 0.0
