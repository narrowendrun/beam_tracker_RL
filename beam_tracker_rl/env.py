from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from beam_tracker_rl.sim import (
    ChannelConfig,
    FeedbackHistory,
    HIST_LEN,
    MAX_ANGLE_DEG,
    MIN_ANGLE_DEG,
    NUM_BEAMS,
    Obstacle,
    RewardConfig,
    UEState,
    advance_ue_state,
    angle_error_deg,
    beam_angle_from_action,
    build_observation,
    compute_reward,
    compute_snr,
    detect_events,
    euclidean_distance,
    info_dict,
    is_blocked,
    is_outage,
    make_codebook,
    make_initial_ue_state,
    max_range_from_bs,
    nearest_beam_index,
    static_world_dict,
    true_angle_deg,
    with_scenario_overrides,
)


class BeamTrackingEnv(gym.Env):
    """Gymnasium environment for feedback-only mmWave beam tracking."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        scenario_name: str = "single_occluder",
        max_steps: int | None = None,
        hist_len: int = HIST_LEN,
        num_beams: int = NUM_BEAMS,
        min_angle_deg: float = MIN_ANGLE_DEG,
        max_angle_deg: float = MAX_ANGLE_DEG,
        obstacles: tuple[Obstacle, ...] | None = None,
        ue_start_xy: tuple[float, float] | None = None,
        ue_velocity_xy: tuple[float, float] | None = None,
        channel_config: ChannelConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.hist_len = int(hist_len)
        self.scenario = with_scenario_overrides(
            scenario_name=scenario_name,
            max_steps=max_steps,
            obstacles=obstacles,
            ue_start_xy=ue_start_xy,
            ue_velocity_xy=ue_velocity_xy,
        )
        self.channel_config = channel_config or ChannelConfig()
        self.reward_config = reward_config or RewardConfig()
        self.codebook = make_codebook(num_beams, min_angle_deg, max_angle_deg)
        self.max_distance = max_range_from_bs(self.scenario)

        self.action_space = gym.spaces.Discrete(len(self.codebook))
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.hist_len + 1,),
            dtype=np.float32,
        )

        self.history = FeedbackHistory(self.hist_len)
        self.ue_state: UEState | None = None
        self.current_action: int | None = None
        self.current_obs: np.ndarray | None = None
        self.prev_blocked = False
        self.prev_outage = False
        self.episode_log: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "scenario_name" in options:
            self.scenario = with_scenario_overrides(str(options["scenario_name"]))
            self.max_distance = max_range_from_bs(self.scenario)

        self.ue_state = make_initial_ue_state(self.scenario)
        info, obs = self._evaluate(action=None, prev_action=None)

        self.current_action = int(info["selected_beam_idx"])
        self.current_obs = obs
        self.prev_blocked = bool(info["blocked"])
        self.prev_outage = bool(info["outage"])
        self.episode_log = []
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.ue_state is None or self.current_action is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action={action}; expected 0..{self.action_space.n - 1}"
            )

        prev_action = self.current_action
        self.ue_state = advance_ue_state(self.ue_state, self.scenario)

        info, obs = self._evaluate(action=action, prev_action=prev_action)
        reward = float(info["reward"])
        self.episode_log.append(info)

        self.current_action = action
        self.current_obs = obs
        self.prev_blocked = bool(info["blocked"])
        self.prev_outage = bool(info["outage"])

        terminated = False
        truncated = bool(self.ue_state.t >= self.scenario.max_steps)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def get_episode_log(self) -> list[dict[str, Any]]:
        return list(self.episode_log)

    def get_static_world(self) -> dict[str, object]:
        return static_world_dict(self.scenario, self.codebook)

    def render(self) -> dict[str, object] | None:
        if self.ue_state is None or self.current_action is None:
            return None

        beam_deg = beam_angle_from_action(self.current_action, self.codebook)
        state = {
            "t": self.ue_state.t,
            "ue_xy": (self.ue_state.x, self.ue_state.y),
            "bs_xy": self.scenario.bs_xy,
            "selected_beam_idx": self.current_action,
            "selected_beam_deg": beam_deg,
        }
        if self.render_mode == "human":
            print(
                f"t={state['t']:3d} "
                f"ue=({state['ue_xy'][0]:7.2f}, {state['ue_xy'][1]:7.2f}) "
                f"beam_idx={state['selected_beam_idx']:2d} "
                f"beam_deg={state['selected_beam_deg']:6.1f}"
            )
        return state

    def _evaluate(
        self,
        action: int | None,
        prev_action: int | None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        assert self.ue_state is not None

        ue_xy = (self.ue_state.x, self.ue_state.y)
        theta_true = true_angle_deg(self.scenario.bs_xy, ue_xy)
        selected_action = (
            nearest_beam_index(theta_true, self.codebook)
            if action is None
            else int(action)
        )
        beam_deg = beam_angle_from_action(selected_action, self.codebook)
        distance = euclidean_distance(self.scenario.bs_xy, ue_xy)
        blocked = is_blocked(self.scenario.bs_xy, ue_xy, self.scenario.obstacles)
        snr_parts = compute_snr(
            distance=distance,
            angle_error=angle_error_deg(theta_true, beam_deg),
            blocked=blocked,
            channel=self.channel_config,
        )
        outage = is_outage(snr_parts["snr_db"], self.reward_config)

        if prev_action is None:
            reward_value = 0.0
            reward_terms = {
                "snr_term": 0.0,
                "outage_penalty": 0.0,
                "switch_penalty": 0.0,
                "reward": 0.0,
            }
            events = {
                "beam_switched": False,
                "occlusion_started": False,
                "occlusion_ended": False,
                "outage_started": False,
                "outage_ended": False,
            }
            self.history.reset(snr_parts["snr_db"], selected_action)
        else:
            reward_value, reward_terms = compute_reward(
                snr_db=snr_parts["snr_db"],
                action=selected_action,
                prev_action=prev_action,
                reward=self.reward_config,
            )
            events = detect_events(
                prev_action=prev_action,
                action=selected_action,
                prev_blocked=self.prev_blocked,
                blocked=blocked,
                prev_outage=self.prev_outage,
                outage=outage,
            )
            self.history.append(snr_parts["snr_db"], selected_action)

        obs = build_observation(
            self.history,
            distance=distance,
            max_distance=self.max_distance,
            num_beams=len(self.codebook),
        )
        info = info_dict(
            t=self.ue_state.t,
            ue_xy=ue_xy,
            bs_xy=self.scenario.bs_xy,
            action=selected_action,
            beam_deg=beam_deg,
            theta_true=theta_true,
            distance=distance,
            blocked=blocked,
            snr_parts=snr_parts,
            reward_value=reward_value,
            events=events,
            reward_terms=reward_terms,
            reward_config=self.reward_config,
        )
        return info, obs
