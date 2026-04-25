from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from beam_tracker_rl.sim import (
    ChannelConfig,
    EPISODE_LOG_COLUMNS,
    FeedbackHistory,
    HIST_LEN,
    MAX_ANGLE_DEG,
    MIN_ANGLE_DEG,
    MovementConfig,
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
    episode_log_row,
    euclidean_distance,
    info_dict,
    is_blocked,
    is_outage,
    make_codebook,
    make_initial_ue_state,
    max_range_from_bs,
    nearest_beam_index,
    simulation_metadata,
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
        movement_config: MovementConfig | None = None,
        data_dir: str | Path | None = None,
        run_name: str | None = None,
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
        self.movement_config = movement_config or MovementConfig()
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
        self._episode_records: list[dict[str, Any]] = []
        self.episode_index = -1
        self.data_root = Path(data_dir) if data_dir is not None else None
        self.run_dir = self._make_run_dir(run_name) if self.data_root else None
        self._episode_dir: Path | None = None
        self._episode_saved = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.save_episode_recording()
        if options and "scenario_name" in options:
            self.scenario = with_scenario_overrides(str(options["scenario_name"]))
            self.max_distance = max_range_from_bs(self.scenario)
        if options and "movement_config" in options:
            self.movement_config = options["movement_config"]

        self.ue_state = make_initial_ue_state(self.scenario)
        info, obs = self._evaluate(action=None, prev_action=None)

        self.current_action = int(info["selected_beam_idx"])
        self.current_obs = obs
        self.prev_blocked = bool(info["blocked"])
        self.prev_outage = bool(info["outage"])
        self.episode_log = []
        self._episode_records = [info]
        self.episode_index += 1
        self._episode_saved = False
        self._start_episode_recording(info)
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
        self.ue_state = advance_ue_state(
            self.ue_state,
            self.scenario,
            self.movement_config,
            self.np_random,
        )

        info, obs = self._evaluate(action=action, prev_action=prev_action)
        reward = float(info["reward"])
        self.episode_log.append(info)
        self._episode_records.append(info)

        self.current_action = action
        self.current_obs = obs
        self.prev_blocked = bool(info["blocked"])
        self.prev_outage = bool(info["outage"])

        terminated = False
        truncated = bool(self.ue_state.t >= self.scenario.max_steps)
        if truncated:
            self.save_episode_recording()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def get_episode_log(self) -> list[dict[str, Any]]:
        return list(self.episode_log)

    def get_static_world(self) -> dict[str, object]:
        return static_world_dict(self.scenario, self.codebook)

    def save_episode_recording(self) -> Path | None:
        if (
            self.run_dir is None
            or self._episode_dir is None
            or self._episode_saved
            or not self._episode_records
        ):
            return None

        steps_path = self._episode_dir / "steps.csv"
        with steps_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(
                episode_log_row(info, self.episode_index)
                for info in self._episode_records
            )

        self._episode_saved = True
        return steps_path

    def close(self) -> None:
        self.save_episode_recording()
        super().close()

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
        optimal_action = nearest_beam_index(theta_true, self.codebook)
        optimal_beam_deg = beam_angle_from_action(optimal_action, self.codebook)
        selected_action = (
            optimal_action
            if action is None
            else int(action)
        )
        beam_deg = beam_angle_from_action(selected_action, self.codebook)
        angle_error = angle_error_deg(theta_true, beam_deg)
        distance = euclidean_distance(self.scenario.bs_xy, ue_xy)
        blocked = is_blocked(self.scenario.bs_xy, ue_xy, self.scenario.obstacles)
        snr_parts = compute_snr(
            distance=distance,
            angle_error=angle_error,
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
            ue_velocity_xy=(self.ue_state.vx, self.ue_state.vy),
            bs_xy=self.scenario.bs_xy,
            action=selected_action,
            optimal_action=optimal_action,
            beam_deg=beam_deg,
            optimal_beam_deg=optimal_beam_deg,
            theta_true=theta_true,
            angle_error=angle_error,
            distance=distance,
            blocked=blocked,
            snr_parts=snr_parts,
            reward_value=reward_value,
            events=events,
            reward_terms=reward_terms,
            reward_config=self.reward_config,
        )
        return info, obs

    def _make_run_dir(self, run_name: str | None) -> Path:
        assert self.data_root is not None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = run_name or self.scenario.name
        safe_prefix = "".join(
            char if char.isalnum() or char in ("-", "_") else "_" for char in prefix
        ).strip("_")
        run_dir = self.data_root / f"{safe_prefix}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _start_episode_recording(self, initial_info: dict[str, Any]) -> None:
        if self.run_dir is None:
            self._episode_dir = None
            return

        self._episode_dir = self.run_dir / f"episode_{self.episode_index:04d}"
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            **simulation_metadata(
                self.scenario,
                self.codebook,
                self.channel_config,
                self.reward_config,
                self.movement_config,
            ),
            "episode": self.episode_index,
            "initial_step": episode_log_row(initial_info, self.episode_index),
        }
        with (self._episode_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
