from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Deque, Iterable, Mapping, Sequence

import numpy as np


Point = tuple[float, float]

ENV_ID = "BeamTracking-v0"
HIST_LEN = 5
NUM_BEAMS = 13
MIN_ANGLE_DEG = -60.0
MAX_ANGLE_DEG = 60.0


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    width: int = 1024
    height: int = 512
    bs_xy: Point = (512.0, 32.0)
    x_bounds: Point = (96.0, 928.0)
    y_bounds: Point = (300.0, 460.0)
    obstacles: tuple[Obstacle, ...] = ()
    ue_start_xy: Point = (128.0, 380.0)
    ue_velocity_xy: Point = (4.0, 0.0)
    max_steps: int = 192


@dataclass(frozen=True)
class ChannelConfig:
    snr_ref_db: float = 30.0
    d0: float = 100.0
    path_loss_exp: float = 2.2
    blockage_loss_db: float = 22.0
    beam_3db_deg: float = 10.0


@dataclass(frozen=True)
class RewardConfig:
    outage_thresh_db: float = 5.0
    snr_weight: float = 1.0
    outage_weight: float = 1.0
    switch_weight: float = 0.02


@dataclass(frozen=True)
class UEState:
    x: float
    y: float
    vx: float
    vy: float
    t: int = 0


class FeedbackHistory:
    def __init__(self, hist_len: int) -> None:
        self.snr: Deque[float] = deque(maxlen=hist_len)
        self.actions: Deque[int] = deque(maxlen=hist_len)

    def reset(self, snr_db: float, action: int) -> None:
        self.snr.clear()
        self.actions.clear()
        for _ in range(self.snr.maxlen or 0):
            self.snr.append(float(snr_db))
            self.actions.append(int(action))

    def append(self, snr_db: float, action: int) -> None:
        self.snr.append(float(snr_db))
        self.actions.append(int(action))


def make_scenario_config(
    name: str = "single_occluder", **overrides: object
) -> ScenarioConfig:
    if name == "single_occluder":
        scenario = ScenarioConfig(
            name=name,
            obstacles=(Obstacle(x=470.0, y=220.0, w=84.0, h=140.0),),
        )
    elif name == "los_straight":
        scenario = ScenarioConfig(name=name)
    else:
        raise ValueError(f"Unknown scenario_name={name!r}")

    if "obstacles" in overrides and overrides["obstacles"] is not None:
        overrides["obstacles"] = tuple(overrides["obstacles"])  # type: ignore[arg-type]
    return replace(scenario, **overrides)


def make_initial_ue_state(scenario: ScenarioConfig) -> UEState:
    return UEState(
        x=float(scenario.ue_start_xy[0]),
        y=float(scenario.ue_start_xy[1]),
        vx=float(scenario.ue_velocity_xy[0]),
        vy=float(scenario.ue_velocity_xy[1]),
        t=0,
    )


def advance_ue_state(state: UEState, scenario: ScenarioConfig) -> UEState:
    x_min, x_max = scenario.x_bounds
    y_min, y_max = scenario.y_bounds
    return UEState(
        x=float(np.clip(state.x + state.vx, x_min, x_max)),
        y=float(np.clip(state.y + state.vy, y_min, y_max)),
        vx=state.vx,
        vy=state.vy,
        t=state.t + 1,
    )


def euclidean_distance(a: Point, b: Point) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def true_angle_deg(bs_xy: Point, ue_xy: Point) -> float:
    theta = math.degrees(math.atan2(ue_xy[0] - bs_xy[0], ue_xy[1] - bs_xy[1]))
    return normalize_angle_deg(theta)


def normalize_angle_deg(theta: float) -> float:
    return float(((theta + 180.0) % 360.0) - 180.0)


def line_intersects_obstacle(p1: Point, p2: Point, obstacle: Obstacle) -> bool:
    if _point_in_obstacle(p1, obstacle) or _point_in_obstacle(p2, obstacle):
        return True

    left, right, bottom, top = _obstacle_bounds(obstacle)
    corners = ((left, bottom), (right, bottom), (right, top), (left, top))
    edges = zip(corners, corners[1:] + corners[:1])
    return any(
        _segments_intersect(p1, p2, edge_start, edge_end)
        for edge_start, edge_end in edges
    )


def is_blocked(bs_xy: Point, ue_xy: Point, obstacles: Iterable[Obstacle]) -> bool:
    return any(
        line_intersects_obstacle(bs_xy, ue_xy, obstacle) for obstacle in obstacles
    )


def make_codebook(
    num_beams: int = NUM_BEAMS,
    min_deg: float = MIN_ANGLE_DEG,
    max_deg: float = MAX_ANGLE_DEG,
) -> np.ndarray:
    return np.linspace(min_deg, max_deg, int(num_beams), dtype=np.float32)


def beam_angle_from_action(
    action: int, codebook: Sequence[float] | np.ndarray
) -> float:
    return float(codebook[int(action)])


def nearest_beam_index(theta_deg: float, codebook: Sequence[float] | np.ndarray) -> int:
    codebook_arr = np.asarray(codebook, dtype=np.float32)
    return int(np.argmin(np.abs(codebook_arr - theta_deg)))


def angle_error_deg(true_angle: float, beam_angle: float) -> float:
    return normalize_angle_deg(true_angle - beam_angle)


def compute_snr(
    distance: float,
    angle_error: float,
    blocked: bool,
    channel: ChannelConfig = ChannelConfig(),
) -> dict[str, float]:
    d = max(float(distance), 1e-6)
    path_loss = 10.0 * channel.path_loss_exp * math.log10(d / channel.d0)
    beam_gain = max(-30.0, -12.0 * (angle_error / channel.beam_3db_deg) ** 2)
    blockage_loss = channel.blockage_loss_db if blocked else 0.0
    snr = channel.snr_ref_db - path_loss + beam_gain - blockage_loss
    return {
        "snr_db": float(snr),
        "path_loss_db": float(path_loss),
        "beam_gain_db": float(beam_gain),
        "blockage_loss_db": float(blockage_loss),
    }


def normalize_snr(
    snr_db: float, clip_min: float = -20.0, clip_max: float = 30.0
) -> float:
    snr_clip = float(np.clip(snr_db, clip_min, clip_max))
    return float(2.0 * (snr_clip - clip_min) / (clip_max - clip_min) - 1.0)


def normalize_action(action: int, num_beams: int) -> float:
    return 0.0 if num_beams <= 1 else float(action / (num_beams - 1))


def normalize_range(distance: float, max_distance: float) -> float:
    return (
        0.0 if max_distance <= 0 else float(np.clip(distance / max_distance, 0.0, 1.0))
    )


def build_observation(
    history: FeedbackHistory,
    distance: float,
    max_distance: float,
    num_beams: int,
) -> np.ndarray:
    snr_hist = [normalize_snr(value) for value in history.snr]
    action_hist = [normalize_action(action, num_beams) for action in history.actions]
    return np.asarray(
        snr_hist + action_hist + [normalize_range(distance, max_distance)],
        dtype=np.float32,
    )


def is_outage(snr_db: float, reward: RewardConfig = RewardConfig()) -> bool:
    return bool(snr_db < reward.outage_thresh_db)


def compute_reward(
    snr_db: float,
    action: int,
    prev_action: int,
    reward: RewardConfig = RewardConfig(),
) -> tuple[float, dict[str, float]]:
    snr_term = reward.snr_weight * normalize_snr(snr_db)
    outage_penalty = reward.outage_weight * float(snr_db < reward.outage_thresh_db)
    switch_penalty = reward.switch_weight * float(action != prev_action)
    value = float(snr_term - outage_penalty - switch_penalty)
    return value, {
        "snr_term": float(snr_term),
        "outage_penalty": float(outage_penalty),
        "switch_penalty": float(switch_penalty),
        "reward": value,
    }


def detect_events(
    prev_action: int,
    action: int,
    prev_blocked: bool,
    blocked: bool,
    prev_outage: bool,
    outage: bool,
) -> dict[str, bool]:
    return {
        "beam_switched": bool(action != prev_action),
        "occlusion_started": bool((not prev_blocked) and blocked),
        "occlusion_ended": bool(prev_blocked and (not blocked)),
        "outage_started": bool((not prev_outage) and outage),
        "outage_ended": bool(prev_outage and (not outage)),
    }


def info_dict(
    *,
    t: int,
    ue_xy: Point,
    bs_xy: Point,
    action: int,
    beam_deg: float,
    theta_true: float,
    distance: float,
    blocked: bool,
    snr_parts: Mapping[str, float],
    reward_value: float,
    events: Mapping[str, bool],
    reward_terms: Mapping[str, float],
    reward_config: RewardConfig,
) -> dict[str, object]:
    return {
        "t": int(t),
        "ue_xy": (float(ue_xy[0]), float(ue_xy[1])),
        "bs_xy": (float(bs_xy[0]), float(bs_xy[1])),
        "true_angle_deg": float(theta_true),
        "selected_beam_idx": int(action),
        "selected_beam_deg": float(beam_deg),
        "distance": float(distance),
        "blocked": bool(blocked),
        "snr_db": float(snr_parts["snr_db"]),
        "beam_gain_db": float(snr_parts["beam_gain_db"]),
        "path_loss_db": float(snr_parts["path_loss_db"]),
        "blockage_loss_db": float(snr_parts["blockage_loss_db"]),
        "outage": is_outage(snr_parts["snr_db"], reward_config),
        "reward": float(reward_value),
        "events": {key: bool(value) for key, value in events.items()},
        "reward_terms": {key: float(value) for key, value in reward_terms.items()},
    }


def static_world_dict(
    scenario: ScenarioConfig, codebook: Sequence[float] | np.ndarray
) -> dict[str, object]:
    return {
        "width": int(scenario.width),
        "height": int(scenario.height),
        "bs_xy": tuple(scenario.bs_xy),
        "obstacles": [obstacle.__dict__.copy() for obstacle in scenario.obstacles],
        "codebook": np.asarray(codebook, dtype=np.float32).copy(),
        "scenario_name": scenario.name,
        "coverage_bounds": {
            "x_min": float(scenario.x_bounds[0]),
            "x_max": float(scenario.x_bounds[1]),
            "y_min": float(scenario.y_bounds[0]),
            "y_max": float(scenario.y_bounds[1]),
        },
    }


def max_range_from_bs(scenario: ScenarioConfig) -> float:
    x_min, x_max = scenario.x_bounds
    y_min, y_max = scenario.y_bounds
    corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    return max(euclidean_distance(scenario.bs_xy, corner) for corner in corners)


def with_scenario_overrides(
    scenario_name: str,
    max_steps: int | None = None,
    obstacles: tuple[Obstacle, ...] | None = None,
    ue_start_xy: Point | None = None,
    ue_velocity_xy: Point | None = None,
) -> ScenarioConfig:
    overrides: dict[str, object] = {}
    if max_steps is not None:
        overrides["max_steps"] = int(max_steps)
    if obstacles is not None:
        overrides["obstacles"] = obstacles
    if ue_start_xy is not None:
        overrides["ue_start_xy"] = ue_start_xy
    if ue_velocity_xy is not None:
        overrides["ue_velocity_xy"] = ue_velocity_xy
    return make_scenario_config(scenario_name, **overrides)


def _obstacle_bounds(obstacle: Obstacle) -> tuple[float, float, float, float]:
    x2 = obstacle.x + obstacle.w
    y2 = obstacle.y + obstacle.h
    return (
        min(obstacle.x, x2),
        max(obstacle.x, x2),
        min(obstacle.y, y2),
        max(obstacle.y, y2),
    )


def _point_in_obstacle(point: Point, obstacle: Obstacle) -> bool:
    x, y = point
    left, right, bottom, top = _obstacle_bounds(obstacle)
    return left <= x <= right and bottom <= y <= top


def _orientation(a: Point, b: Point, c: Point) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    eps = 1e-12
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def _segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False
