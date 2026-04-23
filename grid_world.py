from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# =========================
# Dataclasses and constants
# =========================

@dataclass
class RectObstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass
class WorldConfig:
    width: int
    height: int
    bs_x: float
    bs_y: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    obstacles: List[RectObstacle]
    scenario_name: str


@dataclass
class UEState:
    x: float
    y: float
    vx: float
    vy: float
    t: int


WORLD_W = 1024
WORLD_H = 512
BS_X = 512.0
BS_Y = 32.0

X_MIN, X_MAX = 96.0, 928.0
Y_MIN, Y_MAX = 300.0, 460.0

UE_START_X = 128.0
UE_START_Y = 380.0
UE_VX = 4.0
UE_VY = 0.0

NUM_BEAMS = 13
MIN_ANGLE_DEG = -60.0
MAX_ANGLE_DEG = 60.0
BEAM_3DB_DEG = 10.0

SNR_REF_DB = 30.0
D0 = 100.0
PATH_LOSS_EXP = 2.2
BLOCKAGE_LOSS_DB = 22.0
OUTAGE_THRESH_DB = 5.0

HIST_LEN = 5
MAX_STEPS = 192

ALPHA = 1.0
BETA = 1.0
GAMMA = 0.02


# =========================
# World/scenario helpers
# =========================

def make_world_config(scenario_name: str) -> WorldConfig:
    if scenario_name == "los_straight":
        obstacles: List[RectObstacle] = []
    elif scenario_name == "single_occluder":
        obstacles = [RectObstacle(x=470.0, y=220.0, w=84.0, h=140.0)]
    else:
        raise ValueError(f"Unknown scenario_name={scenario_name!r}")

    return WorldConfig(
        width=WORLD_W,
        height=WORLD_H,
        bs_x=BS_X,
        bs_y=BS_Y,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        obstacles=obstacles,
        scenario_name=scenario_name,
    )


def get_bs_position(world: WorldConfig) -> Tuple[float, float]:
    return (world.bs_x, world.bs_y)


def get_coverage_bounds(world: WorldConfig) -> Tuple[float, float, float, float]:
    return (world.x_min, world.x_max, world.y_min, world.y_max)


# =========================
# Geometry helpers
# =========================

def euclidean_distance(bs_xy: Tuple[float, float], ue_xy: Tuple[float, float]) -> float:
    dx = ue_xy[0] - bs_xy[0]
    dy = ue_xy[1] - bs_xy[1]
    return float(np.hypot(dx, dy))


def normalize_angle_deg(theta: float) -> float:
    return ((theta + 180.0) % 360.0) - 180.0


def true_angle_deg(bs_xy: Tuple[float, float], ue_xy: Tuple[float, float]) -> float:
    # Angle measured from +y boresight.
    theta = np.degrees(np.arctan2(ue_xy[0] - bs_xy[0], ue_xy[1] - bs_xy[1]))
    return float(normalize_angle_deg(theta))


def _point_in_rect(p: Tuple[float, float], rect: RectObstacle) -> bool:
    x, y = p
    return (rect.x <= x <= rect.x + rect.w) and (rect.y <= y <= rect.y + rect.h)


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    # 0 -> colinear, 1 -> clockwise, 2 -> counterclockwise
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(
    p1: Tuple[float, float],
    q1: Tuple[float, float],
    p2: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if (o1 != o2) and (o3 != o4):
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


def line_intersects_rect(
    p1: Tuple[float, float], p2: Tuple[float, float], rect: RectObstacle
) -> bool:
    if _point_in_rect(p1, rect) or _point_in_rect(p2, rect):
        return True

    r1 = (rect.x, rect.y)
    r2 = (rect.x + rect.w, rect.y)
    r3 = (rect.x + rect.w, rect.y + rect.h)
    r4 = (rect.x, rect.y + rect.h)

    edges = [(r1, r2), (r2, r3), (r3, r4), (r4, r1)]
    return any(_segments_intersect(p1, p2, e1, e2) for e1, e2 in edges)


def is_blocked(
    bs_xy: Tuple[float, float], ue_xy: Tuple[float, float], obstacles: List[RectObstacle]
) -> bool:
    return any(line_intersects_rect(bs_xy, ue_xy, rect) for rect in obstacles)


# =========================
# Motion helpers
# =========================

def make_initial_ue_state(scenario_name: str) -> UEState:
    # Same motion for both v1 scenarios.
    _ = scenario_name
    return UEState(x=UE_START_X, y=UE_START_Y, vx=UE_VX, vy=UE_VY, t=0)


def advance_ue_state(state: UEState) -> UEState:
    return UEState(
        x=state.x + state.vx,
        y=state.y + state.vy,
        vx=state.vx,
        vy=state.vy,
        t=state.t + 1,
    )


def clip_state_to_corridor(
    state: UEState, bounds: Tuple[float, float, float, float]
) -> UEState:
    x_min, x_max, y_min, y_max = bounds
    return UEState(
        x=float(np.clip(state.x, x_min, x_max)),
        y=float(np.clip(state.y, y_min, y_max)),
        vx=state.vx,
        vy=state.vy,
        t=state.t,
    )


def get_ue_xy(state: UEState) -> Tuple[float, float]:
    return (state.x, state.y)


# =========================
# Beam/codebook helpers
# =========================

def make_codebook(
    num_beams: int = NUM_BEAMS,
    min_deg: float = MIN_ANGLE_DEG,
    max_deg: float = MAX_ANGLE_DEG,
) -> np.ndarray:
    return np.linspace(min_deg, max_deg, num_beams, dtype=np.float32)


def beam_angle_from_action(action: int, codebook: np.ndarray) -> float:
    return float(codebook[int(action)])


def nearest_beam_index(theta_deg: float, codebook: np.ndarray) -> int:
    idx = int(np.argmin(np.abs(codebook - theta_deg)))
    return idx


def angle_error_deg(theta_true: float, theta_beam: float) -> float:
    return float(normalize_angle_deg(theta_true - theta_beam))


# =========================
# Channel/SNR helpers
# =========================

def path_loss_db(distance: float, d0: float = D0, eta: float = PATH_LOSS_EXP) -> float:
    d = max(distance, 1e-6)
    return float(10.0 * eta * np.log10(d / d0))


def beam_gain_db(angle_err_deg: float, beam_3db_deg: float = BEAM_3DB_DEG) -> float:
    gain = -12.0 * (angle_err_deg / beam_3db_deg) ** 2
    return float(max(-30.0, gain))


def blockage_loss_db(blocked: bool, loss_db: float = BLOCKAGE_LOSS_DB) -> float:
    return float(loss_db if blocked else 0.0)


def compute_snr_components(
    distance: float,
    angle_err_deg: float,
    blocked: bool,
    snr_ref_db: float = SNR_REF_DB,
    d0: float = D0,
    eta: float = PATH_LOSS_EXP,
    beam_3db_deg: float = BEAM_3DB_DEG,
    blockage_loss: float = BLOCKAGE_LOSS_DB,
) -> Dict[str, float]:
    pl_db = path_loss_db(distance, d0=d0, eta=eta)
    bg_db = beam_gain_db(angle_err_deg, beam_3db_deg=beam_3db_deg)
    bl_db = float(blockage_loss if blocked else 0.0)
    snr_db = float(snr_ref_db - pl_db + bg_db - bl_db)
    return {
        "snr_db": snr_db,
        "path_loss_db": pl_db,
        "beam_gain_db": bg_db,
        "blockage_loss_db": bl_db,
    }


# =========================
# Observation/history helpers
# =========================

class HistoryBuffer:
    def __init__(self, hist_len: int) -> None:
        self.snr: Deque[float] = deque(maxlen=hist_len)
        self.actions: Deque[int] = deque(maxlen=hist_len)

    def clear(self) -> None:
        self.snr.clear()
        self.actions.clear()


def normalize_snr(snr_db: float, clip_min: float = -20.0, clip_max: float = 30.0) -> float:
    snr_clip = float(np.clip(snr_db, clip_min, clip_max))
    # Map [clip_min, clip_max] -> [-1, 1]
    return float(2.0 * (snr_clip - clip_min) / (clip_max - clip_min) - 1.0)


def normalize_action(action: int, num_beams: int) -> float:
    if num_beams <= 1:
        return 0.0
    return float(action / (num_beams - 1))


def normalize_range(distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        return 0.0
    return float(np.clip(distance / max_distance, 0.0, 1.0))


def initialize_history(history: HistoryBuffer, init_snr: float, init_action: int) -> None:
    history.clear()
    for _ in range(history.snr.maxlen):
        history.snr.append(init_snr)
        history.actions.append(init_action)


def update_history(history: HistoryBuffer, snr_db: float, action: int) -> None:
    history.snr.append(float(snr_db))
    history.actions.append(int(action))


def build_observation(
    history: HistoryBuffer,
    distance: float,
    max_distance: float,
    num_beams: int,
) -> np.ndarray:
    snr_hist = [normalize_snr(v) for v in history.snr]
    action_hist = [normalize_action(a, num_beams) for a in history.actions]
    range_norm = normalize_range(distance, max_distance)

    obs = np.asarray(snr_hist + action_hist + [range_norm], dtype=np.float32)
    return obs


# =========================
# Reward/event helpers
# =========================

def is_outage(snr_db: float, threshold_db: float = OUTAGE_THRESH_DB) -> bool:
    return bool(snr_db < threshold_db)


def detect_events(
    prev_action: int,
    action: int,
    prev_blocked: bool,
    blocked: bool,
    prev_outage: bool,
    outage: bool,
) -> Dict[str, bool]:
    return {
        "beam_switched": bool(action != prev_action),
        "occlusion_started": bool((not prev_blocked) and blocked),
        "occlusion_ended": bool(prev_blocked and (not blocked)),
        "outage_started": bool((not prev_outage) and outage),
        "outage_ended": bool(prev_outage and (not outage)),
    }


def compute_reward(
    snr_db: float,
    action: int,
    prev_action: int,
    outage_thresh_db: float = OUTAGE_THRESH_DB,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> Tuple[float, Dict[str, float]]:
    snr_term = alpha * normalize_snr(snr_db)
    outage_penalty = beta * float(snr_db < outage_thresh_db)
    switch_penalty = gamma * float(action != prev_action)
    reward = float(snr_term - outage_penalty - switch_penalty)
    breakdown = {
        "snr_term": float(snr_term),
        "outage_penalty": float(outage_penalty),
        "switch_penalty": float(switch_penalty),
        "reward": reward,
    }
    return reward, breakdown


# =========================
# Gym environment
# =========================

class BeamTrackingEnv(gym.Env):
    """
    Single-BS, single-UE beam-tracking environment.

    Observation (shape = 11):
        [last 5 normalized SNR values,
         last 5 normalized beam indices,
         current normalized range]

    Action:
        Discrete beam index in the codebook.

    Notes
    -----
    - The simulator knows full geometry.
    - The agent only observes recent beam/SNR feedback and coarse range.
    - Visualization/evaluation should use `info` and `get_episode_log()`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        scenario_name: str = "single_occluder",
        max_steps: int = MAX_STEPS,
        hist_len: int = HIST_LEN,
        num_beams: int = NUM_BEAMS,
        min_angle_deg: float = MIN_ANGLE_DEG,
        max_angle_deg: float = MAX_ANGLE_DEG,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.hist_len = int(hist_len)

        self.world = make_world_config(scenario_name)
        self.bs_xy = get_bs_position(self.world)
        self.bounds = get_coverage_bounds(self.world)
        self.codebook = make_codebook(
            num_beams=num_beams,
            min_deg=min_angle_deg,
            max_deg=max_angle_deg,
        )

        self.action_space = gym.spaces.Discrete(len(self.codebook))
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.hist_len + 1,),
            dtype=np.float32,
        )

        # Max distance used for normalized range in the observation.
        corners = [
            (self.world.x_min, self.world.y_min),
            (self.world.x_min, self.world.y_max),
            (self.world.x_max, self.world.y_min),
            (self.world.x_max, self.world.y_max),
        ]
        self.max_distance = max(euclidean_distance(self.bs_xy, c) for c in corners)

        # Mutable episode state
        self.ue_state: Optional[UEState] = None
        self.history = HistoryBuffer(hist_len=self.hist_len)
        self.episode_log: List[Dict[str, Any]] = []

        self.current_action: Optional[int] = None
        self.current_obs: Optional[np.ndarray] = None
        self.prev_blocked: bool = False
        self.prev_outage: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "scenario_name" in options:
            self.world = make_world_config(str(options["scenario_name"]))
            self.bs_xy = get_bs_position(self.world)
            self.bounds = get_coverage_bounds(self.world)

        self.ue_state = make_initial_ue_state(self.world.scenario_name)
        ue_xy = get_ue_xy(self.ue_state)

        theta_true = true_angle_deg(self.bs_xy, ue_xy)
        init_action = nearest_beam_index(theta_true, self.codebook)
        beam_deg = beam_angle_from_action(init_action, self.codebook)

        distance = euclidean_distance(self.bs_xy, ue_xy)
        blocked = is_blocked(self.bs_xy, ue_xy, self.world.obstacles)
        angle_err = angle_error_deg(theta_true, beam_deg)
        snr_parts = compute_snr_components(distance, angle_err, blocked)
        outage = is_outage(snr_parts["snr_db"])

        initialize_history(self.history, snr_parts["snr_db"], init_action)
        obs = build_observation(
            self.history,
            distance=distance,
            max_distance=self.max_distance,
            num_beams=len(self.codebook),
        )

        self.current_action = init_action
        self.current_obs = obs
        self.prev_blocked = blocked
        self.prev_outage = outage
        self.episode_log = []

        info = self._build_info_dict(
            t=self.ue_state.t,
            action=init_action,
            beam_deg=beam_deg,
            theta_true=theta_true,
            distance=distance,
            blocked=blocked,
            snr_parts=snr_parts,
            reward=0.0,
            events={
                "beam_switched": False,
                "occlusion_started": False,
                "occlusion_ended": False,
                "outage_started": False,
                "outage_ended": False,
            },
            reward_terms={"snr_term": 0.0, "outage_penalty": 0.0, "switch_penalty": 0.0, "reward": 0.0},
        )

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.ue_state is None or self.current_action is None or self.current_obs is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action={action}; expected 0..{self.action_space.n - 1}")

        prev_action = int(self.current_action)
        prev_blocked = bool(self.prev_blocked)
        prev_outage = bool(self.prev_outage)

        # 1) advance motion
        self.ue_state = advance_ue_state(self.ue_state)
        self.ue_state = clip_state_to_corridor(self.ue_state, self.bounds)
        ue_xy = get_ue_xy(self.ue_state)

        # 2) geometry
        distance = euclidean_distance(self.bs_xy, ue_xy)
        theta_true = true_angle_deg(self.bs_xy, ue_xy)

        # 3) action -> beam angle
        beam_deg = beam_angle_from_action(action, self.codebook)
        angle_err = angle_error_deg(theta_true, beam_deg)

        # 4) blockage
        blocked = is_blocked(self.bs_xy, ue_xy, self.world.obstacles)

        # 5) channel/SNR
        snr_parts = compute_snr_components(distance, angle_err, blocked)
        outage = is_outage(snr_parts["snr_db"])

        # 6) update history
        update_history(self.history, snr_parts["snr_db"], action)

        # 7) build observation
        obs = build_observation(
            self.history,
            distance=distance,
            max_distance=self.max_distance,
            num_beams=len(self.codebook),
        )

        # 8) reward
        reward, reward_terms = compute_reward(
            snr_db=snr_parts["snr_db"],
            action=action,
            prev_action=prev_action,
        )

        # 9) events
        events = detect_events(
            prev_action=prev_action,
            action=action,
            prev_blocked=prev_blocked,
            blocked=blocked,
            prev_outage=prev_outage,
            outage=outage,
        )

        # 10) info/logging
        info = self._build_info_dict(
            t=self.ue_state.t,
            action=action,
            beam_deg=beam_deg,
            theta_true=theta_true,
            distance=distance,
            blocked=blocked,
            snr_parts=snr_parts,
            reward=reward,
            events=events,
            reward_terms=reward_terms,
        )
        self.episode_log.append(info)

        # 11) termination
        terminated = False
        truncated = bool(self.ue_state.t >= self.max_steps)

        # 12) update internal state
        self.current_action = action
        self.current_obs = obs
        self.prev_blocked = blocked
        self.prev_outage = outage

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _build_info_dict(
        self,
        *,
        t: int,
        action: int,
        beam_deg: float,
        theta_true: float,
        distance: float,
        blocked: bool,
        snr_parts: Dict[str, float],
        reward: float,
        events: Dict[str, bool],
        reward_terms: Dict[str, float],
    ) -> Dict[str, Any]:
        assert self.ue_state is not None

        return {
            "t": int(t),
            "ue_xy": (float(self.ue_state.x), float(self.ue_state.y)),
            "bs_xy": (float(self.bs_xy[0]), float(self.bs_xy[1])),
            "true_angle_deg": float(theta_true),
            "selected_beam_idx": int(action),
            "selected_beam_deg": float(beam_deg),
            "distance": float(distance),
            "blocked": bool(blocked),
            "snr_db": float(snr_parts["snr_db"]),
            "beam_gain_db": float(snr_parts["beam_gain_db"]),
            "path_loss_db": float(snr_parts["path_loss_db"]),
            "blockage_loss_db": float(snr_parts["blockage_loss_db"]),
            "outage": bool(is_outage(snr_parts["snr_db"])),
            "reward": float(reward),
            "events": dict(events),
            "reward_terms": dict(reward_terms),
        }

    def get_episode_log(self) -> List[Dict[str, Any]]:
        return list(self.episode_log)

    def get_static_world(self) -> Dict[str, Any]:
        return {
            "width": self.world.width,
            "height": self.world.height,
            "bs_xy": (self.world.bs_x, self.world.bs_y),
            "obstacles": [vars(o).copy() for o in self.world.obstacles],
            "codebook": self.codebook.copy(),
            "scenario_name": self.world.scenario_name,
            "coverage_bounds": {
                "x_min": self.world.x_min,
                "x_max": self.world.x_max,
                "y_min": self.world.y_min,
                "y_max": self.world.y_max,
            },
        }

    def render(self) -> Optional[Dict[str, Any]]:
        if self.ue_state is None or self.current_action is None:
            return None

        beam_deg = beam_angle_from_action(self.current_action, self.codebook)
        state = {
            "t": self.ue_state.t,
            "ue_xy": (self.ue_state.x, self.ue_state.y),
            "bs_xy": self.bs_xy,
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
