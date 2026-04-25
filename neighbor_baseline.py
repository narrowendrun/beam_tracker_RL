#!/usr/bin/env python3
"""
Neighbor-beam tracking baseline for beam_tracker_rl.

Run from the project root:

    python neighbor_baseline.py --scenario single_occluder --num-beams 13 --radius 1

What it does:
    At each time step, probe the current beam and its local neighbors,
    select the beam with the highest instantaneous SNR, log the result,
    and advance the UE.

This is a simple non-RL baseline against PPO.
The CLI defaults to stochastic UE movement; pass
`--movement-model constant_velocity` to recover the original deterministic path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from beam_tracker_rl.env import BeamTrackingEnv
from beam_tracker_rl.sim import (
    MovementConfig,
    advance_ue_state,
    angle_error_deg,
    beam_angle_from_action,
    compute_snr,
    euclidean_distance,
    is_blocked,
    nearest_beam_index,
    true_angle_deg,
)


def local_candidate_indices(center_idx: int, radius: int, num_beams: int) -> list[int]:
    """Return clipped local beam candidates around the current beam."""
    lo = max(0, int(center_idx) - int(radius))
    hi = min(int(num_beams) - 1, int(center_idx) + int(radius))
    return list(range(lo, hi + 1))


def probe_snr_at_current_state(env: BeamTrackingEnv, action: int) -> float:
    """
    Compute the SNR that would be observed if `action` were probed
    at the current UE position.

    This uses the simulator geometry, so this baseline is an optimistic
    local-probing baseline. It is still useful as a sanity-check baseline
    before training PPO.
    """
    if env.ue_state is None:
        raise RuntimeError("env.reset() must be called before probing.")

    ue_xy = (env.ue_state.x, env.ue_state.y)
    theta_true = true_angle_deg(env.scenario.bs_xy, ue_xy)
    distance = euclidean_distance(env.scenario.bs_xy, ue_xy)
    blocked = is_blocked(env.scenario.bs_xy, ue_xy, env.scenario.obstacles)
    beam_deg = beam_angle_from_action(action, env.codebook)

    snr_parts = compute_snr(
        distance=distance,
        angle_error=angle_error_deg(theta_true, beam_deg),
        blocked=blocked,
        channel=env.channel_config,
    )
    return float(snr_parts["snr_db"])


def choose_best_neighbor_beam(
    env: BeamTrackingEnv,
    current_action: int,
    radius: int = 1,
) -> tuple[int, dict[int, float]]:
    """
    Probe current beam +/- radius and choose the candidate with max SNR.
    """
    candidates = local_candidate_indices(
        center_idx=current_action,
        radius=radius,
        num_beams=len(env.codebook),
    )

    candidate_snrs = {
        int(a): probe_snr_at_current_state(env, int(a))
        for a in candidates
    }

    best_action = max(candidate_snrs, key=candidate_snrs.get)
    return int(best_action), candidate_snrs


def run_neighbor_tracker(
    scenario_name: str = "single_occluder",
    max_steps: int = 192,
    num_beams: int = 13,
    radius: int = 1,
    seed: int = 0,
    movement_config: MovementConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Run the local neighbor-probing baseline.

    Time convention:
        The baseline selects a beam using local probes at the current UE
        position, records SNR/reward, then advances the UE.
    """
    env = BeamTrackingEnv(
        scenario_name=scenario_name,
        max_steps=max_steps,
        num_beams=num_beams,
        movement_config=movement_config,
    )

    _, reset_info = env.reset(seed=seed)

    current_action = int(reset_info["selected_beam_idx"])
    env.current_action = current_action

    logs: list[dict[str, Any]] = []

    while env.ue_state is not None and env.ue_state.t < env.scenario.max_steps:
        prev_action = current_action

        action, candidate_snrs = choose_best_neighbor_beam(
            env=env,
            current_action=current_action,
            radius=radius,
        )

        # Evaluate the selected beam at the current UE position.
        # _evaluate is private, but it keeps this baseline exactly consistent
        # with the environment's reward, outage, and event definitions.
        info, obs = env._evaluate(action=action, prev_action=prev_action)  # noqa: SLF001

        theta_true = float(info["true_angle_deg"])
        oracle_action = nearest_beam_index(theta_true, env.codebook)

        row = dict(info)
        row["candidate_snrs_db"] = dict(candidate_snrs)
        row["oracle_beam_idx"] = int(oracle_action)
        row["oracle_beam_deg"] = float(beam_angle_from_action(oracle_action, env.codebook))
        row["tracking_error_deg"] = float(
            abs(angle_error_deg(theta_true, float(info["selected_beam_deg"])))
        )
        logs.append(row)

        # Keep env state coherent.
        env.current_action = int(action)
        env.current_obs = obs
        env.prev_blocked = bool(info["blocked"])
        env.prev_outage = bool(info["outage"])
        current_action = int(action)

        # Advance UE after the decision epoch.
        env.ue_state = advance_ue_state(
            env.ue_state,
            env.scenario,
            env.movement_config,
            env.np_random,
        )

    metrics = summarize_logs(logs)
    return logs, metrics


def summarize_logs(logs: list[dict[str, Any]]) -> dict[str, float]:
    if not logs:
        return {}

    snr = np.asarray([float(row["snr_db"]) for row in logs], dtype=float)
    outage = np.asarray([bool(row["outage"]) for row in logs], dtype=bool)
    blocked = np.asarray([bool(row["blocked"]) for row in logs], dtype=bool)
    reward = np.asarray([float(row["reward"]) for row in logs], dtype=float)
    beam = np.asarray([int(row["selected_beam_idx"]) for row in logs], dtype=int)
    err = np.asarray([float(row["tracking_error_deg"]) for row in logs], dtype=float)

    return {
        "steps": float(len(logs)),
        "mean_snr_db": float(np.mean(snr)),
        "p05_snr_db": float(np.percentile(snr, 5)),
        "min_snr_db": float(np.min(snr)),
        "outage_fraction": float(np.mean(outage)),
        "blocked_fraction": float(np.mean(blocked)),
        "beam_switch_count": float(np.sum(beam[1:] != beam[:-1])) if len(beam) > 1 else 0.0,
        "mean_reward": float(np.mean(reward)),
        "mean_tracking_error_deg": float(np.mean(err)),
        "p95_tracking_error_deg": float(np.percentile(err, 95)),
    }


def write_csv(logs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "t",
        "ue_x",
        "ue_y",
        "ue_vx",
        "ue_vy",
        "ue_speed",
        "true_angle_deg",
        "selected_beam_idx",
        "selected_beam_deg",
        "oracle_beam_idx",
        "oracle_beam_deg",
        "tracking_error_deg",
        "distance",
        "blocked",
        "outage",
        "snr_db",
        "beam_gain_db",
        "path_loss_db",
        "blockage_loss_db",
        "reward",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in logs:
            ue_x, ue_y = row["ue_xy"]
            ue_vx, ue_vy = row["ue_velocity_xy"]
            writer.writerow(
                {
                    "t": row["t"],
                    "ue_x": ue_x,
                    "ue_y": ue_y,
                    "ue_vx": ue_vx,
                    "ue_vy": ue_vy,
                    "ue_speed": row["ue_speed"],
                    "true_angle_deg": row["true_angle_deg"],
                    "selected_beam_idx": row["selected_beam_idx"],
                    "selected_beam_deg": row["selected_beam_deg"],
                    "oracle_beam_idx": row["oracle_beam_idx"],
                    "oracle_beam_deg": row["oracle_beam_deg"],
                    "tracking_error_deg": row["tracking_error_deg"],
                    "distance": row["distance"],
                    "blocked": row["blocked"],
                    "outage": row["outage"],
                    "snr_db": row["snr_db"],
                    "beam_gain_db": row["beam_gain_db"],
                    "path_loss_db": row["path_loss_db"],
                    "blockage_loss_db": row["blockage_loss_db"],
                    "reward": row["reward"],
                }
            )


def plot_logs(logs: list[dict[str, Any]], path: Path, outage_thresh_db: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray([int(row["t"]) for row in logs])
    snr = np.asarray([float(row["snr_db"]) for row in logs])
    beam = np.asarray([int(row["selected_beam_idx"]) for row in logs])
    oracle_beam = np.asarray([int(row["oracle_beam_idx"]) for row in logs])
    blocked = np.asarray([float(bool(row["blocked"])) for row in logs])
    outage = np.asarray([float(bool(row["outage"])) for row in logs])
    ue_x = np.asarray([float(row["ue_xy"][0]) for row in logs])
    ue_y = np.asarray([float(row["ue_xy"][1]) for row in logs])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    axes[0].plot(t, snr, label="neighbor baseline SNR")
    axes[0].axhline(outage_thresh_db, linestyle="--", label="outage threshold")
    axes[0].set_ylabel("SNR [dB]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(t, beam, where="post", label="selected beam")
    axes[1].step(t, oracle_beam, where="post", linestyle="--", label="oracle nearest beam")
    axes[1].set_ylabel("beam index")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].fill_between(t, 0, blocked, step="post", alpha=0.35, label="blocked")
    axes[2].fill_between(t, 0, outage, step="post", alpha=0.35, label="outage")
    axes[2].set_ylabel("indicator")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(ue_x, ue_y, label="UE path")
    axes[3].scatter([ue_x[0]], [ue_y[0]], marker="o", label="start")
    axes[3].scatter([ue_x[-1]], [ue_y[-1]], marker="x", label="end")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="single_occluder")
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=13)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--movement-model",
        choices=("constant_velocity", "stochastic"),
        default="stochastic",
    )
    parser.add_argument("--speed-mean", type=float, default=4.0)
    parser.add_argument("--speed-std", type=float, default=0.5)
    parser.add_argument("--heading-std-deg", type=float, default=8.0)
    parser.add_argument("--velocity-damping", type=float, default=0.85)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--out-dir", default="baseline_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs, metrics = run_neighbor_tracker(
        scenario_name=args.scenario,
        max_steps=args.max_steps,
        num_beams=args.num_beams,
        radius=args.radius,
        seed=args.seed,
        movement_config=MovementConfig(
            model=args.movement_model,
            speed_mean=args.speed_mean,
            speed_std=args.speed_std,
            heading_std_deg=args.heading_std_deg,
            velocity_damping=args.velocity_damping,
            position_noise_std=args.position_noise_std,
        ),
    )

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "neighbor_baseline_log.csv"
    fig_path = out_dir / "neighbor_baseline_plot.png"

    write_csv(logs, csv_path)
    plot_logs(logs, fig_path)

    print("\nNeighbor-beam baseline metrics")
    print("--------------------------------")
    for key, value in metrics.items():
        print(f"{key:28s}: {value:.4f}")

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    main()
