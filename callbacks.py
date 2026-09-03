"""
Callbacks: logging that is IDENTICAL for the baseline and the treatment.

WHY THIS MATTERS FOR THE EXPERIMENT
-----------------------------------
`DoubleDQN.train()` logs `train/mean_target_q`, but stock `DQN.train()` does not.
If that were our only source of Q-statistics we could not compare the two
algorithms on the very quantity Double DQN is designed to fix.

So `QValueProbeCallback` measures Q-values from OUTSIDE the algorithm, on a fixed
set of probe states, using only the public `q_net`. It behaves identically for
`DQN` and `DoubleDQN`, which makes the comparison fair.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback


class QValueProbeCallback(BaseCallback):
    """Track estimated Q-values on a FIXED set of states throughout training.

    This is the evidence for the overestimation story. Double DQN's claim is that
    it reduces over-estimation of action values. To show that, we need Q-values
    measured the same way for both algorithms, on the same states, over time.

    We freeze a set of probe states once at the start of training (collected with
    a random policy) and re-evaluate them periodically. Because the states never
    change, any movement in the curve is movement in the Q-function, not a change
    in which states we happened to visit.

    Note: this measures Q-value MAGNITUDE, not error. To show over-ESTIMATION you
    compare these against the true discounted return achieved by the policy --
    which `scripts/evaluate.py` computes. The gap between the two is the bias.
    """

    def __init__(
        self,
        probe_states: np.ndarray,
        log_freq: int = 1000,
        output_csv: str | Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.probe_states = np.asarray(probe_states, dtype=np.float32)
        self.log_freq = int(log_freq)
        self.output_csv = Path(output_csv) if output_csv else None
        self.records: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        if self.output_csv is not None:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        q_net = self.model.q_net
        was_training = q_net.training
        q_net.eval()
        with th.no_grad():
            states = th.as_tensor(self.probe_states, device=self.model.device)
            q_values = q_net(states)                 # shape (N, n_actions)
            max_q = q_values.max(dim=1).values       # greedy value of each state
            record = {
                "timesteps": float(self.num_timesteps),
                "mean_max_q": float(max_q.mean().item()),
                "std_max_q": float(max_q.std().item()),
                "mean_q_all_actions": float(q_values.mean().item()),
                "max_q": float(max_q.max().item()),
                "min_q": float(max_q.min().item()),
            }
        if was_training:
            q_net.train()

        self.records.append(record)
        self.logger.record("probe/mean_max_q", record["mean_max_q"])
        self.logger.record("probe/mean_q_all_actions", record["mean_q_all_actions"])
        return True

    def _on_training_end(self) -> None:
        if self.output_csv is None or not self.records:
            return
        with self.output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)


class TrainingMetricsCallback(BaseCallback):
    """Capture our five required metrics for every TRAINING episode.

    The `EpisodeMetricsWrapper` attaches an "episode_metrics" dict to `info` when
    an episode ends. Stable-Baselines3 passes those infos through to callbacks, so
    we pick them up here and stream them to a CSV.

    This gives us training curves in terms of the metrics the brief asks about
    (collision rate, speed, distance) rather than only cumulative reward.
    """

    FIELDS = [
        "timesteps",
        "episode",
        "cumulative_reward",
        "length",
        "collided",
        "went_offroad",
        "truncated",
        "mean_speed",
        "distance_travelled",
        "successful_lane_changes",
        "attempted_lane_changes",
        "invalid_lane_changes",
        "min_time_headway",
    ]

    def __init__(self, output_csv: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.output_csv = Path(output_csv)
        self.episode_count = 0
        self._fh = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_csv.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            self.episode_count += 1
            row = {
                "timesteps": self.num_timesteps,
                "episode": self.episode_count,
                "cumulative_reward": metrics.get("cumulative_reward"),
                "length": metrics.get("length"),
                "collided": int(bool(metrics.get("collided"))),
                "went_offroad": int(bool(metrics.get("went_offroad"))),
                "truncated": int(bool(metrics.get("truncated"))),
                "mean_speed": metrics.get("mean_speed"),
                "distance_travelled": metrics.get("distance_travelled"),
                "successful_lane_changes": metrics.get("successful_lane_changes"),
                "attempted_lane_changes": metrics.get("attempted_lane_changes"),
                "invalid_lane_changes": metrics.get("invalid_lane_changes"),
                "min_time_headway": metrics.get("min_time_headway"),
            }
            self._writer.writerow(row)
            self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def collect_probe_states(env, n_states: int = 512, seed: int = 12345) -> np.ndarray:
    """Gather a fixed, reproducible set of states for the Q-value probe.

    Uses a random policy so the probe set covers a broad slice of the state space
    rather than only states a trained agent visits. The seed is fixed and
    recorded, so every algorithm and every training seed probes the SAME states.
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []

    obs, _ = env.reset(seed=seed)
    while len(states) < n_states:
        states.append(np.asarray(obs, dtype=np.float32))
        action = int(rng.integers(0, env.action_space.n))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    return np.stack(states[:n_states])
