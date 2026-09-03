"""
Per-episode metric collection.

This is the measuring instrument. The brief names five evaluation metrics:

    collision rate | average speed | distance travelled |
    successful lane changes | cumulative reward

All five are computed here, at the moment the episode ends, and attached to the
final `info` dict under the key "episode_metrics".

WHY A WRAPPER RATHER THAN COMPUTING THIS IN THE TRAINING LOOP
-------------------------------------------------------------
Because the wrapper sees EVERY step, including steps inside vectorised
environments and inside library code we did not write. If we tried to compute
distance travelled in `train.py` we would have to reach into Stable-Baselines3's
rollout loop. Putting it in a wrapper means the metric is collected identically
during training and during evaluation, which is what makes the numbers comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import gymnasium as gym
import numpy as np

LANE_CHANGE_ACTIONS = frozenset({0, 2})  # LANE_LEFT, LANE_RIGHT


@dataclass
class EpisodeMetrics:
    """The five metrics required by the brief, plus diagnostics."""

    # --- the five required metrics ---------------------------------------
    collided: bool = False           # -> collision RATE when averaged over episodes
    mean_speed: float = 0.0          # m/s
    distance_travelled: float = 0.0  # m
    successful_lane_changes: int = 0
    cumulative_reward: float = 0.0

    # --- diagnostics ------------------------------------------------------
    length: int = 0                       # steps survived
    went_offroad: bool = False
    truncated: bool = False               # hit the time limit (i.e. survived)
    attempted_lane_changes: int = 0
    invalid_lane_changes: int = 0
    min_time_headway: float = float("inf")
    reward_components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        if not np.isfinite(d["min_time_headway"]):
            d["min_time_headway"] = None
        return d


class EpisodeMetricsWrapper(gym.Wrapper):
    """Accumulate per-episode statistics and emit them when the episode ends.

    Place this as the OUTERMOST wrapper so that the reward it records is the final
    reward the agent actually learns from.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_accumulators()

    # --------------------------------------------------------------- lifecycle

    def _reset_accumulators(self) -> None:
        self._speeds: list[float] = []
        self._reward_sum = 0.0
        self._steps = 0
        self._attempted_lane_changes = 0
        self._invalid_lane_changes = 0
        self._successful_lane_changes = 0
        self._min_headway = float("inf")
        self._component_totals: dict[str, float] = {}
        self._start_position = self._ego_x()
        self._last_lane_id = self._ego_lane_id()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_accumulators()
        return obs, info

    # -------------------------------------------------------------------- step

    def step(self, action):
        action = int(action)
        lane_before = self._ego_lane_id()

        obs, reward, terminated, truncated, info = self.env.step(action)

        self._steps += 1
        self._reward_sum += float(reward)
        self._speeds.append(self._ego_speed())

        # Lane-change bookkeeping.
        # "Attempted"  = the agent asked for a lane change.
        # "Successful" = the lane index actually changed, and we did not crash.
        # These differ because a lane change takes time and can be blocked.
        if action in LANE_CHANGE_ACTIONS:
            self._attempted_lane_changes += 1

        lane_after = self._ego_lane_id()
        if (
            lane_before is not None
            and lane_after is not None
            and lane_after != lane_before
            and not self._ego_crashed()
        ):
            self._successful_lane_changes += 1

        # Accumulate reward components if the reward wrapper provided them.
        breakdown = info.get("reward_breakdown")
        if isinstance(breakdown, dict):
            for key, value in breakdown.items():
                self._component_totals[key] = self._component_totals.get(key, 0.0) + float(value)
            if breakdown.get("invalid_action", 0.0) < 0.0:
                self._invalid_lane_changes += 1

        headway = self._current_time_headway()
        if headway is not None:
            self._min_headway = min(self._min_headway, headway)

        if terminated or truncated:
            info = dict(info)
            info["episode_metrics"] = self._finalise(truncated).as_dict()

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------- finalise

    def _finalise(self, truncated: bool) -> EpisodeMetrics:
        end_x = self._ego_x()
        distance = 0.0
        if self._start_position is not None and end_x is not None:
            distance = float(end_x - self._start_position)

        return EpisodeMetrics(
            collided=self._ego_crashed(),
            mean_speed=float(np.mean(self._speeds)) if self._speeds else 0.0,
            distance_travelled=distance,
            successful_lane_changes=self._successful_lane_changes,
            cumulative_reward=float(self._reward_sum),
            length=self._steps,
            went_offroad=not self._ego_on_road(),
            truncated=bool(truncated),
            attempted_lane_changes=self._attempted_lane_changes,
            invalid_lane_changes=self._invalid_lane_changes,
            min_time_headway=self._min_headway,
            reward_components=dict(self._component_totals),
        )

    # ----------------------------------------------------------- ego accessors
    # Each of these reaches through the wrapper stack to the raw simulator.
    # They are defensive (returning None / defaults) so that a metric never
    # crashes a training run.

    def _ego(self):
        return getattr(self.unwrapped, "vehicle", None)

    def _ego_x(self) -> float | None:
        v = self._ego()
        if v is None:
            return None
        try:
            return float(v.position[0])
        except (AttributeError, IndexError, TypeError):
            return None

    def _ego_speed(self) -> float:
        v = self._ego()
        return float(getattr(v, "speed", 0.0)) if v is not None else 0.0

    def _ego_crashed(self) -> bool:
        v = self._ego()
        return bool(getattr(v, "crashed", False)) if v is not None else False

    def _ego_on_road(self) -> bool:
        v = self._ego()
        return bool(getattr(v, "on_road", True)) if v is not None else True

    def _ego_lane_id(self):
        v = self._ego()
        if v is None:
            return None
        lane_index = getattr(v, "lane_index", None)
        if lane_index is None:
            return None
        try:
            return lane_index[2]  # (from_node, to_node, lane_number)
        except (IndexError, TypeError):
            return None

    def _current_time_headway(self) -> float | None:
        """Seconds to the vehicle ahead. Mirrors the reward wrapper's calculation."""
        v = self._ego()
        road = getattr(self.unwrapped, "road", None)
        if v is None or road is None:
            return None
        try:
            front, _ = road.neighbour_vehicles(v, v.lane_index)
        except (AttributeError, TypeError, ValueError):
            return None
        if front is None:
            return None
        try:
            gap = float(v.lane_distance_to(front)) - float(getattr(v, "LENGTH", 5.0))
        except (AttributeError, TypeError):
            return None
        speed = max(float(getattr(v, "speed", 0.0)), 1e-3)
        return max(gap, 0.0) / speed
