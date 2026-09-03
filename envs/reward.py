"""
Our own reward function for the highway driving MDP.

This file implements Section 3 of docs/01_mdp_formulation.md.

WHY THIS FILE EXISTS
--------------------
highway-env ships its own reward (a mix of collision penalty, right-lane bonus and
high-speed bonus). Section 3 of the assignment brief states that the reward
function "must be the original work of the group". So this wrapper THROWS AWAY the
simulator's scalar reward and computes our own from the raw simulator state.

THE EQUATION (see formulation Section 3.1)

    R = w_v*r_v  +  w_h*r_h  +  w_l*r_l  +  w_inv*r_inv  +  w_c*r_c  +  w_o*r_o
        speed       headway     lane        invalid         crash      off-road
                                change      lane change

Each component is documented at its own method below.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import gymnasium as gym
import numpy as np

# Action indices for DiscreteMetaAction, confirmed against highway_env 1.12.1:
#   {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
ACTION_LANE_LEFT = 0
ACTION_IDLE = 1
ACTION_LANE_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4

LANE_CHANGE_ACTIONS = frozenset({ACTION_LANE_LEFT, ACTION_LANE_RIGHT})


@dataclass(frozen=True)
class RewardConfig:
    """Every tunable number in the reward function, in one place.

    Keeping these in a dataclass (rather than scattered as magic numbers) means
    the hyperparameter table required by the brief can be generated directly from
    this object -- so the reported table cannot drift out of sync with the code.
    """

    # --- speed term -------------------------------------------------------
    target_speed_min: float = 20.0   # m/s -- below this, zero speed reward
    target_speed_max: float = 30.0   # m/s -- at/above this, full speed reward

    # --- headway (safety) term -------------------------------------------
    safe_time_headway: float = 1.5   # seconds; the "how many seconds behind" rule

    # --- component weights ------------------------------------------------
    w_speed: float = 1.0
    w_headway: float = 0.4
    w_lane_change: float = 0.1
    w_invalid_action: float = 0.2
    w_collision: float = 10.0
    w_offroad: float = 10.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewardBreakdown:
    """The individual terms for one step, kept for analysis and debugging.

    Logging the breakdown (not just the total) lets the report say things like
    "the agent forfeited 3.2 reward per episode to headway violations", which is
    far more informative than a single scalar.
    """

    speed: float = 0.0
    headway: float = 0.0
    lane_change: float = 0.0
    invalid_action: float = 0.0
    collision: float = 0.0
    offroad: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class CustomHighwayReward(gym.Wrapper):
    """Replace highway-env's built-in reward with the reward defined in Section 3.

    We subclass `gym.Wrapper` rather than `gym.RewardWrapper` because
    `RewardWrapper.reward(r)` only receives the scalar reward -- it does not see
    the action that was taken, and we need the action for the lane-change terms.
    """

    def __init__(self, env: gym.Env, config: RewardConfig | None = None):
        super().__init__(env)
        self.config = config or RewardConfig()
        self._last_breakdown = RewardBreakdown()

    # ------------------------------------------------------------------ step

    def step(self, action):
        # `get_available_actions()` must be queried BEFORE stepping, because it
        # describes which lane changes were legal from the state the agent acted in.
        legal_actions = self._legal_actions()

        obs, _simulator_reward, terminated, truncated, info = self.env.step(action)
        # `_simulator_reward` is deliberately discarded. See module docstring.

        breakdown = self._compute_reward(int(action), legal_actions)
        self._last_breakdown = breakdown

        # Expose the components so the evaluation harness and plots can use them.
        info = dict(info)
        info["reward_breakdown"] = breakdown.as_dict()
        info["custom_reward"] = breakdown.total

        return obs, breakdown.total, terminated, truncated, info

    # -------------------------------------------------------- reward assembly

    def _compute_reward(self, action: int, legal_actions: set[int]) -> RewardBreakdown:
        cfg = self.config
        vehicle = self.unwrapped.vehicle

        r_speed = self._speed_term(vehicle)
        r_headway = self._headway_term(vehicle)
        r_lane_change = -1.0 if action in LANE_CHANGE_ACTIONS else 0.0
        r_invalid = -1.0 if action not in legal_actions else 0.0
        r_collision = -1.0 if bool(vehicle.crashed) else 0.0
        r_offroad = 0.0 if bool(vehicle.on_road) else -1.0

        breakdown = RewardBreakdown(
            speed=cfg.w_speed * r_speed,
            headway=cfg.w_headway * r_headway,
            lane_change=cfg.w_lane_change * r_lane_change,
            invalid_action=cfg.w_invalid_action * r_invalid,
            collision=cfg.w_collision * r_collision,
            offroad=cfg.w_offroad * r_offroad,
        )
        breakdown.total = float(
            breakdown.speed
            + breakdown.headway
            + breakdown.lane_change
            + breakdown.invalid_action
            + breakdown.collision
            + breakdown.offroad
        )
        return breakdown

    # ----------------------------------------------------------- the terms

    def _speed_term(self, vehicle) -> float:
        """r_v = clip((v - v_min) / (v_max - v_min), 0, 1)

        Linear ramp: 0 reward at 20 m/s, full 1.0 at 30 m/s. Crawling earns nothing,
        so the agent has a positive reason to make progress.

        We use `forward_speed` (the component of velocity along the lane) rather than
        raw `speed`, so that sideways motion during a lane change does not count as
        forward progress.
        """
        cfg = self.config
        speed = float(getattr(vehicle, "speed", 0.0))
        # Project onto the vehicle heading so lateral drift is not rewarded.
        forward = speed * float(np.cos(getattr(vehicle, "heading", 0.0)))

        span = cfg.target_speed_max - cfg.target_speed_min
        if span <= 0:
            return 0.0
        return float(np.clip((forward - cfg.target_speed_min) / span, 0.0, 1.0))

    def _headway_term(self, vehicle) -> float:
        """r_h = -clip((tau_safe - tau) / tau_safe, 0, 1),  tau = gap / speed

        This is the safety term and our main original contribution to the reward.

        We use TIME headway rather than raw distance because a 20 m gap is generous
        at 5 m/s and lethal at 30 m/s. Dividing distance by speed makes the term
        scale-free, and it mirrors how real driving guidance is written (the
        "two-second rule").

        Returns 0.0 when the road ahead is clear or the gap is comfortable, and
        ramps to -1.0 as the gap closes to nothing.
        """
        cfg = self.config
        gap = self._gap_to_leader(vehicle)
        if gap is None:
            return 0.0  # nothing ahead in our lane -- nothing to be unsafe about

        speed = max(float(getattr(vehicle, "speed", 0.0)), 1e-3)
        time_headway = max(gap, 0.0) / speed

        deficit = (cfg.safe_time_headway - time_headway) / cfg.safe_time_headway
        return -float(np.clip(deficit, 0.0, 1.0))

    # ---------------------------------------------------------------- helpers

    def _gap_to_leader(self, vehicle) -> float | None:
        """Bumper-to-bumper distance (m) to the vehicle directly ahead in our lane.

        Returns None if there is no vehicle ahead in our lane.
        """
        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return None

        try:
            front, _rear = road.neighbour_vehicles(vehicle, vehicle.lane_index)
        except (AttributeError, TypeError, ValueError):
            return None

        if front is None:
            return None

        # `lane_distance_to` measures along the lane's curvilinear coordinate,
        # which is the correct notion of "ahead" on a curved road.
        try:
            centre_gap = float(vehicle.lane_distance_to(front))
        except (AttributeError, TypeError):
            centre_gap = float(front.position[0] - vehicle.position[0])

        length = float(getattr(vehicle, "LENGTH", 5.0))
        return centre_gap - length

    def _legal_actions(self) -> set[int]:
        """Which action indices are valid right now.

        At the leftmost lane LANE_LEFT is unavailable; at the rightmost, LANE_RIGHT.
        highway-env silently converts an unavailable lane change into IDLE, so
        without this check the agent gets no signal that it made an impossible
        request. We do not mask the action (see formulation Section 2.1) -- we
        penalise it instead, so the agent learns that the road has edges.
        """
        action_type = getattr(self.unwrapped, "action_type", None)
        if action_type is None or not hasattr(action_type, "get_available_actions"):
            return set(range(5))
        try:
            return {int(a) for a in action_type.get_available_actions()}
        except Exception:
            return set(range(5))

    # ------------------------------------------------------------ inspection

    @property
    def last_breakdown(self) -> RewardBreakdown:
        return self._last_breakdown
