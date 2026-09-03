"""
Observation construction for the highway driving MDP.

This file implements Section 1 of docs/01_mdp_formulation.md.

WHAT A GYMNASIUM WRAPPER IS (for beginners)
-------------------------------------------
A "wrapper" sits around an environment like a layer of an onion. When you call
`env.step(action)`, the call passes through your wrapper on the way in, and the
result passes back through it on the way out. That lets you modify what the agent
sees, or what reward it gets, WITHOUT editing the simulator's source code.

`gymnasium.ObservationWrapper` is a convenience base class: you only implement
`observation(obs)`, and Gymnasium calls it automatically on every reset and step.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Column indices into one row of the (5, 5) Kinematics matrix.
# Row layout is [presence, x, y, vx, vy] -- confirmed empirically against
# highway_env 1.12.1, not assumed.
COL_PRESENCE = 0
COL_X = 1
COL_Y = 2
COL_VX = 3
COL_VY = 4

# Row 0 is always the ego vehicle.
ROW_EGO = 0


class DropEgoLongitudinalPosition(gym.ObservationWrapper):
    """Flatten the Kinematics matrix and delete the ego's absolute x position.

    See docs/01_mdp_formulation.md Section 1.5 for the full justification.

    Short version: the ego's `x` is absolute distance along the road, clipped at
    200 m. The ego spawns near x = 177 m and drives forward, so this feature
    saturates at exactly 1.0 within about a second and stays there. It is
    uninformative once saturated AND it is the only non-stationary feature in the
    vector. Feeding a constant-with-a-transient to a neural network is strictly
    worse than not feeding it at all.

    Every other feature is kept, including the ego's `y` (which lane am I in),
    `vx` (how fast am I going) and `vy` (am I drifting sideways).

    Shape change: (5, 5) matrix -> (24,) vector.
                  25 features minus the 1 we drop = 24.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 2:
            raise ValueError(
                "DropEgoLongitudinalPosition expects a 2-D Box observation "
                f"(the Kinematics matrix), got {base}."
            )

        n_vehicles, n_features = base.shape
        self._n_vehicles = n_vehicles
        self._n_features = n_features

        # Build a boolean mask over the FLATTENED matrix marking which entries we keep.
        # Flattened index of element [row, col] is (row * n_features + col).
        keep = np.ones(n_vehicles * n_features, dtype=bool)
        keep[ROW_EGO * n_features + COL_X] = False
        self._keep_mask = keep

        n_kept = int(keep.sum())
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_kept,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        return flat[self._keep_mask]


def describe_feature_names(n_vehicles: int = 5) -> list[str]:
    """Human-readable names for each entry of the 24-dim single-frame vector.

    Useful for debugging, for plots, and for the report -- it lets you say
    "feature 7 is v1's relative vx" instead of "feature 7".
    """
    features = ["presence", "x", "y", "vx", "vy"]
    names: list[str] = []
    for row in range(n_vehicles):
        label = "ego" if row == ROW_EGO else f"v{row}"
        for col, feat in enumerate(features):
            if row == ROW_EGO and col == COL_X:
                continue  # dropped -- see class docstring
            names.append(f"{label}_{feat}")
    return names
