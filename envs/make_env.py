"""
The single place where an environment is built.

WHY A FACTORY FUNCTION
----------------------
Training and evaluation MUST use an identically-constructed environment,
otherwise the numbers are not comparable. Building the env in two places invites
them to drift apart (someone adds a wrapper to training and forgets evaluation).
So there is exactly one function that builds an env, and everything calls it.

THE WRAPPER STACK (innermost first)
-----------------------------------
    highway-v0                      the raw simulator
      -> CustomHighwayReward         our reward replaces the built-in one
      -> DropEgoLongitudinalPosition (5,5) matrix -> (24,) vector
      -> FrameStackObservation       stack k frames -> (k, 24)
      -> FlattenObservation          -> (96,)     [the state space in the formulation]
      -> EpisodeMetricsWrapper       record the five required metrics
      -> Monitor                     Stable-Baselines3' episode logger

Order matters. The reward wrapper sits innermost because it needs the raw
simulator state. The metrics wrapper sits outermost (before Monitor) so the
reward it records is the reward the agent actually learns from.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

import gymnasium as gym
import highway_env  # noqa: F401  -- registers the highway-v0 environment id

from .observation import DropEgoLongitudinalPosition
from .reward import CustomHighwayReward, RewardConfig
from .monitoring import EpisodeMetricsWrapper


@dataclass(frozen=True)
class EnvConfig:
    """Every environment-side setting, in one auditable place.

    The brief requires us to "state ... every hyperparameter that differs from
    the library default". Defaults from highway-env 1.12.1 are noted inline.
    """

    env_id: str = "highway-v0"

    # --- road and traffic -------------------------------------------------
    lanes_count: int = 4          # library default: 4
    vehicles_count: int = 20      # DIFFERS from library default (50).
    # Measured: the simulator steps every vehicle's IDM model 15x per agent step,
    # so wall-clock cost is linear in vehicle count.
    #   50 vehicles ->  3.7 env steps/sec  -> 7.5 h per training run
    #   20 vehicles -> 16.2 env steps/sec  -> 1.0 h per training run
    # A 2-algorithm x 3-seed protocol is infeasible at 50 on CPU.
    vehicles_density: float = 1.3  # DIFFERS from library default (1.0).
    # Raised to compensate for the lower vehicle count: fewer cars overall, but
    # spaced closer, so local traffic density around the ego is preserved.
    duration: int = 40            # library default: 40  (steps == seconds)

    # --- observation ------------------------------------------------------
    observed_vehicles: int = 5    # library default: 5
    frame_stack: int = 4          # OURS -- library has no frame stacking

    # --- timing -----------------------------------------------------------
    policy_frequency: int = 1     # library default: 1 Hz -> 1 step == 1 second
    simulation_frequency: int = 15  # library default: 15 Hz

    # --- termination ------------------------------------------------------
    offroad_terminal: bool = True  # DIFFERS FROM DEFAULT (False).
    # We terminate on leaving the road because our formulation (Section 4.1)
    # treats off-road as an absorbing failure state, the same as a collision.

    def as_dict(self) -> dict:
        return asdict(self)


def build_env_config(cfg: EnvConfig) -> dict[str, Any]:
    """Translate our EnvConfig into the dict highway-env expects."""
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": cfg.observed_vehicles,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,   # relative coords -- formulation Section 1.4
            "normalize": True,   # min-max to [-1, 1] -- formulation Section 1.3
            "order": "sorted",
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": cfg.lanes_count,
        "vehicles_count": cfg.vehicles_count,
        "vehicles_density": cfg.vehicles_density,
        "duration": cfg.duration,
        "policy_frequency": cfg.policy_frequency,
        "simulation_frequency": cfg.simulation_frequency,
        "offroad_terminal": cfg.offroad_terminal,
        # We compute our own reward, so the built-in shaping terms are irrelevant.
        # They are zeroed anyway to make it unambiguous to a reader that the
        # simulator's reward is not contributing anything.
        "normalize_reward": False,
        "collision_reward": 0.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.0,
        "lane_change_reward": 0.0,
    }


def make_env(
    env_cfg: EnvConfig | None = None,
    reward_cfg: RewardConfig | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_path: str | None = None,
) -> gym.Env:
    """Build one fully-wrapped environment.

    Parameters
    ----------
    env_cfg      Road, traffic and timing settings.
    reward_cfg   Weights and thresholds for our reward function.
    seed         If given, seeds this env's reset AND its action space.
    render_mode  "human" or "rgb_array" to watch the agent; None for training.
    monitor_path Optional CSV path for Stable-Baselines3' Monitor wrapper.
    """
    env_cfg = env_cfg or EnvConfig()
    reward_cfg = reward_cfg or RewardConfig()

    env = gym.make(
        env_cfg.env_id,
        config=build_env_config(env_cfg),
        render_mode=render_mode,
    )

    # 1. Our reward replaces the simulator's.
    env = CustomHighwayReward(env, config=reward_cfg)

    # 2. Observation construction (formulation Section 1).
    env = DropEgoLongitudinalPosition(env)

    if env_cfg.frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=env_cfg.frame_stack)
        env = gym.wrappers.FlattenObservation(env)

    # 3. Metrics.
    env = EpisodeMetricsWrapper(env)

    # 4. Stable-Baselines3' Monitor -- gives us ep_rew_mean / ep_len_mean logging.
    try:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env, filename=monitor_path)
    except ImportError:
        pass  # Monitor is optional; the env is fully usable without it.

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_env_fn(
    env_cfg: EnvConfig | None = None,
    reward_cfg: RewardConfig | None = None,
    seed: int | None = None,
    monitor_path: str | None = None,
) -> Callable[[], gym.Env]:
    """Return a zero-argument callable, as Stable-Baselines3' VecEnv requires."""

    def _init() -> gym.Env:
        return make_env(
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            seed=seed,
            monitor_path=monitor_path,
        )

    return _init
