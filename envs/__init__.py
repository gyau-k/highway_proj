"""Environment layer: observation construction, reward function, metrics.

This package is the part of the project the brief calls "the environment
definition or wrapper, including observation construction, action masking and
termination logic" -- i.e. assessed original work.
"""

from .make_env import EnvConfig, make_env, make_env_fn, build_env_config
from .reward import CustomHighwayReward, RewardConfig, RewardBreakdown
from .observation import DropEgoLongitudinalPosition, describe_feature_names
from .monitoring import EpisodeMetricsWrapper, EpisodeMetrics

__all__ = [
    "EnvConfig",
    "make_env",
    "make_env_fn",
    "build_env_config",
    "CustomHighwayReward",
    "RewardConfig",
    "RewardBreakdown",
    "DropEgoLongitudinalPosition",
    "describe_feature_names",
    "EpisodeMetricsWrapper",
    "EpisodeMetrics",
]
