"""Policy factory utilities for configuration-driven creation."""

from __future__ import annotations

from typing import Mapping, Sequence

from .base import Policy
from .path_policy import PathPolicy
from .random_policy import RandomPolicy


def create_policy(config: Mapping[str, object], actions: Sequence[str]) -> Policy:
    """Instantiate a policy based on the supplied configuration mapping."""
    policy_type = str(config.get("type", "path")).lower()

    if policy_type == "path":
        moves = config.get("path", [])
        if not isinstance(moves, Sequence):
            raise ValueError("Path policy expects a sequence of moves")
        return PathPolicy(moves)

    if policy_type == "random":
        seed = config.get("seed")
        return RandomPolicy(actions, seed=seed if isinstance(seed, int) else None)

    raise ValueError(f"Unsupported policy type: {policy_type}")
