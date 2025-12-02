"""Random walk policy for exploratory coverage."""

from __future__ import annotations

import random
from typing import Any, Sequence, Tuple

from ..environment.grid import OccupancyGridEnvironment
from .base import Policy


class RandomPolicy(Policy):
    """Randomly sample actions from the robot's action space."""

    def __init__(self, actions: Sequence[str], seed: int | None = None) -> None:
        super().__init__()
        self.actions = tuple(actions)
        self._rng = random.Random(seed)

    def initialize(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:
        pass

    def observe(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:  # noqa: D401
        """Observation is not used; returns ``None``."""
        return None

    def next_move(self, position: Tuple[int, int], observation: Any) -> str:
        return self._rng.choice(self.actions)
