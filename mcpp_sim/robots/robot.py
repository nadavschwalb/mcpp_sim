"""Robot entity managing state and interaction with policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from ..environment import OccupancyGridEnvironment
from ..logging_utils import get_robot_logger
from ..policies import Policy

Move = Tuple[int, int]

ACTIONS: Dict[str, Move] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


def action_space() -> Tuple[str, ...]:
    """Return the canonical ordered action space."""
    return tuple(ACTIONS.keys())


@dataclass(slots=True)
class Robot:
    """Robot with discrete motions and policy-driven actions."""

    robot_id: str
    position: Tuple[int, int]
    policy: Policy
    observation_range: int = 1
    track: List[Tuple[int, int]] = field(default_factory=list)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = get_robot_logger(self.robot_id)
        self.track.append(self.position)
        self._logger.debug(
            "Robot initialized | id=%s | position=%s | observation_range=%s",
            self.robot_id,
            self.position,
            self.observation_range,
        )

    @property
    def actions(self) -> Iterable[str]:
        """Return the discrete action labels available to the robot."""
        return ACTIONS.keys()

    def step(self, environment: OccupancyGridEnvironment, robots: List["Robot"]) -> bool:
        """Advance the robot by executing the next policy action."""
        observation = self.policy.observe(environment, self)
        if observation is None:
            observation = self._default_observation(environment)
        move_label = self.policy.next_move(self.position, observation)
        self._logger.debug("Policy decision | move=%s", move_label)

        if move_label not in ACTIONS:
            self._logger.warning("Invalid move requested | move=%s", move_label)
            self.policy.update_last_move_result(False)
            return False

        delta_row, delta_col = ACTIONS[move_label]
        new_row = self.position[0] + delta_row
        new_col = self.position[1] + delta_col

        if not self._is_move_legal(environment, robots, new_row, new_col):
            self.policy.update_last_move_result(False)
            return False

        self.position = (new_row, new_col)
        self.track.append(self.position)
        environment.mark_covered(new_row, new_col)
        self.policy.update_last_move_result(True)
        self._logger.debug("Move success | new_position=%s", self.position)
        return True

    def _default_observation(self, environment: OccupancyGridEnvironment) -> Dict[str, object]:
        """Default observation data used when the policy opts out."""
        window = list(environment.observation_window(self.position, self.observation_range))
        observation: Dict[str, object] = {
            "position": self.position,
            "track": tuple(self.track),
            "window": window,
            "covered_ratio": environment.covered_ratio(),
        }
        return observation

    def _is_move_legal(
        self,
        environment: OccupancyGridEnvironment,
        robots: List["Robot"],
        row: int,
        col: int,
    ) -> bool:
        """Validate that the intended move is legal within the environment."""
        if not environment.is_within_bounds(row, col):
            self._logger.info("Out of bounds move rejected | target=(%s,%s)", row, col)
            return False

        if environment.is_occupied(row, col):
            self._logger.info("Occupied cell move rejected | target=(%s,%s)", row, col)
            return False

        for other in robots:
            if other is not self and other.position == (row, col):
                self._logger.info("Collision move rejected | other_robot=%s", other.robot_id)
                return False

        return True
