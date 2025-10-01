"""Simulation orchestration for the MCPP simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .environment import OccupancyGridEnvironment
from .logging_utils import get_simulation_logger
from .robots import Robot

StepCallback = Callable[[int, OccupancyGridEnvironment, List[Robot], Dict[str, bool]], None]


@dataclass
class Simulation:
    """Coordinate environment updates, robot actions, and observers."""

    environment: OccupancyGridEnvironment
    robots: List[Robot] = field(default_factory=list)
    step_limit: Optional[int] = None
    callbacks: List[StepCallback] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._logger = get_simulation_logger()
        self.step_count: int = 0
        self._logger.debug("Simulation initialized | robots=%s", len(self.robots))
        self._initialize_environment_occupancy()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def step(self) -> Dict[str, bool]:
        """Advance the simulation by one timestep and return move outcomes."""
        move_results: Dict[str, bool] = {}
        for robot in self.robots:
            success = robot.step(self.environment, self.robots)
            move_results[robot.robot_id] = success
        self.step_count += 1
        self._logger.debug(
            "Step complete | step=%s | coverage=%.3f",
            self.step_count,
            self.environment.covered_ratio(),
        )
        self._run_callbacks(move_results)
        return move_results

    def run(self, steps: Optional[int] = None, stop_when_complete: bool = True) -> None:
        """Run the simulation for a number of steps or until coverage completes."""
        target_steps = steps or self.step_limit
        while True:
            if target_steps is not None and self.step_count >= target_steps:
                self._logger.info("Step limit reached | steps=%s", self.step_count)
                break
            if stop_when_complete and self.environment.covered_ratio() >= 1.0:
                self._logger.info("Environment fully covered | steps=%s", self.step_count)
                break
            self.step()

    # ------------------------------------------------------------------
    # Administration helpers
    # ------------------------------------------------------------------
    def add_robot(self, robot: Robot) -> None:
        """Dynamically add a robot into the simulation."""
        self.robots.append(robot)
        self.environment.mark_covered(*robot.position)
        self._logger.info("Robot added | id=%s", robot.robot_id)

    def register_callback(self, callback: StepCallback) -> None:
        """Register a callback executed after every step."""
        self.callbacks.append(callback)

    def get_env(self) -> OccupancyGridEnvironment:
        """Return the simulation environment instance."""
        return self.environment

    def get_robots(self) -> Iterable[Robot]:
        """Return an iterable over the registered robots."""
        return tuple(self.robots)

    def coverage_ratio(self) -> float:
        """Convenience accessor for the environment coverage metric."""
        return self.environment.covered_ratio()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_environment_occupancy(self) -> None:
        for robot in self.robots:
            self.environment.mark_covered(*robot.position)
        self._logger.debug("Initial occupancy marked for all robots")

    def _run_callbacks(self, move_results: Dict[str, bool]) -> None:
        for callback in self.callbacks:
            callback(self.step_count, self.environment, self.robots, move_results)
