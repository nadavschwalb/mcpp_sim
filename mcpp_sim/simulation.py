"""Simulation orchestration for the MCPP simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional
import networkx as nx
from .environment import OccupancyGridEnvironment
from .logging_utils import get_simulation_logger
from .robots.robot import Robot
from .network_graph import BaseNetworkGraph
from .network_graph import EucleadianGraph
from math import hypot
from .visualization.environment_visualizer import EnvironmentVisualizer, get_environment_visualizer

StepCallback = Callable[[int, OccupancyGridEnvironment, List[Robot], Dict[str, bool]], None]

@dataclass
class Simulation:
    """Coordinate environment updates, robot actions, and observers."""

    environment: OccupancyGridEnvironment
    robots: List[Robot] = field(default_factory=list)
    step_limit: Optional[int] = None
    callbacks: List[StepCallback] = field(default_factory=list)
    network_graph: Optional[BaseNetworkGraph] = None
    visualize: bool = True

    def __post_init__(self) -> None:
        self._logger = get_simulation_logger()
        self.step_count: int = 0
        self._logger.debug("Simulation initialized | robots=%s", len(self.robots))
        self._initialize_environment_occupancy()
        self.metrics = {'mst_bottleneck': [], 'coverage_ratio': []}

        if self.visualize:
            self.env_visualizer = get_environment_visualizer("simulation", animate=True)
            self.env_visualizer.draw_grid(self.environment)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def initialize(self):
        """an optional apriori initialization step for the robot to preform, call robot init step"""
        for robot in self.robots:
            robot.initialize(self.environment, self.robots)
            if self.visualize:
                self.env_visualizer.draw_robot(robot)

    def step(self) -> Dict[str, bool]:
        """Advance the simulation by one timestep and return move outcomes."""

        move_results: Dict[str, bool] = {}
        for robot_id, robot in enumerate(self.robots):
            
            # update robot step
            success = robot.step(self.environment, self.robots)
            move_results[robot.robot_id] = success

            # update eucleadian graph
            if self.network_graph is not None:
                if robot_id not in self.network_graph.get_graph().nodes:
                    self.network_graph.add_node(robot_id, pos=robot.position)
                else:
                    self.network_graph.update_node(robot_id, pos=robot.position)

        # recalculate edges based on current positions
        if self.network_graph is not None:
            self.network_graph.compute_edges()

        # compute minimum spanning tree
        mst =  nx.minimum_spanning_tree(self.network_graph.get_graph())
        mst_bottleneck = max([attr[2] for attr in mst.edges.data('weight')])

        # vislualize step
        if self.visualize:
            self.env_visualizer.draw_grid(self.environment)
            for robot in self.robots:
                self.env_visualizer.draw_robot(robot)
            self.env_visualizer.draw_graph('connectivity', mst)
            self.env_visualizer.redraw()

        # update metrics
        self.metrics['mst_bottleneck'].append(mst_bottleneck)
        self.metrics['coverage_ratio'].append(self.environment.covered_ratio())

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

        # initialize step
        self.initialize()

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

    def summary(self):
        return self.metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_environment_occupancy(self) -> None:
        for robot in self.robots:
            self.environment.mark_covered(*robot.position)
        self._logger.debug("Initial occupancy marked for all robots")

    def _run_callbacks(self, move_results: Dict[str, bool]) -> None:
        for callback in self.callbacks:
            callback(self.step_count, self.environment, self.robots, self.network_graph, move_results)

    # ------------------------------------------------------------------
    # analysis functions
    # ------------------------------------------------------------------

    def get_network_graph(self) -> Optional[BaseNetworkGraph]:
        """Return the network graph instance."""
        return self.network_graph