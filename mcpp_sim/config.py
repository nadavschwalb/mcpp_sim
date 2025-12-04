"""Configuration handling for the MCPP simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .environment.grid import OccupancyGridEnvironment
from .policies.factory import create_policy
from .robots.robot import Robot, action_space
from .simulation import Simulation
from .utils import GridLoader
from .network_graph import create_network_graph
from .visualization.environment_visualizer import get_environment_visualizer

@dataclass(slots=True)
class EnvironmentConfig:
    """Configuration payload required to materialize an environment."""

    source: str
    path: Optional[str] = None
    cell_size: float = 1.0
    footprint: Sequence[int] = (1, 1)
    scale: int = 1
    threshold: float = 0.5
    grid: Optional[Sequence[Sequence[float]]] = None
    overlays: List[Dict[str, Any]] = field(default_factory=list)

    def build(self) -> OccupancyGridEnvironment:
        """Create the environment instance described by this config."""
        footprint_tuple = (int(self.footprint[0]), int(self.footprint[1]))
        source_type = self.source.lower()

        if source_type in {"bitmap", "png"}:
            if self.path is None:
                raise ValueError("Image-based environment requires a 'path'")
            loader = GridLoader(threshold=self.threshold)
            loader_fn = loader.from_bitmap if source_type == "bitmap" else loader.from_png
            return loader_fn(
                self.path,
                cell_size=self.cell_size,
                footprint=footprint_tuple,
                scale=int(self.scale),
            )

        if source_type == "manual":
            if self.grid is None:
                raise ValueError("Manual environment requires a 'grid' definition")
            array = np.asarray(self.grid, dtype=np.float32)
            return OccupancyGridEnvironment.from_array(
                array,
                cell_size=self.cell_size,
                footprint=footprint_tuple,
            )

        raise ValueError(f"Unsupported environment source type: {self.source}")


@dataclass(slots=True)
class RobotConfig:
    """Configuration required to construct a robot."""

    policy: Mapping[str, Any]
    observation_range: int = 1

    def build(self, robot_id: int, initial_pos : tuple) -> Robot:
        """Instantiate the configured robot."""
        policy_instance = create_policy(self.policy, action_space())
        return Robot(
            robot_id=robot_id,
            position=initial_pos,
            policy=policy_instance,
            observation_range=int(self.observation_range),
        )


@dataclass(slots=True)
class SimulationOptions:
    """Simulation execution options."""
    number_robots : int
    step_limit: Optional[int] = None
    stop_when_complete: bool = True
    step_interval_ms: Optional[float] = None
    network_graph_type: Optional[str] = None
    visualize : bool = True
    animate : bool = True

@dataclass(slots=True)
class VisualizationOptions:
    """Visualization configuration knobs."""

    show_tracks: bool = True
    show_grid: bool = True
    show_robot_labels: bool = True


@dataclass(slots=True)
class LoggingOptions:
    """Logging configuration parameters."""

    level: str = "INFO"


@dataclass(slots=True)
class SimulationConfig:
    """Structured configuration for the simulator."""

    environment: EnvironmentConfig
    robots: RobotConfig
    simulation: SimulationOptions = field(default_factory=SimulationOptions)
    visualization: Optional[VisualizationOptions] = None
    logging: Optional[LoggingOptions] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        env_cfg = EnvironmentConfig(**data["environment"])
        robot_cfgs = RobotConfig(**data["robots"])
        sim_options = SimulationOptions(**data.get("simulation", {}))
        viz_options = (
            VisualizationOptions(**data["visualization"])
            if "visualization" in data
            else None
        )
        logging_options = (
            LoggingOptions(**data["logging"]) if "logging" in data else None
        )
        return cls(
            environment=env_cfg,
            robots=robot_cfgs,
            simulation=sim_options,
            visualization=viz_options,
            logging=logging_options,
        )


def load_config(path: str | Path) -> SimulationConfig:
    """Load a JSON configuration file into a :class:`SimulationConfig`."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    return SimulationConfig.from_dict(data)

def spawn_robot_pos(environment : OccupancyGridEnvironment, n_robots : int) -> np.array:
    """find n_robots unique positions to spawn robots that are unoccupied
    @param environment : OccupancyGridEnvironment - map environment
    @param n_robots : int - number of robots
    @return initial positions : np.array (n_robots, 2)
    """

    # mask unoccupied cells
    grid = environment.grid
    unoccupied_grid = (grid == 1)

    # get free cells list
    xs, ys = np.where(unoccupied_grid)
    coords = np.column_stack((xs, ys))

    # choose n_robots from unoccupied cells
    selected_positions = coords[np.random.choice(len(coords), n_robots, replace=False)]

    return selected_positions

def build_simulation(config: SimulationConfig) -> Simulation:
    """Construct a :class:`Simulation` from a :class:`SimulationConfig`."""
    environment = config.environment.build()

    # spawn robots
    initial_robot_positions = spawn_robot_pos(environment, config.simulation.number_robots)
    robots = [config.robots.build(i, tuple(position.tolist())) for i, position in enumerate(initial_robot_positions)]
    network_graph = create_network_graph(config.simulation.network_graph_type)
    simulation = Simulation(
        environment=environment,
        robots=robots,
        step_limit=config.simulation.step_limit,
        network_graph=network_graph,
        env_visualizer = get_environment_visualizer('simulation', animate=config.simulation.animate) if config.simulation.visualize else None
    )
    return simulation


def build_visualizer(config: SimulationConfig) -> Optional[MatplotlibVisualizer]:
    """Create a visualizer instance if visualization config is present."""
    if config.visualization is None:
        return None
    return config.visualization.build()
