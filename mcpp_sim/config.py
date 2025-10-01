"""Configuration handling for the MCPP simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .environment import OccupancyGridEnvironment
from .policies import create_policy
from .robots import Robot, action_space
from .simulation import Simulation
from .utils import GridLoader
from .visualization import MatplotlibVisualizer


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

    robot_id: str
    start: Sequence[int]
    policy: Mapping[str, Any]
    observation_range: int = 1

    def build(self) -> Robot:
        """Instantiate the configured robot."""
        position = (int(self.start[0]), int(self.start[1]))
        policy_instance = create_policy(self.policy, action_space())
        return Robot(
            robot_id=self.robot_id,
            position=position,
            policy=policy_instance,
            observation_range=int(self.observation_range),
        )


@dataclass(slots=True)
class SimulationOptions:
    """Simulation execution options."""

    step_limit: Optional[int] = None
    stop_when_complete: bool = True
    step_interval_ms: Optional[float] = None


@dataclass(slots=True)
class VisualizationOptions:
    """Visualization configuration knobs."""

    show_tracks: bool = True
    show_grid: bool = True
    show_robot_labels: bool = True

    def build(self) -> MatplotlibVisualizer:
        """Materialize a :class:`MatplotlibVisualizer` with the configured settings."""
        return MatplotlibVisualizer(
            show_tracks=self.show_tracks,
            show_grid=self.show_grid,
            show_robot_labels=self.show_robot_labels,
        )


@dataclass(slots=True)
class LoggingOptions:
    """Logging configuration parameters."""

    level: str = "INFO"


@dataclass(slots=True)
class SimulationConfig:
    """Structured configuration for the simulator."""

    environment: EnvironmentConfig
    robots: List[RobotConfig]
    simulation: SimulationOptions = field(default_factory=SimulationOptions)
    visualization: Optional[VisualizationOptions] = None
    logging: Optional[LoggingOptions] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        env_cfg = EnvironmentConfig(**data["environment"])
        robot_cfgs = [RobotConfig(**item) for item in data.get("robots", [])]
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


def build_simulation(config: SimulationConfig) -> Simulation:
    """Construct a :class:`Simulation` from a :class:`SimulationConfig`."""
    environment = config.environment.build()
    robots = [robot_cfg.build() for robot_cfg in config.robots]
    simulation = Simulation(
        environment=environment,
        robots=robots,
        step_limit=config.simulation.step_limit,
    )
    return simulation


def build_visualizer(config: SimulationConfig) -> Optional[MatplotlibVisualizer]:
    """Create a visualizer instance if visualization config is present."""
    if config.visualization is None:
        return None
    return config.visualization.build()
