"""Matplotlib based visualization for MCPP simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from ..environment import OccupancyGridEnvironment
from ..logging_utils import get_visualization_logger
from ..robots import Robot
from ..network_graph import BaseNetworkGraph
import math

Overlay = Callable[[Axes, OccupancyGridEnvironment, Iterable[Robot]], None]


@dataclass
class MatplotlibVisualizer:
    """Render environment and robot tracks using matplotlib."""

    show_tracks: bool = True
    show_grid: bool = True
    show_robot_labels: bool = True
    overlays: List[Overlay] = field(default_factory=list)
    color_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._logger = get_visualization_logger()
        self._figure: Figure | None = None
        self._ax: Axes | None = None

    def add_overlay(self, overlay: Overlay) -> None:
        """Register a custom overlay callable for additional drawing."""
        self.overlays.append(overlay)
        self._logger.debug("Overlay added | count=%s", len(self.overlays))

    def visualize(self, environment: OccupancyGridEnvironment, robots: Iterable[Robot]) -> None:
        """Draw the current environment and robot states."""
        robots = tuple(robots)
        self._logger.debug("Rendering frame | robots=%s", len(robots))
        self._ensure_canvas()
        ax = self._ax
        assert ax is not None  # for type checkers
        self._draw_scene(ax, environment, robots)
        plt.show(block=True)

    def visualize_step(self, environment: OccupancyGridEnvironment, robots: Iterable[Robot], network_graph: Optional[BaseNetworkGraph]) -> None:
        """Incrementally update the visualization without blocking execution."""
        robots = tuple(robots)
        self._ensure_canvas()
        ax = self._ax
        assert ax is not None
        self._draw_scene(ax, environment, robots, network_graph)
        plt.pause(0.001)

    def _color_for_robot(self, robot_id: str) -> str:
        if robot_id not in self.color_map:
            palette = [
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
            ]
            color = palette[len(self.color_map) % len(palette)]
            self.color_map[robot_id] = color
            self._logger.debug("Color assigned | robot=%s | color=%s", robot_id, color)
        return self.color_map[robot_id]

    def _ensure_canvas(self) -> None:
        if self._figure is None or self._ax is None:
            plt.ion()
            self._figure, self._ax = plt.subplots()

    def _draw_scene(
        self,
        ax: Axes,
        environment: OccupancyGridEnvironment,
        robots: Iterable[Robot],
        network_graph: Optional[BaseNetworkGraph] = None,
    ) -> None:
        ax.clear()
        grid = environment.get_current_env()
        if self.show_grid:
            ax.imshow(
                grid,
                cmap="gray",
                interpolation="nearest",
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )

        for robot in robots:
            color = self._color_for_robot(robot.robot_id)
            row, col = robot.position
            ax.scatter(col, row, c=color, label=robot.robot_id, s=50, marker="s")
            if self.show_tracks and robot.track:
                track = np.array(robot.track)
                ax.plot(track[:, 1], track[:, 0], linestyle="-", linewidth=1.5, color=color)
            if self.show_robot_labels:
                ax.text(col + 0.1, row + 0.1, robot.robot_id, color=color, fontsize=8)

        if network_graph is not None:
            ng_edge_lines = []
            for edge in network_graph.get_graph().edges:
                ng_edge_lines.append([
                    tuple(reversed(network_graph.get_graph().nodes[edge[0]]['pos'])), 
                    tuple(reversed(network_graph.get_graph().nodes[edge[1]]['pos']))
                    ])

            lc = LineCollection(ng_edge_lines, color='blue', linewidth=2, )
            ax.add_collection(lc)

        for overlay in self.overlays:
            overlay(ax, environment, robots)

        if robots:
            ax.legend(loc="upper right", fontsize="small")
        ax.set_title("Multi-Robot Coverage Simulation")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_aspect("equal")
        if self._figure is not None:
            self._figure.canvas.draw_idle()

class SummaryVisualizer:
    def __init__(self, summary : dict):
        self.summary = summary
        num_ax = len(summary.keys())
        rows = math.ceil(math.sqrt(num_ax))
        cols = math.ceil(num_ax / rows)
        self.fig, self.ax = plt.subplots(rows, cols)

        self.ax = self.ax.ravel()

        self.visualize()

    def visualize(self):
        for index, (metric, data) in enumerate(self.summary.items()):
            x = np.arange(0, len(data))
            y = data
            self.ax[index].plot(x,y)
            self.ax[index].set_title(metric)

        plt.show(block=True)