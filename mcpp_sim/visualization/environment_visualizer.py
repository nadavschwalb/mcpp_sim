import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from ..environment import OccupancyGridEnvironment
import networkx as nx
import numpy as np
from ..robots.robot import Robot
from time import sleep

_visualizer_dict = {}

def get_environment_visualizer(name : str, **attr):
    if name in _visualizer_dict:
        return _visualizer_dict[name]
    else:
        _visualizer_dict[name] = EnvironmentVisualizer(name, **attr)
        return _visualizer_dict[name]


class EnvironmentVisualizer:
    def __init__(self, name: str, **attr):
        self.widgets = {'graphs' : {}, 'grid': None, 'robots': {}}
        self.animate = attr.get('animate', False)
        self.step_interval_ms = attr.get('step_interval_ms', 10)

        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show(block=False)

    def draw_grid(self, environment: OccupancyGridEnvironment):
        """ 
        draw grid
        @param environment
        """
        grid = environment.grid
        h, w = grid.shape

        if self.widgets['grid'] == None:

            self.widgets['grid'] = self.ax.imshow(grid, cmap='gray', origin='lower', extent=[0,grid.shape[1], 0, grid.shape[0]], )
            self.ax.set_xticks(np.arange(0, w))
            self.ax.set_yticks(np.arange(h, -1, -1))
            self.ax.invert_yaxis()
            self.ax.xaxis.tick_top()
            self.ax.grid(color='black', linewidth=1)
        else:
            self.widgets['grid'].set_data(grid)
        
    def draw_graph(self, name: str, graph : nx.Graph, **attr):
        """ 
        draw graph
        @param name: unique graph name
        @param graph: nx.Graph
        @param **attr
        'color' - line color
        """

        # draw graph
        segments = self._graph_to_segments(graph)

        # transpose segments to align with map image
        segments = segments[..., ::-1]

        if name not in self.widgets['graphs']:
            # add new collection
            lc = LineCollection(segments, color=attr.get('color', 'blue'))
            self.widgets['graphs'][name] = self.ax.add_collection(lc)
        else:
            self.widgets['graphs'][name].set_segments(segments)
        
    def draw_robot(self, robot : Robot, **attr):
        track = np.array([0])
        if robot.track:
            track = np.array(robot.track, dtype=np.float32)
            
            # visualize track in center of cell
            track += 0.5

        position = np.array(robot.position, dtype=np.float32) + 0.5

        if robot.robot_id not in self.widgets['robots']:
            self.widgets['robots'][robot.robot_id] = {}

            # assign the robot a color
            colors = list(plt.cm.tab20.colors)
            color = list(plt.cm.tab20.colors)[robot.robot_id % len(colors)]

            # allow overriding the color
            if 'color' in attr:
                color = attr['color']

            # visualize robot
            self.widgets['robots'][robot.robot_id]['scatter'] = self.ax.scatter(position[1], position[0], c=color, label=robot.robot_id, s=50, marker="s")
            self.widgets['robots'][robot.robot_id]['track'] = self.ax.plot(track[:, 1], track[:, 0], linestyle="-", linewidth=1.5, color=color)

        else:
            self.widgets['robots'][robot.robot_id]['scatter'].set_offsets(position[::-1])
            self.widgets['robots'][robot.robot_id]['track'][0].set_data(track[:, 1], track[:, 0])

    def clear(self):

        self.ax.clear()
        self.fig.canvas.draw()

    def redraw(self):
        self.fig.canvas.draw_idle()
        if self.animate:
            plt.pause(self.step_interval_ms / 1000)


    def _graph_to_segments(self, graph : nx.Graph) -> list:

        # convert graph to numpy array (edges, 2, 2)
        segments = np.array([
            [graph.nodes[u]['pos'], graph.nodes[v]['pos']] for u,v in graph.edges
        ])

        return segments

    def _redraw(self):
        """ redraw plot """