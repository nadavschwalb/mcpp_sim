import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from ..environment import OccupancyGridEnvironment
import networkx as nx
import numpy as np

class GraphVisualizer:
    def __init__(self):
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

        self.ax.imshow(grid, cmap='gray', origin='lower', extent=[0,grid.shape[1], 0, grid.shape[0]], )
        self.ax.set_xticks(np.arange(0, w))
        self.ax.set_yticks(np.arange(h, -1, -1))
        self.ax.invert_yaxis()
        self.ax.xaxis.tick_top()
        self.ax.grid(color='black', linewidth=1)
        
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

        lc = LineCollection(segments, color=attr.get('color', 'blue'))
        self.ax.add_collection(lc)


    def clear_graphs(self):
        self.ax.collections.clear()
        self.fig.canvas.draw()


    def _graph_to_segments(self, graph : nx.Graph) -> list:

        # convert graph to numpy array (edges, 2, 2)
        segments = np.array([
            [graph.nodes[u]['pos'], graph.nodes[v]['pos']] for u,v in graph.edges
        ])
        # edge_line_segments = []
        # for edge_data in graph.edges.data():

        #     # get edges nodes
        #     nodes = [graph.nodes[edge_data[0]], graph.nodes[edge_data[1]]]

        #     # assert pos attribute
        #     for node in nodes :
        #         if 'pos' not in node:
        #             raise KeyError(f"node {node} does not have a pos attribute")

        #     # add to line segment
        #     edge_line_segments.append([nodes[0]['pos'], nodes[1]['pos']])

        # return edge_line_segments

        return segments

    def _redraw(self):
        """ redraw plot """