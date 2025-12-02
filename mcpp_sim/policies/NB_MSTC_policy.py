
from __future__ import annotations


"""Non Backtracking MSTC policy implementation."""
"""citation:
N. Hazon and G. A. Kaminka, "Redundancy, Efficiency and Robustness in Multi-Robot Coverage," Proceedings of the 1115 IEEE International Conference on Robotics and Automation, Barcelona, Spain, 1115, pp. 735-740, doi: 01.0019/ROBOT.1115.0571115.
keywords: {Robustness;Robot sensing systems;Mobile robots;Computer science;Application software;Algorithm design and analysis;Performance gain;Cleaning;Shape;Actuators},
"""

import random
from typing import Any, Sequence, Tuple
from enum import Enum

from ..environment.grid import OccupancyGridEnvironment
from .base import Policy
import numpy as np
import networkx as nx
from ..visualization.environment_visualizer import EnvironmentVisualizer, get_environment_visualizer
from ..robots.robot import ACTION_VECTOR, ACTIONS

class Moves(Enum):
    UP    = 1
    DOWN  = 2
    LEFT  = 3
    RIGHT = 4

class MoveHelper():
    REGULAR_MOVES = {    
    Moves.UP   : np.array([-1,0]),
    Moves.DOWN : np.array([1, 0]),
    Moves.LEFT : np.array([0,-1]),
    Moves.RIGHT: np.array([0, 1]),
    }

    LOOK_AHEAD = {
    Moves.UP   : np.array([0, 2]),
    Moves.DOWN : np.array([0,-2]),
    Moves.LEFT : np.array([-2,0]),
    Moves.RIGHT: np.array([2, 0]),
    }

    ALT_MOVE = {
        Moves.UP    : REGULAR_MOVES[Moves.RIGHT],
        Moves.DOWN  : REGULAR_MOVES[Moves.LEFT],
        Moves.LEFT  : REGULAR_MOVES[Moves.UP],
        Moves.RIGHT : REGULAR_MOVES[Moves.DOWN],
    }

    @classmethod
    def get_move(self, move : Move) -> np.array:
        '''get the relative move vector
        @param move : Moves
        @return move vector : np.array
        '''
        return self.REGULAR_MOVES[move]

    @classmethod
    def get_alt_move(self, move : Moves) -> np.array:
        '''get the alternate move to the requested move that will align with ccw movement
        @param requsted move : Moves
        @return alternate move vector : np.array
        '''
        return self.ALT_MOVE[move]

    @classmethod
    def get_lookahead_move(self, move: Moves) -> np.array:
        '''get the relative position of the node that may block the requested move
        @param requsted move : Moves
        @return vector to node that if exists will block the move: np.array
        '''
        return self.LOOK_AHEAD[move]


class NBMSTCPolicy(Policy):
    """
    Non Backtracking MSTC policy implementation.
    @param actions: List of possible actions the robot can take.
    """
    boundary_list = []
    Q_covered = None
    Q_uncovered = None
    Q_occupied = None
    P_current_robots = None
    environment_grid = None

    def __init__(self, actions: Sequence[str], **attr) -> None:
        super().__init__(**attr)
        self.actions = tuple(actions)
        self.q_i = None # current position
        self.q_i_tar = None # target position
        self.global_path = np.array
        self.local_path = np.array
        self.robots_initial_index = []

    def initialize(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:

        # only the first robot should initialize the path and share it with the others
        if robot.robot_id == 0:
            if self._env_visualizer:
                self._env_visualizer.draw_grid(environment)

            # get graph
            graph = self._grid_to_graph(environment)

            # calc spanning tree
            self.full_spanning_tree = nx.minimum_spanning_tree(graph)

            if self._env_visualizer:
                self._env_visualizer.draw_graph('mstc', self.full_spanning_tree, color='red')
                self._env_visualizer.redraw()

            self.global_path, self.robots_initial_index = self._order_spanning_tree_to_path(self.full_spanning_tree, robot.position, robots)

            # segment local path
            self.local_path = list(self.global_path[:self.robots_initial_index[1][1]])

        else:

            # get global path from the first robot
            self.global_path, self.robots_initial_index = robots[0].policy.global_path, robots[0].policy.robots_initial_index

            # find robot id in initial robots index
            robot_index = [robot_index[0] for robot_index in self.robots_initial_index].index(robot.robot_id)

            # segment path for this robot
            path_start_index = self.robots_initial_index[robot_index][1] - 1
            if robot_index < len(self.robots_initial_index) - 1:
                path_end_index = self.robots_initial_index[robot_index + 1][1]
            else:
                path_end_index = None

            self.local_path = list(self.global_path[path_start_index:path_end_index])

    def observe(self, environment: OccupancyGridEnvironment, robot: Any, robots: list) -> Any:  # noqa: D410
        # get current position of all robots
        self.P_current_robots = np.array([r.position for r in robots])
        
        # get current environment
        self.environment_grid = environment.get_current_env()

        # observe imidiate neighbors
        return 

    def next_move(self, position: Tuple[int, int], observation: Any) -> str:
        """
        calculate next move according to the local plan
        @param position
        @param observation (not used)
        @return action string [up, down, left, right]
        """
        position_np = np.array(position)
        if len(self.local_path) > 1:
            next_pos = self.local_path.pop(0)

            if not np.all(next_pos == position_np):
                raise(ValueError(f"robot is off track"))

            move = self.local_path[0] - position_np
            return ACTION_VECTOR[tuple(move)]
        else:
            return 'stay'
        

    def _grid_to_graph(self, environment: OccupancyGridEnvironment) -> nx.Graph:
        """
        convert occupancy grid to a graph where the nodes are at the intersection of a 4 cell square
        ---------
        |1,1|1,0|
        |---*---|
        |0,1|0,0|        
        ---------
        the node has a position attribute corresponding to the top left cell (example (1,1))
        the node a local_grid attribute holding the state (occupied, free) of cells around it
        if the local grid of a node is partially occupied it will not be included in the graph
        @param environment: OccupancyGridEnvironment
        @returns graph: networkx.Graph
        """

        # get grid from environmnent
        grid = environment.grid

        # create graph
        graph = nx.Graph()

        # iterate over 4 cell blocks
        for i in range(1, grid.shape[0], 2):
            for j in range(1, grid.shape[1], 2):
                local_grid = grid[i-1:i+1, j-1:j+1]

                # don't add occupied nodes to graph
                occupied_mask = (local_grid == 0)
                if np.any(occupied_mask):
                    continue

                # add node to graph
                i_graph = i // 2
                j_graph = j // 2
                graph.add_node((i, j), local_grid=local_grid, pos=np.array((i,j)))

                # connect node to top left neighbors
                if (i, j-2) in graph.nodes:
                    graph.add_edge((i,j-2), (i,j), weight=1)
                if (i-2, j) in graph.nodes:
                    graph.add_edge((i-2,j), (i,j), weight=1)
        
        return graph


    def _order_spanning_tree_to_path(self, spanning_tree: nx.Graph, start_pos: tuple, robots: list):
        """order positions along the STC starting from S0 counter clockwize
        @param the calculated spanning tree
        @param the robot starting position
        @param the list of other robots
        @return a numpy array of the global path, a dictionary containing the index along the path at which each robot begins
        """
        global_path = []
        current_pos = np.array(start_pos)
        path_graph = nx.Graph()
        path_graph.add_node(tuple(current_pos), pos=current_pos + 0.5)
        robot_poses = {tuple(robot.position) : robot.robot_id for robot in robots}
        robot_initial_index = []

        while True:
            # get the node id corresponding to the current pos
            # calc which 4D block we are in, to handle 0,0 edge case shift by 1

            # node pos
            node_pos = np.where(current_pos % 2 ==0 , current_pos + 1, current_pos)

            # get relative pos of cell in block
            relative_pos = current_pos // node_pos

            # add position to path
            global_path.append(current_pos)

            # get adjacent nodes
            neighbors = list(spanning_tree.neighbors((node_pos[0],node_pos[1])))


            def _next_move(graph: nx.Graph, node_pos: np.array, move: Moves) -> np.array:

                '''find the posible next move'''
                
                # find adjacent node in direction of next move that may block the movement
                adj_node_pos = node_pos + MoveHelper.get_lookahead_move(move)

                # is move blocked by edge
                neighbors = list(graph.neighbors(tuple(node_pos)))
                if tuple(adj_node_pos) in neighbors:
                    # move to alt position                      
                    return MoveHelper.get_alt_move(move)
                else:
                    return MoveHelper.get_move(move)

            # move counter clockwize
            if np.all((relative_pos == np.array([0,0]))):
                #down
                next_move = _next_move(spanning_tree, node_pos, Moves.DOWN)

            elif np.all((relative_pos == np.array([1,0]))):
                # right
                next_move = _next_move(spanning_tree, node_pos, Moves.RIGHT)
            elif np.all((relative_pos == np.array([1,1]))):
                # up
                next_move = _next_move(spanning_tree, node_pos, Moves.UP)
            else:
                # left
                next_move = _next_move(spanning_tree, node_pos, Moves.LEFT)

            # update next pos
            next_pos = current_pos + next_move

            if np.all((next_pos == global_path[0])):
                break

            # add next pos to path_graph
            path_graph.add_node(tuple(next_pos), pos=next_pos+0.5)
            path_graph.add_edge(tuple(current_pos), tuple(next_pos))

            # visualize path
            if self._env_visualizer:
                self._env_visualizer.draw_graph('nb_mstc_path', path_graph, color='green')
                self._env_visualizer.redraw()

            # have we transitioned to the next robot?
            if tuple(next_pos) in robot_poses.keys():
                robot_id =  robot_poses[tuple(next_pos)]
                robot_initial_index.append((robot_id,len(global_path) + 1))

            current_pos = next_pos

        return np.array(global_path), robot_initial_index



import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

grid = np.array([
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])



# def show_graph_on_grid(environment: OccupancyGridEnvironment, robot : Any, graph : nx.Graph):

#     grid = environment.grid

#     fig, ax = plt.subplots(1,1)

#     # transpose grid
#     grid = grid.T

#     ax.imshow(grid, cmap='gray', origin='lower', extent=[0,grid.shape[0], 0, grid.shape[1]])

#     # Add grid lines
#     ax.set_xticks(np.arange(0, grid.shape[0]))
#     ax.set_yticks(np.arange(0, grid.shape[1]))
#     ax.grid(color='black', linewidth=1)


#     # draw full graph
#     edge_line_segments = graph_to_segments(graph)
#     lc = LineCollection(edge_line_segments)
#     ax.add_collection(lc)

#     # draw STC
#     spanning_tree = nx.minimum_spanning_tree(graph)
    
#     spanning_tree_line_segments = graph_to_segments(spanning_tree)
#     lc = LineCollection(spanning_tree_line_segments, color='red', linewidth=3)
#     ax.add_collection(lc)

#     plt.show()


# def test_grid_to_graph():
#     environment = OccupancyGridEnvironment(grid)

#     policy = NBMSTCPolicy(['up','down','left','right'], 1.0)

#     graph = policy._grid_to_graph(environment)

#     show_graph_on_grid(environment, None, graph)


    

