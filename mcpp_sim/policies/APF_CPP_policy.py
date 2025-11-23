
from __future__ import annotations


"""APF CPP policy implementation."""
"""citation:
Z. Wang et al., "APF-CPP: An Artificial Potential Field Based Multi-Robot Online Coverage Path Planning Approach," in IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 9199-9206, Nov. 2024, doi: 10.1109/LRA.2024.3432351.
keywords: {Robot kinematics;Path planning;Multi-robot systems;Planning;Resource management;Autonomous agents;Multi-robot systems;Uncertainty;Scheduling;Autonomous agents;multi-robot systems;path planning for multiple mobile robots or agents;planning;planning under uncertainty;scheduling and coordination},

"""

import random
from typing import Any, Sequence, Tuple

from ..environment import OccupancyGridEnvironment
from .base import Policy
from ..robots import Robot
import numpy as np


class APFCPPPolicy(Policy):
    """
    APF CPP policy implementation.
    @param actions: List of possible actions the robot can take.
    @param velocity: Velocity of the robot. cells per time step.
    """
    boundary_list = []
    Q_covered = None
    Q_uncovered = None
    Q_occupied = None
    P_current_robots = None
    environment_grid = None

    def __init__(self, actions: Sequence[str], velocity: float) -> None:
        super().__init__()
        self.actions = tuple(actions)
        self.velocity = velocity
        self.q_i = None # current position
        self.q_i_tar = None # target position

    def observe(self, environment: OccupancyGridEnvironment, robot: Robot, robots: list) -> Any:  # noqa: D401
        # get current position of all robots
        self.P_current_robots = np.array([r.position for r in robots])
        
        # get current environment
        self.environment_grid = environment.get_current_env()

        # observe imidiate neighbors
        return 

        


    def next_move(self, position: Tuple[int, int], observation: Any) -> str:
        return self._rng.choice(self.actions)



def get_neighbors(grid : np.ndarray, position : tuple):

    """
    Extracts a square window from `arr` centered at `center`
    and fills out-of-bounds with NaN.
    """
    size = 3
    arr = grid
    r, c = position
    half = size // 2
    padded = np.full((arr.shape[0] + 2*half, arr.shape[1] + 2*half), np.nan)
    padded[half:half+arr.shape[0], half:half+arr.shape[1]] = arr
    window = padded[r:r+size, c:c+size]
    return window

    # neighbors = np.zeros((3,3))
    # print(grid.shape)

    # print(f"qi [{position[0]},{position[1]}] [{grid[position[0],position[1]]}]")

    # # try to append neighbor counter clockwize
    # for i,j in [[-1,-1], [-1,0], [-1,1], [0,1], [0,0], [1,1], [1,0], [1,-1], [0,-1]]:
    #     print(f"[i,j] : [{i},{j}]")
    #     q_n_i = position[0] + i
    #     q_n_j = position[1] + j
    #     if 0 <= q_n_i <= grid.shape[0] - 1 and 0 <= q_n_j <= grid.shape[1] - 1:
    #         print(q_n_i,q_n_i)
    #         print(float(grid[q_n_i,q_n_j]))
    #         neighbors[i + 1, j + 1] = grid[q_n_i, q_n_j]
    #     else:
    #         neighbors[i + 1, j + 1] = np.NaN
    # print(neighbors)
    # return neighbors


import pytest

grid = np.array([[11.,12.,13.,14.,15.],
                 [21.,22.,23.,24.,25.],
                 [31.,32.,33.,34.,35.],
                 [41.,42.,43.,44.,45.],
                 [51.,52.,53.,54.,55.]])

print(grid.shape)

def test_get_neighbors_center():
    neighbors = get_neighbors(grid, (2,2))

    np.testing.assert_array_equal(neighbors, np.array([[22,23,24],
                                                       [32,33,34],
                                                       [42,43,44]]))

def test_get_neighbors_quarner():
    neighbors = get_neighbors(grid, (0,0))
    np.testing.assert_array_equal(neighbors, np.array([[np.NAN,np.NAN,np.NAN],
                                                       [np.NAN,11,12],
                                                       [np.NAN,21,22]]))

def test_get_neighbors_left_edge():
    neighbors = get_neighbors(grid, (2,0))
    np.testing.assert_array_equal(neighbors, np.array([[np.NAN,21,22],
                                                       [np.NAN,31,32],
                                                       [np.NAN,41,42]]))


def test_get_neighbors_right_edge():
    neighbors = get_neighbors(grid, (2,4))
    np.testing.assert_array_equal(neighbors, np.array([[24,25,np.NAN],
                                                         [34,35,np.NAN],
                                                         [44,45,np.NAN]]))



