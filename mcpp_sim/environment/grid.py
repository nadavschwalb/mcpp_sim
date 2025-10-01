"""Environment representation for coverage simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
from PIL import Image

from ..logging_utils import get_environment_logger


class CellState(float, Enum):
    """Enumeration of canonical cell states stored as floats."""

    OCCUPIED = 0.0
    COVERED = 0.5
    UNCOVERED = 1.0


GridArray = np.ndarray


@dataclass
class OccupancyGridEnvironment:
    """2D occupancy grid supporting coverage tracking and updates."""

    grid: GridArray
    cell_size: float = 1.0
    footprint: Tuple[int, int] = (1, 1)
    logger_name: str = "mcpp.environment"

    def __post_init__(self) -> None:
        self._logger = get_environment_logger()
        self._logger.debug(
            "Environment initialized | shape=%s | cell_size=%s | footprint=%s",
            self.grid.shape,
            self.cell_size,
            self.footprint,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_array(
        cls,
        array: GridArray,
        *,
        cell_size: float = 1.0,
        footprint: Tuple[int, int] = (1, 1),
    ) -> "OccupancyGridEnvironment":
        """Create an environment from a precomputed array."""
        return cls(grid=array.astype(np.float32), cell_size=cell_size, footprint=footprint)

    @classmethod
    def from_bitmap(
        cls,
        bitmap_path: str | Path,
        *,
        cell_size: float = 1.0,
        footprint: Tuple[int, int] = (1, 1),
    ) -> "OccupancyGridEnvironment":
        """Instantiate the environment by discretizing a bitmap image."""
        array = _load_grayscale_array(bitmap_path)
        grid = _threshold_grid(array)
        return cls.from_array(grid, cell_size=cell_size, footprint=footprint)

    @classmethod
    def from_png(
        cls,
        png_path: str | Path,
        *,
        cell_size: float = 1.0,
        footprint: Tuple[int, int] = (1, 1),
    ) -> "OccupancyGridEnvironment":
        """Instantiate the environment by discretizing a PNG image."""
        return cls.from_bitmap(png_path, cell_size=cell_size, footprint=footprint)

    # ------------------------------------------------------------------
    # Grid queries and updates
    # ------------------------------------------------------------------
    def cell_state(self, row: int, col: int) -> CellState:
        """Return the state of a specific cell."""
        return CellState(self.grid[row, col])

    def update_cells(self, indices: Sequence[Tuple[int, int]], state: CellState) -> None:
        """Update multiple cells to the provided state."""
        for row, col in indices:
            if self.is_within_bounds(row, col) and self.grid[row, col] != CellState.OCCUPIED:
                self.grid[row, col] = state.value
                self._logger.debug("Cell updated | row=%s | col=%s | state=%s", row, col, state.name)

    def update_cell(self, row: int, col: int, state: CellState) -> None:
        """Update a single cell."""
        self.update_cells([(row, col)], state)

    def mark_covered(self, row: int, col: int) -> None:
        """Mark the cell as covered if it is not occupied."""
        if self.grid[row, col] != CellState.OCCUPIED:
            self.grid[row, col] = CellState.COVERED.value
            self._logger.debug("Cell marked covered | row=%s | col=%s", row, col)

    def get_current_env(self) -> GridArray:
        """Return a defensive copy of the grid."""
        return self.grid.copy()

    def get_dimensions(self) -> Tuple[int, int]:
        """Return grid dimensions (rows, columns)."""
        return self.grid.shape

    def covered_ratio(self) -> float:
        """Compute the ratio of covered cells out of all traversable cells."""
        traversable = np.isin(self.grid, [CellState.COVERED.value, CellState.UNCOVERED.value])
        total = np.count_nonzero(traversable)
        if total == 0:
            return 0.0
        covered = np.count_nonzero(self.grid == CellState.COVERED.value)
        return float(covered / total)

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Check whether coordinates are within the discrete grid bounds."""
        rows, cols = self.grid.shape
        return 0 <= row < rows and 0 <= col < cols

    def is_occupied(self, row: int, col: int) -> bool:
        """Check whether a cell is currently marked as occupied."""
        return bool(self.grid[row, col] == CellState.OCCUPIED.value)

    def distance(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
        """Compute Euclidean distance between two cells measured in meters."""
        delta = (np.array(pos_a) - np.array(pos_b)) * self.cell_size
        return float(np.linalg.norm(delta))

    def observation_window(self, center: Tuple[int, int], radius: int) -> Iterator[Tuple[int, int, CellState]]:
        """Yield cell coordinates and states within an observation radius."""
        row_c, col_c = center
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                row, col = row_c + d_row, col_c + d_col
                if self.is_within_bounds(row, col):
                    yield row, col, self.cell_state(row, col)


def _load_grayscale_array(image_path: str | Path) -> GridArray:
    """Load an image file as a normalized grayscale numpy array."""
    image = Image.open(image_path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def _threshold_grid(gray_grid: GridArray, threshold: float = 0.5) -> GridArray:
    """Convert a grayscale grid into OCCUPIED/UNCOVERED states."""
    occupied_mask = gray_grid <= threshold
    grid = np.where(occupied_mask, CellState.OCCUPIED.value, CellState.UNCOVERED.value)
    return grid.astype(np.float32)
