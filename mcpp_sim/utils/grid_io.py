"""Grid loading helpers for environment initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from ..environment import CellState, OccupancyGridEnvironment


class GridLoader:
    """Load occupancy grids from bitmap sources."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def from_bitmap(
        self,
        bitmap_path: str | Path,
        *,
        cell_size: float = 1.0,
        footprint: Tuple[int, int] = (1, 1),
        scale: int = 1,
    ) -> OccupancyGridEnvironment:
        """Load an environment from a monochrome bitmap file."""
        grid = self._load_image(bitmap_path)
        grid = self._downsample_grid(grid, scale)
        return OccupancyGridEnvironment.from_array(grid, cell_size=cell_size, footprint=footprint)

    def from_png(
        self,
        png_path: str | Path,
        *,
        cell_size: float = 1.0,
        footprint: Tuple[int, int] = (1, 1),
        scale: int = 1,
    ) -> OccupancyGridEnvironment:
        """Load an environment from a PNG file."""
        return self.from_bitmap(png_path, cell_size=cell_size, footprint=footprint, scale=scale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_image(self, path: str | Path) -> np.ndarray:
        image = Image.open(path).convert("L")
        gray = np.asarray(image, dtype=np.float32) / 255.0
        occupied_mask = gray <= self.threshold
        grid = np.where(occupied_mask, CellState.OCCUPIED.value, CellState.UNCOVERED.value)
        return grid.astype(np.float32)

    def _downsample_grid(self, grid: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return grid
        rows, cols = grid.shape
        new_rows = rows // scale
        new_cols = cols // scale
        if new_rows == 0 or new_cols == 0:
            raise ValueError("Scale factor too large for the provided grid")

        trimmed = grid[: new_rows * scale, : new_cols * scale]
        reshaped = trimmed.reshape(new_rows, scale, new_cols, scale)
        downsampled = reshaped.min(axis=(1, 3))
        return downsampled
