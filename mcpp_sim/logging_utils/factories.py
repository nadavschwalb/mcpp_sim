"""Logging factory helpers for simulator components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _build_logger(name: str, level: int = logging.INFO, handler: Optional[logging.Handler] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        stream_handler = handler or logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logger.addHandler(stream_handler)
    return logger


def get_environment_logger() -> logging.Logger:
    """Return a logger configured for environment messages."""
    return _build_logger("mcpp.environment")


def get_robot_logger(robot_id: str | None = None) -> logging.Logger:
    """Return a logger for robot-specific messages."""
    name = f"mcpp.robot.{robot_id}" if robot_id else "mcpp.robot"
    return _build_logger(name)


def get_simulation_logger() -> logging.Logger:
    """Return a logger for simulation orchestration messages."""
    return _build_logger("mcpp.simulation")


def get_visualization_logger() -> logging.Logger:
    """Return a logger for visualization pipeline messages."""
    return _build_logger("mcpp.visualization")


def configure_logging(level: int | str = logging.INFO, log_directory: Optional[str | Path] = None) -> None:
    """Configure core simulator loggers with a shared level and optional file output."""
    numeric_level = (
        logging.getLevelName(level.upper()) if isinstance(level, str) else int(level)
    )
    logging.basicConfig(level=numeric_level)

    handlers: list[logging.Handler] = []
    if log_directory is not None:
        log_path = Path(log_directory)
        log_path.mkdir(parents=True, exist_ok=True)
        for name in ("environment", "robot", "simulation", "visualization"):
            file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
            handlers.append(file_handler)
            logger = logging.getLogger(f"mcpp.{name}")
            logger.handlers = []
            logger.addHandler(file_handler)
            logger.setLevel(numeric_level)

    for logger in (
        get_environment_logger(),
        get_robot_logger(),
        get_simulation_logger(),
        get_visualization_logger(),
    ):
        logger.setLevel(numeric_level)
