"""Command-line entry point for running coverage path planning simulations."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


if __package__ is None or __package__ == "":  # pragma: no cover - script execution
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))

from mcpp_sim.config import build_simulation, build_visualizer, load_config
from mcpp_sim.environment import OccupancyGridEnvironment
from mcpp_sim.logging_utils import configure_logging, get_simulation_logger
from mcpp_sim.robots import Robot
from mcpp_sim.visualization import MatplotlibVisualizer


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-robot coverage simulation")
    parser.add_argument("--config", required=True, help="Path to simulation configuration JSON")
    parser.add_argument("--steps", type=int, default=None, help="Override step limit from configuration")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--animate", action="store_true", help="Render each simulation step (implies visualization)")
    parser.add_argument("--log-level", default=None, help="Override logging level (e.g., DEBUG, INFO)")
    parser.add_argument("--log-dir", default=None, help="Optional directory to write component log files")
    parser.add_argument("--no-summary", action="store_true", help="Suppress coverage summary output")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)

    log_level = args.log_level or (config.logging.level if config.logging else "INFO")
    configure_logging(level=log_level, log_directory=args.log_dir)
    logger = get_simulation_logger()

    simulation = build_simulation(config)
    step_limit = args.steps or simulation.step_limit

    step_interval_seconds: Optional[float] = None
    raw_interval = config.simulation.step_interval_ms
    if raw_interval is not None:
        try:
            step_interval_seconds = max(0.0, float(raw_interval) / 1000.0)
        except (TypeError, ValueError):
            logger.warning("Invalid step_interval_ms value: %s", raw_interval)
            step_interval_seconds = None

    visualizer = None
    if not args.headless:
        visualizer = build_visualizer(config) or MatplotlibVisualizer()
        if args.animate:
            def _render_frame(
                step: int,
                env: OccupancyGridEnvironment,
                robots: List[Robot],
                move_results: Dict[str, bool],
            ) -> None:
                if visualizer is not None:
                    visualizer.visualize_step(env, robots)
                if step_interval_seconds:
                    time.sleep(step_interval_seconds)

            simulation.register_callback(_render_frame)

    logger.info("Starting simulation | step_limit=%s", step_limit)
    simulation.run(steps=step_limit, stop_when_complete=config.simulation.stop_when_complete)
    logger.info(
        "Simulation complete | steps=%s | coverage=%.3f",
        simulation.step_count,
        simulation.coverage_ratio(),
    )

    if visualizer and not args.headless and not args.animate:
        visualizer.visualize(simulation.get_env(), simulation.get_robots())

    if not args.no_summary:
        coverage = simulation.coverage_ratio()
        print("Coverage ratio:", f"{coverage:.2%}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main())
