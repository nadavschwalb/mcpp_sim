"""Module entry point for ``python -m mcpp_sim``."""

from __future__ import annotations

from mcpp_sim.cli import main


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main())
