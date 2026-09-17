"""Convenience entry point: `uv run python main.py ...` or `uv run fly-ai ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())