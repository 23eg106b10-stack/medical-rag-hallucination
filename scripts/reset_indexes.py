"""Delete generated corpus and index artifacts from the local workspace.

Usage:
    python scripts/reset_indexes.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _clear_directory(directory: Path) -> None:
    """Remove generated contents from a directory while preserving .gitkeep."""
    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def reset_indexes() -> None:
    """Remove generated corpus and index outputs so they can be rebuilt."""
    root = Path(__file__).resolve().parent.parent
    _clear_directory(root / "data" / "corpus")
    _clear_directory(root / "data" / "indexes")


if __name__ == "__main__":
    reset_indexes()
