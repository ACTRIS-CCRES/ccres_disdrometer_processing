"""Tests preprocessing."""
from pathlib import Path


def test_run(root_dir: Path) -> None:
    print("path ", root_dir)
    print("type", type(root_dir))
