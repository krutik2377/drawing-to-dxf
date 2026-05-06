#!/usr/bin/env python3
"""
Remove generated outputs so you can rerun drawing-to-dxf with a clean tree.

Default removes (if present), relative to repo root:
  - out/
  - out_sheet/
  - out_panels/
  - .pytest_cache/

Use --include-venv only when you intend to recreate the virtualenv.

Usage (from repo root):
  python scripts/clean_outputs.py
  python scripts/clean_outputs.py --include-venv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rm_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed: {path}")
    elif path.exists():
        path.unlink()
        print(f"Removed file: {path}")
    else:
        print(f"Skip (not found): {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete build/output folders for a clean rerun.")
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Also delete .venv (you must recreate: python -m venv .venv && pip install -e \".[ocr]\")",
    )
    args = parser.parse_args()

    root = _repo_root()
    print(f"Repo: {root}")
    print()

    for name in ("out", "out_sheet", "out_panels", ".pytest_cache"):
        _rm_tree(root / name)

    if args.include_venv:
        _rm_tree(root / ".venv")
        print()
        print("Recreate venv:  python -m venv .venv")
        print("Then activate it and run:  pip install -e \".[ocr]\"")
    else:
        print()
        print("Tip: pass --include-venv to also delete .venv, then reinstall deps.")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
