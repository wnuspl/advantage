"""Elo sweeps: the same as scripts/sweep.py with model elo_v1 by default.

Kept so existing commands keep working:
    python archive/v1/scripts/elo_sweep.py sweep [--grid ...] [--name ...]
    python archive/v1/scripts/elo_sweep.py show [--name ...] [--scope ...]   # Elo runs only
    python archive/v1/scripts/elo_sweep.py finalize RUN_ID [--save-predictions]
See archive/v1/scripts/sweep.py for the full help.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep  # noqa: E402


def main(argv=None) -> int:
    return sweep.main(argv, default_model="elo_v1")


if __name__ == "__main__":
    sys.exit(main())
