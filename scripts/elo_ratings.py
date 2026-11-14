"""Elo standings: the same as scripts/ratings.py with model elo_v1 by default.

Kept so existing commands keep working:
    python scripts/elo_ratings.py [--as-of DATE] [--run ID] [--tour atp|wta] [--surface ...]
See scripts/ratings.py for the full help (it also handles Glicko-2 runs).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ratings  # noqa: E402


def main(argv=None) -> int:
    return ratings.main(argv, default_model="elo_v1")


if __name__ == "__main__":
    sys.exit(main())
