"""v1 research modules (archived at the reboot, 2026-11-15).

This package merges with the live `src/tennis/` package (pkgutil.extend_path), so archived
code imports archived modules (elo, tuning, calibrate, matchup, ...) and live ones
(points_rating, markov, context, matches, ...) as `tennis.<name>`, with nothing copied.
Put archive/v1/src BEFORE src on sys.path; the archived scripts and tests do this.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
