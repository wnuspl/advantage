"""The live engine (reboot, docs/REBOOT.md section 6).

    predictor.py  black-box Predictor interface + placeholder (the seam models plug into)
    state.py      persisted rating state: save/load, resume, incremental replay        (planned)
    predict.py    the one canonical pre-match + in-play prediction path: PointsPredictor (A1)
    players.py    name -> player_id resolution, shared by CLI, web and Kalshi          (A1)
    editions.py   tournament / shared edition lookup -> edition_id, surface, best-of   (A1)
    rankings.py   standings from a saved state at any date                             (planned)
    live.py       Kalshi pulls (events, milestones, live scores, books) -> MarketSnapshots (B1)

Model internals (points_rating, context, markov, calibrate) stay in src/tennis/ and are only
reached through a Predictor.
"""
from .predictor import FixedPredictor, MatchQuery, Prediction, Predictor

__all__ = ["FixedPredictor", "MatchQuery", "Prediction", "Predictor"]
