"""The live engine (reboot, docs/REBOOT.md section 6).

    predictor.py  black-box Predictor interface + placeholder (the seam models plug into)
    state.py      persisted rating state: save/load, resume, incremental replay        (planned)
    predict.py    the one canonical pre-match + in-play prediction path (a Predictor)  (planned)
    players.py    name -> player_id resolution, shared by CLI, web and Kalshi          (planned)
    rankings.py   standings from a saved state at any date                             (planned)
    live.py       Kalshi markets + live scores -> priced rows, alerts                  (planned)

Model internals (points_rating, context, markov, calibrate) stay in src/tennis/ and are only
reached through a Predictor.
"""
from .predictor import FixedPredictor, MatchQuery, Prediction, Predictor

__all__ = ["FixedPredictor", "MatchQuery", "Prediction", "Predictor"]
