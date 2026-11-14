"""Registry of walk-forward rating models. A model_run's model_id names its entry here.

Every model provides:
  params_cls    frozen dataclass with as_dict(), update_key() (all but blend), and the
                fields init, surface_init, blend, retired_weight
  run(m, p)     -> result with p_winner, r_/s_ (pre-match overall/surface ratings),
                n_ (counts), optional d_ (deviation), and .state (ratings after the last
                match: rating, count, srating, scount; Glicko adds rd_at()/srd_at())
  predict(res, p) -> P(actual winner wins) per match at another setting of the
                prediction-only parameters of p (blend; points_v1 also unc)
  attach(con, m) optional: load model-specific per-match data into m.extra first
"""
from __future__ import annotations

from dataclasses import dataclass

from . import elo, glicko2, points_rating


@dataclass(frozen=True)
class Model:
    model_id: str
    params_cls: type
    run: object
    predict: object
    default_grid: list
    has_deviation: bool
    attach: object = None


MODELS = {
    "elo_v1": Model(
        "elo_v1", elo.EloParams, elo.run, lambda res, p: elo.predict(res, p.blend),
        # around the best plain-Elo setting of sweep default-v1, with surface ratings
        [{"k": [100, 150, 200], "offset": [5], "shape": [0.2, 0.25, 0.3], "retired_weight": [1.0],
          "surface_init": [0.0, 1.0], "blend": [0.0, 0.25, 0.5, 0.75, 1.0]}],
        False),
    "glicko2_v1": Model(
        "glicko2_v1", glicko2.Glicko2Params, glicko2.run, lambda res, p: glicko2.predict(res, p.blend),
        # ~50-80 s per pass; blend is free (scored from the same pass)
        [{"tau": [0.0, 0.5], "rd_init": [250.0, 350.0], "period_days": [7.0, 30.0],
          "retired_weight": [1.0], "surface_init": [0.0],
          "blend": [0.0, 0.25, 0.35, 0.5, 0.75]}],
        True),
    "points_v1": Model(
        "points_v1", points_rating.PointsParams, points_rating.run, points_rating.predict,
        # blend and unc are prediction-only (scored from the same pass)
        [{"v_init": [0.02, 0.04], "q": [1e-5, 3e-5], "obs_var": [0.01, 0.03],
          "blend": [0.0, 0.5, 1.0], "unc": [0.0, 1.0]}],
        True, points_rating.attach),
}


def get(model_id: str) -> Model:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"unknown model {model_id!r}; known: {sorted(MODELS)}") from None
