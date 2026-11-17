"""Why the model gives the probability it gives: a step-by-step breakdown of P(A wins)
from two tour-average players to the price, for the live page's "Why" tab.

Each step changes one input of the model's own arithmetic (engine.predict) and re-prices,
so the steps add up exactly to the output; the order is fixed, so a step's size depends
on the steps before it (as in any waterfall):

  start    both players at the tour's mean serve and return rating (ref): about 50%
  serve    each player's overall serve rating instead of the mean
  return   each player's overall return rating instead of the mean
  surface  the surface deviations (the run's blend)
  ctx      the context offsets, one step per player and group: rust (time off), load
           (recent matches), age (context.FEATURES)
  doubt    the rating variances: the model's shrink toward 50% for its own uncertainty
  calib    the calibration (pre-match only)
  -> the pre-match probability
  score    in play: the Markov roll-up from the current score, pre-match serve rates
           (in-play prices are not calibrated, so this step also takes calibration off)
  form_a   A's serve stream on the day (in-match form), then form_b B's
  -> the in-play probability

Steps before `doubt` are priced without the shrink (variances 0). Ratings are logits
relative to `base`, the serve-point logit of an average server against an average
returner on the surface.
"""
from __future__ import annotations

from datetime import date

from .. import context, points_rating

GROUPS = {"rust": ("back1", "back2_5", "gap5_12"), "load": ("gap1", "load28"),
          "age": ("age_lt22", "age_lt26", "age_gt28", "age_gt32")}
IDX = {f: i for i, f in enumerate(context.FEATURES)}


def _age(dob: str | None, on: str) -> float | None:
    try:
        return (date.fromisoformat(on).toordinal() - date.fromisoformat(dob).toordinal()) / 365.25 if dob else None
    except ValueError:
        return None


def explain(model, q, d: dict, live=None, live_x: tuple | None = None) -> dict:
    """model: engine.predict.ModelState; q: the MatchQuery; d: PointsPredictor.prematch()
    detail (with ctx_c_a / ctx_c_b); live / live_x: when priced in play, the live state and
    (x_a, x_b, v_a, v_b) after the form update, the in-play output being their roll-up."""
    st, prm = model.state, model.params
    bo, fin, day = q.best_of, q.final, d["rating_day"]
    a, b = q.player_a, q.player_b
    oa = points_rating.player_ratings(st, a, q.tour, None, day)
    ob = points_rating.player_ratings(st, b, q.tour, None, day)
    ref = d.get("ref") or {"serve": (oa["serve"] + ob["serve"]) / 2, "ret": (oa["ret"] + ob["ret"]) / 2}
    base = d["base"]

    def wp(xa, xb):
        return points_rating.win_probability(st, xa, xb, 0.0, 0.0, bo, fin)

    steps = []
    p_prev = None

    def add(key, p, **kw):
        nonlocal p_prev
        steps.append({"key": key, "p": p, "d": None if p_prev is None else p - p_prev, **kw})
        p_prev = p

    x0 = base + ref["serve"] - ref["ret"]
    add("start", wp(x0, x0), serve=ref["serve"], ret=ref["ret"])
    xa, xb = base + oa["serve"] - ref["ret"], base + ob["serve"] - ref["ret"]
    add("serve", wp(xa, xb), a=oa["serve"] - ref["serve"], b=ob["serve"] - ref["serve"])
    xa, xb = base + oa["serve"] - ob["ret"], base + ob["serve"] - oa["ret"]
    add("return", wp(xa, xb), a=oa["ret"] - ref["ret"], b=ob["ret"] - ref["ret"])
    xa, xb = d["x_a"] - d["ctx_a"], d["x_b"] - d["ctx_b"]
    add("surface", wp(xa, xb), surface=d.get("surface"),
        a=(d["rating_a"]["serve"] - oa["serve"], d["rating_a"]["ret"] - oa["ret"]),
        b=(d["rating_b"]["serve"] - ob["serve"], d["rating_b"]["ret"] - ob["ret"]))
    ca, cb = d.get("ctx_c_a"), d.get("ctx_c_b")
    if ca is not None and cb is not None and prm.has_context():
        bs, br = prm.context_betas(q.tour)
        for side, c, pid in (("a", ca, a), ("b", cb, b)):
            for g, feats in GROUPS.items():
                own = sum(c[IDX[f]] * bs[IDX[f]] for f in feats)          # on this player's serve
                opp = -sum(c[IDX[f]] * br[IDX[f]] for f in feats)         # on the opponent's serve
                if not own and not opp:
                    continue
                if side == "a":
                    xa, xb = xa + own, xb + opp
                else:
                    xb, xa = xb + own, xa + opp
                add("ctx", wp(xa, xb), side=side, group=g,
                    feat={f: c[IDX[f]] for f in feats if c[IDX[f]]},
                    age=_age(model.dob.get(pid), day) if g == "age" else None)
    p_raw = points_rating.win_probability(st, d["x_a"], d["x_b"], d["v_a"], d["v_b"], bo, fin)
    add("doubt", p_raw, sd_a=d["v_a"] ** 0.5, sd_b=d["v_b"] ** 0.5)
    add("calib", d["p_prematch"], alpha=d["alpha"])
    out = {"steps": steps, "p_prematch": d["p_prematch"], "in_play": False}
    if live is not None and live_x is not None:
        tb = getattr(live, "deciding_tb_points", 7)

        def ip(xa, xb, va, vb):
            return points_rating.in_play_probability(st, xa, xb, va, vb, bo, fin, live.score(), live.a_serving,
                                                     deciding_tb_points=tb)
        xal, xbl, val, vbl = live_x
        add("score", ip(d["x_a"], d["x_b"], d["v_a"], d["v_b"]), score=getattr(live, "raw", ""))
        add("form_a", ip(xal, d["x_b"], val, d["v_b"]), form=xal - d["x_a"], pts=list(live.serve_a))
        add("form_b", ip(xal, xbl, val, vbl), form=xbl - d["x_b"], pts=list(live.serve_b))
        out["in_play"] = True
    out["p"] = steps[-1]["p"]
    return out
