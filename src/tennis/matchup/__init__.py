"""Matchup layer (MIDDLE_LAYER_SPEC.md): interaction models on two players' style vectors.

    logit P(p1 wins) = alpha * glicko_logit + delta(s_p1, s_p2, surface)

- data.py     match table (pre-match Glicko-2 logits) and style vectors projected through a
              fixed set of heavy-layer fits per fold
- models.py   Glicko-only, antisymmetric bilinear, Siamese MLP, LightGBM benchmark
- harness.py  walk-forward folds, alpha calibration, scoring, persistence
"""
