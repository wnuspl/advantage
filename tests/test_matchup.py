"""Matchup layer: antisymmetry, Glicko-only reproduction, null vectors, fit consistency,
leakage. Synthetic data only (no tennis.db / models.db)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis import elo, heavy                                       # noqa: E402
from tennis.matchup import data, harness, models                    # noqa: E402
from tennis.matchup.data import FoldData, MatchTable, StyleSource   # noqa: E402
from tennis.style_features import FEATURES, SIDES, MatchCounts      # noqa: E402

DIMS = {"serve": 4, "return": 3, "rally": 5}


def random_fold(n=400, seed=0, dims=DIMS, has_frac=1.0) -> FoldData:
    rng = np.random.default_rng(seed)
    gl = rng.normal(0, 1, n)
    has = rng.random(n) < has_frac
    return FoldData(rows=np.arange(n), y=(rng.random(n) < data.sigmoid(gl)).astype(float),
                    glicko_logit=gl, offset=0.9 * gl, surf=data.surface_onehot(rng.integers(-1, 3, n)),
                    xa={s: rng.normal(0, 1, (n, k)) for s, k in dims.items()},
                    xb={s: rng.normal(0, 1, (n, k)) for s, k in dims.items()},
                    has=has, na=rng.integers(1, 30, n), nb=rng.integers(1, 30, n), fit_ids={})


def with_signal(fd: FoldData, seed=9) -> FoldData:
    """Labels driven by a serve x return interaction on top of the offset."""
    z = fd.offset + 1.5 * (fd.xa["serve"][:, 0] * fd.xb["return"][:, 0] - fd.xb["serve"][:, 0] * fd.xa["return"][:, 0])
    fd.y = (np.random.default_rng(seed).random(len(fd)) < data.sigmoid(z)).astype(float)
    return fd


def swapped(fd: FoldData) -> FoldData:
    return FoldData(fd.rows, 1.0 - fd.y, -fd.glicko_logit, -fd.offset, fd.surf, fd.xb, fd.xa, fd.has,
                    fd.nb, fd.na, fd.fit_ids)


def randomize(module, seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g))


class TestAntisymmetry(unittest.TestCase):
    def check(self, m, fd):
        d_ab, d_ba = m.delta(fd), m.delta(swapped(fd))
        self.assertGreater(np.abs(d_ab).max(), 1e-3)          # non-trivial
        np.testing.assert_allclose(d_ab, -d_ba, atol=1e-5)
        p_ab, p_ba = m.predict_proba(fd), m.predict_proba(swapped(fd))
        np.testing.assert_allclose(p_ab + p_ba, 1.0, atol=1e-6)

    def test_bilinear_random_weights(self):
        fd = random_fold()
        for terms in [("sr", "rr"), ("sr",), ("rr",)]:
            m = models.BilinearModel(terms=terms)
            m.dims, m.module = fd.dims(), m.build(fd.dims())
            randomize(m.module)
            self.check(m, fd)

    def test_rally_matrix_is_antisymmetric(self):
        m = models.BilinearModel()
        m.module = m.build(DIMS)
        randomize(m.module)
        W = m.module.w_rr().detach().numpy()
        np.testing.assert_allclose(W, -W.T, atol=1e-6)

    def test_siamese_random_weights(self):
        fd = random_fold()
        for head in (16, 0):
            m = models.SiameseModel(head_hidden=head)
            m.dims, m.module = fd.dims(), m.build(fd.dims())
            randomize(m.module)
            m.module.eval()
            self.check(m, fd)

    def test_lgbm_symmetrized(self):
        tr, va, te = random_fold(2000, 1), random_fold(500, 2), random_fold(300, 3)
        with_signal(tr), with_signal(va)
        m = models.LGBMModel(models.LGBMConfig(num_boost_round=50, min_data_in_leaf=20))
        m.fit(tr, va)
        self.check(m, te)


class TestGlickoReproduction(unittest.TestCase):
    def test_untrained_models_equal_glicko_only(self):
        fd = random_fold()
        base = data.sigmoid(fd.offset)
        for m in (models.SiameseModel(), models.BilinearModel()):
            m.dims, m.module = fd.dims(), m.build(fd.dims())
            np.testing.assert_array_equal(m.predict_proba(fd), base)
        np.testing.assert_array_equal(models.GlickoOnly().predict_proba(fd), base)

    def test_null_vector_rows_get_glicko_only(self):
        tr, va = with_signal(random_fold(800, 1)), with_signal(random_fold(200, 2))
        te = random_fold(300, 3, has_frac=0.5)
        for name in ("bilinear", "siamese"):
            m = models.make(name)
            m.train_cfg = models.TrainConfig(max_epochs=20, lr=0.05)
            m.fit(tr, va)
            p = m.predict_proba(te)
            np.testing.assert_array_equal(p[~te.has], data.sigmoid(te.offset[~te.has]))
            self.assertFalse(np.allclose(p[te.has], data.sigmoid(te.offset[te.has])))

    def test_fit_alpha_recovers_scale(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.5, 200_000)
        y = (rng.random(len(x)) < data.sigmoid(0.8 * x)).astype(float)
        self.assertAlmostEqual(data.fit_alpha(x, y), 0.8, delta=0.02)


# ------------------------------------------------------------------ synthetic world

N_PLAYERS = 40
START = date(2015, 1, 5)
WEEKS = 5 * 52


def synthetic(seed=0):
    """Weekly editions 2015-2019, 40 ATP players with latent style, ~half the matches charted."""
    rng = np.random.default_rng(seed)
    style = rng.normal(0, 1, (N_PLAYERS, 2))
    strength = rng.normal(0, 0.8, N_PLAYERS)
    m = elo.empty_matches()
    mc_rows = []
    mid = 0
    for w in range(WEEKS):
        d = (START + timedelta(weeks=w)).isoformat()
        surf = ("hard", "clay", "grass")[w % 3]
        for _ in range(8):
            a, b = rng.choice(N_PLAYERS, 2, replace=False)
            pa = data.sigmoid(strength[a] - strength[b] + 0.6 * (style[a, 0] * style[b, 1] - style[b, 0] * style[a, 1]))
            win, lose = (a, b) if rng.random() < pa else (b, a)
            mid += 1
            m.match_id.append(mid); m.winner.append(int(win)); m.loser.append(int(lose))
            m.start.append(d); m.status.append("completed"); m.tour_level.append(1)
            m.tour.append("atp"); m.surface.append(surf)
            if rng.random() < 0.5:
                mc_rows += [(mid, int(a), d), (mid, int(b), d)]
    mc_rows.sort(key=lambda r: (r[2], r[0], r[1]))
    n = len(mc_rows)
    mc = MatchCounts(match_id=np.array([r[0] for r in mc_rows]), player=np.array([r[1] for r in mc_rows]),
                     date=np.array([r[2] for r in mc_rows]), tour=np.array(["atp"] * n),
                     surface=np.array([""] * n))
    for s in SIDES:
        F = len(FEATURES[s])
        den = rng.integers(20, 80, (n, F)).astype(float)
        base = 0.5 + 0.15 * np.tanh(style[mc.player] @ rng.normal(0, 1, (2, F)))
        mc.num[s] = rng.binomial(den.astype(int), base).astype(float)
        mc.den[s] = den
        mc.points[s] = den[:, 0].astype(np.int64)
    pkey = {i: f"syn:{i:03d}" for i in range(N_PLAYERS)}
    flag = bytearray(1 if pkey[w] < pkey[l] else 0 for w, l in zip(m.winner, m.loser))
    # stand-in light layer: noisy pre-match strengths
    p_w = [float(data.sigmoid(strength[w] - strength[l] + rng.normal(0, 0.3)))
           for w, l in zip(m.winner, m.loser)]
    return m, p_w, flag, pkey, mc


HCFG = heavy.HeavyConfig(min_fit_players=15, min_fit_matches=2, refit_months=3)
HARNESS = harness.HarnessConfig(test_from="2018-07-01", test_to="2019-12-31", step_months=6, val_months=6,
                                tours=("atp",))


def world(seed=0, mutate=None):
    m, p_w, flag, pkey, mc = synthetic(seed)
    if mutate:
        mutate(m, p_w, mc)
        flag = bytearray(1 if pkey[w] < pkey[l] else 0 for w, l in zip(m.winner, m.loser))
    t = data.build_table(m, p_w, flag, pkey)
    sm = heavy.StyleModel(mc, HCFG)
    sm.fit_all(first="2015-01-01", last="2019-12-31")
    return t, StyleSource(sm)


class TestBuildTable(unittest.TestCase):
    def test_orientation_and_status_filter(self):
        m = elo.empty_matches()
        for i, (w, l, st) in enumerate([(1, 2, "completed"), (2, 1, "retired"), (1, 2, "walkover")]):
            m.match_id.append(i); m.winner.append(w); m.loser.append(l); m.start.append("2020-01-06")
            m.status.append(st); m.tour_level.append(1); m.tour.append("atp"); m.surface.append("carpet")
        pkey = {1: "a", 2: "b"}
        flag = bytearray(1 if pkey[w] < pkey[l] else 0 for w, l in zip(m.winner, m.loser))
        t = data.build_table(m, [0.7, 0.6, 0.5], flag, pkey)
        self.assertEqual(len(t), 2)                                  # walkover dropped
        np.testing.assert_array_equal(t.p1, [1, 1])                 # smaller key first, not winner
        np.testing.assert_array_equal(t.y, [1.0, 0.0])
        np.testing.assert_allclose(data.sigmoid(t.glicko_logit), [0.7, 0.4])
        np.testing.assert_array_equal(t.surface, [0, 0])            # carpet -> hard


class TestPointsBase(unittest.TestCase):
    """Milestone 10b: a points_v1 run as the base logit instead of Glicko-2."""

    def test_points_run_probabilities_feed_the_table(self):
        from array import array
        from tennis import points_rating as pr, ratingmodels
        m = elo.empty_matches()
        d = pr.PointsData(bytearray(), array("l"), array("l"), array("l"), array("l"), {})
        rows = [(1, 2, "2020-01-06", 60, 48, 60, 33), (2, 1, "2020-02-03", 60, 40, 60, 42),
                (1, 3, "2020-03-02", -1, -1, -1, -1), (3, 2, "2020-04-06", 60, 45, 60, 36)]
        for i, (w, l, st, *sp) in enumerate(rows):
            m.match_id.append(i + 1); m.winner.append(w); m.loser.append(l); m.start.append(st)
            m.status.append("completed"); m.surface.append("hard"); m.tour.append("atp"); m.tour_level.append(1)
            d.fmt.append(pr.FORMAT_INDEX[(3, "tb")])
            for a, v in zip((d.w_sp, d.w_sw, d.l_sp, d.l_sw), sp):
                a.append(v)
        m.extra["points"] = d
        spec, params = ratingmodels.get("points_v1"), pr.PointsParams(blend=0.7, unc=5.0)
        pw = data.base_p_winner(spec, params, m)
        expect = pr.predict(pr.run(m, params), params)
        np.testing.assert_allclose(list(pw), list(expect))
        pkey = {1: "a", 2: "b", 3: "c"}
        flag = bytearray(1 if pkey[w] < pkey[l] else 0 for w, l in zip(m.winner, m.loser))
        t = data.build_table(m, pw, flag, pkey, glicko_run_id=605, base_model_id="points_v1")
        self.assertEqual(t.base_model_id, "points_v1")
        p1 = np.where(t.y == 1, np.asarray(expect), 1 - np.asarray(expect))
        np.testing.assert_allclose(data.sigmoid(t.glicko_logit), p1, atol=1e-9)

    def test_alpha_rows_tour_uses_only_tour_level_rows(self):
        t, src = world()
        t.tour_level = t.tour_level.copy()
        t.tour_level[::2] = False                          # half the matches below tour level
        cfg = harness.HarnessConfig(**{**HARNESS.as_dict(), "tours": ("atp",), "alpha_rows": "tour"})
        f = harness.make_folds(t, src, cfg, log=lambda *_: None)[0]
        lo = harness.add_months(f.cutoff, -12 * cfg.alpha_years)
        sel = (t.tour == "atp") & t.tour_level & (t.date < f.cutoff) & (t.date >= lo)
        self.assertEqual(f.n_alpha, int(sel.sum()))
        self.assertEqual(f.alpha, data.fit_alpha(t.glicko_logit[sel], t.y[sel]))
        # run() keeps that alpha (only alpha_rows='train' refits it on the training rows)
        r = harness.run(t, src, cfg, ["glicko"], log=lambda *_: None, keep_models=False)
        self.assertEqual(r.folds[0].alpha, f.alpha)


class TestHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.src = world()
        cls.res = harness.run(cls.t, cls.src, HARNESS, ["glicko", "bilinear"], log=lambda *_: None)

    def test_folds_and_scores(self):
        self.assertEqual([f.test_from for f in self.res.folds], ["2018-07-01", "2019-01-01", "2019-07-01"])
        pooled = harness.metrics(self.t, self.res)
        self.assertEqual(pooled["glicko"]["tour"].n, 3 * 26 * 8)
        # the synthetic world has a real serve x return interaction; bilinear should find some of it
        self.assertLess(pooled["bilinear"]["tour_both"].log_loss, pooled["glicko"]["tour_both"].log_loss)
        text = harness.report(self.t, self.res)
        self.assertIn("tour_both", text)

    def test_fit_consistency(self):
        """One fit per side per fold, used for every train/val/test vector, and recorded."""
        for f in self.res.folds:
            self.assertEqual(set(f.fit_ids), set(SIDES))
            for s, fid in f.fit_ids.items():
                self.assertLessEqual(self.src.fit_by_id[fid].fit_date, f.cutoff)
            # test rows come later than newer fits exist; vectors must still use the fold's fit
            rows = f.test[:50]
            v = self.src.vectors("atp", self.t.p1[rows], self.t.date[rows], f.fit_ids)
            self.assertEqual(v.fit_ids, f.fit_ids)
            newer = self.src.model.fit_for("atp", "serve", self.t.date[rows[0]])
            self.assertNotEqual(newer.fit_id, f.fit_ids["serve"])        # a newer fit existed ...
            fit = self.src.fit_by_id[f.fit_ids["serve"]]
            i = int(np.flatnonzero(v.has)[0])
            tot = self.src.model.hist[("atp", "serve")].player_totals(int(self.t.p1[rows[i]]), self.t.date[rows[i]])
            np.testing.assert_allclose(v.x["serve"][i], fit.transform(tot[0], tot[1])[0])  # ... but wasn't used

    def test_bundles_record_fit_ids(self):
        import sqlite3
        from tennis import models_db
        with tempfile.TemporaryDirectory() as tmp:
            con = models_db.connect(Path(tmp) / "models.db")
            res = self.res
            # predictions need match keys; skip them here
            ids = harness.save(con, self.t, res, sweep_id="test", bundle_dir=Path(tmp) / "b",
                               save_predictions=False)
            rows = con.execute("SELECT run_id, test_from, fit_ids_json, alpha, path FROM matchup_bundle "
                               "WHERE run_id=?", (ids["bilinear"],)).fetchall()
            self.assertEqual(len(rows), len(res.folds))
            import json
            import torch
            for (rid, tf, fj, alpha, path), f in zip(rows, res.folds):
                self.assertEqual(json.loads(fj), f.fit_ids)
                blob = torch.load(path, weights_only=False)
                self.assertEqual(blob["fit_ids"], f.fit_ids)
                self.assertAlmostEqual(blob["alpha"], f.alpha)
            n = con.execute("SELECT COUNT(*) FROM run_metric WHERE run_id=? AND block='wf_test'",
                            (ids["glicko"],)).fetchone()[0]
            self.assertGreater(n, 0)
            con.close()


class TestLeakage(unittest.TestCase):
    def test_future_data_cannot_change_predictions(self):
        """Scramble every input dated on/after a date D inside the first test block (results,
        light-layer probabilities, charted counts). What the matchup layer adds (alpha and
        delta) for every block match dated <= D must not move. (Matches later in the block
        legitimately see charts from earlier in the block.)"""
        D = "2018-09-03"                                  # a Monday, 10th week of the block

        def mutate(m, p_w, mc):
            rng = np.random.default_rng(99)
            for i in range(len(m)):
                if m.start[i] >= D:
                    p_w[i] = float(rng.random())
                    if rng.random() < 0.5:
                        m.winner[i], m.loser[i] = m.loser[i], m.winner[i]
            late = mc.date >= D
            for s in SIDES:
                mc.num[s][late] = np.floor(mc.den[s][late] * rng.random(mc.den[s][late].shape))

        cfg = harness.HarnessConfig(**{**HARNESS.as_dict(), "tours": ("atp",), "test_to": "2018-12-31"})
        t1, src1 = world()
        t2, src2 = world(mutate=mutate)
        r1 = harness.run(t1, src1, cfg, ["glicko", "bilinear"], log=lambda *_: None)
        r2 = harness.run(t2, src2, cfg, ["glicko", "bilinear"], log=lambda *_: None)
        rows = r1.folds[0].test
        np.testing.assert_array_equal(rows, r2.folds[0].test)
        early = rows[t1.date[rows] <= D]
        self.assertGreater(len(early), 50)
        self.assertTrue(np.all(t1.p1[early] == t2.p1[early]))      # orientation is by key
        self.assertEqual(r1.folds[0].alpha, r2.folds[0].alpha)
        self.assertEqual(r1.folds[0].fit_ids, r2.folds[0].fit_ids)
        moved = 0
        for name in ("glicko", "bilinear"):
            # the light-layer input of matches dated D is itself scrambled: compare the
            # matchup layer's own contribution, logit(p) - alpha * glicko_logit
            d1 = np.log(r1.pred[name] / (1 - r1.pred[name])) - r1.folds[0].alpha * t1.glicko_logit
            d2 = np.log(r2.pred[name] / (1 - r2.pred[name])) - r2.folds[0].alpha * t2.glicko_logit
            np.testing.assert_allclose(d1[early], d2[early], atol=1e-6)
            late = rows[t1.date[rows] > D]
            moved += int(np.sum(np.abs(d1[late] - d2[late]) > 1e-6))
        self.assertGreater(moved, 0)                               # the scramble did reach later rows

    def test_vectors_use_only_earlier_charts(self):
        t, src = world()
        f = harness.make_folds(t, src, HARNESS, log=lambda *_: None)[0]
        harness.check_fold(t, src, f)
        for side in SIDES:
            hist = src.model.hist[("atp", side)]
            for p, d in zip(t.p1[f.test[:100]], t.date[f.test[:100]]):
                got = hist.player_totals(int(p), d)
                if got is not None:
                    self.assertLess(got[4], d)


class TestMonths(unittest.TestCase):
    def test_add_months(self):
        self.assertEqual(harness.add_months("2022-01-01", -6), "2021-07-01")
        self.assertEqual(harness.add_months("2022-11-01", 3), "2023-02-01")


if __name__ == "__main__":
    unittest.main()
