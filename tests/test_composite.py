"""Composite model (light + heavy + matchup): end-to-end predictions rebuilt from stored
bundles reproduce the walk-forward harness; evaluation metrics. Synthetic data only."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tennis import composite, composite_eval, glicko2, heavy, models_db   # noqa: E402
from tennis.matchup import data, harness                                  # noqa: E402
from tennis.matchup.data import StyleSource                               # noqa: E402

import test_matchup as tm                                                 # noqa: E402

GP = glicko2.Glicko2Params(tau=0.0, rd_init=200.0, period_days=4.0, blend=0.3)
MODELS = ["glicko", "bilinear", "siamese", "lgbm"]


class TestEndToEnd(unittest.TestCase):
    """The harness run on a synthetic world with a REAL Glicko-2 light layer, saved to a
    temporary models.db; the composite predictor then rebuilds predictions from the saved
    bundles and a Glicko state replayed to each date."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        m, _, flag, pkey, mc = tm.synthetic()
        cls.m = m
        res = glicko2.run(m, GP)
        cls.p_winner = res.p_winner
        cls.t = data.build_table(m, res.p_winner, flag, pkey)
        sm = heavy.StyleModel(mc, tm.HCFG)
        sm.fit_all(first="2015-01-01", last="2019-12-31")
        cls.src = StyleSource(sm)
        cls.res = harness.run(cls.t, cls.src, tm.HARNESS, MODELS, log=lambda *_: None)
        cls.con = models_db.connect(tmp / "models.db")
        cls.run_ids = harness.save(cls.con, cls.t, cls.res, sweep_id="syn", bundle_dir=tmp / "b",
                                   save_predictions=False)
        cls.row_of = {int(mid): i for i, mid in enumerate(cls.t.match_id)}

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_glicko_prob_matches_run(self):
        """Light layer: state replayed before D gives the run's pre-match probability for
        each player's first match of the editions starting on D."""
        for d in ("2016-03-07", "2018-08-06", "2019-05-06"):
            st = composite.glicko_state(self.m, GP, d)
            idx = composite.first_appearances(self.m, d)
            self.assertGreater(len(idx), 3)
            for i in idx:
                g = composite.glicko_prob(st, self.m.winner[i], self.m.loser[i], d, self.m.surface[i])
                self.assertAlmostEqual(g["p"], self.p_winner[i], places=10)

    def test_unrated_player_starts_at_init(self):
        st = composite.glicko_state(self.m, GP, "2016-01-04")
        g = composite.glicko_prob(st, 10_000, 10_001, "2016-01-04", "clay")
        self.assertAlmostEqual(g["p"], 0.5)
        self.assertFalse(g["a"]["rated"])

    def test_composite_reproduces_harness(self):
        for name in MODELS:
            bundles, params = composite.load_bundles(self.con, self.run_ids[name])
            self.assertEqual(params["model"], name)
            checked = moved = 0
            for d in ("2018-07-02", "2018-11-05", "2019-02-04", "2019-09-02"):
                pred = composite.CompositePredictor(composite.glicko_state(self.m, GP, d), self.src, bundles,
                                                    state_as_of=d)
                idx = composite.first_appearances(self.m, d)
                rows = [self.row_of[int(self.m.match_id[i])] for i in idx]
                got = pred.predict_many(self.t.p1[rows], self.t.p2[rows], [d] * len(rows),
                                        [self.m.surface[i] for i in idx], "atp")
                for r, g in zip(rows, got):
                    self.assertAlmostEqual(g.p, self.res.pred[name][r], places=6, msg=f"{name} {d}")
                    self.assertAlmostEqual(g.glicko_logit, self.t.glicko_logit[r], places=9)
                    fold = self.res.folds[self.res.fold_of[r]]
                    self.assertEqual(g.bundle["fit_ids"], fold.fit_ids)
                    self.assertAlmostEqual(g.alpha, fold.alpha)
                    if not g.has_vectors:
                        self.assertEqual(g.delta, 0.0)
                    moved += abs(g.delta) > 1e-6
                    checked += 1
            self.assertGreater(checked, 10)
            if name != "glicko":
                self.assertGreater(moved, 0, name)

    def test_antisymmetric_end_to_end(self):
        bundles, _ = composite.load_bundles(self.con, self.run_ids["bilinear"])
        d = "2019-03-04"
        pred = composite.CompositePredictor(composite.glicko_state(self.m, GP, d), self.src, bundles)
        ab = pred.predict_many(range(0, 10), range(10, 20), [d] * 10, ["hard"] * 10, "atp")
        ba = pred.predict_many(range(10, 20), range(0, 10), [d] * 10, ["hard"] * 10, "atp")
        for x, y in zip(ab, ba):
            self.assertAlmostEqual(x.p + y.p, 1.0, places=6)

    def test_bundle_choice(self):
        bundles, _ = composite.load_bundles(self.con, self.run_ids["glicko"])
        b = bundles["atp"]
        self.assertEqual(composite.pick_bundle(b, "2018-07-01").test_from, "2018-07-01")
        self.assertEqual(composite.pick_bundle(b, "2019-01-15").test_from, "2019-01-01")
        self.assertEqual(composite.pick_bundle(b, "2030-01-01").test_from, b[-1].test_from)   # live: latest
        with self.assertRaises(ValueError):
            composite.pick_bundle(b, "2017-01-01")

    def test_state_must_precede_dates(self):
        bundles, _ = composite.load_bundles(self.con, self.run_ids["glicko"])
        pred = composite.CompositePredictor(composite.glicko_state(self.m, GP, "2019-01-07"), self.src, bundles,
                                            state_as_of="2019-01-07")
        with self.assertRaises(ValueError):
            pred.predict(1, 2, "2018-12-31", "hard", "atp")


class TestFirstAppearances(unittest.TestCase):
    def test_skips_players_already_seen_that_day(self):
        from tennis import elo
        m = elo.empty_matches()
        for i, (w, l, s) in enumerate([(1, 2, "2020-01-06"), (3, 4, "2020-01-13"), (1, 3, "2020-01-13"),
                                       (5, 6, "2020-01-13"), (7, 8, "2020-01-13")]):
            m.match_id.append(i); m.winner.append(w); m.loser.append(l); m.start.append(s)
            m.status.append("walkover" if w == 7 else "completed")
            m.tour_level.append(1); m.tour.append("atp"); m.surface.append("hard")
        self.assertEqual(composite.first_appearances(m, "2020-01-13"), [1, 3])


class TestEvalMetrics(unittest.TestCase):
    def test_recalibration_and_ece(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1.5, 200_000)
        y = (rng.random(len(z)) < data.sigmoid(z)).astype(float)
        a, b = composite_eval.recalibration(data.sigmoid(z), y)
        self.assertAlmostEqual(a, 0.0, delta=0.02)
        self.assertAlmostEqual(b, 1.0, delta=0.02)
        a, b = composite_eval.recalibration(data.sigmoid(2 * z), y)       # overconfident
        self.assertAlmostEqual(b, 0.5, delta=0.02)
        self.assertLess(composite_eval.ece(data.sigmoid(z), y), 0.01)
        self.assertGreater(composite_eval.ece(data.sigmoid(2 * z), y), 0.05)

    def test_paired_difference_and_verdict(self):
        rng = np.random.default_rng(1)
        n = 40_000
        z = rng.normal(0, 1, n)
        y = (rng.random(n) < data.sigmoid(z)).astype(float)
        good, weak = data.sigmoid(z), data.sigmoid(0.5 * z)
        dates = np.array(["2022-0%d-01" % (1 + 3 * (i % 3)) for i in range(n)])
        r = composite_eval.Rows(match_key=np.arange(n).astype(str), date=dates, y=y,
                                p={"weak": weak, "good": good, "over": data.sigmoid(3 * z)},
                                tour=np.array(["atp"] * n), tour_level=np.ones(n, bool),
                                surface=np.array(["hard"] * n), has=np.arange(n) % 2 == 0,
                                n_min=np.full(n, 20), block=composite_eval.block_of(dates))
        res = composite_eval.evaluate(r, "weak")
        m = res["tour"]["good"]
        self.assertLess(m.d_ref, 0)
        self.assertLess(m.d_ref / m.d_se, -5)
        self.assertEqual((m.blocks_won, m.blocks), (3, 3))
        self.assertEqual(res["tour"]["weak"].d_ref, 0.0)
        self.assertTrue(composite_eval.verdict(res, "good", "weak").startswith("better"))
        self.assertTrue(composite_eval.verdict(res, "weak", "good").startswith("not better"))
        self.assertEqual(res["tour_both"]["good"].n, n // 2)
        text = composite_eval.report(r, res, "weak", {"weak": 1, "good": 2, "over": 3}, min_charted=10,
                                     title="t")
        self.assertIn("tour_both", text)

    def test_load_predictions_inner_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            con = models_db.connect(Path(tmp) / "models.db")
            r1 = models_db.start_run(con, "x", {})
            r2 = models_db.start_run(con, "y", {})
            rows = [("k1", "2022-01-03", "a", "b", 0.6, 1), ("k2", "2022-01-03", "a", "c", 0.3, 0),
                    ("k3", "2021-01-04", "a", "c", 0.3, 0)]
            con.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)", [(r1, *r) for r in rows])
            con.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)",
                            [(r2, "k1", "2022-01-03", "a", "b", 0.7, 1), (r2, "k9", "2022-01-03", "a", "b", 0.7, 1)])
            keys, dates, y, p, k1, k2 = composite_eval.load_predictions(con, {"x": r1, "y": r2},
                                                                        "2022-01-01", "2022-12-31")
            self.assertEqual(list(keys), ["k1"])
            self.assertEqual(list(p["y"]), [0.7])
            con.execute("UPDATE prediction SET p1_won=0 WHERE run_id=? AND match_key='k1'", (r2,))
            with self.assertRaises(ValueError):
                composite_eval.load_predictions(con, {"x": r1, "y": r2}, "2022-01-01", "2022-12-31")
            con.close()


if __name__ == "__main__":
    unittest.main()
