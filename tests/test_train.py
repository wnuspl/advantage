"""Milestone A2: the training harness (src/tennis/train, scripts/train.py)."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import sys
import unittest
from array import array
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from tennis import context, matches, models_db, points_rating as pr   # noqa: E402
from tennis.train import compare, context_fit, runs, scoring           # noqa: E402
from tennis.train.data import from_matches                             # noqa: E402


def synthetic(n_years=(2016, 2017, 2018, 2019, 2020), per_q=60, players=40, age_effect=0.0, seed=0):
    """Matches with serve stats drawn from true serve skills; with age_effect, a player under 22
    serves age_effect * (22 - age) logits better than their skill."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.3, players + 1)
    birth_year = rng.integers(1990, 2002, players + 1)
    dob = {k: f"{birth_year[k]}-06-01" for k in range(1, players + 1)}
    m = matches.empty_matches()
    d = pr.PointsData(bytearray(), array("l"), array("l"), array("l"), array("l"), {})
    eds = []
    i = 0
    for y in n_years:
        for q in range(4):
            start = f"{y}-{1 + 3 * q:02d}-01"
            for _ in range(per_q):
                a, b = rng.choice(np.arange(1, players + 1), 2, replace=False)
                xs = []
                for s, r in ((a, b), (b, a)):
                    age = y + (3 * q) / 12 - birth_year[s] - 0.5
                    xs.append(0.5 + skill[s] - skill[r] + age_effect * max(0.0, 22 - age))
                n = 70
                ka, kb = (int(rng.binomial(n, 1 / (1 + np.exp(-x)))) for x in xs)
                w, l, wsw, lsw = (a, b, ka, kb) if ka - kb + rng.normal(0, 3) > 0 else (b, a, kb, ka)
                i += 1
                m.match_id.append(i); m.winner.append(int(w)); m.loser.append(int(l))
                m.start.append(start); m.status.append("completed"); m.surface.append("hard")
                m.tour.append("atp" if i % 3 else "wta"); m.tour_level.append(1)
                d.fmt.append(pr.FORMAT_INDEX[(5 if i % 7 == 0 else 3, "tb")])
                for arr, v in zip((d.w_sp, d.w_sw, d.l_sp, d.l_sw), (n, wsw, n, lsw)):
                    arr.append(v)
                eds.append(y * 10 + q)
    m.extra["points"] = d
    m.extra["context"] = context.features(m, eds, dob)
    return m


class ScoringTest(unittest.TestCase):
    def test_known_values(self):
        p, y = np.array([0.8, 0.3, 0.5, 0.9]), np.array([1.0, 0.0, 1.0, 0.0])
        mt = scoring.metrics(p, y)
        want = -np.mean([np.log(0.8), np.log(0.7), np.log(0.5), np.log(0.1)])
        self.assertAlmostEqual(mt.log_loss, want)
        self.assertAlmostEqual(mt.brier, np.mean([0.04, 0.09, 0.25, 0.81]))
        self.assertAlmostEqual(mt.accuracy, (1 + 1 + 0.5 + 0) / 4)

    def test_paired_difference_and_blocks(self):
        rng = np.random.default_rng(3)
        y = (rng.random(4000) < 0.6).astype(float)
        good = np.where(y == 1, 0.7, 0.3)
        bad = np.full(4000, 0.6)
        blk = np.repeat(["2020-01-01", "2020-04-01"], 2000)
        mt = scoring.metrics(good, y, ref=bad, block=blk)
        self.assertLess(mt.d_ref, 0)
        self.assertGreater(mt.d_se, 0)
        self.assertEqual((mt.blocks_won, mt.blocks), (2, 2))
        self.assertEqual(scoring.metrics(bad, y, ref=bad).d_ref, 0.0)

    def test_recalibration_slope(self):
        rng = np.random.default_rng(4)
        lg = rng.normal(0, 1.5, 30000)
        y = (rng.random(30000) < 1 / (1 + np.exp(-lg))).astype(float)
        over = 1 / (1 + np.exp(-2 * lg))                       # overconfident by 2x
        self.assertAlmostEqual(scoring.metrics(over, y).cal_slope, 0.5, delta=0.03)
        self.assertLess(scoring.metrics(1 / (1 + np.exp(-lg)), y).ece, 0.01)

    def test_orientation_is_not_winner_first(self):
        p, y = scoring.oriented([0.7, 0.6], bytearray([1, 0]), [0, 1])
        np.testing.assert_allclose(p, [0.7, 0.4])
        np.testing.assert_array_equal(y, [1, 0])


class RunsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = from_matches(synthetic())
        cls.block = ("2019-01-01", "2020-12-31")

    def test_expand_grid(self):
        base = pr.PointsParams(q=7e-6)
        g = runs.expand_grid({"unc": [1, 3], "blend": [0.7]}, base)
        self.assertEqual(len(g), 2)
        self.assertTrue(all(p.q == 7e-6 and p.blend == 0.7 for p in g))
        self.assertEqual(len(runs.expand_grid([{"unc": [1, 3]}, {"unc": [3]}])), 2)
        with self.assertRaises(ValueError):
            runs.expand_grid({"nope": [1]})

    def test_scored_raw_and_calibrated(self):
        res = pr.run(self.prep.m, pr.PointsParams())
        s = runs.Scored(self.prep, res.p_winner, self.block)
        sc = s.scores()
        self.assertEqual(set(sc), set(runs.RAW_SCOPES + runs.CAL_SCOPES))
        self.assertEqual(sc["tour"].n, sc["tour_atp"].n + sc["tour_wta"].n)
        self.assertEqual(sc["tour_cal"].n, sc["tour"].n)
        self.assertEqual(sc["all"].n, sc["tour"].n)
        # raw log loss = mean -log P(winner) over the same matches
        want = -np.mean(np.log([res.p_winner[i] for i in s.idx]))
        self.assertAlmostEqual(sc["tour"].log_loss, want, places=12)
        self.assertNotEqual(sc["tour_cal"].log_loss, sc["tour"].log_loss)
        # paired against itself: zero difference
        self.assertEqual(s.scores(ref=s)["tour_cal"].d_ref, 0.0)

    def test_record_sweep_finalize(self):
        tmp = Path(tempfile.mkdtemp(prefix="tennis-train-"))
        con = models_db.connect(tmp / "models.db")
        try:
            grid = runs.expand_grid({"q": [1e-5, 2e-5], "blend": [0.5, 1.0]})
            out = runs.sweep(con, self.prep, grid, sweep_id="t", block=self.block, log=lambda s: None)
            self.assertEqual(len(out), 4)
            # blend is prediction-only: (q, 0.5) and (q, 1.0) come from one pass but differ
            self.assertNotEqual(out[0][1].log_loss, out[1][1].log_loss)
            board = runs.leaderboard(con, sweep_id="t")
            self.assertEqual(len(board), 4)
            self.assertEqual([r[4] for r in board], sorted(r[4] for r in board))
            rid = board[0][0]
            self.assertFalse(runs.has_test(con, rid))
            runs.finalize(con, self.prep, rid, block=self.block)
            self.assertTrue(runs.has_test(con, rid))
            with self.assertRaises(ValueError):
                runs.finalize(con, self.prep, rid, block=self.block)
            # the recorded params replay to the recorded val score
            _, p = runs.params_of(con, rid)
            again = runs.Scored(self.prep, pr.run(self.prep.m, p).p_winner, self.block).scores()
            self.assertAlmostEqual(again["tour_cal"].log_loss, board[0][4], places=12)
        finally:
            con.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dominance_bins_and_subset(self):
        from tennis.train import diagnose
        p = pr.PointsParams()
        res = pr.run(self.prep.m, p)
        s = runs.Scored(self.prep, res.p_winner, self.block)
        sub = s.subset(s.select("tour_atp"))
        self.assertEqual(len(sub.y), s.scores()["tour_atp"].n)
        dom = diagnose.serve_dominance(res, p, np.asarray(sub.idx))
        rows = diagnose.by_dominance(sub, dom, bins=4)
        self.assertEqual(sum(r["n"] for r in rows), len(sub.y))
        self.assertTrue(all(r["atp_share"] == 1.0 for r in rows))
        self.assertIn("slope cal", diagnose.format_rows(rows))

    def test_compare_table(self):
        sc = {lab: runs.Scored(self.prep, pr.run(self.prep.m, p).p_winner, self.block)
              for lab, p in (("a", pr.PointsParams()), ("b", pr.PointsParams(unc=0.0)))}
        tab = compare.table(sc)
        first = tab["tour"][0][1]
        self.assertEqual(first.d_ref, 0.0)
        self.assertEqual(len(tab["tour"]), 4)
        self.assertIn("[tour_wta]", compare.format_table(tab))


class ContextFitTest(unittest.TestCase):
    def test_recovers_an_age_effect_and_converges(self):
        prep = from_matches(synthetic(age_effect=0.08, per_q=80, seed=5))
        start = pr.PointsParams(v_init=0.02, q=7e-6, obs_var=0.003)
        k = context.FEATURES.index("age_lt22")
        r0 = pr.run(prep.m, start)
        ds, dr, ses, ser, n = context_fit.residual_fit(prep.m, r0, start, "2000-01-01", "2019-01-01")
        self.assertGreater(n, 500)
        self.assertGreater(ds[k], 0)                           # young servers outperform
        p, hist, res, sc = context_fit.fit(prep, start, iters=3, cutoff="2019-01-01",
                                           block=("2019-01-01", "2020-12-31"), log=lambda s: None)
        self.assertEqual(len(hist), 4)
        self.assertGreater(p.context_betas("atp")[0][k], 0)
        steps = []
        for it in range(3):
            a = np.array(hist[it]["params"]["cs_age_lt22"])
            b = np.array(hist[it + 1]["params"]["cs_age_lt22"])
            steps.append(abs(b - a))
        self.assertLess(steps[-1], steps[0])                    # the iteration settles
        # the fit used only matches before the cutoff
        none_ = context_fit.residual_fit(prep.m, r0, start, "2000-01-01", "2016-01-01")
        self.assertEqual(none_[4], 0)
        np.testing.assert_array_equal(none_[0], np.zeros(context.K))

    def test_per_tour(self):
        prep = from_matches(synthetic(seed=6))
        p, hist, _, _ = context_fit.fit(prep, pr.PointsParams(), iters=1, per_tour=True,
                                        block=("2019-01-01", "2020-12-31"), log=lambda s: None)
        self.assertEqual(p.ctx_per_tour, 1.0)
        self.assertFalse(np.array_equal(p.context_betas("atp")[0], p.context_betas("wta")[0]))


class TrainCliTest(unittest.TestCase):
    """scripts/train.py end to end on the fixture tennis.db."""

    @classmethod
    def setUpClass(cls):
        from test_build import Workspace
        cls.ws = Workspace()
        cls.ws.build()
        cls.models = cls.ws.tmp / "models.db"
        con = models_db.connect(cls.models)
        cls.base = models_db.start_run(con, "points_v1", pr.PointsParams(score_weight=1.5).as_dict())
        models_db.finish_run(con, cls.base)
        con.close()

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def cli(self, *argv) -> str:
        import train
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(train.main(["--models", str(self.models), "--data", str(self.ws.db), *argv]), 0)
        return buf.getvalue()

    def test_commands(self):
        out = self.cli("score", str(self.base))
        self.assertIn("tour_cal", out)
        self.cli("sweep", "--base", str(self.base), "--grid", json.dumps({"unc": [1, 2]}), "--name", "cli")
        self.assertIn("[tour_cal]", self.cli("show", "--name", "cli"))
        js = self.ws.tmp / "ctx.json"
        out = self.cli("context", "--base", str(self.base), "--set", "q=7e-6", "--iters", "1", "--record",
                       "--model-id", "points_v2", "--json", str(js))
        self.assertIn("recorded run", out)
        hist = json.loads(js.read_text(encoding="utf-8"))["history"]
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["params"]["q"], 7e-6)
        con = models_db.connect(self.models)
        try:
            rid, mid = con.execute("SELECT run_id, model_id FROM model_run ORDER BY run_id DESC LIMIT 1").fetchone()
        finally:
            con.close()
        self.assertEqual(mid, "points_v2")
        self.assertIn("[tour_atp]", self.cli("compare", str(self.base), str(rid)))
        self.assertIn("[wta]", self.cli("diagnose", str(self.base), "--bins", "2"))


if __name__ == "__main__":
    unittest.main()
