"""Milestone A1: the canonical prediction path (engine/predict.py), players and editions.

The key property: PointsPredictor on a final state reproduces, for the next match, exactly
the walk-forward prediction run() makes for it (logits, context, roll-up, shrink)."""
from __future__ import annotations

import json
import math
import pickle
import shutil
import sqlite3
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import calibrate, context, matches, models_db, points_rating as pr   # noqa: E402
from tennis.engine import MatchQuery, Predictor                                  # noqa: E402
from tennis.engine.editions import find_edition                                  # noqa: E402
from tennis.engine.predict import ModelState, PointsPredictor, build_model, live_from_score   # noqa: E402
from tennis.kalshi import LiveState                                             # noqa: E402

# run 790's context coefficients (docs/REBOOT.md 3.2)
CS = (-0.063, -0.023, 0.015, 0.016, 0.036, -0.099, 0.042, -0.012, -0.013)
CR = (-0.047, -0.036, -0.060, 0.001, -0.024, 0.037, 0.017, -0.019, 0.008)
P790 = pr.PointsParams(v_init=0.02, q=7e-6, obs_var=0.003, surf_var=0.003, result_weight=0.3,
                       score_weight=0.0, blend=0.7, unc=3.0).with_context(CS, CR)

# (winner, loser, start, status, surface, w_sp, w_sw, l_sp, l_sw, edition); sp -1 = no stats
ROWS = [
    (1, 2, "2019-01-07", "completed", "hard", 60, 45, 60, 36, 1),
    (3, 4, "2019-01-07", "completed", "hard", 70, 47, 66, 40, 1),
    (1, 3, "2019-01-07", "completed", "hard", 55, 40, 58, 38, 1),
    (2, 4, "2019-05-27", "completed", "clay", -1, -1, -1, -1, 2),
    (1, 4, "2019-05-27", "retired", "clay", 30, 20, 31, 17, 2),
    (3, 2, "2019-06-24", "completed", "grass", 80, 60, 75, 50, 3),
    (4, 1, "2019-11-04", "completed", "hard", 62, 44, 64, 41, 4),
    (2, 3, "2020-01-20", "completed", "hard", 90, 64, 88, 60, 5),
    (1, 2, "2020-01-20", "completed", "hard", 70, 50, 72, 49, 5),
]
EDITIONS = {1: ("2019-01-07", 10, "hard", 3), 2: ("2019-05-27", 20, "clay", 3), 3: ("2019-06-24", 30, "grass", 3),
            4: ("2019-11-04", 40, "hard", 3), 5: ("2020-01-20", 10, "hard", 3), 6: ("2020-03-02", 50, "hard", 3)}
DOB = {1: "1990-03-01", 2: "2001-07-15", 3: "1985-10-10", 4: "1996-01-01"}


def make(rows):
    m = matches.empty_matches()
    d = pr.PointsData(bytearray(), array("l"), array("l"), array("l"), array("l"), {})
    for i, r in enumerate(rows):
        m.match_id.append(i + 1); m.winner.append(r[0]); m.loser.append(r[1])
        m.start.append(r[2]); m.status.append(r[3]); m.surface.append(r[4])
        m.tour.append("atp"); m.tour_level.append(1)
        d.fmt.append(pr.FORMAT_INDEX[(5 if r[9] == 5 else 3, "tb")])
        for a, v in zip((d.w_sp, d.w_sw, d.l_sp, d.l_sw), r[5:9]):
            a.append(v)
    m.extra["points"] = d
    m.extra["context"] = context.features(m, [r[9] for r in rows], DOB)
    return m


def prefix(m, k):
    return matches.Matches(m.match_id[:k], m.winner[:k], m.loser[:k], m.start[:k], m.status[:k],
                           m.tour_level[:k], m.tour[:k], m.surface[:k], m.extra)


def predictor(rows, k, params=P790, **kw):
    m = make(rows)
    md = build_model(prefix(m, k), params, [r[9] for r in rows[:k]], DOB, EDITIONS, today="2020-02-01", run_id=1)
    return PointsPredictor(md, **kw), m


def query(row, **kw):
    bo = 5 if row[9] == 5 else 3
    return MatchQuery(tour="atp", player_a=row[0], player_b=row[1], on=row[2], surface=row[4],
                      best_of=bo, final="tb", edition_id=row[9], **kw)


class ReproducesRunTest(unittest.TestCase):
    def test_every_match_from_the_state_before_it(self):
        """For each k: replay matches [0, k), predict match k; must equal run()'s p_winner[k],
        whether k opens a new edition or is a later round of a running one."""
        full = pr.run(make(ROWS), P790)
        for k in range(1, len(ROWS)):
            pp, _ = predictor(ROWS, k, calibrated=False)
            pred = pp.predict(query(ROWS[k]))
            self.assertAlmostEqual(pred.p_a, full.p_winner[k], places=12, msg=f"match {k}")
            self.assertEqual(pred.p_raw, pred.p_a)

    def test_without_context_too(self):
        plain = pr.PointsParams(v_init=0.02, q=7e-6, blend=0.7, unc=3.0)
        full = pr.run(make(ROWS), plain)
        pp, _ = predictor(ROWS, 8, plain, calibrated=False)
        self.assertAlmostEqual(pp.predict(query(ROWS[8])).p_a, full.p_winner[8], places=12)
        self.assertEqual(pp.model.ctx_states, {})

    def test_context_is_applied(self):
        pp, _ = predictor(ROWS, 6, calibrated=False)
        d = pp.predict(query(ROWS[6])).detail
        self.assertNotEqual(d["ctx_a"], 0.0)
        no_ctx = pr.PointsParams(**{**P790.as_dict(), **{f"c{s}_{f}": 0.0 for s in "sr" for f in context.FEATURES}})
        pp2 = PointsPredictor(ModelState(**{**pp.model.__dict__, "params": no_ctx}), calibrated=False)
        self.assertAlmostEqual(pp2.predict(query(ROWS[6])).detail["x_a"], d["x_a"] - d["ctx_a"], places=12)


class PredictorTest(unittest.TestCase):
    def setUp(self):
        self.pp, _ = predictor(ROWS, len(ROWS))
        self.q = MatchQuery(tour="atp", player_a=1, player_b=3, on="2020-03-05", surface="hard",
                            best_of=3, edition_id=6)

    def test_is_a_predictor(self):
        self.assertIsInstance(self.pp, Predictor)
        self.assertEqual(self.pp.name, "points_v1/run1")

    def test_symmetric(self):
        a = self.pp.predict(self.q).p_a
        b = self.pp.predict(MatchQuery(**{**self.q.__dict__, "player_a": 3, "player_b": 1})).p_a
        self.assertAlmostEqual(a + b, 1.0, places=12)

    def test_calibration_alpha(self):
        self.pp.model.alphas = {"atp": 1.1, ("atp", 3): 0.8, ("atp", 5): 1.3}
        p = self.pp.predict(self.q)
        self.assertAlmostEqual(p.p_a, calibrate.apply(p.p_raw, 0.8), places=14)
        self.assertEqual(p.detail["alpha"], 0.8)
        p5 = self.pp.predict(MatchQuery(**{**self.q.__dict__, "best_of": 5}))
        self.assertEqual(p5.detail["alpha"], 1.3)
        raw = PointsPredictor(self.pp.model, calibrated=False).predict(self.q)
        self.assertEqual(raw.p_a, p.p_raw)
        self.assertTrue(raw.model.endswith("/raw"))

    def test_rating_day_is_the_edition_start(self):
        late = self.pp.predict(MatchQuery(**{**self.q.__dict__, "on": "2020-03-08"}))
        self.assertEqual(late.detail["rating_day"], "2020-03-02")
        self.assertEqual(late.p_a, self.pp.predict(self.q).p_a)
        new = self.pp.predict(MatchQuery(**{**self.q.__dict__, "edition_id": None}))
        self.assertEqual(new.detail["rating_day"], "2020-03-05")

    def test_surface_from_edition(self):
        q = MatchQuery(**{**self.q.__dict__, "surface": None})
        self.assertEqual(self.pp.predict(q).detail["surface"], "hard")
        q2 = MatchQuery(**{**self.q.__dict__, "surface": None, "edition_id": None})
        p2 = self.pp.predict(q2)
        self.assertIsNone(p2.detail["surface"])
        self.assertTrue(any("surface unknown" in n for n in p2.notes))

    def test_stale_and_unknown_players(self):
        self.assertFalse(self.pp.predict(self.q).stale)
        old = self.pp.predict(MatchQuery(**{**self.q.__dict__, "on": "2020-09-01", "edition_id": None}))
        self.assertTrue(old.stale)
        new = self.pp.predict(MatchQuery(**{**self.q.__dict__, "player_b": 99}))
        self.assertTrue(new.stale)
        self.assertIsNone(new.detail["last_b"])
        with self.assertRaises(ValueError):
            self.pp.predict(MatchQuery(**{**self.q.__dict__, "player_b": 1}))

    def test_not_point_in_time_note(self):
        p = self.pp.predict(MatchQuery(**{**self.q.__dict__, "on": "2019-12-01", "edition_id": None}))
        self.assertTrue(any("not point-in-time" in n for n in p.notes))
        self.assertFalse(any("not point-in-time" in n for n in self.pp.predict(self.q).notes))

    def test_deciding_rule(self):
        self.pp.model.edition_rule[5] = "tb12"
        self.pp.model.tournament_rule[10] = "tb12"
        self.assertEqual(self.pp.deciding_rule(5), "tb12")
        self.assertEqual(self.pp.deciding_rule(None, 10), "tb12")
        self.assertEqual(self.pp.deciding_rule(6), "tb")          # tournament 50: never seen
        self.assertEqual(self.pp.deciding_rule(None, 999), "tb")

    def test_pickle_round_trip(self):
        md = pickle.loads(pickle.dumps(self.pp.model))
        self.assertEqual(PointsPredictor(md).predict(self.q).p_a, self.pp.predict(self.q).p_a)


class InPlayTest(unittest.TestCase):
    def setUp(self):
        self.pp, _ = predictor(ROWS, len(ROWS))
        self.q = MatchQuery(tour="atp", player_a=1, player_b=3, on="2020-03-05", surface="hard", edition_id=6)
        self.pre = self.pp.predict(self.q)

    def live(self, ls):
        return self.pp.predict(MatchQuery(**{**self.q.__dict__, "live": ls}))

    def test_start_of_match_equals_raw_prematch(self):
        p = self.live(live_from_score("", "", True))
        self.assertTrue(p.in_play)
        self.assertAlmostEqual(p.p_a, self.pre.p_raw, delta=2e-4)       # grid interpolation only

    def test_v1_composition(self):
        """In play = in_match_form on each serve logit, then in_play_probability (v1 Pricer)."""
        ls = live_from_score("6-4 2-3", "30-40", False, serve_a=(31, 45), serve_b=(25, 42))
        p = self.live(ls)
        d = self.pre.detail
        xa, va = pr.in_match_form(d["x_a"], d["v_a"], 31, 45)
        xb, vb = pr.in_match_form(d["x_b"], d["v_b"], 25, 42)
        want = pr.in_play_probability(self.pp.model.state, xa, xb, va, vb, 3, "tb", ls.score(), False)
        self.assertEqual(p.p_a, want)
        self.assertEqual(p.p_raw, want)
        self.assertTrue(any("not calibrated" in n for n in p.notes))
        no_form = PointsPredictor(self.pp.model, live_form=False).predict(MatchQuery(**{**self.q.__dict__, "live": ls}))
        self.assertEqual(no_form.p_a, pr.in_play_probability(self.pp.model.state, d["x_a"], d["x_b"], d["v_a"],
                                                              d["v_b"], 3, "tb", ls.score(), False))

    def test_score_moves_the_probability(self):
        up = self.live(live_from_score("6-1 5-1", "40-0", True))
        down = self.live(live_from_score("1-6 1-5", "0-40", False))
        self.assertGreater(up.p_a, 0.95)
        self.assertLess(down.p_a, 0.05)

    def test_unusable_live_state_is_prematch_and_flagged(self):
        for ls in (LiveState(status="live", problem="server unknown"), LiveState(status="live", a_serving=None),
                   LiveState(status="not_started")):
            p = self.live(ls)
            self.assertFalse(p.in_play)
            self.assertEqual(p.p_a, self.pre.p_a)
            self.assertTrue(p.notes)

    def test_deciding_tiebreak_points(self):
        a = self.live(live_from_score("6-4 4-6 6-6", "5-5", True, deciding_tb_points=7))
        b = self.live(live_from_score("6-4 4-6 6-6", "5-5", True, deciding_tb_points=10))
        self.assertNotEqual(a.p_a, b.p_a)


class LoadFromDbTest(unittest.TestCase):
    """load(): models.db run -> replay of a fixture tennis.db -> cache -> same predictions."""

    @classmethod
    def setUpClass(cls):
        from test_build import Workspace
        cls.ws = Workspace()
        cls.ws.build()
        cls.models = cls.ws.tmp / "models.db"
        con = models_db.connect(cls.models)
        cls.run_id = models_db.start_run(con, "points_v1", P790.as_dict())
        con.execute("INSERT INTO model_run (model_id, params_json, started_at) VALUES ('elo_v1', '{}', 'x')")
        cls.elo_run = con.execute("SELECT MAX(run_id) FROM model_run").fetchone()[0]
        con.commit()
        con.close()
        cls.cache = cls.ws.tmp / "cache"
        cls.logs = []
        cls.pp = PointsPredictor.load(cls.run_id, today="2020-03-01", models_path=cls.models,
                                      data_path=cls.ws.db, cache_dir=cls.cache, log=cls.logs.append)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_players_and_prediction(self):
        res = self.pp.players
        fed, how = res.resolve("atp", "roger federer")
        nad, _ = res.resolve("atp", "Rafael NADAL")
        self.assertEqual(how, "exact")
        self.assertEqual(res.name(fed), "Roger Federer")
        self.assertEqual(res.resolve("atp", "R. Federer")[0], fed)          # last name + initial
        self.assertEqual(res.resolve("atp", "Nobody Here"), (None, "unresolved"))
        self.assertEqual(res.search("fede")[0][0], fed)
        p = self.pp.predict(MatchQuery("atp", fed, nad, on="2020-02-10", surface="hard"))
        self.assertTrue(0 < p.p_a < 1)
        self.assertEqual(p.model, f"points_v1/run{self.run_id}")

    def test_cache_round_trip(self):
        self.assertTrue(any("replaying" in s for s in self.logs))
        logs = []
        again = PointsPredictor.load(self.run_id, today="2020-03-01", models_path=self.models,
                                     data_path=self.ws.db, cache_dir=self.cache, log=logs.append)
        self.assertTrue(any("cached replay" in s for s in logs))
        fed = self.pp.players.resolve("atp", "Roger Federer")[0]
        nad = self.pp.players.resolve("atp", "Rafael Nadal")[0]
        q = MatchQuery("atp", fed, nad, on="2020-02-10", surface="clay")
        self.assertEqual(again.predict(q).p_a, self.pp.predict(q).p_a)
        # a new calibration quarter invalidates the cache
        logs.clear()
        PointsPredictor.load(self.run_id, today="2020-04-01", models_path=self.models, data_path=self.ws.db,
                             cache_dir=self.cache, log=logs.append)
        self.assertTrue(any("replaying" in s for s in logs))

    def test_as_of_replays_a_prefix(self):
        early = PointsPredictor.load(self.run_id, as_of="2020-01-15", models_path=self.models,
                                     data_path=self.ws.db, cache_dir=None, log=lambda s: None)
        self.assertLess(early.model.n_matches, self.pp.model.n_matches)
        self.assertLess(early.model.last_start, "2020-01-15")
        self.assertIn("@2020-01-15", early.name)
        self.assertTrue(all(v[0] < "2020-01-15" for v in early.model.editions.values()))

    def test_rejects_other_models(self):
        with self.assertRaises(ValueError):
            PointsPredictor.load(self.elo_run, models_path=self.models, data_path=self.ws.db, cache_dir=None)
        with self.assertRaises(ValueError):
            PointsPredictor.load(10 ** 6, models_path=self.models, data_path=self.ws.db, cache_dir=None)

    def test_find_edition(self):
        con = sqlite3.connect(self.ws.db)
        try:
            ao = find_edition(con, "atp", "australian", "2020-01-25", alias="main")
            self.assertTrue(ao["running"])
            self.assertEqual(ao["surface"], "hard")
            self.assertIsNotNone(ao["edition_id"])
            later = find_edition(con, "atp", "Australian Open", "2020-06-01", alias="main")
            self.assertFalse(later["running"])
            self.assertIsNone(later["edition_id"])
            self.assertEqual(later["latest_edition_id"], ao["edition_id"])
            self.assertIsNone(find_edition(con, "atp", "australian", "2019-01-01", alias="main"))
            self.assertIsNone(find_edition(con, "atp", "no such event", "2020-06-01", alias="main"))
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
