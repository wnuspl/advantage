"""Milestone A5: saved state at cutoffs; incremental replay equals full replay."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import context, matches, models_db, points_rating as pr          # noqa: E402
from tennis.engine import MatchQuery                                        # noqa: E402
from tennis.engine import state as st_mod                                   # noqa: E402
from tennis.engine.predict import PointsPredictor, model_from_db            # noqa: E402
from test_predict import P790                                               # noqa: E402

PARAMS = pr.PointsParams(**{**P790.as_dict(), "score_weight": 1.5})
CUTOFFS = ["2020-01-01", "2020-01-15"]   # fixture editions start 2020-01-06 and 2020-01-20


def same_ctx(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(all(getattr(a[k], f) == getattr(b[k], f) for f in st_mod._P_FIELDS) for k in a)


class StateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_build import Workspace
        cls.ws = Workspace()
        cls.ws.build()
        cls.models = cls.ws.tmp / "models.db"
        con = models_db.connect(cls.models)
        cls.run_id = models_db.start_run(con, "points_v1", PARAMS.as_dict())
        con.close()
        cls.state_dir = cls.ws.tmp / "state"
        cls.con = sqlite3.connect(cls.ws.db)
        cls.con.execute("ATTACH DATABASE ? AS data", (str(cls.ws.db),))
        st_mod.build(cls.con, cls.run_id, PARAMS, CUTOFFS, out_dir=cls.state_dir, log=lambda s: None)
        # full replay reference
        m = matches.load_matches(cls.con)
        pr.attach(cls.con, m)
        cls.full_m = m
        cls.full = pr.run(m, PARAMS)
        edmap = dict(cls.con.execute("SELECT match_id, edition_id FROM data.matches"))
        cls.full_ctx = context.player_states(m, [edmap[i] for i in m.match_id])

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.ws.cleanup()

    def test_fixture_spans_the_cutoffs(self):
        starts = self.full_m.start
        self.assertTrue(any(CUTOFFS[0] <= s < CUTOFFS[1] for s in starts) and any(s >= CUTOFFS[1] for s in starts))
        self.assertEqual([p.stem for p in st_mod.snapshots(self.run_id, self.state_dir)], CUTOFFS)

    def test_resume_equals_full_replay(self):
        for path in st_mod.snapshots(self.run_id, self.state_dir):
            snap = st_mod.latest_valid(self.con, self.run_id, PARAMS, before=path.stem, state_dir=self.state_dir,
                                       log=lambda s: None)
            self.assertEqual(snap.cutoff, path.stem)
            w, _, _ = st_mod.resume(self.con, snap)
            s, f = w.res.state, self.full.state
            for fld in ("serve", "ret", "vs", "vr", "last", "count", "sserve", "sret", "svs", "svr",
                        "slast", "scount", "base"):
                self.assertEqual(getattr(s, fld), getattr(f, fld), f"{fld} after resuming at {snap.cutoff}")
            self.assertTrue(same_ctx(w.ctx, self.full_ctx))
            # the window's predictions are the full run's, match for match
            k = len(self.full_m) - len(w.m)
            self.assertEqual(list(w.res.p_winner), list(self.full.p_winner[k:]))

    def test_model_from_state_equals_full_build(self):
        a = model_from_db(self.con, self.run_id, "points_v1", PARAMS, as_of=None, today="2026-11-15",
                          state_dir=self.state_dir, log=lambda s: None)
        b = model_from_db(self.con, self.run_id, "points_v1", PARAMS, as_of=None, today="2026-11-15",
                          use_state=False)
        for fld in ("alphas", "edition_rule", "tournament_rule", "last_start", "n_matches", "editions", "dob"):
            self.assertEqual(getattr(a, fld), getattr(b, fld), fld)
        self.assertEqual(a.state.serve, b.state.serve)
        self.assertTrue(same_ctx(a.ctx_states, b.ctx_states))
        pa, pb = PointsPredictor(a), PointsPredictor(b)
        fed = pa.players.resolve("atp", "Roger Federer")[0]
        nad = pa.players.resolve("atp", "Rafael Nadal")[0]
        q = MatchQuery("atp", fed, nad, on="2026-11-20", surface="clay")
        self.assertEqual(pa.predict(q).p_a, pb.predict(q).p_a)

    def test_as_of_uses_an_earlier_snapshot(self):
        a = model_from_db(self.con, self.run_id, "points_v1", PARAMS, as_of="2020-03-01", today="2020-03-01",
                          state_dir=self.state_dir, log=lambda s: None)
        b = model_from_db(self.con, self.run_id, "points_v1", PARAMS, as_of="2020-03-01", today="2020-03-01",
                          use_state=False)
        self.assertEqual(a.state.serve, b.state.serve)
        self.assertEqual(a.n_matches, b.n_matches)

    def test_changed_history_makes_a_snapshot_stale(self):
        logs = []
        con = sqlite3.connect(":memory:")
        con.execute("ATTACH DATABASE ? AS data", (str(self.ws.db),))
        try:
            mid = con.execute("""SELECT m.match_id FROM data.matches m JOIN data.tournament_edition e USING (edition_id)
                                 WHERE e.start_date < ? AND m.score IS NOT NULL LIMIT 1""", (CUTOFFS[1],)).fetchone()[0]
            old = con.execute("SELECT score FROM data.matches WHERE match_id=?", (mid,)).fetchone()[0]
            con.execute("UPDATE data.matches SET score = ? WHERE match_id = ?", (old + " ", mid))
            try:
                snap = st_mod.latest_valid(con, self.run_id, PARAMS, state_dir=self.state_dir, log=logs.append)
                self.assertTrue(any("stale" in s for s in logs))
                self.assertTrue(snap is None or snap.cutoff <= CUTOFFS[0])
            finally:
                con.execute("UPDATE data.matches SET score = ? WHERE match_id = ?", (old, mid))
                con.commit()
        finally:
            con.close()
        self.assertEqual(st_mod.latest_valid(self.con, self.run_id, PARAMS, state_dir=self.state_dir,
                                             log=lambda s: None).cutoff, CUTOFFS[1])

    def test_stable_keys_survive_new_ids(self):
        snap = st_mod.latest_valid(self.con, self.run_id, PARAMS, state_dir=self.state_dir, log=lambda s: None)
        keys = st_mod.key_maps(self.con)
        shifted = {kind: {i + 1000: k for i, k in mp.items()} for kind, mp in keys.items()}
        st, ctx, rl, er = st_mod.from_stable(snap, shifted)
        st0, ctx0, rl0, er0 = st_mod.from_stable(snap, keys)
        self.assertEqual(st.serve, {p + 1000: v for p, v in st0.serve.items()})
        self.assertEqual(set(ctx), {p + 1000 for p in ctx0})
        self.assertTrue(all((ctx[p + 1000].ed is None) == (ctx0[p].ed is None) for p in ctx0))
        self.assertEqual(er, {e + 1000: r for e, r in er0.items()})

    def test_predictor_load_resumes(self):
        logs = []
        pp = PointsPredictor.load(self.run_id, today="2026-11-15", models_path=self.models, data_path=self.ws.db,
                                  cache_dir=None, state_dir=self.state_dir, log=logs.append)
        self.assertTrue(any("resuming from saved state 2020-01-15" in s for s in logs))
        full = PointsPredictor.load(self.run_id, today="2026-11-15", models_path=self.models, data_path=self.ws.db,
                                    cache_dir=None, use_state=False, log=lambda s: None)
        self.assertEqual(pp.model.state.serve, full.model.state.serve)


if __name__ == "__main__":
    unittest.main()
