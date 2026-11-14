"""Heavy layer: shrinkage, point-in-time correctness, deterministic refits, get_style API."""
from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import heavy, models_db, style_features as sf   # noqa: E402
from test_build import Workspace                           # noqa: E402

DATES = ["2019-01-07", "2019-03-04", "2019-05-27", "2019-08-26", "2020-01-20", "2020-05-25"]


def synthetic_counts(n_players: int = 60, seed: int = 7, tour: str = "atp") -> sf.MatchCounts:
    """Each player plays one charted match on every date in DATES, with their own true rates."""
    rng = np.random.RandomState(seed)
    rows, meta, mid = {}, {}, 0
    true = {s: rng.uniform(0.2, 0.8, size=(n_players, len(sf.FEATURES[s]))) for s in sf.SIDES}
    for d in DATES:
        for p in range(n_players):
            mid += 1
            meta[mid] = (d, tour, "hard")
            t = sf.Tally()
            for s in sf.SIDES:
                den = rng.randint(20, 80, size=len(sf.FEATURES[s]))
                t.den[s] = list(den)
                t.num[s] = list(rng.binomial(den, true[s][p]))
                t.points[s] = int(den.max())
            rows[(mid, 1000 + p)] = t
    return sf.build_match_counts(rows, meta)


CFG = heavy.HeavyConfig(min_fit_players=20, min_fit_matches=2)


class ShrinkageTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        true = rng.beta(20, 30, size=300)                 # tour mean 0.4, sd ~0.07
        den = np.full(300, 400.0)
        num = rng.binomial(400, true).astype(float)
        # one thin player far above the mean, one thick player far above the mean
        self.num = np.append(num, [5.0, 9000.0])[:, None]
        self.den = np.append(den, [5.0, 10000.0])[:, None]

    def test_prior_recovers_beta(self):
        mean, k = sf.estimate_priors(self.num, self.den)
        self.assertAlmostEqual(mean[0], 0.4, delta=0.02)
        self.assertGreater(k[0], 25)                      # true a + b = 50
        self.assertLess(k[0], 90)

    def test_thin_player_near_mean_thick_player_near_raw(self):
        mean, k = sf.estimate_priors(self.num, self.den)
        x = sf.shrink(self.num, self.den, mean, k)[:, 0]
        thin, thick = x[-2], x[-1]                        # raw rates 1.0 and 0.9
        self.assertLess(abs(thin - mean[0]), 0.1 * abs(1.0 - mean[0]) + 0.02)
        self.assertAlmostEqual(thick, 0.9, delta=0.01)

    def test_zero_denominator_is_tour_mean(self):
        x = sf.shrink(np.zeros((1, 2)), np.zeros((1, 2)), np.array([0.3, 0.6]), np.array([10.0, 1.0]))
        np.testing.assert_allclose(x[0], [0.3, 0.6])

    def test_no_between_player_variance_gives_max_k(self):
        num = np.full((50, 1), 40.0); den = np.full((50, 1), 100.0)
        _, k = sf.estimate_priors(num, den, k_max=1e4)
        self.assertEqual(k[0], 1e4)


class PointInTimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mc = synthetic_counts()

    def test_totals_use_only_rows_strictly_before(self):
        h = sf.History(self.mc, "atp", "serve")
        for as_of in DATES + ["2030-01-01"]:
            tot = h.totals(as_of)
            if len(tot.player):
                self.assertLess(tot.max_date, as_of)
            self.assertTrue(all(n == sum(d < as_of for d in DATES) for n in tot.n_matches))
        self.assertIsNone(h.player_totals(1000, DATES[0]))           # nothing before the first

    def test_no_input_row_on_or_after_as_of(self):
        """Every row feeding a vector (and its fit) is dated strictly before as_of."""
        m = heavy.StyleModel(self.mc, CFG)
        m.fit_all(first="2019-01-01", last="2020-12-31")
        for as_of in ["2019-05-27", "2019-06-01", "2020-01-20", "2020-07-01"]:
            for side in sf.SIDES:
                sv = m.style(1000, as_of, side)
                fit = m.fit_for("atp", side, as_of)
                self.assertIsNotNone(sv)
                self.assertLess(fit.data_to, as_of)
                self.assertLessEqual(fit.fit_date, as_of)
                rows = (self.mc.player == 1000) & (self.mc.date < as_of)
                self.assertEqual(sv.n_matches, int(rows.sum()))

    def test_future_data_cannot_change_a_vector(self):
        a = heavy.StyleModel(self.mc, CFG)
        a.fit_all(first="2019-01-01", last="2020-12-31")
        mc2 = synthetic_counts()
        late = mc2.date >= "2020-01-20"
        for s in sf.SIDES:                      # rewrite everything on/after the cut-off
            mc2.num[s][late] = 0
            mc2.den[s][late] = 500
        b = heavy.StyleModel(mc2, CFG)
        b.fit_all(first="2019-01-01", last="2020-12-31")
        for side in sf.SIDES:
            for p in (1000, 1017, 1059):
                np.testing.assert_array_equal(a.style(p, "2020-01-20", side).vector,
                                              b.style(p, "2020-01-20", side).vector)
            self.assertFalse(np.array_equal(a.style(1000, "2020-01-21", side).vector,
                                            b.style(1000, "2020-01-21", side).vector))

    def test_fit_dates(self):
        self.assertEqual(heavy.fit_dates("2019-02-15", "2019-12-31", 3),
                         ["2019-04-01", "2019-07-01", "2019-10-01"])
        self.assertEqual(heavy.fit_dates("2019-11-01", "2020-02-01", 1),
                         ["2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01"])


class FitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mc = synthetic_counts()

    def test_refits_are_deterministic(self):
        runs = []
        for _ in range(2):
            m = heavy.StyleModel(synthetic_counts(), CFG)
            m.fit_all(first="2019-01-01", last="2020-12-31")
            runs.append(np.vstack([m.style(p, "2020-07-01", s).vector
                                   for p in range(1000, 1060) for s in ("serve",)]))
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_sign_convention_and_component_count(self):
        m = heavy.StyleModel(self.mc, CFG)
        m.fit_all(first="2019-01-01", last="2020-12-31")
        for fits in m.fits.values():
            for f in fits:
                L = f.loadings()
                top = L[np.argmax(np.abs(L), axis=0), np.arange(L.shape[1])]
                self.assertTrue(np.all(top > 0))
                cum = np.cumsum(f.explained)
                self.assertLessEqual(f.n_components, CFG.max_components)
                if f.n_components < CFG.max_components:
                    self.assertGreaterEqual(cum[-1], CFG.variance - 1e-9)
                    self.assertTrue(len(cum) == 1 or cum[-2] < CFG.variance)

    def test_too_few_players_means_no_fit(self):
        m = heavy.StyleModel(synthetic_counts(n_players=10), CFG)
        self.assertEqual(m.fit_all(first="2019-01-01", last="2020-12-31"), [])
        self.assertIsNone(m.style(1000, "2020-07-01", "serve"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            heavy.HeavyConfig(max_components=11)
        self.assertNotEqual(heavy.HeavyConfig().config_hash,
                            heavy.HeavyConfig(refit_months=1).config_hash)


class ApiTest(unittest.TestCase):
    """End to end on the fixture build: fit, store, materialize, read back."""

    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.ws.build()
        cls.style_dir = Path(tempfile.mkdtemp(prefix="style-test-", dir=cls.ws.tmp))
        cls.models = cls.ws.tmp / "models.db"
        cls.cfg = heavy.HeavyConfig(min_fit_players=2, min_fit_matches=1)
        with closing(models_db.connect(cls.models)) as con:
            models_db.attach_data(con, cls.ws.db)
            cls.mc = heavy.load_counts(con, built_at="fixture", style_dir=cls.style_dir)
            model = heavy.StyleModel(cls.mc, cls.cfg)
            fits = model.fit_all(first="2020-01-01", last="2020-12-31")
            for f in fits:
                heavy.save_fit(con, f, cls.cfg, style_dir=cls.style_dir)
            pkeys = models_db.player_keys(con)
            cls.keys = sorted(pkeys[int(p)] for p in set(cls.mc.player))
            cls.first = min(str(d) for d in cls.mc.date)
            cls.n_rows = heavy.materialize(con, model, [(int(p), "2020-07-01") for p in set(cls.mc.player)],
                                           pkeys)

    @classmethod
    def tearDownClass(cls):
        heavy._default.clear()
        cls.ws.cleanup()

    def get(self, key, as_of, side="serve"):
        return heavy.get_style(key, as_of, side, cfg=self.cfg, models_path=self.models,
                               data_path=self.ws.db, style_dir=self.style_dir)

    def test_fixture_charted_matches(self):
        self.assertEqual(len(set(self.mc.match_id)), 2)     # Djokovic-Nadal, Groth-Bedene
        self.assertEqual(len(self.keys), 4)
        self.assertGreater(self.n_rows, 0)

    def test_none_without_charted_data_before_as_of(self):
        for key in self.keys:
            self.assertIsNone(self.get(key, self.first))           # the match itself is not before
            self.assertIsNone(heavy.get_style_vector(
                key, "2000-01-01", "return", cfg=self.cfg, models_path=self.models,
                data_path=self.ws.db, style_dir=self.style_dir))
        self.assertIsNone(self.get("sackmann_atp:does-not-exist", "2020-07-01"))

    def test_materialized_row_equals_on_demand(self):
        for key in self.keys:
            stored = self.get(key, "2020-07-01")                    # materialized row
            fresh = self.get(key, "2020-07-02")                     # computed on demand
            self.assertIsNotNone(stored)
            self.assertEqual((stored.n_matches, stored.fit_id), (1, fresh.fit_id))
            self.assertGreaterEqual(stored.n_points, 0)          # Bedene never served
            np.testing.assert_allclose(stored.vector, fresh.vector, atol=1e-12)

    def test_fit_tables(self):
        with closing(models_db.connect(self.models)) as con:
            fid, n = con.execute("SELECT fit_id, n_components FROM style_fit LIMIT 1").fetchone()
            nf = con.execute("SELECT COUNT(*) FROM style_fit_feature WHERE fit_id=?", (fid,)).fetchone()[0]
            nl = con.execute("SELECT COUNT(*) FROM style_loading WHERE fit_id=?", (fid,)).fetchone()[0]
            ver = con.execute("SELECT value FROM models_meta WHERE key='models_schema_version'").fetchone()[0]
        self.assertEqual(nl, nf * n)
        self.assertGreaterEqual(int(ver), 2)   # v2 = heavy layer; later versions keep it


if __name__ == "__main__":
    unittest.main()
