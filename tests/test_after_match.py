"""The match panel's "After the match" block: final Kalshi stats vs the pre-match expected
serve-point rates (engine/livefeed.after_match, versus, final_stats), the noise labels, the
missing-prematch fallback, and each serve stream's final in-match form recorded at
settlement (engine/predict stream_form via Predictor.predict, recorder.record_final_form)."""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import points_rating as pr                                   # noqa: E402
from tennis.engine import livefeed                                       # noqa: E402
from tennis.engine.predictor import FixedPredictor, MatchQuery           # noqa: E402
from tennis.engine.recorder import Recorder                             # noqa: E402

from test_predict import ROWS, predictor                                 # noqa: E402

EV = "KXATPMATCH-20MAR05AAACCC"


def stats(won, lost, **kw):
    return {"service_points_won": won, "service_points_lost": lost, **kw}


def details(sa: dict, sb: dict, swap: bool = True) -> dict:
    """Kalshi live details as the settlement blob keeps them; swap puts B in competitor1, so
    a correct mapping must go by id, not by order."""
    one, two = (("cb", sb), ("ca", sa)) if swap else (("ca", sa), ("cb", sb))
    return {"status": "closed", "match_status": "ended", "competitor1_id": one[0], "competitor2_id": two[0],
            "competitor1_statistics": one[1], "competitor2_statistics": two[1]}


class VersusTest(unittest.TestCase):
    """Hand-computed edges: p = 0.6, n = 100, v = 0.04.
    binomial var 0.6*0.4/100 = 0.0024; rating var (0.24)^2 * 0.04 = 0.002304;
    sd = sqrt(0.004704) = 0.068586; 2 sd = 0.137171 (13.7 points of rate)."""

    def test_numbers(self):
        c = livefeed.versus(70, 106, 0.63, None)
        self.assertAlmostEqual(c["rate"], 70 / 106)
        self.assertAlmostEqual(c["pts"], 70 - 106 * 0.63)               # +3.2 points over expectation
        self.assertAlmostEqual(c["sd"], math.sqrt(0.63 * 0.37 / 106))  # no variance: binomial only
        self.assertEqual(c["sd_rating"], 0.0)
        c = livefeed.versus(74, 100, 0.6, 0.04)
        self.assertAlmostEqual(c["sd"], math.sqrt(0.004704))
        self.assertAlmostEqual(c["sd_binom"], math.sqrt(0.0024))
        self.assertAlmostEqual(c["sd_rating"], math.sqrt(0.002304))

    def test_labels_at_the_edges(self):
        lab = lambda won: livefeed.versus(won, 100, 0.6, 0.04)["label"]   # noqa: E731
        self.assertEqual(lab(74), "above")      # +0.14 > 0.1372
        self.assertEqual(lab(73), "within")     # +0.13
        self.assertEqual(lab(60), "within")
        self.assertEqual(lab(47), "within")     # -0.13
        self.assertEqual(lab(46), "below")      # -0.14
        # without the rating's uncertainty the band is narrower: 2 * sqrt(0.0024) = 0.098
        self.assertEqual(livefeed.versus(70, 100, 0.6, None)["label"], "above")
        self.assertEqual(livefeed.versus(70, 100, 0.6, 0.04)["label"], "within")


class AfterMatchTest(unittest.TestCase):
    SA = stats(70, 36, aces=7, double_faults=2, first_serve_successful=66, first_serve_points_won=52,
               second_serve_points_won=18, breakpoints_won=3, total_breakpoints=8, service_games_won=15,
               tiebreaks_won=1, points_won=120)
    SB = stats(60, 50, aces=3, double_faults=4, first_serve_successful=70, first_serve_points_won=45,
               second_serve_points_won=15, breakpoints_won=2, total_breakpoints=4, service_games_won=13,
               tiebreaks_won=0, points_won=96)
    PRE = {"serve_a": 0.63, "serve_b": 0.6, "v_a": 0.03, "v_b": 0.05, "x_a": 0.53, "x_b": 0.4}

    def test_maps_by_competitor_id(self):
        for swap in (True, False):
            st = livefeed.final_stats(details(self.SA, self.SB, swap), "ca", "cb")
            self.assertEqual((st["a"]["serve_won"], st["a"]["serve_played"]), (70, 106))
            self.assertEqual((st["b"]["serve_won"], st["b"]["serve_played"]), (60, 110))
        a = st["a"]
        self.assertEqual((a["first_in"], a["first_won"], a["second_played"], a["second_won"]), (66, 52, 40, 18))
        self.assertEqual((a["aces"], a["double_faults"], a["bp_won"], a["bp_total"], a["tiebreaks_won"]), (7, 2, 3, 8, 1))
        # service games = held + the opponent's breaks
        self.assertEqual((a["service_games_won"], a["service_games"]), (15, 17))
        self.assertEqual(st["b"]["service_games"], 13 + 3)
        self.assertIsNone(livefeed.final_stats(details(self.SA, self.SB), "ca", "zz"))

    def test_serve_and_return_against_expectation(self):
        x = livefeed.after_match(details(self.SA, self.SB), "ca", "cb", self.PRE)
        sa, ra, sb, rb = x["serve"]["a"], x["ret"]["a"], x["serve"]["b"], x["ret"]["b"]
        self.assertEqual((sa["won"], sa["played"], sa["expected"]), (70, 106, 0.63))
        self.assertAlmostEqual(sa["pts"], 70 - 106 * 0.63)
        # A's return points are B's serve points A won, against 1 - B's expected serve rate
        self.assertEqual((ra["won"], ra["played"]), (50, 110))
        self.assertAlmostEqual(ra["expected"], 0.4)
        self.assertAlmostEqual(ra["gap"], -sb["gap"])                   # one stream, two sides
        self.assertAlmostEqual(ra["sd"], sb["sd"])
        self.assertAlmostEqual(sb["sd"], math.sqrt(0.24 / 110 + 0.24 ** 2 * 0.05))   # B's variance
        self.assertEqual(rb["won"], 36)
        self.assertEqual({sa["label"], sb["label"], ra["label"], rb["label"]}, {"within"})
        self.assertIsNone(x["form"])
        json.dumps(x)

    def test_expected_from_logit_when_rate_missing(self):
        pre = {"x_a": 0.53, "x_b": 0.4, "v_a": 0.03, "v_b": 0.05}
        x = livefeed.after_match(details(self.SA, self.SB), "ca", "cb", pre)
        self.assertAlmostEqual(x["serve"]["a"]["expected"], 1 / (1 + math.exp(-0.53)))

    def test_no_prematch_gives_actuals_only(self):
        x = livefeed.after_match(details(self.SA, self.SB), "ca", "cb", None)
        self.assertIsNone(x["serve"])
        self.assertIsNone(x["ret"])
        self.assertEqual(x["stats"]["a"]["serve_won"], 70)

    def test_void_and_walkover_show_nothing(self):
        self.assertIsNone(livefeed.after_match(None, "ca", "cb", self.PRE))
        self.assertIsNone(livefeed.after_match({"status": "closed"}, "ca", "cb", self.PRE))
        self.assertIsNone(livefeed.after_match(details(stats(0, 0), stats(0, 0)), "ca", "cb", self.PRE))


class FinalFormTest(unittest.TestCase):
    """At settlement the recorder asks the predictor (Predictor.predict only) for each serve
    stream's form on the final stats, and the match panel shows it."""

    @classmethod
    def setUpClass(cls):
        cls.pp, _ = predictor(ROWS, len(ROWS))
        cls.q = MatchQuery(tour="atp", player_a=1, player_b=3, on="2020-03-05", surface="hard", edition_id=6)
        cls.pre = cls.pp.predict(cls.q).detail

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "live.db"
        self.rec = Recorder(self.db)
        c = self.rec.con
        c.execute("INSERT INTO watch VALUES (1, '2020-03-05T09:00:00+00:00', 'test', '{}', 'KXATPMATCH', 20, NULL, NULL)")
        c.execute("INSERT INTO poll (poll_id, watch_id, as_of, done_at, n_events) VALUES (1, 1, ?, ?, 1)",
                  ("2020-03-05T10:00:00+00:00",) * 2)
        c.execute("INSERT INTO event VALUES (?, 'KXATPMATCH', 'atp', 'Aaa vs Ccc', 'Test Open', NULL, ?, ?, 'Ann Aaa', "
                  "'Cy Ccc', 'ca', 'cb', NULL, ?, ?)", (EV, EV + "-A", EV + "-C", "2020-03-05T10:00:00+00:00",
                                                         "2020-03-05T10:00:00+00:00"))
        c.execute("""INSERT INTO snapshot (poll_id, event, status, skip, p_model, a_pid, b_pid, day, surface, best_of,
                         final, edition_id) VALUES (1, ?, 'live', '', 0.6, 1, 3, '2020-03-05', 'hard', 3, 'tb', 6)""", (EV,))
        c.commit()
        self.ev = dict(c.execute("SELECT * FROM event WHERE event = ?", (EV,)).fetchone())
        self.final = details(stats(48, 22), stats(30, 34))

    def tearDown(self):
        self.rec.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detail_has_stream_form_in_and_after_play(self):
        ls = self.rec.last_query(EV, None)
        self.assertEqual(ls, self.q)                                    # rebuilt from the snapshot
        from tennis.kalshi import live_state
        fin = live_state(self.final, "ca", "cb")
        self.assertEqual((fin.status, fin.serve_a, fin.serve_b), ("closed", (48, 70), (30, 64)))
        d = self.pp.predict(MatchQuery(**{**self.q.__dict__, "live": fin})).detail
        xa, va = pr.in_match_form(self.pre["x_a"], self.pre["v_a"], 48, 70, pr.IN_PLAY_FORM_VAR)
        xb, _ = pr.in_match_form(self.pre["x_b"], self.pre["v_b"], 30, 64, pr.IN_PLAY_FORM_VAR)
        self.assertAlmostEqual(d["form_a"], xa - self.pre["x_a"])
        self.assertAlmostEqual(d["form_b"], xb - self.pre["x_b"])
        self.assertAlmostEqual(d["serve_a_form"], 1 / (1 + math.exp(-xa)))
        self.assertAlmostEqual(d["v_a_form"], va)
        self.assertEqual(d["form_points_a"], [48, 70])
        for s, rate in (("a", 48 / 70), ("b", 30 / 64)):               # form moves toward the actual rate
            pre = 1 / (1 + math.exp(-self.pre["x_" + s]))
            self.assertEqual(d["form_" + s] > 0, rate > pre)
            self.assertLess(abs(d[f"serve_{s}_form"] - pre), abs(rate - pre))   # shrunk
        self.assertNotIn("form_a", self.pre)                             # pre-match: no live, no form

    def test_recorded_at_settlement_and_shown(self):
        self.assertEqual([e["event"] for e, _ in self.rec.missing_final_form()], [])   # not settled yet
        sha = self.rec.put_blob({"event": {}, "live": self.final, "milestone": None})
        self.rec.con.execute("INSERT INTO settlement (event, seen_at, winner, final_score, match_status, raw_sha) "
                             "VALUES (?, '2020-03-05T12:00:00+00:00', 'a', '6-4 6-3', 'ended', ?)", (EV, sha))
        self.rec.con.commit()
        [(ev, det)] = self.rec.missing_final_form()
        self.assertEqual(det, self.final)
        row = self.rec.record_final_form(ev, det, self.pp, now="2020-03-05T12:00:00+00:00")
        xa, _ = pr.in_match_form(self.pre["x_a"], self.pre["v_a"], 48, 70, pr.IN_PLAY_FORM_VAR)
        self.assertAlmostEqual(row["form_a"], xa - self.pre["x_a"])
        self.assertEqual((row["serve_b_won"], row["serve_b_played"], row["model"]), (30, 64, self.pp.name))
        self.assertAlmostEqual(row["x_a"], self.pre["x_a"])
        self.assertEqual(self.rec.missing_final_form(), [])
        con = livefeed.connect(self.db)
        try:
            v = livefeed.match_view(con, EV)
        finally:
            con.close()
        f = v["after"]["form"]
        self.assertEqual(f["a"]["form"], round(row["form_a"], 4))
        self.assertEqual((f["a"]["won"], f["a"]["played"]), (48, 70))
        self.assertEqual(f["form_var"], pr.IN_PLAY_FORM_VAR)
        self.assertIsNone(v["after"]["serve"])                         # no prematch row in this db

    def test_nothing_recorded_without_points_or_form(self):
        self.assertIsNone(self.rec.record_final_form(self.ev, None, self.pp))
        self.assertIsNone(self.rec.record_final_form(self.ev, details(stats(0, 0), stats(0, 0)), self.pp))
        self.assertIsNone(self.rec.record_final_form(self.ev, self.final, FixedPredictor(0.5)))
        self.assertIsNone(self.rec.record_final_form({**self.ev, "event": "KXATPMATCH-NEVERPRICED"}, self.final, self.pp))
        # ratings that already include the match day (player 1 last rated 2020-01-20): not recorded
        self.rec.con.execute("UPDATE snapshot SET day = '2020-01-20' WHERE event = ?", (EV,))
        self.assertIsNone(self.rec.record_final_form(self.ev, self.final, self.pp))
        self.assertEqual(self.rec.con.execute("SELECT COUNT(*) FROM final_form").fetchone()[0], 0)

    def test_old_db_without_final_form_table(self):
        self.rec.con.execute("DROP TABLE final_form")
        self.rec.con.commit()
        con = sqlite3.connect(self.db)
        con.row_factory = sqlite3.Row
        try:
            self.assertIsNone(livefeed._final_form(con, EV))
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
