from __future__ import annotations

import sys
import unittest
from array import array
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import elo, markov, points_rating as pr   # noqa: E402


class MarkovTest(unittest.TestCase):
    def test_game_known_values(self):
        self.assertAlmostEqual(markov.game(0.5), 0.5)
        self.assertAlmostEqual(markov.game(0.6), 0.735729, places=6)

    def test_equal_players_are_even(self):
        for bo in (1, 3, 5):
            for f in markov.FINALS:
                self.assertAlmostEqual(markov.match(0.62, 0.62, bo, f), 0.5)

    def test_serve_order_does_not_matter(self):
        # P(A wins the set | A serves first) = 1 - P(B wins | B serves first)
        for f in ("tb", "adv", "tb12"):
            self.assertAlmostEqual(markov.set_(0.66, 0.59, f), 1 - markov.set_(0.59, 0.66, f))
        for n in (7, 10):
            self.assertAlmostEqual(markov.tiebreak(0.66, 0.59, n), 1 - markov.tiebreak(0.59, 0.66, n))

    def test_complementary_and_longer_formats_favour_the_better_player(self):
        a, b = 0.66, 0.62
        for f in markov.FINALS:
            self.assertAlmostEqual(markov.match(a, b, 3, f) + markov.match(b, a, 3, f), 1.0)
        self.assertGreater(markov.match(a, b, 5), markov.match(a, b, 3))
        self.assertGreater(markov.match(a, b, 3), markov.match(a, b, 3, "mtb"))

    def test_vectorized_equals_scalar(self):
        pa, pb = np.array([0.55, 0.7]), np.array([0.6, 0.65])
        v = markov.match(pa, pb, 5, "adv")
        for k in range(2):
            self.assertAlmostEqual(v[k], markov.match(float(pa[k]), float(pb[k]), 5, "adv"))

    def test_grid_matches_exact_and_gradient_signs(self):
        g = markov.MatchGrid(3, "tb")
        xa, xb = markov.logit(0.67), markov.logit(0.61)
        p, da, db = g.lookup(xa, xb)
        self.assertAlmostEqual(p, markov.match(0.67, 0.61), delta=1e-4)
        self.assertGreater(da, 0)
        self.assertLess(db, 0)
        arr = g.lookup_array(np.array([xa]), np.array([xb]))
        self.assertAlmostEqual(float(arr[0]), p)
        # clamped outside the grid, never an index error
        self.assertTrue(0 <= g.lookup(50.0, -50.0)[0] <= 1)


class DecidingSetTest(unittest.TestCase):
    def test_evidence(self):
        ev = pr.deciding_set_evidence
        self.assertEqual(ev("6-3 3-6 10-7", 3), "mtb")
        self.assertEqual(ev("6-4 3-6 [10-8]", 3), "mtb")
        self.assertEqual(ev("6-4 3-6 7-6(5) 4-6 13-12(5)", 5), "tb12")
        self.assertEqual(ev("6-4 3-6 7-6(5) 4-6 10-8", 5), "adv")
        self.assertEqual(ev("6-4 3-6 7-6(5)", 3), "tb")
        self.assertIsNone(ev("6-4 6-3", 3))           # no deciding set
        self.assertIsNone(ev("6-4 3-6 6-2", 3))       # deciding set says nothing
        self.assertIsNone(ev(None, 3))

    def test_infer_rules_uses_tournament_history_then_default(self):
        rows = [(1, 10, "2018-06-01", "6-4 3-6 4-6 6-3 12-10", 5),    # adv, by evidence
                (2, 10, "2019-06-01", "6-4 6-3 6-2", 5),       # no evidence: previous edition
                (3, 20, "2019-06-01", "6-4 6-3", 3)]           # nothing at all: default
        rules, counts = pr.infer_rules(rows)
        self.assertEqual(rules, {1: "adv", 2: "adv", 3: "tb"})
        self.assertEqual(counts[("adv", "tournament_history")], 1)


def make(rows):
    """rows: (winner, loser, start, status, surface, w_sp, w_sw, l_sp, l_sw); sp=-1: no stats."""
    m = elo.empty_matches()
    d = pr.PointsData(bytearray(), array("l"), array("l"), array("l"), array("l"), {})
    for i, r in enumerate(rows):
        m.match_id.append(i + 1); m.winner.append(r[0]); m.loser.append(r[1])
        m.start.append(r[2]); m.status.append(r[3]); m.surface.append(r[4])
        m.tour.append("atp"); m.tour_level.append(1)
        d.fmt.append(pr.FORMAT_INDEX[(3, "tb")])
        for a, v in zip((d.w_sp, d.w_sw, d.l_sp, d.l_sw), r[5:9]):
            a.append(v)
    m.extra["points"] = d
    return m


class PointsModelTest(unittest.TestCase):
    P = pr.PointsParams()

    def state_after(self, rows, p=None):
        return pr.run(make(rows), p or self.P).state

    def test_points_update_moves_serve_and_return(self):
        # 1 wins 80% of serve points (far above base), 2 wins 55% of theirs
        st = self.state_after([(1, 2, "2020-01-06", "completed", "hard", 60, 48, 60, 33)])
        self.assertGreater(st.serve[1], 0)
        self.assertLess(st.ret[2], 0)
        self.assertLess(st.vs[1], self.P.v_init)
        self.assertGreater(st.sserve[(1, "hard")], 0)       # surface deviation moves too
        self.assertGreater(st.base[("atp", "hard")], pr.BASE_INIT["atp"])

    def test_points_not_result_drive_the_update(self):
        # the winner won fewer points than expected on serve: their serve rating drops
        st = self.state_after([(1, 2, "2020-01-06", "completed", "hard", 80, 40, 80, 56)])
        self.assertLess(st.serve[1], 0)
        self.assertGreater(st.serve[2], 0)

    def test_result_only_fallback(self):
        st = self.state_after([(1, 2, "2020-01-06", "completed", "clay", -1, -1, -1, -1)])
        self.assertGreater(st.serve[1], 0)
        self.assertGreater(st.ret[1], 0)
        self.assertLess(st.serve[2], 0)
        self.assertAlmostEqual(st.serve[1] + st.ret[1], -(st.serve[2] + st.ret[2]))
        none = self.state_after([(1, 2, "2020-01-06", "completed", "clay", -1, -1, -1, -1)],
                                pr.PointsParams(nostat_weight=0.0))
        self.assertEqual(none.serve[1], 0.0)

    def test_walkover_never_updates_and_retired_weight(self):
        st = self.state_after([(1, 2, "2020-01-06", "walkover", "hard", -1, -1, -1, -1)])
        self.assertNotIn(1, st.serve)
        st = self.state_after([(1, 2, "2020-01-06", "retired", "hard", 30, 20, 30, 15)],
                              pr.PointsParams(retired_weight=0.0))
        self.assertNotIn(1, st.serve)

    def test_prediction_is_pre_match_and_symmetric(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 45, 60, 36),
                (3, 4, "2020-01-06", "completed", "hard", 60, 40, 60, 38),
                (1, 2, "2020-02-03", "completed", "hard", 60, 44, 60, 37)]
        res = pr.run(make(rows), self.P)
        self.assertAlmostEqual(res.p_winner[0], 0.5)
        self.assertGreater(res.p_winner[2], 0.5)
        flipped = rows[:2] + [(2, 1, "2020-02-03", "completed", "hard", 60, 37, 60, 44)]
        res2 = pr.run(make(flipped), self.P)
        self.assertAlmostEqual(res.p_winner[2], 1 - res2.p_winner[2], places=12)
        # a later match never changes an earlier prediction
        longer = pr.run(make(rows + [(2, 1, "2020-03-02", "completed", "hard", 60, 50, 60, 20)]), self.P)
        self.assertEqual(list(res.p_winner), list(longer.p_winner[:3]))

    def test_predict_reproduces_run_and_prediction_only_params(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 45, 60, 36),
                (1, 2, "2020-03-02", "completed", "grass", 60, 45, 60, 36),
                (1, 2, "2020-05-04", "completed", "hard", -1, -1, -1, -1)]
        m = make(rows)
        res = pr.run(m, self.P)
        self.assertEqual(list(res.p_winner), list(pr.predict(res, self.P)))
        other = pr.PointsParams(blend=0.3, unc=0.0)
        self.assertEqual(list(pr.run(m, other).p_winner), list(pr.predict(res, other)))
        self.assertEqual(other.update_key(), self.P.update_key())
        # uncertainty shrinks toward 0.5
        self.assertLess(pr.predict(res, self.P)[2], pr.predict(res, pr.PointsParams(unc=0.0))[2])

    def test_blackout_hides_stats_from_the_date_only(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 48, 60, 33),
                (1, 2, "2021-01-04", "completed", "hard", 60, 30, 60, 50),
                (3, 4, "2021-02-01", "completed", "hard", 60, 40, 60, 38)]
        m = make(rows)
        d = m.extra["points"]
        b = pr.blackout(m, d, "2021-01-04")
        self.assertEqual(list(b.w_sp), [60, -1, -1])
        self.assertEqual(list(b.l_sw), [33, -1, -1])
        self.assertEqual(list(d.w_sp), [60, 60, 60])          # original untouched
        full = pr.run(m, self.P)
        m.extra["points"] = b
        dark = pr.run(m, self.P)
        # identical before the blackout (pre-match predictions of matches 0 and 1)
        self.assertEqual(list(full.p_winner[:2]), list(dark.p_winner[:2]))
        # after it, the winner of match 1 gains from the result instead of losing on points
        self.assertGreater(dark.state.serve[1] + dark.state.ret[1],
                           full.state.serve[1] + full.state.ret[1])

    def test_variance_grows_with_inactivity_and_is_capped(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 45, 60, 36),
                (1, 3, "2020-02-03", "completed", "hard", 60, 45, 60, 36),
                (1, 3, "2030-02-04", "completed", "hard", 60, 45, 60, 36)]
        res = pr.run(make(rows), self.P)
        self.assertLess(res.d_winner[1] ** 2, 2 * self.P.v_init)
        self.assertAlmostEqual(res.d_winner[2] ** 2, 2 * self.P.v_init)

    def test_until_keeps_extras(self):
        m = make([(1, 2, "2020-01-06", "completed", "hard", 60, 45, 60, 36),
                  (1, 2, "2021-01-04", "completed", "hard", 60, 45, 60, 36)])
        sub = elo.until(m, "2020-06-01")
        self.assertEqual(len(sub), 1)
        self.assertEqual(pr.run(sub, self.P).state.count[1], 1)


if __name__ == "__main__":
    unittest.main()


class CalibrateTest(unittest.TestCase):
    def test_alpha_uses_only_earlier_matches(self):
        from tennis import calibrate, tuning
        rng = np.random.default_rng(0)
        m = elo.empty_matches()
        rows = []
        for i in range(400):
            start = f"{2018 + i // 100}-0{1 + (i % 100) // 25 * 3:01d}-01".replace("-010-", "-10-")
            rows.append(start)
        rows.sort()
        for i, start in enumerate(rows):
            m.match_id.append(i + 1); m.winner.append(2 * i + 1); m.loser.append(2 * i + 2)
            m.start.append(start); m.status.append("completed"); m.surface.append("hard")
            m.tour.append("atp"); m.tour_level.append(1)
        flag = bytearray(int(rng.random() < 0.5) for _ in rows)
        prep = tuning.Prepared(m, flag, {})
        pw = array("d", (0.5 + 0.3 * rng.random() for _ in rows))
        p1, y, idx, alphas = calibrate.calibrated(prep, pw, "2021-01-01", "2021-12-31")
        self.assertEqual(len(idx), 100)
        # changing predictions inside the test year never changes the alphas
        pw2 = array("d", pw)
        for i in idx:
            pw2[i] = 0.99
        _, _, _, alphas2 = calibrate.calibrated(prep, pw2, "2021-01-01", "2021-01-31")
        self.assertEqual(alphas[("atp", "2021-01-01")], alphas2[("atp", "2021-01-01")])
        # per best-of: all best-of-3 reproduces the per-tour calibration exactly
        bo3 = [3] * len(rows)
        p3, _, _, _ = calibrate.calibrated(prep, pw, "2021-01-01", "2021-12-31", best_of=bo3, min_rows=10)
        np.testing.assert_array_equal(p3, p1)
        # a best-of with too little history falls back to the tour alpha; enough history: its own
        mixed = [5 if i % 2 else 3 for i in range(len(rows))]
        _, _, _, a_few = calibrate.calibrated(prep, pw, "2021-01-01", "2021-03-31", best_of=mixed, min_rows=1000)
        self.assertEqual(a_few[("atp", "2021-01-01", 5)], a_few[("atp", "2021-01-01")])
        _, _, _, a_own = calibrate.calibrated(prep, pw, "2021-01-01", "2021-03-31", best_of=mixed, min_rows=10)
        self.assertNotEqual(a_own[("atp", "2021-01-01", 5)], a_own[("atp", "2021-01-01")])


class ScoreTierTest(unittest.TestCase):
    def test_parse_score(self):
        ps = pr.parse_score
        self.assertEqual(ps("6-0 6-0", 3, "tb", "completed"), ((0, 0), (0, 0)))
        self.assertEqual(ps("7-6(5) 6-7(3) 7-6(8)", 3, "tb", "completed"), ((0, 6), (0, 13), (0, 6)))
        self.assertEqual(ps("6-3 3-6 10-7", 3, "mtb", "completed"), ((0, 3), (0, 10), (2, 0)))
        self.assertEqual(ps("6-4 3-6 6-7(5) 6-3 12-10", 5, "adv", "completed"),
                         ((0, 4), (0, 10), (0, 13), (0, 3), (1, 6)))
        # retirement: completed sets kept, the set in progress dropped
        self.assertEqual(ps("6-4 2-1 RET", 3, "tb", "retired"), ((0, 4),))
        self.assertEqual(ps("6-2 2-0 ret.", 3, "tb", "retired"), ((0, 2),))
        self.assertIsNone(ps("3-2 RET", 3, "tb", "retired"))
        # unusable: not complete sets, wrong winner, unknown tokens, too many sets
        self.assertIsNone(ps("6-5 6-3", 3, "tb", "completed"))
        self.assertIsNone(ps("3-6 4-6", 3, "tb", "completed"))
        self.assertIsNone(ps("W/O", 3, "tb", "walkover"))
        self.assertIsNone(ps("6-3 6-3 Played and unfinished", 3, "tb", "completed"))
        self.assertIsNone(ps("8-6 6-3", 3, "tb", "completed"))       # 8-6 is not a tiebreak set
        self.assertIsNone(ps(None, 3, "tb", "completed"))

    def test_set_score_distribution(self):
        for f in ("tb", "adv"):
            d = markov.set_score_dist(0.66, 0.61, f)
            self.assertAlmostEqual(sum(d), 1.0)
            self.assertAlmostEqual(sum(d[:7]), markov.set_(0.66, 0.61, f))
        g = markov.score_grid("tb")
        xa, xb, h = markov.logit(0.66), markov.logit(0.61), 1e-5
        for c in (0, 3, 6, 11):
            p, da, db = g.lookup(xa, xb, c)
            exact = lambda a, b: markov.set_score_dist(1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b)))[c]  # noqa: E731
            self.assertAlmostEqual(p, exact(xa, xb), delta=2e-4)
            self.assertAlmostEqual(da, (exact(xa + h, xb) - exact(xa - h, xb)) / (2 * h), delta=0.01)
            self.assertAlmostEqual(db, (exact(xa, xb + h) - exact(xa, xb - h)) / (2 * h), delta=0.01)

    def run_scored(self, score, w=1.0):
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1)]
        m = make(rows)
        m.extra["scores"] = [pr.parse_score(score, 3, "tb", "completed")]
        return pr.run(m, pr.PointsParams(score_weight=w)).state

    def test_score_tier_scales_with_the_margin(self):
        blowout = self.run_scored("6-0 6-0")
        close = self.run_scored("7-6(5) 6-7(3) 7-6(8)")
        gain = lambda st: st.serve[1] + st.ret[1]          # noqa: E731
        self.assertGreater(gain(close), 0)
        self.assertGreater(gain(blowout), 2 * gain(close))
        # expected (Fisher) information: depends on the sets played, not the margin
        self.assertLess(close.vs[1], blowout.vs[1])
        # a close match says both served well: serve up, return down for both, nearly
        self.assertGreater(close.serve[2], blowout.serve[2])

    def test_score_weight_zero_is_result_only(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (1, 2, "2020-02-03", "completed", "hard", -1, -1, -1, -1)]
        m = make(rows)
        plain = pr.run(m, pr.PointsParams())
        m.extra["scores"] = [pr.parse_score("6-0 6-0", 3, "tb", "completed")] * 2
        self.assertEqual(list(plain.p_winner), list(pr.run(m, pr.PointsParams()).p_winner))
        scored = pr.run(m, pr.PointsParams(score_weight=1.0))
        self.assertGreater(scored.p_winner[1], plain.p_winner[1])
        # stats still take precedence over the score
        m2 = make([(1, 2, "2020-01-06", "completed", "hard", 60, 30, 60, 50)])
        m2.extra["scores"] = [pr.parse_score("6-0 6-0", 3, "tb", "completed")]
        self.assertEqual(pr.run(m2, pr.PointsParams(score_weight=1.0)).state.serve,
                         pr.run(m2, pr.PointsParams()).state.serve)


class ScorelineTest(unittest.TestCase):
    def test_scorelines_add_up_to_the_win_probability(self):
        for bo, f in ((3, "tb"), (3, "mtb"), (5, "adv"), (5, "tb12")):
            counts = markov.set_count_dist(0.66, 0.62, bo, f)
            self.assertAlmostEqual(sum(counts.values()), 1.0)
            won = sum(v for k, v in counts.items() if int(k.split("-")[0]) > int(k.split("-")[1]))
            # tb12 deciding sets are approximated as advantage sets in the scorelines
            self.assertAlmostEqual(won, markov.match(0.66, 0.62, bo, f), delta=1e-3 if f == "tb12" else 1e-12)
        d = markov.match_score_dist(0.66, 0.62, 3, "tb", min_prob=0.0)
        self.assertAlmostEqual(sum(d.values()), 1.0)
        self.assertAlmostEqual(d["6-4 6-4"], markov.match_score_dist(0.62, 0.66, 3, "tb")["4-6 4-6"])
        self.assertEqual(markov.top_scores(markov.match_score_dist(0.5, 0.5), 1)[0][1],
                         max(markov.match_score_dist(0.5, 0.5).values()))
        self.assertIn("[10-x]", " ".join(markov.match_score_dist(0.66, 0.62, 3, "mtb")))
        self.assertIn("8-6+", " ".join(markov.match_score_dist(0.66, 0.62, 5, "adv")))

    def test_matchup_reproduces_the_pre_match_prediction(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 45, 60, 36),
                (1, 3, "2020-02-03", "completed", "clay", 70, 44, 70, 40),
                (3, 2, "2020-03-02", "completed", "hard", -1, -1, -1, -1),
                (1, 2, "2020-05-04", "completed", "hard", 60, 41, 60, 39)]
        p = pr.PointsParams(blend=0.7, unc=5.0)
        res = pr.run(make(rows), p)
        st = pr.run(elo.until(make(rows), "2020-05-04"), p).state
        x = pr.matchup(st, 1, 2, "atp", "hard", "2020-05-04")
        self.assertAlmostEqual(pr.win_probability(st, *x), res.p_winner[3], places=12)
        # a player never seen is a new player
        x_new = pr.matchup(st, 99, 1, "atp", "hard", "2020-05-04")[0]
        self.assertAlmostEqual(x_new, st.base[("atp", "hard")] - st.ret[1] - p.blend * st.sret[(1, "hard")])
