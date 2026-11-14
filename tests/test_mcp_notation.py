"""MCP shot notation parser and per-point feature tallies, checked by hand.

Examples come from the 'Instructions' tab of the MCP charting spreadsheet (MatchChart 0.3.2).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis.mcp_notation import NotationError, parse_cell, parse_point   # noqa: E402
from tennis import style_features as sf                                   # noqa: E402


class CellTest(unittest.TestCase):
    def test_key_example_first_serve_in(self):
        # "the first serve lands in ("6") and the ensuing rally ("f27b1*")"
        c = parse_cell("6f27b1*")
        self.assertEqual((c.serve_dir, c.serve_outcome, c.ending), (6, "in", "winner"))
        self.assertEqual([s.kind for s in c.shots], ["f", "b"])
        self.assertEqual((c.shots[0].direction, c.shots[0].depth), (2, 7))
        self.assertEqual((c.shots[1].direction, c.shots[1].depth), (1, None))

    def test_key_example_second_serve(self):
        c = parse_cell("5d")
        self.assertEqual((c.serve_dir, c.serve_outcome, c.fault), (5, "fault", "d"))
        c = parse_cell("4s39b3b1w@")
        self.assertEqual([s.kind for s in c.shots], ["s", "b", "b"])
        self.assertEqual((c.shots[0].depth, c.ending, c.error_type), (9, "unforced", "w"))

    def test_lets_unknown_fault_and_unknown_direction(self):
        c = parse_cell("cc4e")
        self.assertEqual((c.lets, c.serve_dir, c.fault), (2, 4, "e"))
        self.assertIsNone(parse_cell("0*").serve_dir)

    def test_ace_unreturnable_serve_and_volley(self):
        self.assertEqual(parse_cell("5*").serve_outcome, "ace")
        self.assertEqual(parse_cell("6#").serve_outcome, "unreturnable")
        c = parse_cell("4+b27v1*")
        self.assertTrue(c.serve_and_volley)
        self.assertTrue(c.shots[1].at_net)
        self.assertFalse(c.shots[0].at_net)

    def test_position_codes(self):
        c = parse_cell("4b+2f-1o=2s;3z^2*")
        b, f, o, s, z = c.shots
        self.assertTrue(b.approach and not b.at_net)
        self.assertTrue(f.at_net)
        self.assertFalse(o.at_net)            # '=' : overhead hit from the baseline
        self.assertTrue(s.net_cord)
        self.assertTrue(z.stop and z.at_net)

    def test_forced_error_needs_only_shot_type(self):
        c = parse_cell("6f2b#")
        self.assertEqual((c.ending, c.error_type), ("forced", None))

    def test_conventions_seen_in_data(self):
        self.assertEqual(parse_cell("5b28f39s18f+39*").shots[2].depth, 8)   # rally-shot depth
        self.assertTrue(parse_cell("4f18;b2n@").shots[0].net_cord)          # mark after direction
        self.assertTrue(parse_cell("6n").fault == "n")
        self.assertEqual(parse_cell("g").fault, "g")                        # fault, no direction
        c = parse_cell("4f2f@")
        self.assertEqual((c.ending, c.error_type), ("unforced", None))

    def test_undefined_codes_raise(self):
        for code in ("", "q*", "6f2", "6f2x", "3f1*", "6s278f18*", "6f1*3b2*"):
            with self.assertRaises(NotationError, msg=code):
                parse_cell(code)


class PointTest(unittest.TestCase):
    def test_double_fault(self):
        pt = parse_point("6x", "5n", server_won=False)
        self.assertTrue(pt.double_fault)
        self.assertEqual((pt.serve_no, pt.n_strokes), (2, 0))

    def test_rally_strokes_and_winner(self):
        pt = parse_point("5d", "4s39b3b1w@", server_won=True)   # returner's 3rd shot error
        self.assertEqual((pt.serve_no, pt.n_strokes), (2, 4))

    def test_winner_must_match_notation(self):
        with self.assertRaises(NotationError):
            parse_point("6f27b1*", None, server_won=False)       # server hit the winner
        with self.assertRaises(NotationError):
            parse_point("6#", None, server_won=False)

    def test_no_rally_data_points_raise(self):
        for first, second in (("S", None), ("R", None), ("P", None), ("6b29C", None),
                              ("6*", "5d"), ("5d", None)):
            with self.assertRaises(NotationError, msg=first):
                parse_point(first, second, server_won=True)

    def test_time_violation_moves_to_second_serve(self):
        pt = parse_point("V", "4*", server_won=True)
        self.assertEqual((pt.serve_no, pt.deciding.serve_outcome), (2, "ace"))

    def test_reason_strips_code(self):
        try:
            parse_cell("6f1*3b2*")
        except NotationError as e:
            self.assertEqual(e.reason, "unknown rally ending")


class CourtTest(unittest.TestCase):
    def test_court_side(self):
        self.assertTrue(sf.is_deuce_court("0-0", 0))
        self.assertFalse(sf.is_deuce_court("15-0", 0))
        self.assertTrue(sf.is_deuce_court("40-40", 0))
        self.assertFalse(sf.is_deuce_court("AD-40", 0))
        self.assertFalse(sf.is_deuce_court("40-AD", 0))
        self.assertTrue(sf.is_deuce_court("3-3", 1))
        self.assertFalse(sf.is_deuce_court("4-3", 1))
        self.assertIsNone(sf.is_deuce_court(None, 0))


class TallyTest(unittest.TestCase):
    """Five service points by one player, every count worked out by hand."""

    POINTS = [  # (1st, 2nd, score before, server won)
        ("6f27b1*", None, "0-0", True),          # deuce T; server backhand winner
        ("5d", "4s39b3b1w@", "15-0", True),      # ad; 2nd wide; returner UE on his 2nd rally shot
        ("4*", None, "30-0", True),              # deuce wide ace
        ("6x", "5n", "40-0", False),             # ad; double fault
        ("4+b27v1*", None, "40-15", True),       # deuce wide; serve-and-volley winner
    ]

    @classmethod
    def setUpClass(cls):
        rows = [(1, 10, 20, 10 if won else 20, a, b, score, 0) for a, b, score, won in cls.POINTS]
        t = sf.count_points(rows)
        cls.srv, cls.ret = t[(1, 10)], t[(1, 20)]

    def check(self, tally, side, expected):
        for feat, (num, den) in expected.items():
            i = sf.INDEX[side][feat]
            self.assertEqual((tally.num[side][i], tally.den[side][i]), (num, den), f"{side}.{feat}")

    def test_serve(self):
        self.assertEqual(self.srv.points["serve"], 5)
        self.check(self.srv, "serve", {
            "first_in": (3, 5), "second_in": (1, 2), "ace": (1, 5),
            "unret_1st": (1, 3), "unret_2nd": (0, 1), "serve_volley": (1, 4),
            "dir1_deuce_wide": (2, 3), "dir1_deuce_t": (1, 3),
            "dir1_ad_wide": (0, 2), "dir1_ad_t": (1, 2),
            "dir2_wide": (1, 2), "dir2_t": (0, 2),
            "plus1_forehand": (1, 3),
            "len_le3": (3, 4), "len_4_6": (1, 4), "len_7_9": (0, 4)})

    def test_return(self):
        self.assertEqual(self.ret.points["return"], 5)
        self.check(self.ret, "return", {
            "in_vs_1st": (2, 2), "in_vs_2nd": (1, 1),
            "depth_shallow": (2, 3), "depth_deep": (1, 3),
            "forehand": (1, 3), "slice": (1, 3), "approach": (0, 3),
            "len_le3": (3, 4), "len_4_6": (1, 4)})

    def test_rally(self):
        self.check(self.srv, "rally", {
            "forehand": (1, 3), "slice": (0, 2), "drop": (0, 3), "lob": (0, 3),
            "net": (1, 4), "net_won": (1, 1),
            "fh_winner": (1, 1), "bh_winner": (1, 2), "fh_unforced": (0, 1), "bh_unforced": (0, 2)})
        self.check(self.ret, "rally", {
            "forehand": (0, 1), "slice": (0, 1), "net": (0, 4), "net_won": (0, 0),
            "bh_winner": (0, 1), "bh_unforced": (1, 1), "fh_unforced": (0, 0)})

    def test_bad_points_are_counted_not_tallied(self):
        from collections import Counter
        skipped = Counter()
        t = sf.count_points([(1, 10, 20, 10, "S", None, "0-0", 0),
                             (1, 10, 20, 10, "6f1*3b2*", None, "15-0", 0)], skipped)
        self.assertEqual(t, {})
        self.assertEqual(skipped, Counter({"no rally data": 1, "unknown rally ending": 1}))


if __name__ == "__main__":
    unittest.main()
