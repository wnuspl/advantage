from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tennis.load_mcp import parse_point_codes                     # noqa: E402
from tennis.load_slam_pbp import parse_match_num                   # noqa: E402
from tennis.normalize import (norm_name, norm_round, parse_ymd,   # noqa: E402
                              score_status, is_placeholder_name)


class NormalizeTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(norm_name("Juan Martín del Potro"), "juan martin del potro")
        self.assertEqual(norm_name("Félix Auger-Aliassime"), "felix auger aliassime")
        self.assertEqual(norm_name("Caroline Wozniacki "), "caroline wozniacki")
        self.assertEqual(norm_name("Iga Świątek"), "iga swiatek")
        self.assertEqual(norm_name("Holger Vitus Nødskov Rune"), "holger vitus nodskov rune")
        self.assertEqual(norm_name("R. Nadal"), "r nadal")

    def test_dates(self):
        self.assertEqual(parse_ymd("20240115"), "2024-01-15")
        self.assertIsNone(parse_ymd("19000000"))
        self.assertIsNone(parse_ymd("2024011"))
        self.assertIsNone(parse_ymd(""))

    def test_status(self):
        self.assertEqual(score_status("6-4 6-4"), "completed")
        self.assertEqual(score_status("2-6 6-2 4-1 RET"), "retired")
        self.assertEqual(score_status("W/O"), "walkover")
        self.assertEqual(score_status("Walkover"), "walkover")
        self.assertEqual(score_status("6-3 Def."), "default")
        self.assertEqual(score_status("6-4 ABD"), "unfinished")
        self.assertEqual(score_status("BYE"), "bye")

    def test_rounds(self):
        self.assertEqual(norm_round("f"), "F")
        self.assertEqual(norm_round("R64 "), "R64")
        self.assertIsNone(norm_round("Unipol Arena"))

    def test_placeholders(self):
        self.assertTrue(is_placeholder_name("U", "Unknown"))
        self.assertTrue(is_placeholder_name("X", "X"))
        self.assertFalse(is_placeholder_name("Unknown", "Garcia Cors"))

    def test_slam_match_num(self):
        self.assertEqual(parse_match_num("1101"), ("atp", "R128"))
        self.assertEqual(parse_match_num("2701"), ("wta", "F"))
        self.assertEqual(parse_match_num("MS501"), ("atp", "QF"))
        self.assertEqual(parse_match_num("WS101"), ("wta", "R128"))
        self.assertEqual(parse_match_num("3101"), (None, None))


class McpCodesTest(unittest.TestCase):
    def test_ace(self):
        c = parse_point_codes("4*", "", True)
        self.assertEqual((c["is_ace"], c["end_type"], c["rally_len"], c["serve_no"]), (1, "ace", 1, 1))

    def test_double_fault(self):
        c = parse_point_codes("6n", "5d", False)
        self.assertEqual((c["is_double_fault"], c["end_type"], c["serve_no"]), (1, "double_fault", 2))

    def test_let_then_ace(self):
        self.assertEqual(parse_point_codes("c4*", "", True)["is_ace"], 1)

    def test_unreturned_serve(self):
        c = parse_point_codes("6#", "", True)
        self.assertEqual((c["end_type"], c["rally_len"]), ("forced_error", 1))

    def test_rally(self):
        c = parse_point_codes("4f1b2f3*", "", False)
        self.assertEqual((c["end_type"], c["rally_len"]), ("winner", 4))
        c = parse_point_codes("5b28f1n@", "", False)
        self.assertEqual((c["end_type"], c["rally_len"]), ("unforced_error", 2))

    def test_empty(self):
        self.assertIsNone(parse_point_codes("", "", True)["serve_no"])


if __name__ == "__main__":
    unittest.main()
