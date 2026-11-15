"""Tournament surface from history (engine/surfaces.py), and its use in the retriever."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis.engine import FixedPredictor                                     # noqa: E402
from tennis.engine.live import EditionInfo, load_pull, snapshots             # noqa: E402
from tennis.engine.surfaces import TournamentSurfaces, kalshi_tournament     # noqa: E402
from test_live import FIX, by_event, resolver                                # noqa: E402

# (tournament_id, tour, name, [(level, start_date, surface)])
TOURNAMENTS = [
    (1, "atp", "Kobe CH", [("C", "2024-11-11", "hard"), ("C", "2025-11-10", "hard")]),
    (2, "atp", "Lima CH", [("C", "2025-06-23", "clay")]),
    (3, "atp", "Lima Los Inkas CH", [("C", "2025-10-27", "hard")]),      # pretend: autumn edition on hard
    (4, "atp", "Metz", [("A", "2024-11-04", "hard"), ("A", "2025-11-02", "hard")]),
    (5, "atp", "M25 Champaign IL", [("25", "2025-07-21", "hard")]),
    (6, "atp", "Tour Finals", [("F", "2024-11-10", "hard"), ("F", "2025-11-09", "hard")]),
    (7, "atp", "Next Gen Finals", [("F", "2025-12-17", "carpet")]),
    (8, "atp", "Davis Cup Finals QF: ITA vs AUT", [("D", "2025-11-19", "hard")]),
    (9, "atp", "Wimbledon", [("G", "2025-06-30", "grass")]),
    (10, "wta", "Wimbledon", [("G", "2025-06-30", "grass")]),
    (11, "wta", "Montevideo 125", [("C", "2023-12-04", "clay")]),
    (12, "atp", "Brisbane 1 CH", [("C", "2026-02-02", "hard")]),
    (13, "atp", "Old Town CH", [("C", "2015-11-09", "clay")]),           # outside the 6-year window
    (14, "atp", "Kobe CH 2027", [("C", "2026-12-01", "clay")]),         # after the match: never used
]
ALIASES = [("wimbledon", 9), ("wimbledon", 10), ("tour finals", 6)]


def surfaces() -> TournamentSurfaces:
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE tournament (tournament_id INTEGER PRIMARY KEY, tour TEXT, canonical_name TEXT);
        CREATE TABLE tournament_edition (edition_id INTEGER PRIMARY KEY, tournament_id INTEGER, level TEXT,
                                         start_date TEXT, surface TEXT);
        CREATE TABLE tournament_alias (alias_normalized TEXT, tournament_id INTEGER);""")
    for tid, tour, name, eds in TOURNAMENTS:
        con.execute("INSERT INTO tournament VALUES (?,?,?)", (tid, tour, name))
        con.executemany("INSERT INTO tournament_edition (tournament_id, level, start_date, surface) VALUES (?,?,?,?)",
                        [(tid, *e) for e in eds])
    con.executemany("INSERT INTO tournament_alias VALUES (?,?)", ALIASES)
    return TournamentSurfaces.from_db(con)


class NameTest(unittest.TestCase):
    def test_kalshi_tournament(self):
        self.assertEqual(kalshi_tournament({"details": {"tournament_name": "ATP Challenger Kobe"}}), "ATP Challenger Kobe")
        rules = "If X wins the X vs Y professional tennis match in the 2026 {} after a ball has been played, ..."
        for raw, want in [("ATP Challenger Metz Round Of 32", "ATP Challenger Metz"),
                          ("ATP Challenger Kobe Qualification Final", "ATP Challenger Kobe"),
                          ("ATP Challenger Champaign Qualification Round 1", "ATP Challenger Champaign"),
                          ("ATP Challenger Lima Final", "ATP Challenger Lima"),
                          ("ATP World Tour Finals Men Singles Ivan Lendl Group", "ATP World Tour Finals Men Singles"),
                          ("WTA Finals Women Singles Final", "WTA Finals Women Singles")]:
            self.assertEqual(kalshi_tournament(None, rules.format(raw)), want)
        self.assertEqual(kalshi_tournament(None, "no tournament here"), "")


class LookupTest(unittest.TestCase):
    def setUp(self):
        self.ts = surfaces()

    def test_challenger_by_city(self):
        g = self.ts.lookup("atp", "ATP Challenger Kobe", "2026-11-16")
        self.assertEqual((g.surface, g.how), ("hard", "challenger: Kobe CH 2025-11-10"))
        self.assertEqual(self.ts.lookup("atp", "ATP Challenger Brisbane", "2026-11-16").tournament_id, 12)
        self.assertEqual(self.ts.lookup("wta", "WTA Challenger Montevideo", "2026-11-30").surface, "clay")

    def test_nearest_time_of_year(self):
        self.assertEqual(self.ts.lookup("atp", "ATP Challenger Lima", "2026-06-22").surface, "clay")
        self.assertEqual(self.ts.lookup("atp", "ATP Challenger Lima", "2026-10-26").surface, "hard")

    def test_fallback_tiers(self):
        g = self.ts.lookup("atp", "ATP Challenger Metz", "2026-11-15")
        self.assertEqual((g.surface, g.how), ("hard", "tour-level same city: Metz 2025-11-02"))
        g = self.ts.lookup("atp", "ATP Challenger Champaign", "2026-11-15")
        self.assertEqual(g.how, "itf same city: M25 Champaign IL 2025-07-21")
        self.assertIsNone(self.ts.lookup("atp", "ATP Challenger Nowhere", "2026-11-15"))
        self.assertIsNone(self.ts.lookup("atp", "ATP Challenger Old Town", "2026-11-15"))    # > 6 years old
        self.assertIsNone(self.ts.lookup("atp", "", "2026-11-15"))

    def test_no_future_editions(self):
        g = self.ts.lookup("atp", "ATP Challenger Kobe", "2026-11-16")
        self.assertNotEqual(g.tournament_id, 14)

    def test_finals_and_tour(self):
        g = self.ts.lookup("atp", "ATP World Tour Finals Men Singles", "2026-11-16")
        self.assertEqual(g.how, "finals: Tour Finals 2025-11-09")                 # not NextGen, not Davis Cup
        g = self.ts.lookup("wta", "WTA Wimbledon Women Singles", "2026-06-29")
        self.assertEqual((g.surface, g.tournament_id), ("grass", 10))           # the WTA one
        self.assertIsNone(self.ts.lookup("wta", "WTA World Tour Finals Women Singles", "2026-11-16"))


class RetrieverTest(unittest.TestCase):
    def test_history_between_edition_and_fallback(self):
        pull = load_pull(FIX)

        def editions(a, b, day):
            return EditionInfo(99, "clay", 3, "tb") if {a, b} == {3, 4} else None   # Medvedev v Cobolli
        r = by_event(snapshots(pull, FixedPredictor(), resolver(), editions, surface="grass", surfaces=surfaces()))
        alc = r["KXATPMATCH-26NOV16ALCTIA"]
        self.assertEqual(alc.tournament, "ATP World Tour Finals Men Singles")
        self.assertEqual((alc.surface, alc.surface_from), ("hard", "history: finals: Tour Finals 2025-11-09"))
        self.assertEqual(alc.query.surface, "hard")
        self.assertEqual(r["KXATPMATCH-26NOV16MEDCOB"].surface_from, "edition")
        sh = r["KXATPCHALLENGERMATCH-26NOV15SHAMON"]
        self.assertEqual((sh.surface, sh.surface_from), ("hard", "history: tour-level same city: Metz 2025-11-02"))
        gg = r["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]                            # Drummondville: not in this db
        self.assertEqual((gg.tournament, gg.surface, gg.surface_from), ("ATP Challenger Drummondville", "grass", "fallback"))
        self.assertEqual(alc.to_json()["tournament"], "ATP World Tour Finals Men Singles")


if __name__ == "__main__":
    unittest.main()
