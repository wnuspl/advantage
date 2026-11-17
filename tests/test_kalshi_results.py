"""Kalshi results export (src/tennis/kalshi_results.py): live.db settlements -> immutable
raw files a tennis.db loader can read later. In memory / temp folders, no network."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import kalshi_results as kr                                  # noqa: E402
from tennis.engine.live import load_pull                                 # noqa: E402
from test_live import FIX                                                # noqa: E402
from tennis.engine.decide import Policy, decide_all                       # noqa: E402
from tennis.engine.live import snapshots                                 # noqa: E402
from tennis.engine.recorder import Recorder                              # noqa: E402
from test_live import NoNetwork, resolver                                # noqa: E402
from test_recorder import C1, C2, FINAL_LIVE, GG, Const, finished_event  # noqa: E402

MS = {"id": "ms-goggur", "source_id": "sr:sport_event:1", "start_date": "2026-11-15T18:00:00Z",
      "details": {"best_of": "3", "round": "Round of 32", "tournament_name": "ATP Challenger Champaign",
                  "first_competitor_id": C1, "second_competitor_id": C2}}


def stats(spw, spl, f_in, s_in, df, f_won, s_won, aces, bpw, bpt, sgw):
    return {"service_points_won": spw, "service_points_lost": spl, "first_serve_successful": f_in,
            "second_serve_successful": s_in, "double_faults": df, "first_serve_points_won": f_won,
            "second_serve_points_won": s_won, "aces": aces, "breakpoints_won": bpw,
            "total_breakpoints": bpt, "service_games_won": sgw}


# 7-6 6-4 for C1. C1 served 11 games, all held; C2 served 11, broken once.
LIVE = {**FINAL_LIVE, "competitor1_seed": 3,
        "competitor1_statistics": stats(50, 20, 45, 22, 3, 38, 12, 9, 1, 4, 11),
        "competitor2_statistics": stats(45, 27, 50, 20, 2, 33, 12, 4, 0, 2, 10)}


class ExportTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.pull = load_pull(FIX)
        self.rec = Recorder(":memory:")
        self.wid = self.rec.start_watch("const", Policy(), ["KXATPMATCH"], 60)

    def tearDown(self):
        self.rec.close()
        super().tearDown()

    def record(self, pull):
        rows = snapshots(pull, Const(), resolver())
        self.rec.record_poll(self.wid, pull, rows, decide_all(rows, pull.books, Policy(in_play=True)))

    def settle(self, live=LIVE, milestone=MS, **kw):
        self.record(self.pull)
        ev = next(t for t in self.rec.to_settle(set(), now=self.pull.as_of) if t["event"] == GG)
        return self.rec.settle(ev, finished_event(**kw), live, seen_at="2026-11-15T21:00:00+00:00",
                               milestone=milestone)

    def doc(self, skip=frozenset()):
        return kr.export_docs(self.rec.con, set(skip))

    def test_settled_match_round_trips(self):
        self.settle()
        (d,) = self.doc()
        self.assertEqual((d["format"], d["live_from"], d["milestone_from"]), (kr.FORMAT, "settlement", "settlement"))
        r, why = kr.parse(json.loads(json.dumps(d)))
        self.assertEqual(why, "")
        self.assertEqual((r["tour"], r["level"], r["tournament"], r["round"], r["best_of"], r["day_utc"]),
                         ("atp", "challenger", "ATP Challenger Champaign", "Round of 32", 3, "2026-11-15"))
        self.assertEqual((r["winner_comp"], r["loser_comp"], r["status"], r["score"], r["winner_seed"]),
                         (C1, C2, "completed", "7-6 6-4", 3))
        self.assertEqual(r["winner_name"], d["event"]["a_name"])
        self.assertEqual(r["known_at"], "2026-11-15T21:00:00+00:00")
        w, l = r["winner_stats"], r["loser_stats"]
        self.assertEqual((w["serve_points"], w["first_won"] + w["second_won"], w["aces"], w["serve_games"]),
                         (70, 50, 9, 11))
        self.assertEqual((l["bp_faced"], l["bp_saved"], l["serve_games"]), (4, 3, 11))
        self.assertTrue(w["exact"] and l["exact"])

    def test_old_settlement_recovers_milestone_and_live_from_polls(self):
        pull = copy.deepcopy(self.pull)
        pull.milestones[GG] = MS
        pull.live[MS["id"]] = LIVE
        self.record(pull)
        self.settle(live=None, milestone=None)          # as written before 2026-11-16
        (d,) = self.doc()
        self.assertTrue(d["milestone_from"].startswith("poll ") and d["live_from"].startswith("poll "))
        self.assertEqual(kr.parse(d)[0]["score"], "7-6 6-4")

    def test_write_is_immutable_and_exports_once(self):
        self.settle()
        with tempfile.TemporaryDirectory() as tmp:
            out = kr.write_export(self.doc(), Path(tmp), {"live_db": ":memory:"}, stamp="20261116T000000Z")
            self.assertEqual(out.name, "kalshi_results-20261116T000000Z")
            files = list(out.glob("matches/*/*.json"))
            self.assertEqual([f.relative_to(out).as_posix() for f in files],
                             [f"matches/KXATPCHALLENGERMATCH/{GG}.json"])
            self.assertEqual(json.loads((out / "export.json").read_text(encoding="utf-8"))["matches"], 1)
            self.assertEqual(kr.read(files[0])["event"]["event"], GG)
            self.assertEqual(kr.exported_events(Path(tmp)), {GG})
            self.assertEqual(self.doc(kr.exported_events(Path(tmp))), [])
            self.assertIsNone(kr.write_export([], Path(tmp), {}))
            with self.assertRaises(FileExistsError):
                kr.write_export(self.doc(), Path(tmp), {}, stamp="20261116T000000Z")

    def test_skips_and_flags(self):
        self.settle()
        (d,) = self.doc()
        void = copy.deepcopy(d)
        void["settlement"]["winner"] = None
        self.assertEqual(kr.parse(void), (None, "no_winner"))
        ret = copy.deepcopy(d)
        ret["live"]["match_status"] = "retired"
        self.assertEqual(kr.parse(ret)[0]["score"], "7-6 6-4 RET")
        odd = copy.deepcopy(d)
        odd["live"]["match_status"] = "suspended"
        self.assertEqual(kr.parse(odd), (None, "match_status:suspended"))
        bad = copy.deepcopy(d)
        bad["live"]["competitor2_id"] = "other"
        self.assertEqual(kr.parse(bad), (None, "live_competitors_mismatch"))
        off = copy.deepcopy(d)
        off["live"]["competitor1_statistics"]["first_serve_successful"] += 1   # feed off by one
        self.assertFalse(kr.parse(off)[0]["winner_stats"]["exact"])
        broken = copy.deepcopy(d)
        broken["live"]["competitor1_statistics"]["first_serve_points_won"] += 1  # won no longer adds up
        self.assertIsNone(kr.parse(broken)[0]["winner_stats"])


if __name__ == "__main__":
    unittest.main()
