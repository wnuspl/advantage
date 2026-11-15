"""calibrate.py (archived v1): walk-forward calibration alphas. Moved out of tests/test_points.py
at the reboot; calibration is re-added in milestone A1."""
from __future__ import annotations

import sys
import unittest
from array import array
from pathlib import Path

import numpy as np

# archived test: archive src first (merges with the live tennis package), then live src,
# live tests (test_build.Workspace), archived scripts, this folder
_A = Path(__file__).resolve().parents[1]
for _p in (_A.parents[1] / "tests", _A.parents[1] / "src", _A / "scripts", _A / "tests", _A / "src"):
    sys.path.insert(0, str(_p))

from tennis import matches   # noqa: E402


class CalibrateTest(unittest.TestCase):
    def test_alpha_uses_only_earlier_matches(self):
        from tennis import calibrate, tuning
        rng = np.random.default_rng(0)
        m = matches.empty_matches()
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


if __name__ == "__main__":
    unittest.main()
