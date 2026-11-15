import sys
import unittest
from pathlib import Path

import numpy as np

# archived test: archive src first (merges with the live tennis package), then live src,
# live tests (test_build.Workspace), archived scripts, this folder
_A = Path(__file__).resolve().parents[1]
for _p in (_A.parents[1] / "tests", _A.parents[1] / "src", _A / "scripts", _A / "tests", _A / "src"):
    sys.path.insert(0, str(_p))

from tennis import phase  # noqa: E402
from tennis.mcp_notation import parse_point  # noqa: E402


def _pts(rows):
    """rows: (match, server, returner, server_won, phase, srv_net, ret_net)."""
    a = np.array(rows, dtype=np.int64)
    return phase.Points(a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.int8), a[:, 4].astype(np.int8),
                        a[:, 5].astype(np.int8), a[:, 6].astype(np.int8))


class PhaseTest(unittest.TestCase):
    def test_point_phases(self):
        self.assertEqual(phase.point_phases(parse_point("4*", None, True)), (0, False, False))
        self.assertEqual(phase.point_phases(parse_point("6f2b3f1*", None, False))[0], 1)
        self.assertEqual(phase.point_phases(parse_point("5f1+b2v3*", None, False)), (1, False, True))
        self.assertEqual(phase.point_phases(parse_point("6f2f3f1b2b3f1f2n@", None, True))[0], 2)

    def test_tally_and_point_in_time(self):
        meta = {1: ("2020-01-06", "atp", "clay"), 2: ("2020-02-03", "atp", "hard")}
        pts = _pts([(1, 10, 20, 1, 0, 0, 0), (1, 10, 20, 0, 2, 1, 0), (1, 20, 10, 1, 1, 0, 1),
                    (2, 10, 30, 1, 0, 0, 0)])
        t = phase.tally(pts, meta)
        self.assertEqual(len(t.match_id), 4)     # (1,10) (1,20) (2,10) (2,30)
        h = phase.History(t)
        # strictly before: nothing on the first match's own date
        self.assertEqual(h.player(10, "2020-01-06")[..., 0].sum(), 0)
        st = h.player(10, "2020-02-03")
        clay = phase.SURF_INDEX["clay"]
        self.assertEqual(st[0, clay, 0], 2)          # serve points
        self.assertEqual(st[0, clay, 1], 1)          # serve points won
        self.assertEqual(st[0, clay, 2 + 3], 1)      # server-net points
        self.assertEqual(st[1, clay, 0], 1)          # return points
        self.assertEqual(st[1, clay, 1], 0)          # return point lost
        self.assertEqual(st[1, clay, 2 + 4], 1)      # returner-net point
        self.assertEqual(h.player(10, "2021-01-01")[0, phase.SURF_INDEX["hard"], 0], 1)
        # population for year Y only uses earlier years
        self.assertEqual(h.population("atp", "2020-06-01").sum(), 0)
        self.assertGreater(h.population("atp", "2021-06-01").sum(), 0)

    def _stats(self, n_by_phase, w_by_phase, surf=0):
        st = np.zeros((2, phase.NS, phase.NF))
        for side in (0, 1):
            n = np.asarray(n_by_phase[side], float); w = np.asarray(w_by_phase[side], float)
            st[side, surf, 0] = n[:3].sum(); st[side, surf, 1] = w[:3].sum()
            st[side, surf, 2:2 + phase.NK] = n; st[side, surf, 2 + phase.NK:] = w
        return st

    def test_profile_population_player_is_zero(self):
        n = [[500, 300, 200, 50, 20], [500, 300, 200, 20, 50]]
        w = [[350, 180, 100, 35, 8], [150, 120, 100, 12, 30]]
        st = self._stats(n, w)
        for v in ("pooled", "surface", "same"):
            pr = phase.profile(st, st * 10, v, 0)
            np.testing.assert_allclose(pr["d"], 0, atol=1e-12)
            np.testing.assert_allclose(pr["e"], 0, atol=1e-12)
            np.testing.assert_allclose(pr["t"], 0, atol=1e-12)

    def test_surface_variant_removes_surface_mix(self):
        # population: clay long points more common; player plays only clay, like the clay pop.
        pop = np.zeros((2, phase.NS, phase.NF))
        hard = self._stats([[600, 300, 100, 0, 0]] * 2, [[420, 180, 50, 0, 0], [180, 120, 50, 0, 0]], 0)
        clay = self._stats([[300, 400, 300, 0, 0]] * 2, [[200, 230, 160, 0, 0], [100, 170, 140, 0, 0]], 1)
        pop = (hard + clay) * 20
        player = clay.copy()
        pooled = phase.profile(player, pop, "pooled", 1)
        surf = phase.profile(player, pop, "surface", 1)
        self.assertGreater(np.abs(pooled["t"]).max(), 0.05)       # pooled sees "steering"
        self.assertLess(np.abs(surf["t"]).max(), 1e-12)           # surface-adjusted does not
        self.assertLess(np.abs(surf["e"]).max(), 1e-12)

    def test_matchup_sign(self):
        z = np.zeros((2, phase.NK))
        # A wins more short serve points than its level; B as returner steers to short points
        a = {"e": z.copy(), "t": z.copy()}; b = {"e": z.copy(), "t": z.copy()}
        a["e"][0, 0] = 0.05
        b["t"][1, 0] = 0.10
        self.assertAlmostEqual(phase.matchup(a, b), 0.005)
        # A steers its serve points long, where B returns well: hurts A
        a2 = {"e": z.copy(), "t": z.copy()}; b2 = {"e": z.copy(), "t": z.copy()}
        a2["t"][0, 2] = 0.1; b2["e"][1, 2] = 0.04
        self.assertAlmostEqual(phase.matchup(a2, b2), -0.004)


if __name__ == "__main__":
    unittest.main()
