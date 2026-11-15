from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis.engine import FixedPredictor, MatchQuery, Prediction, Predictor   # noqa: E402
from tennis.kalshi import LiveState                                          # noqa: E402


class PredictorSeamTest(unittest.TestCase):
    def setUp(self):
        self.q = MatchQuery(tour="atp", player_a=1, player_b=2, on="2026-11-20", surface="hard")

    def test_fixed_is_a_predictor(self):
        self.assertIsInstance(FixedPredictor(), Predictor)

    def test_fixed_prediction(self):
        pred = FixedPredictor(0.6).predict(self.q)
        self.assertIsInstance(pred, Prediction)
        self.assertEqual(pred.p_a, 0.6)
        self.assertFalse(pred.in_play)
        self.assertEqual(pred.model, "fixed(0.6)")

    def test_live_state_accepted_but_not_used(self):
        q = MatchQuery(**{**self.q.__dict__, "live": LiveState(status="live", sets_a=1)})
        pred = FixedPredictor().predict(q)
        self.assertFalse(pred.in_play)
        self.assertIn("live state ignored", pred.notes)

    def test_bounds(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                FixedPredictor(bad)


if __name__ == "__main__":
    unittest.main()
