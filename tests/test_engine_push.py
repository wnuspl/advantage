"""scripts/engine_push.py: what goes up, and that the server verifies the cache key before
it stops the watch or touches a file (no network here)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))

import engine_push                                                     # noqa: E402


class EnginePushTest(unittest.TestCase):
    def test_files(self):
        arcs = [a for _, a in engine_push.files_to_push(790, Path("tennis.db"))]
        self.assertEqual(arcs[:2], ["tennis.db", "models/engine/run790.pkl"])
        self.assertTrue(all(a.startswith("models/engine/state/run790/") for a in arcs[2:]))

    def test_verify_before_anything_changes(self):
        s = engine_push.install_script("advantage", 790, "ab" * 32, restart=True)
        v, stop, mv, start = (s.index(x) for x in ("cache_key(", "systemctl stop", "mv -f", "systemctl start"))
        self.assertLess(v, stop)
        self.assertLess(stop, mv)
        self.assertLess(mv, start)
        self.assertIn("exit 3", s[:stop])                               # mismatch: leave before the stop
        self.assertNotIn("systemctl", engine_push.install_script("advantage", 790, "ab" * 32, restart=False))

    def test_snippet_compiles(self):
        compile(engine_push.verify_snippet(790, "ab" * 32), "verify", "exec")


if __name__ == "__main__":
    unittest.main()
