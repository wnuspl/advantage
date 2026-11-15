"""Track B1: the Kalshi retriever on saved raw pulls (no network) and the player resolver.

Fixture: tests/fixtures/kalshi/20261115T223132Z, a subset of a real pull (7 events: two
ATP Finals not started, three Challenger matches live, two not started; with live entries
of other matches of the same players, e.g. Alcaraz's other round-robin matches). It is
a pre-milestones.json pull, so live data maps by competitor pair alone.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import kalshi                                                     # noqa: E402
from tennis.engine import FixedPredictor, MatchQuery, Prediction             # noqa: E402
from tennis.engine.live import (DbEditions, EditionInfo, load_pull, parse_events,   # noqa: E402
                                plain_name, save_pull, snapshots)
from tennis.engine.players import Resolver                                   # noqa: E402

FIX = HERE / "fixtures" / "kalshi" / "20261115T223132Z"
NAMES = ["Carlos Alcaraz", "Frances Tiafoe", "Daniil Medvedev", "Flavio Cobolli", "Vignesh Gogineni",
         "Volodymyr Gurenko", "Giles Hussey", "Lucas Horve", "Enzo Aguiard", "Christopher Oconnell",
         "Marat Sharipov", "Inaki Montes De La Torre", "Keshav Chopra"]          # Mekhael missing


SCHEMA = """
    CREATE TABLE player (player_id INTEGER PRIMARY KEY, tour TEXT, canonical_name TEXT,
                         name_first TEXT, name_last TEXT, is_placeholder INTEGER);
    CREATE TABLE tournament_edition (edition_id INTEGER PRIMARY KEY, tournament_id INTEGER,
                         start_date TEXT, end_date TEXT, surface TEXT, best_of INTEGER);
    CREATE TABLE matches (match_id INTEGER PRIMARY KEY, edition_id INTEGER, match_date TEXT,
                         winner_id INTEGER, loser_id INTEGER, score TEXT, status TEXT, best_of INTEGER);"""


def mini_db(players, activity) -> sqlite3.Connection:
    """In-memory tennis.db subset. players: (pid, tour, canonical, first, last, placeholder);
    activity: pid -> (n matches, ISO date), as wins over a placeholder opponent (id 0)."""
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO player VALUES (?,?,?,?,?,?)", [(0, "atp", "X X", "X", "X", 1), *players])
    days = sorted({d for _, d in activity.values()})
    con.executemany("INSERT INTO tournament_edition VALUES (?,?,?,?,?,?)",
                    [(k, 900 + k, d, d, "hard", 3) for k, d in enumerate(days, 1)])
    ed = {d: k for k, d in enumerate(days, 1)}
    con.executemany("INSERT INTO matches (edition_id, match_date, winner_id, loser_id, score, status, best_of) "
                    "VALUES (?,?,?,0,'6-1 6-1','completed',3)",
                    [(ed[d], d, pid) for pid, (n, d) in activity.items() for _ in range(n)])
    return con


def resolver(today="2026-11-15"):
    players = []
    for k, n in enumerate(NAMES, 1):
        first, last = n.split(" ", 1)
        players.append((k, "atp", n, first, last, 0))
    con = mini_db(players, {k: (10, "2026-11-01") for k in range(1, len(NAMES) + 1)})
    return Resolver.from_db(con, today, alias="main")


class Recorder:
    """A predictor that records its queries."""
    name = "recorder"

    def __init__(self):
        self.queries: list[MatchQuery] = []

    def predict(self, q):
        self.queries.append(q)
        return Prediction(p_a=0.6, model=self.name, in_play=q.live is not None)


def by_event(rows):
    return {r.event.event: r for r in rows}


class NoNetwork(unittest.TestCase):
    def setUp(self):
        self._get = kalshi.get
        kalshi.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used"))

    def tearDown(self):
        kalshi.get = self._get


class ParseTest(NoNetwork):
    def test_events_and_rejects(self):
        pull = load_pull(FIX)
        evs, rej = parse_events(pull.events)
        self.assertEqual(len(evs), 7)
        self.assertEqual(rej, [])
        e = next(x for x in evs if x.event == "KXATPMATCH-26NOV16ALCTIA")
        self.assertEqual((e.tour, e.a.name, e.b.name), ("atp", "Carlos Alcaraz", "Frances Tiafoe"))
        self.assertEqual(e.a.comp, "527915ea-e368-4c7f-a203-c83ebb6f6572")
        self.assertEqual(e.start, "2026-11-16T16:00:00Z")
        bad = copy.deepcopy(pull.events)
        bad["KXATPMATCH"][0]["markets"][1]["status"] = "closed"
        evs, rej = parse_events(bad)
        self.assertEqual(len(evs), 6)
        self.assertEqual(rej, [(bad["KXATPMATCH"][0]["event_ticker"], "1 active markets, expected 2")])

    def test_legacy_pull_loads(self):
        pull = load_pull(FIX)
        self.assertEqual(pull.as_of, "2026-11-15T22:31:32+00:00")
        self.assertEqual(pull.milestones, {})
        self.assertEqual(len(pull.books), 14)


class SnapshotTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.pull = load_pull(FIX)

    def test_every_event_listed_and_priced_or_explained(self):
        rows = snapshots(self.pull, FixedPredictor(), resolver())
        self.assertEqual(len(rows), 7)
        r = by_event(rows)
        self.assertEqual(r["KXATPCHALLENGERMATCH-26NOV15MEKCHO"].skip, "unresolved: Nicholas Mekhael")
        priced = [x for x in rows if not x.skip]
        self.assertEqual(len(priced), 6)
        for x in priced:
            self.assertEqual(x.p_model, 0.5)                     # the placeholder has no opinion
            self.assertIsNotNone(x.p_market)
            self.assertAlmostEqual(x.deviation, 0.5 - x.p_market)

    def test_market_matches_books(self):
        x = by_event(snapshots(self.pull, FixedPredictor(), resolver()))["KXATPMATCH-26NOV16ALCTIA"]
        ba, bb = self.pull.books[x.event.a.ticker], self.pull.books[x.event.b.ticker]
        self.assertAlmostEqual(x.p_market, kalshi.fair_prob(ba, bb))
        self.assertGreater(x.p_market, 0.8)                      # Alcaraz the favourite
        self.assertEqual(x.ask_a, kalshi.ladder(ba, "", bb, "")[0].price)

    def test_live_by_competitor_pair(self):
        rec = Recorder()
        r = by_event(snapshots(self.pull, rec, resolver()))
        alc = r["KXATPMATCH-26NOV16ALCTIA"]                      # Alcaraz is in 3 live entries
        self.assertEqual((alc.status, alc.milestone), ("not_started", "cf1f8ea2-e6fc-4a6a-bf67-4d7ff8ac158e"))
        self.assertIsNone(alc.query.live)
        gg = r["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        self.assertEqual(gg.status, "live")
        self.assertEqual(gg.live.problem, "")
        self.assertEqual((gg.live.sets_a, gg.live.sets_b, gg.live.games_a, gg.live.games_b), (1, 0, 3, 2))
        self.assertEqual(gg.live.raw, "7-6 3-2 40-0 *A serving")
        self.assertIs(gg.query.live, gg.live)
        self.assertTrue(gg.prediction.in_play)
        sh = r["KXATPCHALLENGERMATCH-26NOV15SHAMON"]             # past its start time, feed says not started
        self.assertEqual((sh.status, sh.skip), ("not_started", ""))
        mt = sh.event.a if sh.event.a.name.startswith("Inaki") else sh.event.b
        self.assertEqual(mt.resolved, "exact")                    # "Montes-de la Torre"
        oc = r["KXATPCHALLENGERMATCH-26NOV15AGUOCO"]
        self.assertEqual(oc.event.b.resolved, "exact")           # "O'Connell" == "Oconnell"
        self.assertEqual(len(rec.queries), 6)

    def test_market_order_does_not_matter(self):
        flipped = copy.deepcopy(self.pull)
        for e in flipped.events["KXATPCHALLENGERMATCH"]:
            e["markets"].reverse()
        for a, b in zip(snapshots(self.pull, FixedPredictor(), resolver()),
                        snapshots(flipped, FixedPredictor(), resolver())):     # sides ordered by ticker
            self.assertEqual(a.to_json(), b.to_json())
            self.assertLess(a.event.a.ticker, a.event.b.ticker)
        gg = by_event(snapshots(flipped, FixedPredictor(), resolver()))["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        self.assertEqual((gg.event.a.name, gg.live.sets_a, gg.live.games_a), ("Vignesh Gogineni", 1, 3))

    def test_closed_and_started_without_feed(self):
        pull = copy.deepcopy(self.pull)
        gg = by_event(snapshots(pull, FixedPredictor(), resolver()))["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        pull.live[gg.milestone]["status"] = "closed"
        sh = by_event(snapshots(self.pull, FixedPredictor(), resolver()))["KXATPCHALLENGERMATCH-26NOV15SHAMON"]
        del pull.live[sh.milestone]
        r = by_event(snapshots(pull, FixedPredictor(), resolver()))
        self.assertEqual(r["KXATPCHALLENGERMATCH-26NOV15GOGGUR"].skip, "finished")
        self.assertEqual(r["KXATPCHALLENGERMATCH-26NOV15SHAMON"].skip, "started, no live score")
        r = by_event(snapshots(pull, FixedPredictor(), resolver(), include_started=True))
        self.assertEqual((r["KXATPCHALLENGERMATCH-26NOV15SHAMON"].status,
                          r["KXATPCHALLENGERMATCH-26NOV15SHAMON"].skip), ("no_feed", ""))

    def test_unreadable_live_score_is_not_priced(self):
        pull = copy.deepcopy(self.pull)
        gg = by_event(snapshots(pull, FixedPredictor(), resolver()))["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        pull.live[gg.milestone]["server"] = ""
        rec = Recorder()
        x = by_event(snapshots(pull, rec, resolver()))["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        self.assertEqual(x.skip, "live score unreadable: server unknown")
        self.assertIsNone(x.prediction)
        self.assertEqual(len([q for q in rec.queries if q.live is not None]), 1)   # HUSHOR only

    def test_rules_and_surface(self):
        def editions(a, b, day):
            return EditionInfo(99, "clay", 5, "adv") if {a, b} == {1, 2} else None
        rec = Recorder()
        r = by_event(snapshots(self.pull, rec, resolver(), editions, surface="hard"))
        alc = r["KXATPMATCH-26NOV16ALCTIA"]
        self.assertEqual((alc.surface, alc.surface_from, alc.best_of, alc.best_of_from, alc.final, alc.edition_id),
                         ("clay", "edition", 5, "edition", "adv", 99))
        q = alc.query
        self.assertEqual((q.tour, q.player_a, q.player_b, q.on, q.surface, q.best_of, q.final, q.edition_id),
                         ("atp", 1, 2, "2026-11-16", "clay", 5, "adv", 99))
        med = r["KXATPMATCH-26NOV16MEDCOB"]
        self.assertEqual((med.surface, med.surface_from, med.best_of, med.best_of_from, med.final),
                         ("hard", "fallback", 3, "default", "tb"))
        med = by_event(snapshots(self.pull, rec, resolver()))["KXATPMATCH-26NOV16MEDCOB"]
        self.assertEqual((med.surface, med.surface_from), (None, "unknown"))

    def test_saved_pull_with_milestones(self):
        pull = copy.deepcopy(self.pull)
        rows = by_event(snapshots(pull, FixedPredictor(), resolver()))
        alc = rows["KXATPMATCH-26NOV16ALCTIA"]
        gg = rows["KXATPCHALLENGERMATCH-26NOV15GOGGUR"]
        pull.milestones = {
            alc.event.event: {"id": alc.milestone, "details": {"best_of": "5",
                              "first_competitor_id": alc.event.a.comp, "second_competitor_id": alc.event.b.comp}},
            gg.event.event: {"id": gg.milestone, "details": {"best_of": "3",       # wrong pair: refused
                             "first_competitor_id": gg.event.a.comp, "second_competitor_id": alc.event.b.comp}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            d = save_pull(pull, Path(tmp))
            self.assertEqual(d.name, "20261115T223132Z")
            meta = json.loads((d / "pull.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["events"], meta["milestones"]), (7, 2))
            with self.assertRaises(FileExistsError):              # raw pulls are immutable
                save_pull(pull, Path(tmp))
            back = load_pull(d)
        self.assertEqual(back.as_of, pull.as_of)
        r = by_event(snapshots(back, FixedPredictor(), resolver()))
        self.assertEqual((r[alc.event.event].best_of, r[alc.event.event].best_of_from), (5, "milestone"))
        self.assertEqual(r[gg.event.event].skip, "milestone competitors do not match the markets")
        js = r[alc.event.event].to_json()
        json.dumps(js)
        self.assertEqual((js["prediction"]["p_a"], js["best_of"], js["a"]["player_id"]), (0.5, 5, 1))


class ResolverTest(unittest.TestCase):
    def test_birth_year_suffix(self):
        self.assertEqual(plain_name("Cezar Cretu (b. 2001)"), "Cezar Cretu")
        self.assertEqual(plain_name("Carlos Alcaraz"), "Carlos Alcaraz")

    def test_rules(self):
        players = [(1, "atp", "Christopher Oconnell", "Christopher", "Oconnell", 0),
                   (2, "atp", "Chris Wells", "Chris", "Wells", 0),                  # inactive
                   (3, "atp", "Inaki Montes De La Torre", "Inaki", "Montes De La Torre", 0),
                   (4, "atp", "John Smith", "John", "Smith", 0),
                   (5, "atp", "John Smith", "John", "Smith", 0),
                   (6, "atp", "Unknown", "", "Unknown", 1),
                   (7, "wta", "Anna Bondar", "Anna", "Bondar", 0)]
        act = {1: (50, "2026-10-01"), 2: (80, "1975-06-01"), 3: (40, "2026-10-01"), 4: (5, "2026-01-01"),
               5: (90, "2026-09-01"), 7: (30, "2026-10-01")}
        r = Resolver.from_db(mini_db(players, act), "2026-11-15", alias="main")
        self.assertEqual(r("atp", "Christopher O'Connell"), (1, "exact"))
        self.assertEqual(r("atp", "Chris O'Connell"), (1, "last+initial"))
        self.assertEqual(r("atp", "Colson Wells"), (None, "unresolved"))           # recency guard
        self.assertEqual(r("atp", "Inaki Montes-de la Torre"), (3, "exact"))
        self.assertEqual(r("atp", "I. Montes-de la Torre")[0], 3)
        self.assertEqual(r("atp", "John Smith"), (5, "exact (busiest of 2)"))
        self.assertEqual(r("atp", "Unknown"), (None, "unresolved"))
        self.assertEqual(r("atp", "Anna Bondar"), (None, "unresolved"))            # other tour


class DbTest(unittest.TestCase):
    """Resolver.from_db and DbEditions on the minimal in-memory schema."""

    def setUp(self):
        self.con = con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO player VALUES (?,?,?,?,?,0)",
                        [(1, "atp", "Ann A", "Ann", "A"), (2, "atp", "Bob B", "Bob", "B"), (3, "atp", "Cy C", "Cy", "C")])
        con.executemany("INSERT INTO tournament_edition VALUES (?,?,?,?,?,?)",
                        [(10, 1, "2025-11-09", "2025-11-16", "hard", 3),
                         (11, 1, "2026-11-08", "2026-11-15", "hard", 3),
                         (12, 2, "2026-11-08", "2026-11-15", "clay", 3)])
        con.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?)",
                        [(1, 10, "2025-11-10", 1, 2, "6-4 3-6 7-6(5)", "completed", 3),
                         (2, 11, "2026-11-09", 1, 3, "6-3 6-4", "completed", 3),
                         (3, 11, "2026-11-10", 2, 3, "6-3 6-4", "completed", 3),
                         (4, 12, "2026-11-10", 3, 1, "6-3 6-4", "completed", 3)])

    def test_editions(self):
        ed = DbEditions(self.con)
        self.assertEqual(ed(1, 2, "2026-11-14"), EditionInfo(11, "hard", 3, "tb"))
        self.assertIsNone(ed(2, 3, "2026-12-20"))                   # nothing running
        self.assertEqual(ed(1, 3, "2026-11-14").edition_id, 11)     # both in 11 and 12: latest start, then any

    def test_resolver_from_db(self):
        r = Resolver.from_db(self.con, "2026-11-15", alias="main")
        self.assertEqual(r("atp", "Bob B"), (2, "exact"))
        self.assertEqual(r("atp", "Robert B"), (None, "unresolved"))
        self.assertEqual(r("atp", "B. Bob")[0], None)


if __name__ == "__main__":
    unittest.main()
