"""Odds benchmark: xlsx reader, tennis-data parsing, name matching, linking, devig,
blend fit and backtest. Synthetic data only (a tiny .xlsx is written on the fly)."""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis import odds                       # noqa: E402
from tennis.xlsx_lite import read_first_sheet, serial_to_iso   # noqa: E402


def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def write_xlsx(path: Path, header: list, rows: list) -> None:
    """Minimal valid .xlsx: strings shared, numbers inline."""
    strings, sidx = [], {}

    def cell(ref, v):
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            return f'<c r="{ref}"><v>{v}</v></c>'
        if v not in sidx:
            sidx[v] = len(strings)
            strings.append(v)
        return f'<c r="{ref}" t="s"><v>{sidx[v]}</v></c>'

    xml_rows = []
    for r, vals in enumerate([header] + rows, start=1):
        cells = "".join(cell(f"{_col(c)}{r}", v) for c, v in enumerate(vals))
        xml_rows.append(f'<row r="{r}">{cells}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    sheet = f'<?xml version="1.0"?><worksheet {ns}><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    sst = (f'<?xml version="1.0"?><sst {ns}>'
           + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    wb = (f'<?xml version="1.0"?><workbook {ns} {rns}><sheets>'
          '<sheet name="2024" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)


HEADER = ["ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface", "Round",
          "Best of", "Winner", "Loser", "WRank", "LRank", "W1", "L1", "Wsets", "Lsets",
          "Comment", "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
# 45306 = 2024-01-15 (Excel serial)
ROWS = [
    [1, "Melbourne", "Australian Open", 45306, "Grand Slam", "Outdoor", "Hard", "1st Round",
     5, "Auger-Aliassime F.", "De Minaur A.", 20, 10, 6, 4, 3, 1, "Completed",
     1.9, 1.9, 1.95, 1.92, 2.05, 2.0, 1.9, 1.88],
    [1, "Melbourne", "Australian Open", 45307, "Grand Slam", "Outdoor", "Hard", "1st Round",
     5, "Mpetshi G.", "Wolf J.J.", 50, 60, 7, 6, 3, 2, "Retired",
     2.5, 1.5, None, None, 2.6, 1.55, 2.45, 1.52],
]


class TestXlsx(unittest.TestCase):
    def test_reads_header_rows_and_dates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2024.xlsx"
            write_xlsx(p, HEADER, ROWS)
            header, rows, d1904 = read_first_sheet(p)
        self.assertEqual(header, HEADER)
        self.assertEqual(len(rows), 2)
        self.assertFalse(d1904)
        self.assertEqual(serial_to_iso(rows[0][3]), "2024-01-15")
        self.assertEqual(rows[0][9], "Auger-Aliassime F.")
        self.assertIsNone(rows[1][20])

    def test_rejects_xls(self):
        with self.assertRaises(ValueError):
            read_first_sheet(Path("old.xls"))


class TestParse(unittest.TestCase):
    def test_odds_pairs_from_header(self):
        self.assertEqual(odds.odds_pairs(HEADER), ["B365", "PS", "Max", "Avg"])

    def test_read_td_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2024.xlsx"
            write_xlsx(p, HEADER, ROWS)
            rows, header = odds.read_td_file(p, "atp")
        self.assertEqual(rows[0].date, "2024-01-15")
        self.assertEqual(rows[0].odds["PS"], (1.95, 1.92))
        self.assertNotIn("PS", rows[1].odds)          # blank odds are dropped, not zero
        self.assertEqual(rows[1].comment, "Retired")
        self.assertEqual(rows[0].source_key, "2024:2")

    def test_names(self):
        self.assertEqual(odds.parse_td_name("Auger-Aliassime F."), (["auger", "aliassime"], "f"))
        self.assertEqual(odds.parse_td_name("Wolf J.J."), (["wolf"], "j"))
        self.assertGreater(odds.name_score("Mpetshi G.", "Mpetshi Perricard", "Giovanni",
                                           "Giovanni Mpetshi Perricard"), 0)
        self.assertGreater(odds.name_score("Muller A.", "Müller", "Alexandre", "Alexandre Müller"), 0)
        self.assertEqual(odds.name_score("Murray A.", "Moutet", "Corentin", "Corentin Moutet"), 0)


class TestLink(unittest.TestCase):
    NAMES = {1: ("Auger-Aliassime", "Felix", "Felix Auger-Aliassime"),
             2: ("De Minaur", "Alex", "Alex De Minaur"),
             3: ("Mpetshi Perricard", "Giovanni", "Giovanni Mpetshi Perricard"),
             4: ("Wolf", "Jeffrey John", "Jeffrey John Wolf"),
             5: ("Sinner", "Jannik", "Jannik Sinner")}

    def _rows(self, **kw):
        base = dict(source_key="2024:2", tour="atp", date="2024-01-15", tournament="AO",
                    round="1st Round", winner="Auger-Aliassime F.", loser="De Minaur A.",
                    comment="Completed", surface="Hard")
        base.update(kw)
        return [odds.TdRow(**base)]

    def test_links_by_pair_and_window(self):
        ours = [odds.OurMatch(10, "atp", "2024-01-14", None, 1, 2, "completed"),
                odds.OurMatch(11, "atp", "2024-01-14", None, 3, 4, "completed")]
        lk = odds.link(self._rows(), ours, self.NAMES)
        self.assertEqual((lk[0].match_id, lk[0].reason), (10, "linked"))

    def test_every_row_gets_a_reason(self):
        ours = [odds.OurMatch(10, "atp", "2024-01-14", None, 2, 1, "completed")]
        swapped = odds.link(self._rows(), ours, self.NAMES)[0]
        self.assertEqual((swapped.match_id, swapped.reason), (None, "winner_disagrees"))
        far = odds.link(self._rows(date="2024-03-01"), ours, self.NAMES)[0]
        self.assertEqual(far.reason, "no_candidate")
        wta = odds.link(self._rows(tour="wta"), ours, self.NAMES)[0]
        self.assertEqual(wta.reason, "no_candidate")
        nodate = odds.link(self._rows(date=None), ours, self.NAMES)[0]
        self.assertEqual(nodate.reason, "no_date")

    def test_exact_day_respected(self):
        ours = [odds.OurMatch(10, "atp", "2024-01-14", "2024-01-25", 1, 2, "completed")]
        self.assertEqual(odds.link(self._rows(), ours, self.NAMES)[0].reason, "no_candidate")

    def test_match_claimed_once(self):
        ours = [odds.OurMatch(10, "atp", "2024-01-14", None, 1, 2, "completed")]
        rows = self._rows() + self._rows(source_key="2024:3")
        got = odds.link(rows, ours, self.NAMES)
        self.assertEqual(sorted(l.reason for l in got), ["duplicate_claim", "linked"])


class TestValuebetennis(unittest.TestCase):
    CSV = ("\ufeffmatch_id;date;tournoi;tournoi_id;categorie;genre;surface;tour;joueur1;joueur1_id;"
           "joueur2;joueur2_id;cote1_ouverture;cote2_ouverture;cote1_cloture;cote2_cloture;"
           "vainqueur_id;score;duree_min\n"
           '1;"2024-01-15 09:00:00";"Australian Open";9;"Grand Chelem";atp;dur;4;"Alex de Minaur";'
           '20;"Felix Auger-Aliassime";10;1.900;1.950;1.800;2.050;10;"6-4 6-4 6-4";\n'
           '2;"2024-01-15 09:00:00";"M15 Monastir";12;ITF;atp;dur;2;"A B";1;"C D";2;;;;;1;;\n'
           '3;"2024-01-16 09:00:00";"Australian Open";9;"Grand Chelem";atp;dur;4;"Giovanni Mpetshi Perricard";'
           '30;"J.J. Wolf";40;;;;;30;"7-6 6-3 6-4";\n')

    def test_read_orients_odds_to_winner(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "valuebetennis-matchs-2024.csv"
            p.write_text(self.CSV, encoding="utf-8")
            rows, counts = odds.read_vb_file(p)
        self.assertEqual(counts, {"read": 3, "not_tour_category": 1, "no_winner": 0, "kept": 2})
        r = rows[0]
        self.assertEqual((r.winner, r.loser, r.date), ("Felix Auger-Aliassime", "Alex de Minaur", "2024-01-15"))
        self.assertEqual(r.odds["PS_close"], (2.05, 1.8))     # winner first
        self.assertEqual(r.odds["PS_open"], (1.95, 1.9))
        self.assertEqual(rows[1].odds, {})                     # no odds: kept, empty

    def test_full_name_scorer_and_link(self):
        f = odds.name_score_full
        self.assertEqual(f("Alex de Minaur", "De Minaur", "Alex", "Alex De Minaur"), 4)
        self.assertGreater(f("J.J. Wolf", "Wolf", "Jeffrey John", "Jeffrey John Wolf"), 0)
        self.assertEqual(f("Andy Murray", "Moutet", "Corentin", "Corentin Moutet"), 0)
        names = TestLink.NAMES
        ours = [odds.OurMatch(10, "atp", "2024-01-14", None, 1, 2, "completed")]
        row = odds.TdRow("vb:1", "atp", "2024-01-15", "AO", "4", "Felix Auger-Aliassime",
                         "Alex de Minaur", "", "dur", {})
        lk = odds.link([row], ours, names, scorer=f)[0]
        self.assertEqual((lk.match_id, lk.reason), (10, "linked"))


class TestDevig(unittest.TestCase):
    def test_methods_sum_to_one_and_order(self):
        o1, o2 = np.array([1.5, 1.2, 1.95]), np.array([2.7, 4.6, 1.92])
        for m in ("multiplicative", "additive", "power", "shin"):
            p = odds.devig(o1, o2, m)
            p2 = odds.devig(o2, o1, m)
            np.testing.assert_allclose(p + p2, 1.0, atol=1e-9, err_msg=m)
            self.assertTrue(np.all(p[:2] > 0.5))
        # favourite-longshot adjustment: power and Shin give the favourite more than proportional
        self.assertGreater(odds.devig(o1, o2, "power")[1], odds.devig(o1, o2, "multiplicative")[1])
        self.assertGreater(odds.devig(o1, o2, "shin")[1], odds.devig(o1, o2, "multiplicative")[1])

    def test_fair_odds_unchanged(self):
        for m in ("multiplicative", "power", "shin"):
            self.assertAlmostEqual(float(odds.devig(np.array([1.25]), np.array([5.0]), m)[0]), 0.8, 6)


class TestBlendAndBacktest(unittest.TestCase):
    def test_blend_finds_added_information(self):
        rng = np.random.default_rng(0)
        n = 60_000
        a, h = rng.normal(0, 1, n), rng.normal(0, 0.6, n)
        y = (rng.random(n) < odds.sigmoid(a + h)).astype(float)
        market = a
        useless = a + rng.normal(0, 0.5, n)
        useful = a + h + rng.normal(0, 0.5, n)
        (_, b0), (_, se0) = odds.blend_fit(y, np.column_stack([market, useless]))
        (_, b1), (_, se1) = odds.blend_fit(y, np.column_stack([market, useful]))
        self.assertLess(abs(b0 / se0), 3)
        self.assertGreater(b1 / se1, 10)

    def test_backtest_accounting(self):
        # one bet on side 1 at 2.0 that wins, one on side 2 at 3.0 that loses
        bt = odds.backtest([0.6, 0.2], [1, 1], [2.0, 1.3], [1.8, 3.0])
        self.assertEqual(bt.n_bets, 2)
        self.assertAlmostEqual(bt.profit, 0.0)
        self.assertAlmostEqual(bt.roi, 0.0)
        none = odds.backtest([0.5], [1], [1.9], [1.9])
        self.assertEqual(none.n_bets, 0)

    def test_no_information_loses_the_margin(self):
        rng = np.random.default_rng(1)
        L = rng.normal(0, 1.2, 5000)
        pts = odds.simulate(L, sigma_hidden=0.5, kappas=[0.0], noises=[0.4], margins=[0.05],
                            n=200_000, seed=2)
        self.assertLess(pts[0].roi, 0)
        self.assertAlmostEqual(pts[0].blend_gain, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
