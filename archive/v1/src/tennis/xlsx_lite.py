"""Minimal .xlsx reader (standard library only): the first worksheet as header + rows.

Enough for flat data sheets such as tennis-data.co.uk's yearly files. Handles shared
strings, inline strings, booleans and numbers; dates come back as Excel serial numbers
(use `serial_to_iso`). Not a general spreadsheet library: no formulas, no styles, no .xls.
"""
from __future__ import annotations

import re
import zipfile
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
_REF = re.compile(r"([A-Z]+)(\d+)")


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text(el) -> str:
    """Concatenated text of an <si> or <is> element (handles rich-text runs)."""
    return "".join(t.text or "" for t in el.iter(f"{{{NS['m']}}}t"))


def serial_to_iso(serial: float, date1904: bool = False) -> str:
    base = date(1904, 1, 1) if date1904 else date(1899, 12, 30)
    return (base + timedelta(days=int(serial))).isoformat()


def read_first_sheet(path: Path) -> tuple[list[str], list[list], bool]:
    """Return (header, rows, date1904). Cells are str, float, bool or None; rows are padded
    to the header width. Blank rows are dropped."""
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"{path.name}: only .xlsx is supported")
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        pr = wb.find("m:workbookPr", NS)
        date1904 = pr is not None and pr.get("date1904") in ("1", "true")
        first = wb.find("m:sheets/m:sheet", NS)
        rid = first.get(f"{{{NS['r']}}}id")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels.findall("rel:Relationship", NS)
                      if r.get("Id") == rid)
        target = target.lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [_text(si) for si in sst.findall("m:si", NS)]
        raw_rows = []
        with z.open(sheet_path) as fh:
            for _, el in ET.iterparse(fh):
                if el.tag != f"{{{NS['m']}}}row":
                    continue
                cells = {}
                for c in el.findall("m:c", NS):
                    m = _REF.match(c.get("r", ""))
                    col = _col_index(m.group(1)) if m else len(cells)
                    t = c.get("t")
                    v = c.find("m:v", NS)
                    if t == "s":
                        val = shared[int(v.text)] if v is not None else None
                    elif t == "inlineStr":
                        is_ = c.find("m:is", NS)
                        val = _text(is_) if is_ is not None else None
                    elif t == "b":
                        val = v is not None and v.text == "1"
                    elif t in ("str", "e"):
                        val = v.text if v is not None else None
                    else:
                        val = float(v.text) if v is not None and v.text not in (None, "") else None
                    if isinstance(val, str):
                        val = val.strip() or None
                    cells[col] = val
                el.clear()
                if any(x is not None for x in cells.values()):
                    raw_rows.append(cells)
    if not raw_rows:
        return [], [], date1904
    width = max(raw_rows[0]) + 1
    header = [str(raw_rows[0].get(i) or "").strip() for i in range(width)]
    rows = [[r.get(i) for i in range(width)] for r in raw_rows[1:]]
    return header, rows, date1904
