"""Pure value normalizers shared by the loaders. No database access here."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

# Letters NFKD does not decompose into base letter + accent.
_TRANSLATE = str.maketrans({
    "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ß": "ss",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ı": "i", "þ": "th", "Þ": "th",
})


def norm_name(s: str | None) -> str:
    """'Juan Martín del Potro' -> 'juan martin del potro'. Hyphens/underscores/dots -> space."""
    s = (s or "").translate(_TRANSLATE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[-_.'’`]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def parse_ymd(s: str | None) -> str | None:
    """'20240115' -> '2024-01-15'; anything else (incl. impossible dates) -> None."""
    s = (s or "").strip()
    if not re.fullmatch(r"\d{8}", s):
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:])).isoformat()
    except ValueError:
        return None


def add_days(iso: str, n: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days=n)).isoformat()


def to_int(s: str | None) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else None


# Chronological order of rounds within an edition. PR = pre-round / play-off (US Open
# 1970s), ER = early round, CR = consolation round.
ROUND_ORDER = {
    "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5, "PR": 6, "ER": 7,
    "R128": 8, "R64": 9, "R32": 10, "R16": 11, "RR": 12, "CR": 12,
    "QF": 13, "SF": 14, "BR": 15, "F": 16,
}


def norm_round(s: str | None) -> str | None:
    r = (s or "").strip().upper()
    return r if r in ROUND_ORDER else None


SURFACES = {"hard", "clay", "grass", "carpet"}


def norm_surface(s: str | None) -> str | None:
    v = (s or "").strip().lower()
    return v if v in SURFACES else None


def norm_hand(s: str | None) -> str | None:
    v = (s or "").strip().upper()
    if v in ("R", "L", "U"):
        return v
    if v == "A":              # ambidextrous in a few Sackmann rows
        return "U"
    return None


_RET = {"RET", "RET.", "RTD", "RTD.", "RE", "RETIRED"}
_WO = {"W/O", "WO", "W/O.", "WALKOVER"}
_DEF = {"DEF", "DEF.", "DEFAULT", "DEFAULTED"}
_UNF = {"ABD", "ABANDONED", "UNFINISHED"}


def score_status(score: str | None) -> str:
    """Match status from a Sackmann score string. Returns 'bye' for byes (caller skips)."""
    toks = set((score or "").upper().split())
    if "BYE" in toks:
        return "bye"
    if toks & _WO:
        return "walkover"
    if toks & _DEF:
        return "default"
    if toks & _RET:
        return "retired"
    if toks & _UNF:
        return "unfinished"
    return "completed"


def is_placeholder_name(first: str | None, last: str | None) -> bool:
    """Sackmann uses 'Unknown' / 'X X' rows as stand-ins for unidentified players."""
    f, l = (first or "").strip().lower(), (last or "").strip().lower()
    return l == "unknown" or (f, l) == ("x", "x")
