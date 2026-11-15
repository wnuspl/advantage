"""Point-in-time context features for the points_v1 context offsets (milestone 9).

Each player in each match gets a feature vector c (FEATURES), computed only from matches
before it in walk-forward order (earlier editions, or earlier rounds of the same edition),
at all levels. points_rating.run() turns them into an offset on each serve-point logit:

    x_A' = x_A + beta_serve . c_A - beta_ret . c_B      (A serving, B returning)

used for the prediction and for the update, never stored in the rating state. beta_serve
is the feature's effect on the player's own serve, beta_ret on their own return (so a
rusty returner, beta_ret < 0, raises the server's x).

Sackmann dates matches only by the event's start week, so gaps are edition start to
edition start (as in scripts/context_report.py, whose findings chose these features).

Features (0 when unknown, which is neutral):
  back1      first match played after a > 12-week gap between events
  back2_5    2nd-5th match played after such a gap
  gap5_12    this event started 5-12 weeks after the player's previous one
  gap1       ... 1 week after (back-to-back weeks)
  load28     matches played in events of the previous 4 weeks, / 5
  age_lt22   max(0, 22 - age)  \
  age_lt26   max(0, 26 - age)   | piecewise-linear age curve in years, flat 26-28:
  age_gt28   max(0, age - 28)   | smooth, so no expectation jumps on a birthday
  age_gt32   max(0, age - 32)  /
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date

import numpy as np

FEATURES = ("back1", "back2_5", "gap5_12", "gap1", "load28",
            "age_lt22", "age_lt26", "age_gt28", "age_gt32")
K = len(FEATURES)
PLAYED = {"completed", "retired"}
LAYOFF_DAYS = 84


class _P:
    __slots__ = ("ed", "start", "n", "prev_start", "eds", "since_return")

    def __init__(self):
        self.ed = None            # current edition
        self.start = None         # its start ordinal
        self.n = 0                # matches played in it so far
        self.prev_start = None    # start ordinal of the previous edition
        self.eds = []             # (start ordinal, matches played) of finished editions, last 12
        self.since_return = 99    # matches played since a > 12-week gap (99 = none)


def features(m, edition, dob: dict) -> tuple[np.ndarray, np.ndarray]:
    """(C_winner, C_loser), each len(m) x K, for matches in m's walk-forward order.
    edition[i]: edition id of match i; dob: player_id -> ISO birth date (or missing)."""
    n = len(m)
    C = (np.zeros((n, K)), np.zeros((n, K)))
    st = defaultdict(_P)
    ords: dict[str, int] = {}
    birth = {}
    for pid, d in dob.items():
        try:
            birth[pid] = date.fromisoformat(d).toordinal() if d else None
        except ValueError:
            birth[pid] = None
    for i in range(n):
        s0 = m.start[i]
        day = ords.get(s0)
        if day is None:
            day = ords[s0] = date.fromisoformat(s0).toordinal()
        eid = edition[i]
        pair = (m.winner[i], m.loser[i])
        for side, pid in enumerate(pair):
            s = st[pid]
            if s.ed != eid:                                  # player enters a new edition
                if s.ed is not None:
                    s.eds.append((s.start, s.n))
                    if len(s.eds) > 12:
                        s.eds.pop(0)
                    if day - s.start > LAYOFF_DAYS:
                        s.since_return = 0
                    s.prev_start = s.start
                s.ed, s.start, s.n = eid, day, 0
            c = C[side][i]
            if s.since_return == 0:
                c[0] = 1.0
            elif 1 <= s.since_return <= 4:
                c[1] = 1.0
            if s.prev_start is not None:
                gap = day - s.prev_start
                c[2] = 1.0 if 28 < gap <= LAYOFF_DAYS else 0.0
                c[3] = 1.0 if 0 < gap <= 7 else 0.0
            c[4] = sum(k for d0, k in s.eds if 0 < day - d0 <= 28) / 5.0
            b = birth.get(pid)
            if b is not None:
                age = (day - b) / 365.25
                c[5] = max(0.0, 22 - age)
                c[6] = max(0.0, 26 - age)
                c[7] = max(0.0, age - 28)
                c[8] = max(0.0, age - 32)
        if m.status[i] in PLAYED:                            # after the match
            for pid in pair:
                s = st[pid]
                s.n += 1
                if s.since_return < 99:
                    s.since_return += 1
    return C


def attach(con: sqlite3.Connection, m, alias: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """Context features aligned with m, cached on m.extra['context']."""
    if "context" in m.extra:
        return m.extra["context"]
    ed = dict(con.execute(f"SELECT match_id, edition_id FROM {alias}.matches"))
    dob = dict(con.execute(f"SELECT player_id, dob FROM {alias}.player"))
    C = features(m, [ed[mid] for mid in m.match_id], dob)
    m.extra["context"] = C
    return C


def offsets(C, beta_serve: np.ndarray, beta_ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(o_a, o_b): offsets on x_a (winner serving) and x_b (loser serving)."""
    cw, cl = C
    return cw @ beta_serve - cl @ beta_ret, cl @ beta_serve - cw @ beta_ret
