"""Match Charting Project shot notation: parse one point's '1st'/'2nd' cells into shots.

Grammar follows the 'Instructions' tab of the MCP charting spreadsheet
(`MatchChart 0.3.2.xlsm`, MCP commit 1813a130). Summary of what that key defines:

- Serve cell: optional lets `c...`, then the serve direction 4 = wide, 5 = body, 6 = T,
  0 = unknown; optional `+` = serve-and-volley attempt. Then exactly one of:
    fault letter  n net, w wide, d deep, x wide+deep, g foot fault, e unknown, ! shank
    `*` ace, `#` unreturnable (returner touched it but could not return it)
    a rally: one or more shots.
  Whole-cell codes: S = point to server, R = point to returner (missed point),
  P = penalty against server, Q = penalty against returner, V = time violation (loses a serve).
- Shot: one letter
    f b forehand/backhand groundstroke   r s forehand/backhand slice (incl. chips)
    v z volley                           o p overhead / backhand overhead
    u y drop shot                        l m lob
    h i half-volley                      j k swinging volley
    t trick shot                         q unknown shot
  then optional position codes (`+` approach, `-` at net, `=` at baseline, `;` net cord,
  `^` stop/drop volley), then direction 1/2/3 (0 unknown), and on the return only, a depth
  digit 7 (service boxes), 8 (behind the service line), 9 (near the baseline), 0 unknown.
- Endings: `*` winner; an error letter (n w d x e !) then `@` unforced / `#` forced;
  a forced error may omit the letter (`b#`). `C` = play stopped by an incorrect challenge.

Conventions found in the data but not written in the key, accepted because no other
reading is possible (survey of all 1.84M charted points, 2026-11-13):
- a depth digit 7/8/9 on rally shots, not just the return (`f39`, ~233k points);
- position marks after the direction (`f1;`, `f18;`) and `!` as a shank mark on a shot
  before its direction (`f!2d@`);
- an unforced error without an error letter (`f@`): error_type stays None;
- a fault letter with no direction (`g`, `e`): a fault of unknown direction; `;` after a
  fault letter (net cord on a missed serve).
Anything else raises NotationError: callers count and skip the point instead of guessing.
Pure logic, no database access.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SHOT_LETTERS = "fbrsvzopuylmhijktq"
FAULT_LETTERS = "nwdxge!"
ERROR_LETTERS = "nwdxe!"
FOREHAND_SIDE = set("frvoulhj")     # t (trick) and q (unknown) belong to neither side
BACKHAND_SIDE = set("bszpymik")
GROUND = set("fb")                  # topspin/flat groundstrokes
SLICE = set("rs")
DROP = set("uy")
LOB = set("lm")
NET_SHOTS = set("vzopijhk")         # assumed at the net unless marked '=' (key: 'Baseline/net')
SPECIAL = {"S", "R", "P", "Q", "V"}


class NotationError(ValueError):
    """A code the MCP key does not define (or that contradicts the point's other cell)."""

    @property
    def reason(self) -> str:
        """Message without the quoted code, for counting skips by kind."""
        return re.sub(r"\s*(in )?['\"].*$|\s*\(.*\)$", "", str(self)).strip()


@dataclass
class Shot:
    kind: str                       # one of SHOT_LETTERS
    direction: int | None = None    # 1/2/3, None if missing or 0
    depth: int | None = None        # returns only: 7/8/9, None if missing or 0
    approach: bool = False          # '+'
    at_net: bool = False            # net shot letter or '-', unless '='
    net_cord: bool = False          # ';'
    stop: bool = False              # '^'
    shank: bool = False             # '!' used as a shot mark


@dataclass
class Cell:
    """One serve and (if it went in) the rally that followed."""
    special: str | None = None      # S/R/P/Q/V whole-cell code
    lets: int = 0
    serve_dir: int | None = None    # 4/5/6, None if unknown (0)
    serve_and_volley: bool = False
    fault: str | None = None        # fault letter if the serve missed
    serve_outcome: str | None = None  # 'fault', 'ace', 'unreturnable', 'in'
    shots: list[Shot] = field(default_factory=list)   # after the serve: shots[0] is the return
    ending: str | None = None       # 'winner', 'forced', 'unforced', 'challenge' (rally only)
    error_type: str | None = None   # n/w/d/x/e/! on an error ending


_SHOT_RE = re.compile(r"([fbrsvzopuylmhijktq])([+\-=;^!]*)([0-3])?([0789])?([+\-=;^]*)")


def parse_cell(code: str) -> Cell:
    s = (code or "").strip()
    if not s:
        raise NotationError("empty cell")
    cell = Cell()
    if s in SPECIAL:
        cell.special = s
        return cell
    i = 0
    while i < len(s) and s[i] == "c":
        i += 1
    cell.lets = i
    if s[i:].rstrip(";") in tuple(FAULT_LETTERS):     # fault of unknown direction
        cell.fault, cell.serve_outcome = s[i:].rstrip(";"), "fault"
        return cell
    if i >= len(s) or s[i] not in "0456":
        raise NotationError(f"no serve direction in {code!r}")
    cell.serve_dir = int(s[i]) if s[i] != "0" else None
    i += 1
    if i < len(s) and s[i] == "+":
        cell.serve_and_volley = True
        i += 1
    rest = s[i:]
    if not rest:
        raise NotationError(f"serve with no outcome in {code!r}")
    if rest.rstrip(";") in tuple(FAULT_LETTERS):     # ';' = net cord on the missed serve
        cell.fault, cell.serve_outcome = rest.rstrip(";"), "fault"
        return cell
    if rest == "*":
        cell.serve_outcome = "ace"
        return cell
    if rest == "#":
        cell.serve_outcome = "unreturnable"
        return cell
    cell.serve_outcome = "in"
    _parse_rally(rest, cell, code)
    return cell


def _parse_rally(s: str, cell: Cell, code: str) -> None:
    i = 0
    while i < len(s):
        m = _SHOT_RE.match(s, i)
        if not m:
            break
        letter, mods, direction, depth, mods2 = m.groups()
        mods += mods2
        shot = Shot(kind=letter,
                    direction=int(direction) if direction and direction != "0" else None,
                    depth=int(depth) if depth and depth != "0" else None,
                    approach="+" in mods, net_cord=";" in mods, stop="^" in mods,
                    shank="!" in mods)
        shot.at_net = ("-" in mods) or (letter in NET_SHOTS and "=" not in mods)
        cell.shots.append(shot)
        i = m.end()
    tail = s[i:]
    if not cell.shots:
        raise NotationError(f"rally does not start with a shot in {code!r}")
    if tail == "*":
        cell.ending = "winner"
    elif tail == "C":
        cell.ending = "challenge"
    elif tail in ("@", "#"):
        cell.ending = "unforced" if tail == "@" else "forced"
    elif len(tail) == 2 and tail[0] in ERROR_LETTERS and tail[1] in "@#":
        cell.error_type = tail[0]
        cell.ending = "unforced" if tail[1] == "@" else "forced"
    else:
        raise NotationError(f"unknown rally ending {tail!r} in {code!r}")


@dataclass
class Point:
    """A parsed point from the server's point of view.

    shots: every stroke after the deciding serve; shots[0] is the return, so shot k is hit
    by the returner when k is even and by the server when k is odd.
    """
    serve_no: int                   # serve the point was decided on (1 or 2)
    first: Cell
    second: Cell | None
    deciding: Cell
    double_fault: bool
    server_won: bool

    @property
    def shots(self) -> list[Shot]:
        return self.deciding.shots

    @property
    def n_strokes(self) -> int:
        """Strokes including the deciding serve and the final (error) shot; 0 on a double fault."""
        if self.double_fault:
            return 0
        return 1 + len(self.deciding.shots)


def parse_point(first: str | None, second: str | None, server_won: bool) -> Point:
    """Parse both cells of a point. Raises NotationError for anything unusable.

    Points coded only with S/R/P/Q (missed or penalty points) carry no style information
    and raise too; so do points stopped by a challenge ('C').
    """
    c1 = parse_cell(first or "")
    second = (second or "").strip()
    if c1.special in ("S", "R", "P", "Q"):
        raise NotationError(f"no rally data ({c1.special})")
    if c1.special == "V" or c1.serve_outcome == "fault":
        if not second:
            raise NotationError("first serve missed but no second cell")
        c2 = parse_cell(second)
        if c2.special is not None:
            raise NotationError(f"no rally data ({c2.special} on 2nd)")
        dec, serve_no = c2, 2
    else:
        if second:
            raise NotationError("first serve in but a second cell is present")
        c2, dec, serve_no = None, c1, 1
    if dec.ending == "challenge":
        raise NotationError("rally stopped by a challenge")
    df = dec.serve_outcome == "fault"
    pt = Point(serve_no=serve_no, first=c1, second=c2, deciding=dec, double_fault=df,
               server_won=server_won)
    _check_winner(pt)
    return pt


def _check_winner(pt: Point) -> None:
    """The notation implies who won; it must agree with the PtWinner column."""
    dec = pt.deciding
    if pt.double_fault:
        implied = False
    elif dec.serve_outcome in ("ace", "unreturnable"):
        implied = True
    else:
        last_is_server = len(dec.shots) % 2 == 0
        implied = last_is_server if dec.ending == "winner" else not last_is_server
    if implied != pt.server_won:
        raise NotationError("notation and point winner disagree")
