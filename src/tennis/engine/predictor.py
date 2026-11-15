"""The black-box predictor interface: the one seam between models and everything that uses them.

Consumers (the Kalshi retriever, the CLI, the web app) only ever call `Predictor.predict(query)`.
They never import points_rating, context or markov directly. So the model behind the seam can
be swapped (a placeholder, run 790, the retrained points_v2) without touching the consumers,
and both can be developed at the same time.

    query = MatchQuery(tour="atp", player_a=..., player_b=..., on="2026-11-20", surface="hard")
    pred = predictor.predict(query)          # pred.p_a: P(player A wins the match)

`live` carries the current score when the match is in play (kalshi.LiveState or anything with
the same fields); a predictor that cannot price in play returns `in_play=False` and the
pre-match probability, and says so in `notes`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MatchQuery:
    tour: str                         # "atp" / "wta"
    player_a: int                     # tennis.db player_id (resolve names with engine.players)
    player_b: int
    on: str                           # ISO date of the match
    surface: str | None = None        # hard / clay / grass / carpet; None = unknown
    best_of: int = 3
    final: str = "tb"                 # deciding-set rule: tb / adv / tb12 / mtb
    edition_id: int | None = None     # when the match belongs to a known edition (context, rules)
    live: Any = None                  # in-play state (kalshi.LiveState-like) or None


@dataclass
class Prediction:
    p_a: float                        # P(player A wins), the number consumers act on
    p_raw: float | None = None        # before calibration, when the model has one
    in_play: bool = False             # True if `live` was used
    model: str = ""                   # e.g. "points_v1/run790", "fixed"
    stale: bool = False               # a player's last rated match is too old to trust
    notes: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)   # model-specific breakdown (serve/return, context)


@runtime_checkable
class Predictor(Protocol):
    name: str

    def predict(self, query: MatchQuery) -> Prediction: ...


class FixedPredictor:
    """Placeholder: a constant probability (0.5 = no opinion). Lets the Kalshi pipeline be built
    and tested end to end before a real model sits behind the seam."""

    def __init__(self, p_a: float = 0.5):
        if not 0.0 < p_a < 1.0:
            raise ValueError("p_a must be in (0, 1)")
        self.p_a = p_a
        self.name = f"fixed({p_a:g})"

    def predict(self, query: MatchQuery) -> Prediction:
        notes = ["placeholder predictor: not a model"]
        if query.live is not None:
            notes.append("live state ignored")
        return Prediction(p_a=self.p_a, p_raw=self.p_a, in_play=False, model=self.name, notes=notes)
