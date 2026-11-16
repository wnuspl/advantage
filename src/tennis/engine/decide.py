"""Decisions on one market snapshot (track B3a, docs/PLAN.md): the bet the policy would
place, or why it places none. Pure: no network, no state, no orders (read-only project).

    pol = Policy(bankroll=1000, kelly=0.25)
    ds = decide_all(rows, pull.books, pol)        # rows: live.snapshots(...)

Per snapshot (rewritten from v1 `kalshi_edges.Pricer.price`):
  1. refusals, in order: not priced (the snapshot's skip), placeholder prediction
     (FixedPredictor), stale rating, in play while `in_play` is off, no order book;
  2. p = logit blend of model and market (`model_weight`; 1 = model only, 0 = market only);
  3. for each side, the log-optimal stake on `kelly` * bankroll walking the merged ask
     ladder (YES on the player or NO on the opponent), taker fee included, capped at
     `max_stake` dollars; the side with the larger expected profit wins.
Then, across snapshots, `max_total` (a fraction of bankroll) scales every stake down pro
rata when their sum exceeds it (v1's portfolio cap).

Open positions are not known here: the B3b ledger tracks exposure across polls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import kalshi
from .live import MarketSnapshot


@dataclass(frozen=True)
class Policy:
    bankroll: float = 1000.0          # dollars
    kelly: float = 0.25               # Kelly fraction
    model_weight: float = 1.0         # logit blend weight on the model (1 = model only)
    max_stake: float | None = None    # dollars per match
    max_total: float = 0.5            # summed stakes, as a fraction of bankroll
    in_play: bool = False             # bet on live matches (feed lag behind the court unknown)
    allow_stale: bool = False         # bet when a player's rating is stale
    fee_rate: float = kalshi.FEE_RATE


@dataclass
class Decision:
    event: str
    p_model: float | None = None      # P(A wins), from the predictor
    p_market: float | None = None     # P(A wins), de-overrounded mid
    p_use: float | None = None        # P(A wins) used for sizing (blend)
    side: str | None = None           # "a" / "b": the player the bet is on
    name: str | None = None
    p_side: float | None = None       # P(that player wins), under p_use
    ask: float | None = None          # cheapest ask on that side, before fee
    sizing: kalshi.Sizing = field(default_factory=kalshi.Sizing)
    reason: str = ""                  # why no bet ("" = a bet)

    @property
    def bet(self) -> bool:
        return self.sizing.contracts > 0

    def to_json(self) -> dict:
        sz = self.sizing
        return {"event": self.event, "p_model": self.p_model, "p_market": self.p_market, "p_use": self.p_use,
                "side": self.side, "name": self.name, "p_side": self.p_side, "ask": self.ask,
                "contracts": sz.contracts, "cost": sz.cost, "fees": sz.fees, "ev": sz.ev,
                "log_growth": sz.log_growth, "fills": [vars(f) for f in sz.fills], "reason": self.reason}


def decide(s: MarketSnapshot, books: dict, pol: Policy) -> Decision:
    e = s.event
    d = Decision(e.event, s.p_model, s.p_market)
    pred = s.prediction
    if s.skip or pred is None:
        d.reason = "not priced: " + (s.skip or "no prediction")
        return d
    if pred.detail.get("placeholder"):
        d.reason = "placeholder predictor"
        return d
    if pred.stale and not pol.allow_stale:
        d.reason = "stale rating"
        return d
    if s.status == "live" and not pol.in_play:
        d.reason = "in play (policy off)"
        return d
    ba, bb = books.get(e.a.ticker), books.get(e.b.ticker)
    if ba is None or bb is None:
        d.reason = "no order book"
        return d
    d.p_use = kalshi.blend(pred.p_a, s.p_market, pol.model_weight)
    best = None
    for key, own, opp, bo, bp, p in (("a", e.a, e.b, ba, bb, d.p_use), ("b", e.b, e.a, bb, ba, 1 - d.p_use)):
        lad = kalshi.ladder(bo, own.ticker, bp, opp.ticker)
        sz = kalshi.kelly_walk(p, lad, pol.bankroll, pol.kelly, pol.fee_rate, pol.max_stake)
        cand = (sz.ev if sz.contracts else float("-inf"), key, own.name, p, lad[0].price if lad else None, sz)
        if best is None or cand[0] > best[0]:
            best = cand
    _, d.side, d.name, d.p_side, d.ask, d.sizing = best
    if not d.bet:
        d.side = d.name = None
        d.reason = "no edge after spread and fees"
    return d


def decide_all(rows: list[MarketSnapshot], books: dict, pol: Policy) -> list[Decision]:
    """decide() per snapshot, then the `max_total` cap pro rata across all bets."""
    out = [decide(s, books, pol) for s in rows]
    total = sum(d.sizing.cost for d in out)
    cap = pol.max_total * pol.bankroll
    if total > cap > 0:
        scale = cap / total
        for d in out:
            if d.bet:
                d.sizing = kalshi.scale_sizing(d.sizing, d.p_side, scale, pol.bankroll * pol.kelly, pol.fee_rate)
                if not d.bet:
                    d.side = d.name = None
                    d.reason = "scaled to zero by the total cap"
    return out


CLI_BANKROLL = 10.0                   # scripts/live.py, scripts/watch.py default (user, 2026-11-15)
CLI_MODEL_WEIGHT = 0.15              # B5 Pinnacle blend fit 0.15-0.19 (2026-11-16); refit on settled Kalshi matches


def add_policy_args(ap) -> None:
    """Policy flags for an argparse parser (scripts/live.py, scripts/watch.py)."""
    g = ap.add_argument_group("policy (engine/decide.py)")
    g.add_argument("--bankroll", type=float, default=CLI_BANKROLL, help="dollars (default $%(default)g)")
    g.add_argument("--kelly", type=float, default=Policy.kelly, help="Kelly fraction (1 = full Kelly)")
    g.add_argument("--model-weight", type=float, default=CLI_MODEL_WEIGHT,
                   help="logit blend weight on the model (1 = model only, 0 = market only; default %(default)g)")
    g.add_argument("--max-stake", type=float, default=None, help="cap per match, dollars")
    g.add_argument("--max-total", type=float, default=Policy.max_total,
                   help="cap on summed stakes, fraction of bankroll (scaled down pro rata)")
    g.add_argument("--in-play-bets", action="store_true",
                   help="bet on live matches too (off by default: the feed's lag behind the court is unknown)")
    g.add_argument("--allow-stale", action="store_true", help="bet even when a player's rating is stale")


def policy_from_args(args) -> Policy:
    return Policy(bankroll=args.bankroll, kelly=args.kelly, model_weight=args.model_weight,
                  max_stake=args.max_stake, max_total=args.max_total, in_play=args.in_play_bets,
                  allow_stale=args.allow_stale)
