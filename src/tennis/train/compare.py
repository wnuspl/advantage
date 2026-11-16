"""Compare runs on identical matches, raw and calibrated (from v1 compare_runs.py).

Each run is replayed from its stored params. The reference is the first run, calibrated.
Rows: log loss, paired difference vs the reference with SE, quarters won, Brier, ECE and
recalibration slope (< 1 = overconfident), per scope tour / tour_atp / tour_wta.
"""
from __future__ import annotations

from .. import points_rating
from .data import BLOCKS, Prepared
from .runs import Scored, params_of
from .scoring import metrics


def replay(con, prep: Prepared, run_ids, block=BLOCKS["val"], by_best_of: bool = True, log=print) -> dict:
    out = {}
    for rid in run_ids:
        model_id, params = params_of(con, rid)
        res = points_rating.run(prep.m, params)
        out[f"{rid}:{model_id}"] = Scored(prep, res.p_winner, block, by_best_of)
        log(f"run {rid} ({model_id}) replayed")
    return out


def table(scored: dict) -> dict[str, list[tuple[str, object]]]:
    """{scope: [(label, Metrics)]}: every run raw and calibrated, vs the first one calibrated."""
    first = next(iter(scored.values()))
    out = {}
    for scope in ("tour", "tour_atp", "tour_wta"):
        sel = first.select(scope)
        ref = first.cal[sel]
        rows = []
        for lab, s in scored.items():
            for kind, p in (("cal", s.cal), ("raw", s.raw)):
                rows.append((f"{lab} {kind}", metrics(p[sel], s.y[sel], ref=ref, block=s.quarter[sel])))
        out[scope] = rows
    return out


def format_table(tab: dict) -> str:
    lines = []
    for scope, rows in tab.items():
        lines.append(f"\n[{scope}] n={rows[0][1].n:,}")
        lines.append(f"  {'run':28s} {'log loss':>8} {'delta':>8} {'SE':>7} {'quarters':>8} {'Brier':>7} "
                     f"{'ECE':>7} {'slope':>6}")
        for lab, mt in rows:
            lines.append(f"  {lab:28s} {mt.log_loss:8.4f} {mt.d_ref:+8.4f} {mt.d_se:7.4f} "
                         f"{mt.blocks_won:>3}/{mt.blocks:<4} {mt.brier:7.4f} {mt.ece:7.4f} {mt.cal_slope:6.3f}")
    return "\n".join(lines)
