# Archive

Finished work kept as history, not part of the live engine. Nothing here is imported by `src/`.

## v1: pre-match research (2026-11-13 to 2026-11-15)

Everything that produced runs 132-790: the Elo/Glicko baselines, style/matchup/phase
experiments (negative), points_v1 + score tier + context offsets (run 790, the baseline the
reboot starts from), in-play form test, Kalshi edges.

| Path | What |
|---|---|
| `v1/CLAUDE.md` | Project instructions as of the end of v1: full findings and milestone log 1-11 |
| `v1/scripts/` | All research and model scripts (sweep, compare_runs, context_fit, kalshi_edges, predict_match, ...) |
| `v1/results/`, `v1/reports/` | Their JSON/CSV outputs and figures |
| `v1/docs/` | Research reports (composite, context, matchup, style, odds benchmark) |
| `v1/*_SPEC.md` | The original heavy/middle layer specs |

The archived scripts still run from the project root, against the live `tennis.db`,
`models.db` and `src/tennis/`, and write outputs into `archive/v1/`:

```
python archive/v1/scripts/compare_runs.py 694 790
python archive/v1/scripts/kalshi_edges.py --run 790
```

(`ROOT` in each script is the project root; `ARCHIVE` is `archive/v1`.)

**The library stayed in `src/tennis/`.** The research modules (`elo`, `glicko2`, `ratingmodels`,
`tuning`, `composite_eval`, `matchup/`, `style_features`, `heavy`, ...) are still imported by the
model core (`points_rating`, `calibrate`), so they were not moved. They are cut loose as the
engine takes over (see the plan in `CLAUDE.md`); then they move here.

`models.db` is never rebuilt: all v1 runs stay in it and stay the comparison baselines.
