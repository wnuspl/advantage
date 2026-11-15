# Archive

Finished work kept as history. The live project (`src/`, `scripts/`, `tests/`) never imports from here. What comes back, and when, is listed in `docs/PLAN.md`.

## v1: pre-match research (2026-11-13 to 2026-11-15)

Everything that produced runs 132-790: Elo/Glicko-2 baselines, style/matchup/phase experiments (negative), points_v1 + score tier + context offsets (run 790, the reboot's baseline), the in-play form test, Kalshi edges.

| Path | What |
|---|---|
| `v1/CLAUDE.md` | Project instructions at the end of v1: all findings, milestone log 1-11 |
| `v1/src/tennis/` | Research and v1 tooling modules: elo, glicko2, ratingmodels, tuning, evaluate, calibrate, composite(_eval), matchup/, style_features, heavy, phase, standings, mcp_notation, odds, xlsx_lite |
| `v1/scripts/` | All research and model scripts (sweep, compare_runs, context_fit, kalshi_edges, predict_match, inplay_form_test, ingest_daily, ...) |
| `v1/tests/` | Tests of the archived modules and scripts |
| `v1/results/`, `v1/reports/`, `v1/docs/` | Outputs, figures, research reports |
| `v1/models/` | Fitted artifacts and caches (style, matchup, phase, odds, inplay, kalshi incl. `live_*.jsonl` watch logs). Gitignored, 154 MB |
| `v1/*_SPEC.md` | The original heavy/middle layer specs |

## Running archived code

`archive/v1/src/tennis/` merges into the live `tennis` package (`pkgutil.extend_path`), so archived code sees archived modules and live ones (`points_rating`, `markov`, `context`, `matches`, ...) under one name, with nothing copied. Scripts and tests put `archive/v1/src` first on `sys.path` themselves. From the project root:

```
python archive/v1/scripts/compare_runs.py 694 790
python archive/v1/scripts/kalshi_edges.py --run 790
cd archive/v1/tests; python -m unittest discover -s .
```

In the scripts, `ROOT` is the project root (tennis.db, models.db, data/, src/) and `ARCHIVE` is `archive/v1` (outputs, models/). Archived code runs against the *live* core, so if a later milestone changes `points_rating` or `context`, v1 numbers may no longer reproduce from here; the stored runs in `models.db` remain the record.

`models.db` is never rebuilt: all v1 runs stay in it and stay the comparison baselines.
