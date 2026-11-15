# Advantage: tennis match prediction (reboot)

## Goal
**A live (in-play) betting engine for ATP/WTA singles**, read-only against Kalshi: a pre-match rating gives each player's serve and return going into the match; what happens on court updates it point by point; the engine prices Kalshi match-winner markets and flags where model and market disagree. It never places orders.

Start point: **run 790** (`points_v1` + context offsets), tour-level test 2022-26 log loss **0.5944** (calibrated Glicko-2 0.6045). How it works, exactly as coded: `docs/REBOOT.md` section 3. The v1 history (runs 132-790, milestones 1-11, all findings) is in `archive/v1/CLAUDE.md`; see `archive/README.md`.

## Plan
`docs/PLAN.md`: milestones as a list of what gets added back to the blank project (tracks A model, B Kalshi). The project restarted blank on the data layer; v1 code comes back only when a milestone needs it, rewritten against the engine rather than copied.

## Model (kept from v1; parameters get re-tuned)
- **Ratings:** per player serve rating `s` and return rating `r`, per (player, surface) deviations, Kalman variances; updated on serve points won (stats), else set scores, else the result. Walkovers/unfinished never update; retirements count fully (`retired_weight` 1).
- **Point probabilities -> match probability by Markov chain**, best-of and deciding-set rule from the results data, never assumed. `markov.match_from` rolls up from any in-play score.
- **Context is a per-match offset on each serve-point logit** (rust, load, age), used in prediction and in the update, never written into the state.
- **In-match form:** today's serve points update each serve logit with a shared per-match form term (`IN_PLAY_FORM_VAR` 0.02). Serve vs return form is not yet separated (plan item).
- **Calibrate and report raw and calibrated.** Calibration fitted on past data only.
- **No double-counting:** any layer trained on another's residuals uses a cutoff.

## Measured negatives (v1; don't redo)
PCA style vectors and every matchup layer on them (bilinear, siamese, LightGBM; also on a points_v1 base); personal style edges (don't repeat, r ≈ 0.02); phase-profile correction (-0.0001); per-tour context coefficients; `rust_weight` < 1; home advantage and in-event workload. Details in `archive/v1/CLAUDE.md` "Findings" and `docs/DESIGN.md`.

## Evaluation rules
- Walk-forward only. Never shuffle. Train on the past, predict the next block, roll forward. Val 2019-21, test 2022-26; test is scored once per finished model.
- Features and ratings for a match use only data strictly before it (point-in-time correctness). In play: only points already played.
- When building training rows, randomize or canonicalize player order so the model never sees "winner first."
- Report log loss and Brier, plus calibration. Accuracy alone is not a metric here.
- New models are new `model_run` rows under a new `model_id` (e.g. `points_v2`); run 790 stays the baseline.

## Data sources (as of 2026-11-15)
Jeff Sackmann's `tennis_atp`, `tennis_wta` and slam point-by-point repos were removed from GitHub (404 as of Oct 2026). Only the Match Charting Project repo is still upstream.

| Source | Use | Notes |
|---|---|---|
| `Aneeshers/tennis-sackmann-archive` (mirror) | Backbone: players, matches, per-match serve stats, rankings; slam point-by-point | We record our own commit + sha256. Matches end 2026-05-25, rankings 2026-06-08, slam point data 2011-2024. |
| `JeffSackmann/tennis_MatchChartingProject` | Point/shot data (in-play evaluation) | Keep our own pinned copy. |
| AcesAPI (free plan: 500 credits/month) | Results after the mirror's cutoff: tour-level + qualifying + WTA 125 | `scripts/fetch_acesapi.py --pin` (key `ACESAPI_KEY` in `.env`), then `build_db.py`. No serve stats, no Challenger/ITF. Storage and research use allowed, no redistribution. |
| Kalshi public API | Live markets, order books, live scores (Sportradar-fed) | Public endpoints, no key. Read-only. Raw pulls saved under `data/raw/kalshi/<UTC>/`. |
| tennis-data.co.uk / valuebetennis | Closing-odds benchmark | Odds are never a training feature unless capture times are verified. |

All Sackmann-derived data is **CC BY-NC-SA 4.0**: attribution required, non-commercial only, share-alike on redistribution.

## Architecture rules
- SQLite (`PRAGMA foreign_keys=ON`, WAL, `STRICT` tables, ISO-8601 text dates). `schema.sql` is the schema; `docs/DESIGN.md` explains it.
- Own surrogate IDs (`player_id`, `tournament_id`, `edition_id`, `match_id`). Source IDs live only in `*_xref(source, source_key, id, method, confidence, verified)`.
- Resolve in order: players and tournaments, then editions, then matches. Match join key: unordered player pair + edition + date window (±2 days), round as tiebreaker.
- Fuzzy matching only inside blocks. Low confidence goes to `review_queue`, never auto-merged. Hand-confirmed links and forced non-matches live in `data/overrides.csv`, applied last, always winning.
- Snapshots: `ingest_snapshot` per raw file, `data/build_manifest.json` pins folders, `load_run` accounts for every row (`v_inv_load_counts` empty), `data/snapshot_lock.json` stops the build if pinned bytes change. Raw files are immutable. See DESIGN.md "Build protocol".
- `tennis.db` is disposable: `python scripts/build_db.py` rebuilds it from `data/raw/` plus the manifest (~10 min).
- Model results live in `models.db` (`models_schema.sql`, `src/tennis/models_db.py`), never in `tennis.db`. Never rebuilt (additive schema only); rows keyed by stable source keys (`sackmann_atp:104925`), never surrogate IDs.
- **Engine (`src/tennis/engine/`):** consumers (Kalshi retriever, CLI, web) call only `Predictor.predict(MatchQuery) -> Prediction` (`engine/predictor.py`). They never import `points_rating`/`context`/`markov` directly. Any model sits behind that seam.
- **Read-only against Kalshi.** No order placement code.

## Invariants (must be tested)
- Each `(source, source_key)` maps to exactly one entity.
- Every source row resolves, is skipped with a reason, or lands in `review_queue`. Nothing is silently dropped.
- No player appears in two matches on the same exact day in different events, or twice in one round (`v_inv_same_day_conflicts` empty). Two rounds of one event on one day are allowed (rain doubleheaders).
- Point rows: server and returner are the two players of the match.
- No feature, rating or fit uses data from after the match (or point) being predicted.
- Incremental replay from a saved state equals a full replay (once state persistence exists).
- Live scores map to markets by competitor id, never by name or order.

## Environment and conventions
- **Windows / PowerShell.** Use `curl.exe`, not `curl`. Open files with `encoding="utf-8"` (player names have accents). Use `pathlib`. No `sqlite3` CLI: use Python's `sqlite3` module.
- Python 3.12+. venv at project root (`.venv`, gitignored); `pip install -r requirements.txt` (numpy, scikit-learn).
- Layout: `src/tennis/` (data layer; `matches.py` match table; model core `points_rating`, `context`, `markov`; `kalshi.py` client), `src/tennis/engine/` (live engine), `scripts/` (data-layer scripts; engine CLIs as they're built), `tests/` with small fixtures in `tests/fixtures/` (never test on full data), `archive/` (finished work; see `archive/README.md`). Live code never imports from `archive/`. `data/raw/`, `tennis.db`, `models.db`, `models/` are gitignored.
- Tests: `python -m unittest discover -s tests` (standard library; no pytest).
- Ask before downloading large datasets or adding dependencies. Standard library where reasonable.
- Verify before trusting: never assume CSV column names, ID formats or API field structure. Read the real data first, then write the loader.
- Commit after each milestone with tests passing.
