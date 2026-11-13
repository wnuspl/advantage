# Advantage: tennis match prediction

## Goal
Predict ATP/WTA singles match outcomes with a two-layer model, judged by walk-forward log loss / Brier against (1) surface-aware Elo/Glicko and (2) closing odds.

- **Heavy layer (retrain weekly/monthly):** per-player style embeddings built with **PCA** over hand-built style features from Match Charting Project (MCP) point/shot data (serve direction mix, ace/unreturned rate, rally-length distribution, shot-type mix, net frequency, return depth, etc.), computed per surface.
- **Light layer (update after every match):** a **Glicko-style** rating per player with a rating deviation (uncertainty) that grows with inactivity and shrinks with results. It models recent form as a *delta from the heavy layer's expectation*, not a standalone strength number.
- Point probabilities (serve/return) roll up analytically (Markov chain) to game, set, match win probability.

## Modeling decisions (agreed, and why)
- **PCA, not a learned embedding, first.** Interpretable, cheap, and a baseline the learned model must beat. Standardize features, shrink thin-data players toward the population mean *before* PCA, keep the number of components by explained variance and validation loss (not a fixed number).
- **PCA is fit on training data only**, inside each walk-forward fold. Fitting it on all years leaks the future.
- **PCA is linear and has no matchup term.** Style vectors for the two players feed a downstream interaction model (e.g. product/difference of vectors into logistic regression or gradient boosting). Do not assume the embeddings alone capture "style A beats style B."
- **MCP is a biased sample** (volunteer-charted, skews to top players and big matches). Track charted-match count per player. Below a threshold, the style vector falls back to the population mean and the light layer carries the player.
- **Glicko-style light layer:** state per player (and per surface where data allows): rating, deviation, optionally volatility (Glicko-2). Update from each result; inflate deviation after inactivity and on surface change. Prefer an update on serve/return point outcomes when per-match stats exist (more information per match than win/loss), and fall back to the win/loss update when only a result exists. The light layer must tolerate matches with no stats.
- **No double-counting:** the heavy layer is trained with a cutoff so the light layer's residuals aren't already absorbed by it.
- Retirements/walkovers: exclude from light-layer updates (or down-weight); decide once, document here. (Data side done: `matches.status` is `retired`/`walkover`/`default`/`unfinished`; the model decision is still open.)

## Evaluation rules
- Walk-forward only. Never shuffle. Train on the past, predict the next block, roll forward.
- Features and ratings for a match use only data strictly before it (point-in-time correctness).
- When building training rows, randomize or canonicalize player order so the model never sees "winner first."
- Report log loss and Brier, plus calibration. Accuracy alone is not a metric here.

## Data sources (as of 2026-11-13)
Jeff Sackmann's `tennis_atp`, `tennis_wta` and slam point-by-point repos were **removed from GitHub (404 as of Oct 2026)**. Only the Match Charting Project repo is still upstream.

| Source | Use | Notes |
|---|---|---|
| `Aneeshers/tennis-sackmann-archive` (mirror) | Backbone: players, matches, per-match serve stats, rankings; slam point-by-point | Archive, not a fork: upstream commit hashes aren't recorded, so we record our own commit + sha256. Layout verified 2026-11-13: `atp/`, `wta/`, `slam_pointbypoint/`. Matches end at tourney_date 2026-05-25, rankings at 2026-06-08, slam point data 2011-2024. |
| `JeffSackmann/tennis_MatchChartingProject` | Heavy layer: point/shot data | Still upstream but could vanish: keep our own pinned copy. |
| Daily results feed (deferred) | Light layer after the mirror's cutoff | `scripts/ingest_daily.py` is a skeleton. Field names are guesses; no loader into `matches` yet. Third-party scraping: check terms and pricing before using. |
| tennis-data.co.uk, odds APIs | Benchmark (closing odds) | Later. Odds are never a training feature unless capture times are verified. |

All Sackmann-derived data is **CC BY-NC-SA 4.0**: attribution required, non-commercial only, share-alike on redistribution. Keep the licence in `source` and attribute in any write-up.

## Architecture rules
- SQLite (`PRAGMA foreign_keys=ON`, WAL, `STRICT` tables, ISO-8601 text dates). `schema.sql` is the schema; `docs/DESIGN.md` explains it.
- Own surrogate IDs (`player_id`, `tournament_id`, `edition_id`, `match_id`). Source IDs live only in `*_xref(source, source_key, id, method, confidence, verified)`.
- Resolve in order: players and tournaments, then editions, then matches. Match join key: unordered player pair + edition + date window (±2 days), round as tiebreaker.
- Fuzzy matching only inside blocks. Low confidence goes to `review_queue`, never auto-merged. Hand-confirmed links and forced non-matches live in a version-controlled `data/overrides.csv`, applied last, always winning.
- **Snapshots (implemented: `src/tennis/snapshots.py`, `loadrun.py`, `build.py`; see DESIGN.md "Build protocol"):**
  - `ingest_snapshot`: one row per raw *file* (source, source_ref, file_relpath, sha256, raw_path, row_count), unique on `(source, file_relpath, sha256)`.
  - `data/build_manifest.json` pins which downloaded folder each source uses. A build never silently picks "the newest."
  - Registering a file is separate from loading it. A `load_run` table records rows read / inserted / skipped / queued per loader per file. A view (`v_inv_load_counts`) must return zero rows: every source row is accounted for.
  - If a registered file's bytes change under the same commit, the build stops (hashes kept in tracked `data/snapshot_lock.json`). Raw files are immutable.
  - Bulk tables (`match_player_stats`, `ranking`, `point`) carry a `snapshot_id`.
- The database is disposable: `python scripts/build_db.py` rebuilds `tennis.db` from `data/raw/` plus the manifest.
- Model results live in `models.db` (`models_schema.sql`, `src/tennis/models_db.py`), never in `tennis.db`. It is never rebuilt (additive schema only) and keys rows by stable source keys (`sackmann_atp:104925`), never surrogate IDs, which change on every rebuild.

## Invariants (must be tested)
- Each `(source, source_key)` maps to exactly one entity.
- Every source row resolves, is skipped with a reason, or lands in `review_queue`. Nothing is silently dropped.
- No player appears in two matches on the same exact day (`v_inv_same_day_conflicts` empty).
- Point rows: server and returner are the two players of the match.
- No feature, rating, or PCA fit uses data from after the match being predicted.

## Environment and conventions
- **Windows / PowerShell.** Use `curl.exe`, not `curl`. Open files with `encoding="utf-8"` (player names have accents). Use `pathlib`. No `sqlite3` CLI: use Python's `sqlite3` module.
- Python 3.12+. venv at project root (`.venv`, gitignored). Code in `src/tennis/`, scripts in `scripts/`, tests in `tests/` with small fixtures in `tests/fixtures/` (never test on full data).
- Layout: `schema.sql`, `docs/DESIGN.md`, `data/raw/` (gitignored), `data/SOURCES.md`, `data/build_manifest.json`, `data/snapshot_lock.json`, `data/overrides.csv` (all tracked), `data/build_report.json`, `tennis.db` and `models.db` (gitignored).
- Tests: `python -m unittest discover -s tests` (standard library; no pytest dependency). Full build: `python scripts/build_db.py` (~10 min).
- Ask before downloading large datasets or adding dependencies. Standard library where reasonable.
- Verify before trusting: never assume CSV column names, ID formats, or MCP match-identifier structure. Read the real files first, then write the loader.
- Commit after each milestone with tests passing.

## Milestones
1. **Scaffold + schema + snapshot registration** (`init_db.py`, `snapshots.py`, manifest, `load_run`), with tests on tiny fixtures.
2. **Fetch** both sources (`scripts/fetch_data.py`), then **inspect** the real files: players file, a yearly matches file (date format, round labels, score format, stats columns), the mirror's actual layout, and how an MCP match identifier encodes date, tournament, round and names.
3. **Load Sackmann backbone:** players, tournaments/editions, matches, `match_player_stats`, rankings. Check counts against the CSVs via `load_run`.
4. **MCP resolution:** players, editions, matches through the cascade, then `point`. Report coverage and review-queue size. Spot-check actual merges by hand.
5. **Baseline:** surface-aware Elo/Glicko from `v_player_match`, walk-forward log loss. Plain Elo done (`elo_v1`, `scripts/elo_sweep.py`, results in `models.db` and DESIGN.md); surface split, calibration and Glicko-2 to follow.
6. **Style features + PCA** (fit per fold), then the interaction model on top of the baseline.
7. **Glicko-style light layer** on residuals vs the heavy layer. Compare against the baseline.
8. **Odds benchmark**, and the daily feed once the backbone is stable.

## Open questions
- How does the MCP match identifier encode date/tournament/round/names, and how often does it disagree with the mirror?
- Which feature set and how many PCA components survive validation? Which charted-match threshold triggers the fallback?
- Should the Glicko-style layer run per surface, per component (serve vs return), or both?
- How to treat retirements in the light-layer update and in point-level stats.
- Women's-tour name variants (marriage, transliteration) need their own alias pass.
