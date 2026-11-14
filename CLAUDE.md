# Advantage: tennis match prediction

## Goal
Predict ATP/WTA singles match outcomes, judged by walk-forward log loss / Brier against (1) the Glicko-2 baseline and (2) closing odds. **Target: tour-level test log loss below 0.60** (calibrated Glicko-2: 0.6045 on 2022-26). First measure the closing-odds log loss on the same matches, to know whether the target is reachable.

**Ratings are the model; style is an optional correction** (re-prioritized 2026-11-14 after the style experiments in "Findings" below). Order of importance:

1. **Ratings layer (primary, updated after every match):** serve and return ratings per player (per surface where data allows) with Glicko-style deviation. Updated on **serve/return points won** wherever `match_player_stats` has them, on the result otherwise. Point probabilities roll up analytically (Markov chain) to game, set and match win probability, so best-of-3 vs best-of-5 and tiebreak rules come for free.
2. **Context features (cheap):** best-of-5, rest days and recent workload, home country, tournament level, layoffs. Added on top of the ratings by a small calibrated model.
3. **Style layer (optional, small):** per-player phase profiles from MCP point data (rally-length steering and win rate by phase), contributing a heavily shrunk matchup correction. Kept only if it lowers walk-forward log loss once 1 and 2 are in. The PCA style vectors (`style_pca_v1`) and the matchup layer built on them stay in the repo as a measured negative result; don't extend them.

"Heavy" and "light" in older docs and code mean slow-to-retrain and per-match, not important and unimportant.

## Modeling decisions (agreed, and why)
- **Ratings layer:** state per player (and per surface where data allows): serve rating, return rating, deviation, optionally volatility (Glicko-2). Inflate deviation after inactivity and on surface change. Update on serve/return points won when per-match stats exist (far more information per match than win/loss); fall back to the win/loss update when only a result exists. It must tolerate matches with no stats: AcesAPI results after the mirror's cutoff (2026-05-25) have none.
- **Point probabilities -> match probability by Markov chain**, with best-of and final-set tiebreak rules taken from the results data, never assumed.
- **Calibrate the output.** A scalar on the Glicko logit alone gained 0.0006 log loss (run 203 vs 187): raw Glicko-2 is slightly overconfident. Every new model is scored before and after a calibration fitted on past data only.
- **Context model on top of the ratings:** a small logistic or gradient-boosted model on the ratings' logit plus context features. Antisymmetric in the two players (or trained on both orderings), so P(A beats B) = 1 - P(B beats A).
- **Style layer only as a shrunk correction.** Point-in-time phase profiles (shrunk toward the population), combined A x B, with the shrinkage fitted on validation folds. MCP is a biased sample (top players, big matches): players without enough charted points get no correction, never an imputed one.
- **No double-counting:** any layer trained on residuals of another uses a cutoff, so the residuals aren't already absorbed.
- Retirements/walkovers: exclude from rating updates (or down-weight); decide once, document here. (Data side done: `matches.status` is `retired`/`walkover`/`default`/`unfinished`; the model decision is still open.)

## Findings (2026-11-14; details in DESIGN.md and docs/composite_report.md)
- **PCA style vectors + matchup layer** (bilinear, siamese, LightGBM; sweep `matchup-v1-full`, runs 202-208): no significant gain over calibrated Glicko-2 (best: siamese -0.0003 ± 0.0003 log loss on tour-level 2022-26). Not even in close matches (|p - 0.5| < 0.1). Most of the siamese signal is per-player ("better than the rating says"), not pairing-specific.
- **Personal style edges don't repeat:** a player's over-performance against an opponent-style cluster correlates r ≈ 0.02 between periods (inside the shuffle-noise band); general over-performance vs Glicko repeats at r ≈ 0.17. Graph: `reports/personal_style_matchups.png`.
- **Phase profiles are real but small:** win rate by phase (serve/return x short/medium/long rally, net) relative to the player's own level repeats at r = 0.27-0.70 across periods. A phase model's A x B term predicts points won beyond Glicko at ~3 SE, but its size is ~0.1 pp of points (estimated ~0.0002 log loss). The per-player part is stronger (~7 SE, ~0.001 estimated). Not yet controlled for surface. Graph: `reports/phase_test.png`.
- **Phase correction (milestone 10, done, negative):** surface control changes nothing (edge stability and residual signal identical pooled vs surface-adjusted vs same-surface). The matchup term still predicts points beyond points_v1 (r = 0.04, t ~ 7), but as a shrunk correction on points_v1 run 273 it moves tour-level log loss by -0.0001 ± 0.0001 on test and +0.0001 on val. Not kept. The integration is not the limit: an oracle fit directly on test match outcomes gains only 0.0001. Charted level's t ~ 20 on points is mostly points_v1 compression that calibration already fixes. See DESIGN.md "Phase style correction", `results/phase_test.json`.
- **Implication:** the information Glicko misses is mostly per-player and visible in points won, which `match_player_stats` has for nearly every tour-level match. Points-based ratings are the main lever.
- **Confirmed (milestone 8):** points-based serve/return ratings beat calibrated Glicko-2 by 0.0077 log loss on tour-level test (0.5968 vs 0.6045), 25x the best style model's gain.

## Evaluation rules
- Walk-forward only. Never shuffle. Train on the past, predict the next block, roll forward.
- Features and ratings for a match use only data strictly before it (point-in-time correctness).
- When building training rows, randomize or canonicalize player order so the model never sees "winner first."
- Report log loss and Brier, plus calibration. Accuracy alone is not a metric here.

## Data sources (as of 2026-11-14)
Jeff Sackmann's `tennis_atp`, `tennis_wta` and slam point-by-point repos were **removed from GitHub (404 as of Oct 2026)**. Only the Match Charting Project repo is still upstream.

| Source | Use | Notes |
|---|---|---|
| `Aneeshers/tennis-sackmann-archive` (mirror) | Backbone: players, matches, per-match serve stats, rankings; slam point-by-point | Archive, not a fork: upstream commit hashes aren't recorded, so we record our own commit + sha256. Layout verified 2026-11-13: `atp/`, `wta/`, `slam_pointbypoint/`. Matches end at tourney_date 2026-05-25, rankings at 2026-06-08, slam point data 2011-2024. |
| `JeffSackmann/tennis_MatchChartingProject` | Style layer (optional): point/shot data | Still upstream but could vanish: keep our own pinned copy. |
| AcesAPI (`acesapi`, free plan: 500 credits/month) | Results after the mirror's cutoff: tour-level singles + qualifying + WTA 125, daily | `scripts/fetch_acesapi.py --pin` (key `ACESAPI_KEY` in `.env`), then `build_db.py`. One immutable pull folder per fetch; the build loads the latest pull per tournament; `match_seen` keeps first-seen times. No serve stats, no Challenger/ITF. Terms: storage and research use allowed, no redistribution. See DESIGN.md "AcesAPI". First pull 2026-11-14 (109 draws, 221 credits). |
| Scraped daily feed (shelved) | Stats / Challenger coverage | `scripts/ingest_daily.py` is a skeleton (Apify). Only if Challenger coverage or serve stats become worth it. |
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
- No player appears in two matches on the same exact day in different events, or twice in one round (`v_inv_same_day_conflicts` empty). Two rounds of one event on one day are allowed: real rain doubleheaders (85 in the first AcesAPI pull).
- Point rows: server and returner are the two players of the match.
- No feature, rating, or PCA fit uses data from after the match being predicted.

## Environment and conventions
- **Windows / PowerShell.** Use `curl.exe`, not `curl`. Open files with `encoding="utf-8"` (player names have accents). Use `pathlib`. No `sqlite3` CLI: use Python's `sqlite3` module.
- Python 3.12+. venv at project root (`.venv`, gitignored); `pip install -r requirements.txt` (numpy, scikit-learn). Code in `src/tennis/`, scripts in `scripts/`, tests in `tests/` with small fixtures in `tests/fixtures/` (never test on full data).
- Layout: `schema.sql`, `docs/DESIGN.md`, `data/raw/` (gitignored), `data/SOURCES.md`, `data/build_manifest.json`, `data/snapshot_lock.json`, `data/overrides.csv` (all tracked), `data/build_report.json`, `tennis.db`, `models.db` and `models/` (fitted pipelines, cached counts; gitignored).
- Tests: `python -m unittest discover -s tests` (standard library; no pytest dependency). Full build: `python scripts/build_db.py` (~10 min).
- Ask before downloading large datasets or adding dependencies. Standard library where reasonable.
- Verify before trusting: never assume CSV column names, ID formats, or MCP match-identifier structure. Read the real files first, then write the loader.
- Commit after each milestone with tests passing.

## Milestones
1. **Scaffold + schema + snapshot registration** (`init_db.py`, `snapshots.py`, manifest, `load_run`), with tests on tiny fixtures.
2. **Fetch** both sources (`scripts/fetch_data.py`), then **inspect** the real files: players file, a yearly matches file (date format, round labels, score format, stats columns), the mirror's actual layout, and how an MCP match identifier encodes date, tournament, round and names.
3. **Load Sackmann backbone:** players, tournaments/editions, matches, `match_player_stats`, rankings. Check counts against the CSVs via `load_run`.
4. **MCP resolution:** players, editions, matches through the cascade, then `point`. Report coverage and review-queue size. Spot-check actual merges by hand.
5. **Baseline:** surface-aware Elo/Glicko from `v_player_match`, walk-forward log loss. Done: surface Elo (`elo_v1`, run 132, tour test log loss 0.6076) and surface Glicko-2 (`glicko2_v1`, run 187, 0.6051, better calibrated). **Baseline to beat = run 187.** Sweeps: `scripts/sweep.py --model ...`; standings: `scripts/ratings.py`. See DESIGN.md.
6. **Style experiments (done, negative/small; see "Findings"):** PCA style vectors (`style_pca_v1`, `scripts/style_fit.py`), matchup layer (`src/tennis/matchup/`, `scripts/matchup.py`), composite predict/eval/check (`scripts/composite.py`). Kept as tooling: `composite.py eval` is the standard way to compare any model against the baselines on identical matches.
7. **Odds benchmark:** closing odds (tennis-data.co.uk; check terms, ask before downloading) joined to the backbone; log loss of the de-vigged market on the same 2022-26 tour-level matches. Tells us whether < 0.60 is reachable.
8. **Points-based ratings (main model): done (`points_v1`, run 605).** Serve/return ratings updated on points won (Kalman), win/loss fallback (extended Kalman), Markov roll-up, walk-forward calibration. Tour-level test 2022-26: **0.5968 calibrated (0.5971 raw) vs 0.6045 for run 203**, -0.0077 ± 0.0010, 17/18 quarters. **Baseline to beat = run 605.** Compare models with `scripts/compare_runs.py`. Worse than Glicko-2 on all levels (0.5752 vs 0.5734). See DESIGN.md "Points-based ratings". Checked 2026-11-14: scored on the identical 21,362 matches as runs 187/203 (`compare_runs.py` inner-joins stored 203 and reproduces it exactly); calibration alphas use only matches before each quarter (`tests/test_points.py` CalibrateTest); every prediction is made from the state before the match, and a match's stats update only afterwards (PointsModelTest). The only same-edition input is the deciding-set format, inferred from the edition's scores (an announced rule, not an outcome). Earlier sweep runs 273/602 (used for milestones 7 and 10) are superseded by 605: same model, tour val 0.5945 vs 0.5984 (273) and 0.5955 (602); test raw 0.5971 vs 0.5986 (602).
   - **Fallback tiers:** points won (mirror, to 2026-05-25) > score (games, holds/breaks; AcesAPI stores `score`) > result only.
   - **Stats blackout test (do once 8 is done):** rerun walk-forward with per-match stats hidden after a past date (e.g. 2023-01-01), so only score/result updates apply from then on; plot log loss by month since blackout vs the full-stats run. Decides how urgently a post-2026-05 stats source is needed.
9. **Context features:** best-of-5 (if the Markov roll-up doesn't already cover it), rest days and workload, home country, level, layoffs; a small calibrated model on top of milestone 8.
10. **Phase style correction (done, negative; not kept):** surface-controlled test (`scripts/phase_test.py`, `src/tennis/phase.py`) and the shrunk A x B correction on points_v1. Log loss change -0.0001 ± 0.0001 on test, worse on val. Revisit only on top of milestone 9.
10b. **Siamese matchup on a points_v1 base:** retrain the siamese model (`src/tennis/matchup/`, `scripts/matchup.py`) on the heavy-layer PCA style vectors (`style_pca_v1`), with points_v1 (run 273, calibrated) as the base logit in place of Glicko-2. Same walk-forward folds as `matchup-v1-full`, validation first, test only once the settings are fixed. Why: a stacking check on the stored run 207 predictions found the siamese deviation from Glicko is nearly uncorrelated with points_v1's gain (r = 0.07) and adds ~0.0002 ± 0.00013 on top of it (cross-fitted on test, so not yet a claim; DESIGN.md "Matchup layer"). Keep only if walk-forward log loss beats calibrated points_v1 on identical matches (`composite.py eval`).
11. **Live predictions:** daily feed (AcesAPI) into the ratings; `composite.py predict` rebuilt on the new model.

## Open questions
- How does the MCP match identifier encode date/tournament/round/names, and how often does it disagree with the mirror?
- Ratings per surface, per component (serve vs return), or both? How to blend points-based and win/loss updates?
- How to treat retirements in rating updates and in point-level stats.
- What closing-odds log loss do we face on 2022-26 tour-level matches?
- Women's-tour name variants (marriage, transliteration) need their own alias pass.
