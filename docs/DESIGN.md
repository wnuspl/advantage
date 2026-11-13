# Tennis data backbone: design

Companion to `schema.sql` (SQLite, validated on 3.45). Requires SQLite 3.37+ for `STRICT` tables and generated columns.

## Purpose

One local store that merges several tennis data sources behind stable internal IDs, so the two-layer model (heavy: style embeddings from point data; light: per-match recent-form updates) reads from a single, leakage-safe source of truth.

## Sources and what populates what

| Source | Populates | Notes |
|---|---|---|
| Sackmann `tennis_atp` / `tennis_wta` | `player` (seed), `tournament`, `tournament_edition`, `matches`, `match_player_stats`, `ranking` | Backbone. Verify current coverage and update cadence before relying on it. |
| Match Charting Project (MCP) | `point`, `shot`, match/player xrefs | CC BY-NC-SA 4.0: attribution, non-commercial, share-alike on redistribution. Charted matches skew toward top players. |
| Sackmann slam point-by-point | `point` | AO/FO from 2018 have no serve speed and a placeholder rally length. Leave those columns NULL. |
| tennis-data.co.uk | `odds` (benchmark) | Resolve to `match_id` through `match_xref`. |

Register each in `source` with its license (declared in `data/build_manifest.json`). Every raw *file* gets an `ingest_snapshot` row (sha256, row count, commit). Raw files are immutable and parsers are re-runnable.

## ID strategy

- Own surrogate IDs: `player_id`, `tournament_id`, `edition_id`, `match_id`. Never reuse a source ID as a key.
- Source IDs live only in `player_xref`, `edition_xref`, `match_xref`, keyed `(source, source_key)`. A source key maps to exactly one entity. Many source keys may map to one entity.
- Every link records `method`, `confidence`, and `verified`, so a bad rule can be found and undone in one query.
- Sources without IDs (MCP) use their name string as `source_key`.

## Resolution pipeline (run in this order)

1. **Players and tournaments first.** Matches depend on both.
   1. Seed `player` from Sackmann.
   2. Exact match on normalized name, disambiguated by hand/DOB/country where the source has them.
   3. Fuzzy match, only inside blocks (same tour, same last-name initial or same edition). Below the confidence threshold, write to `review_queue`. Never auto-merge.
   4. Context evidence: two sources agreeing on opponent, tournament, and date confirms both players.
2. **Matches.** Key on the unordered pair (`pair_lo_id`, `pair_hi_id`) + edition + a date window of ±2 days, with round as tiebreaker.
3. **Overrides always win.** `data/overrides.csv` (version-controlled) is loaded into `override_link` before any name-based source is resolved; resolvers consult it first (`link` forces a target, `forbid` removes a candidate), and the verify step re-checks that every override was honoured. Targets are written as **stable source keys** (`target_source,target_key`, e.g. `sackmann_atp,104925`), never surrogate IDs, because surrogate IDs change on every rebuild. An override whose target does not resolve stops the build.

Resolve means: either a xref row exists, or a `review_queue` row exists. Nothing is dropped silently.

## Build protocol (`python scripts/build_db.py`)

Code: `src/tennis/build.py`. The database is disposable; a build always starts from an empty file `tennis.db.building` and swaps it in only after every check passes. On failure the previous `tennis.db` is untouched.

| Step | What | Stops the build when |
|---|---|---|
| 0 preflight | Read `data/build_manifest.json`. Classify every file in each pinned folder into exactly one group (which names its loader) or an `ignore` pattern with a reason. | A pinned folder is missing; a file matches no group/ignore or two groups; a group pattern matches nothing (layout drift). |
| 1 schema | Apply `schema.sql`. | SQLite < 3.37. |
| 2 register | Insert `source` rows; hash + count every planned file; one `ingest_snapshot` row per file. Compare with `data/snapshot_lock.json` (tracked) and add new files to it. | A locked file's bytes changed or it disappeared. |
| 3 backbone | Sackmann players, then editions + matches + `match_player_stats`, then rankings. | A loader's outcomes don't add up to rows read. |
| 4 overrides | `data/overrides.csv` -> `override_link`. | Bad header, unknown target, link+forbid on the same target. |
| 5 MCP | Charted matches: resolve both names, then the match; then points. | (as 3) |
| 6 slam pbp | Slam matches: resolve within the backbone slam edition; then points. | (as 3) |
| 7 verify | `v_inv_load_counts`, `v_inv_same_day_conflicts`, `v_inv_point_players`, foreign keys, overrides honoured, no xref key spanning two tours. | Any of them returns rows. |
| 8 publish | `ANALYZE`, write `data/build_report.json`, swap the file in. | `tennis.db` is held open by another program. |

**Load accounting.** Registering a file is separate from loading it. Each loader opens one `load_run` per file and records one outcome per source row: `inserted` (new entity), `linked` (xref to an existing entity), `skipped` (reason counted in `load_skip`, with example line numbers), or `queued` (`review_queue`). `v_inv_load_counts` must be empty: every registered file has a finished run, `rows_read = row_count`, and the four outcomes sum to `rows_read`.

**Backbone decisions (from inspecting the real files, mirror commit 83733587):**
- ATP and WTA player IDs overlap numerically, so `sackmann_atp` and `sackmann_wta` are separate sources.
- `(tourney_id, match_num)` is not unique: WTA main and qualifying files reuse numbers, and the qual/ITF files repeat ~12k matches verbatim. The match source key is `tourney_id:match_num:round:winner_id:loser_id`. A row that repeats an already-loaded (edition, round, pair) with the same winner and score is skipped as `exact_duplicate_row` (or linked if its key is new). A different result is queued as `duplicate_key_conflict` (~24 rows: pro-tour round-robins where a pair met twice in one round).
- Placeholder players (`Unknown`, `X X`) are kept in `player` with `is_placeholder = 1`, but matches involving them are skipped (they would act as one very busy fake opponent in ratings). Byes are skipped.
- Sackmann has only the event's Monday (`tourney_date`), so matches are `date_precision = 'week'`, `match_date` NULL. The same-day invariant is therefore trivially empty for backbone rows.
- Two draws with the same name in the same week (e.g. `Argentina F5` a/b) become separate tournaments (`name [tourney_id]`).
- Tournament identity is the normalized tournament name per tour. Sponsor renames therefore split a tournament's history (open question).
- Doubles, MCP `-stats-` aggregates and docs are ignored by manifest rule.

**MCP resolution.**
- Player source key: `m:<name>` / `w:<name>`. The cascade is: override; exact normalized name if unique; context (exactly one same-name candidate whose backbone career span ±60 days covers the charted dates); otherwise `review_queue` with up to 5 fuzzy candidates from surname/token blocks. Fuzzy is never auto-linked.
- Match: unordered pair, plus a backbone edition starting 16 days before to 9 days after the charted date. Tiebreaks: a main-draw match cannot be before its edition starts (±2 days); same round; tournament-name similarity; edition start equal to the charted date (old MCP charts often carry the event's start date). A pair meets once per event, so a single candidate with a different round label still links (confidence 0.9).
- Charts dated after the backbone's last edition + 16 days are skipped as `after_backbone_cutoff` (they resolve once the daily feed extends the backbone). A second chart of an already-charted match is linked, but its points are skipped (`duplicate_chart`).
- Point parsing is v0: serve number, ace, double fault, end type, and rally length ("including the serve, excluding errors") read from the deciding serve's code. `raw_first`/`raw_second` are kept for the shot parser.

**Slam point-by-point resolution.**
- `match_num` is `[12]R..` (1 = men, 2 = women; R = round digit 1..7 = R128..F), or `MS/WS R..` in 2021 AO.
- Player source keys are scoped to the edition (`atp:2020:ausopen:A Bedene`), because this source writes names differently across years. Context = players in that backbone edition. That resolves exact-name collisions and initial+surname forms (2018-22 AO/FO).
- Initials forms resolved against the edition's participants: `A Bedene`, `J. L. Struff`/`Jl Struff` (initials of `Jan Lennard`), `Ka. Pliskova` (first letters of `Karolina`, which separates the Pliskova twins), `Mt.Torro-Flor`, `B.Zahlavova Strycova` (surname tokens overlap). Also tried when the exact name belongs only to players who are not in the event (`D. Collins`).

**Opponent anchoring (both name-based sources).** Suppose one side of a match row resolves and the other doesn't. The resolved player's backbone opponents in the same event and round are looked up (slam: the edition; MCP: the date window, main-draw start rule applied). If exactly one of them has a compatible name, the name is linked (`method='context'`, confidence 0.9). Compatible means a shared name token of 3+ letters, or an initials form. This catches first-name variants (Stanislas/Stan, Soonwoo/Soon Woo, Cori/Coco) and married names (Gavrilova/Saville, Kozlova/Baindl, Zakopalova/Koukalova). The name's `review_queue` row stays, marked `resolved`, as an audit trail. On the 2026-11-13 build this resolved 215 names; a random sample of 30 was hand-checked and all were correct.

**Hand review.** `python scripts/export_review.py` writes the open queue to `data/review_open.csv` (gitignored), with candidates as stable source keys. Confirmed rows go into `data/overrides.csv`.
- Rally length is NULL for AO/FO 2018+ (placeholder values in the source).

## models.db (modeling results)

`models.db` (schema `models_schema.sql`, code `src/tennis/models_db.py`) holds everything a model run produces. It is **never rebuilt**: the schema is applied idempotently on every connect and changes only additively. `build_db.py` does not touch it, so sweeps and chosen parameters survive any number of data rebuilds.

- **Stable keys, not surrogate IDs.** Rows refer to players and matches by the source key of their backbone (seed) xref, e.g. `sackmann_atp:104925` and `sackmann_atp:2024-0339:300:F:105777:208029`. `models_db.match_keys()` / `player_keys()` map the current `tennis.db` IDs to them. Analysis code attaches `tennis.db` read-only as `data`.
- **Tables:** `model_run` (model id, canonical parameter JSON, sweep id, the data build and source commits it read, git commit), `run_metric` (log loss / Brier / accuracy / calibration per evaluation block and match scope), `prediction` (canonical player order `p1_key < p2_key`, never winner-first), `pre_match_rating` (each player's rating going into each match: the feature later layers read).
- **Volume policy:** sweeps write only `model_run` + `run_metric`; per-match rows are written only for runs explicitly saved (one run is ~1.8M predictions and ~3.5M ratings). Deleting a run cascades to its rows.

## Point-in-time rules (leakage)

- Features and ratings for a match may use only matches that happened strictly before it. Order by `(edition start_date, round_order)` when `match_date` is NULL.
- `matches.date_precision` says whether `match_date` is exact (`day`) or only the week is known (`week`). Same-day logic applies to `day` rows only.
- Rankings: use the latest `ranking_date` strictly before the match, never the rank stored on the match row for features.
- Model outputs (pre-match ratings, predictions) live in `models.db`, not here. A prediction is made from state strictly before the match and evaluated walk-forward only.
- Odds are a benchmark. Using them as features needs verified capture times.

## Table map

| Group | Tables |
|---|---|
| Plumbing | `schema_meta`, `source`, `ingest_snapshot`, `load_run`, `load_skip` |
| Entities | `player`, `player_alias`, `tournament`, `tournament_alias`, `tournament_edition` |
| Crosswalks | `player_xref`, `edition_xref`, `match_xref` |
| Matches | `matches`, `match_player_stats`, `ranking` |
| Points | `point`, `shot` (phase 2) |
| Benchmark | `odds` |
| Resolution | `review_queue`, `override_link` |
| Views | `v_player_match`, `v_inv_same_day_conflicts`, `v_inv_load_counts`, `v_inv_point_players`, `v_open_review` |

Design notes:
- The table is `matches`, not `match`, because MATCH is an SQL keyword.
- `matches` stores winner/loser (how results arrive) plus generated `pair_lo_id`/`pair_hi_id` for order-independent joins. When building training rows, randomize or canonicalize player order so the model never sees "winner first."
- Surface and indoor live on `tournament_edition`, since they change year to year.
- `point` has a `data_source` column, so the same match charted in two sources coexists, with `UNIQUE (match_id, data_source, point_idx)`.
- `raw_first` / `raw_second` keep the original MCP notation, so parsers can be fixed and re-run without re-ingesting.

## Invariants (write these as tests first)

```sql
-- 1. No player in two matches on the same exact day (must return 0 rows)
SELECT * FROM v_inv_same_day_conflicts;

-- 2. A match's winner and loser are both distinct players (enforced by CHECK; test the rejection)

-- 3. Xref uniqueness is enforced by PRIMARY KEY (source, source_key); test that a second
--    entity for the same source key is rejected.

-- 4. Unresolved source rows: for each source row ingested, it must appear in a xref table
--    or in review_queue. Implement as a pipeline-level check comparing the snapshot's
--    row_count against xref rows + queue rows for that snapshot.

-- 5. Point sanity: server and returner are the two players of the match
SELECT p.point_id
FROM point p JOIN matches m USING (match_id)
WHERE NOT (p.server_id IN (m.winner_id, m.loser_id)
       AND p.returner_id IN (m.winner_id, m.loser_id)
       AND p.server_id <> p.returner_id);

-- 6. Per-tournament match counts roughly agree across sources (report, not hard fail)
```

## Build order

1. Apply `schema.sql`, add a test harness with small fixtures (a few dozen rows per source, including a name variant, a same-name collision, and a retirement).
2. Snapshot and load Sackmann: players, editions, matches, stats, rankings.
3. MCP resolution: players, editions, matches, then `point`. Report coverage and review-queue size.
4. Baseline: surface-aware Elo from `v_player_match`, walk-forward log loss. This is the bar the heavy and light layers must beat.
5. Odds loading and benchmark comparison.
6. `shot` table and style features, then the heavy and light layers.

## Open questions

- MCP: how does its identifier encode date, tournament, round, and names, and how often do those disagree with Sackmann? Check on a sample before writing the matcher.
- How to treat retirements in Elo and in point-level stats (exclude from the light-layer update, or weight down).
- Women's tour name variants (marriage, transliteration) need their own alias pass.
- Whether `ranking` needs per-source provenance if two sources disagree.
