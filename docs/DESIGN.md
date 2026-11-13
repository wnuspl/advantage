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

Register each in `source` with its license. Every raw file gets an `ingest_snapshot` row (hash, fetch time, commit/URL). Raw files are immutable and parsers are re-runnable.

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
3. **Overrides last.** `override_link` is loaded from a version-controlled file (`overrides.csv` or YAML) and always wins. `forbid` rows block specific links.

Resolve means: either a xref row exists, or a `review_queue` row exists. Nothing is dropped silently.

## Point-in-time rules (leakage)

- Features and ratings for a match may use only matches that happened strictly before it. Order by `(edition start_date, round_order)` when `match_date` is NULL.
- `matches.date_precision` says whether `match_date` is exact (`day`) or only the week is known (`week`). Same-day logic applies to `day` rows only.
- Rankings: use the latest `ranking_date` strictly before the match, never the rank stored on the match row for features.
- `rating_snapshot` rows are append-only: the state after all matches before `as_of_date`.
- `prediction` is written before the result is known, and evaluated walk-forward only.
- Odds are a benchmark. Using them as features needs verified capture times.

## Table map

| Group | Tables |
|---|---|
| Plumbing | `schema_meta`, `source`, `ingest_snapshot` |
| Entities | `player`, `player_alias`, `tournament`, `tournament_alias`, `tournament_edition` |
| Crosswalks | `player_xref`, `edition_xref`, `match_xref` |
| Matches | `matches`, `match_player_stats`, `ranking` |
| Points | `point`, `shot` (phase 2) |
| Benchmark | `odds` |
| Resolution | `review_queue`, `override_link` |
| Modeling | `rating_snapshot`, `prediction` |
| Views | `v_player_match`, `v_inv_same_day_conflicts`, `v_open_review` |

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
