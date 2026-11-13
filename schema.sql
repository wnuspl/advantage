-- Tennis data backbone: SQLite schema, v2
-- Applied by src/tennis/db.py (executescript). The database is disposable:
-- `python scripts/build_db.py` rebuilds tennis.db from data/raw/ + data/build_manifest.json.
-- Conventions: STRICT tables, ISO-8601 TEXT for dates/times, own surrogate IDs,
-- source IDs only ever live in *_xref tables.
--
-- v2 (2026-11-13): per-file ingest_snapshot, load_run/load_skip accounting and
-- v_inv_load_counts; snapshot_id on bulk tables; rounds Q4/Q5/PR/CR and best-of-1
-- (both occur in the Sackmann files); player.is_placeholder; override targets stored
-- as stable source keys; review_queue.detail.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

------------------------------------------------------------------------
-- 0. Meta, sources, raw snapshots, load accounting
------------------------------------------------------------------------
CREATE TABLE schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;
INSERT INTO schema_meta (key, value) VALUES ('schema_version', '2');

CREATE TABLE source (
  source      TEXT PRIMARY KEY,   -- 'sackmann_atp','sackmann_wta','sackmann_slam_pbp','mcp', ...
  description TEXT,
  license     TEXT,
  url         TEXT
) STRICT;

-- One row per raw FILE registered for a build. Raw files are immutable on disk:
-- data/snapshot_lock.json remembers each file's sha256, and a changed hash stops the build.
CREATE TABLE ingest_snapshot (
  snapshot_id   INTEGER PRIMARY KEY,
  source        TEXT NOT NULL REFERENCES source(source),
  source_ref    TEXT NOT NULL,      -- commit of the downloaded repo snapshot
  file_relpath  TEXT NOT NULL,      -- path inside the snapshot folder, forward slashes
  raw_path      TEXT NOT NULL,      -- path relative to the project root
  sha256        TEXT NOT NULL,
  row_count     INTEGER NOT NULL,   -- CSV data records (header excluded)
  file_group    TEXT NOT NULL,      -- manifest group; names the loader
  loader        TEXT NOT NULL,
  fetched_at    TEXT,
  registered_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE (source, file_relpath, sha256)
) STRICT;

-- Loading is separate from registering. One row per loader per file.
-- Every row read ends in exactly one outcome: inserted (new entity), linked (xref to an
-- existing entity), skipped (with a reason in load_skip), or queued (review_queue).
CREATE TABLE load_run (
  load_run_id   INTEGER PRIMARY KEY,
  snapshot_id   INTEGER NOT NULL REFERENCES ingest_snapshot(snapshot_id),
  loader        TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  rows_read     INTEGER NOT NULL DEFAULT 0,
  rows_inserted INTEGER NOT NULL DEFAULT 0,
  rows_linked   INTEGER NOT NULL DEFAULT 0,
  rows_skipped  INTEGER NOT NULL DEFAULT 0,
  rows_queued   INTEGER NOT NULL DEFAULT 0,
  UNIQUE (snapshot_id, loader)
) STRICT;

CREATE TABLE load_skip (
  load_run_id INTEGER NOT NULL REFERENCES load_run(load_run_id),
  reason      TEXT NOT NULL,
  n           INTEGER NOT NULL,
  example     TEXT,                -- first few CSV line numbers / keys, for spot checks
  PRIMARY KEY (load_run_id, reason)
) STRICT;

------------------------------------------------------------------------
-- 1. Players
------------------------------------------------------------------------
CREATE TABLE player (
  player_id      INTEGER PRIMARY KEY,
  tour           TEXT NOT NULL CHECK (tour IN ('atp','wta')),
  canonical_name TEXT NOT NULL,
  name_first     TEXT,
  name_last      TEXT,
  dob            TEXT CHECK (dob IS NULL OR dob GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
  hand           TEXT CHECK (hand IS NULL OR hand IN ('R','L','U')),
  height_cm      INTEGER,
  country        TEXT,
  is_placeholder INTEGER NOT NULL DEFAULT 0 CHECK (is_placeholder IN (0,1)),  -- 'Unknown', 'X X'
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
) STRICT;
CREATE INDEX idx_player_name ON player (tour, name_last, name_first);

CREATE TABLE player_xref (
  source      TEXT NOT NULL REFERENCES source(source),
  source_key  TEXT NOT NULL,      -- the source's own ID, or its name string if it has no ID
  player_id   INTEGER NOT NULL REFERENCES player(player_id),
  method      TEXT NOT NULL CHECK (method IN ('seed','exact_name','fuzzy','context','override')),
  confidence  REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0,1)),
  snapshot_id INTEGER REFERENCES ingest_snapshot(snapshot_id),
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (source, source_key)
) STRICT;
CREATE INDEX idx_player_xref_player ON player_xref (player_id);

-- Normalized name variants (lowercased, accents stripped, punctuation removed).
CREATE TABLE player_alias (
  alias_normalized TEXT NOT NULL,
  player_id        INTEGER NOT NULL REFERENCES player(player_id),
  source           TEXT REFERENCES source(source),
  PRIMARY KEY (alias_normalized, player_id)
) STRICT;

------------------------------------------------------------------------
-- 2. Tournaments and editions
------------------------------------------------------------------------
CREATE TABLE tournament (
  tournament_id  INTEGER PRIMARY KEY,
  tour           TEXT NOT NULL CHECK (tour IN ('atp','wta')),
  canonical_name TEXT NOT NULL,
  city           TEXT,
  country        TEXT,
  UNIQUE (tour, canonical_name)
) STRICT;

CREATE TABLE tournament_alias (
  alias_normalized TEXT NOT NULL,
  tournament_id    INTEGER NOT NULL REFERENCES tournament(tournament_id),
  PRIMARY KEY (alias_normalized, tournament_id)
) STRICT;

-- One row per tournament-year. Surface/indoor live here because they change over time.
CREATE TABLE tournament_edition (
  edition_id    INTEGER PRIMARY KEY,
  tournament_id INTEGER NOT NULL REFERENCES tournament(tournament_id),
  year          INTEGER NOT NULL,
  start_date    TEXT NOT NULL,
  end_date      TEXT,
  level         TEXT,             -- raw Sackmann tourney_level: 'G','M','A','C','S','D','15', ...
  surface       TEXT CHECK (surface IS NULL OR surface IN ('hard','clay','grass','carpet')),
  indoor        INTEGER CHECK (indoor IS NULL OR indoor IN (0,1)),
  draw_size     INTEGER,
  best_of       INTEGER CHECK (best_of IS NULL OR best_of IN (1,3,5)),
  UNIQUE (tournament_id, start_date)
) STRICT;
CREATE INDEX idx_edition_start ON tournament_edition (start_date);

CREATE TABLE edition_xref (
  source      TEXT NOT NULL REFERENCES source(source),
  source_key  TEXT NOT NULL,
  edition_id  INTEGER NOT NULL REFERENCES tournament_edition(edition_id),
  method      TEXT NOT NULL CHECK (method IN ('seed','exact_name','fuzzy','context','override')),
  confidence  REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0,1)),
  snapshot_id INTEGER REFERENCES ingest_snapshot(snapshot_id),
  PRIMARY KEY (source, source_key)
) STRICT;
CREATE INDEX idx_edition_xref_edition ON edition_xref (edition_id);

------------------------------------------------------------------------
-- 3. Matches (table is "matches" because MATCH is an SQL keyword)
------------------------------------------------------------------------
CREATE TABLE matches (
  match_id       INTEGER PRIMARY KEY,
  edition_id     INTEGER NOT NULL REFERENCES tournament_edition(edition_id),
  round          TEXT NOT NULL CHECK (round IN
                   ('Q1','Q2','Q3','Q4','Q5','PR','ER','R128','R64','R32','R16',
                    'RR','CR','QF','SF','BR','F')),
  round_order    INTEGER NOT NULL,           -- chronological order within an edition
  match_date     TEXT,                       -- exact day if known, else NULL
  date_precision TEXT NOT NULL CHECK (date_precision IN ('day','week')),
  winner_id      INTEGER NOT NULL REFERENCES player(player_id),
  loser_id       INTEGER NOT NULL REFERENCES player(player_id),
  score          TEXT,
  status         TEXT NOT NULL DEFAULT 'completed'
                   CHECK (status IN ('completed','retired','walkover','default','unfinished')),
  best_of        INTEGER CHECK (best_of IS NULL OR best_of IN (1,3,5)),
  minutes        INTEGER,
  snapshot_id    INTEGER NOT NULL REFERENCES ingest_snapshot(snapshot_id),
  -- Order-independent pair key: use for joins across sources and for leakage-safe features.
  pair_lo_id     INTEGER GENERATED ALWAYS AS (MIN(winner_id, loser_id)) STORED,
  pair_hi_id     INTEGER GENERATED ALWAYS AS (MAX(winner_id, loser_id)) STORED,
  CHECK (winner_id <> loser_id),
  UNIQUE (edition_id, round, pair_lo_id, pair_hi_id)
) STRICT;
CREATE INDEX idx_matches_edition ON matches (edition_id, round_order);
CREATE INDEX idx_matches_date    ON matches (match_date);
CREATE INDEX idx_matches_winner  ON matches (winner_id);
CREATE INDEX idx_matches_loser   ON matches (loser_id);
CREATE INDEX idx_matches_pair    ON matches (pair_lo_id, pair_hi_id);

CREATE TABLE match_xref (
  source      TEXT NOT NULL REFERENCES source(source),
  source_key  TEXT NOT NULL,      -- MCP match_id, slam match_id, or Sackmann tid:num:round:w:l
  match_id    INTEGER NOT NULL REFERENCES matches(match_id),
  method      TEXT NOT NULL CHECK (method IN ('seed','pair_tournament_date','fuzzy','context','override')),
  confidence  REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0,1)),
  snapshot_id INTEGER REFERENCES ingest_snapshot(snapshot_id),
  PRIMARY KEY (source, source_key)
) STRICT;
CREATE INDEX idx_match_xref_match ON match_xref (match_id);

-- Long format: two rows per match (one per player) per source.
CREATE TABLE match_player_stats (
  match_id       INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  player_id      INTEGER NOT NULL REFERENCES player(player_id),
  source         TEXT NOT NULL REFERENCES source(source),
  is_winner      INTEGER NOT NULL CHECK (is_winner IN (0,1)),
  aces           INTEGER,
  double_faults  INTEGER,
  serve_points   INTEGER,
  first_in       INTEGER,
  first_won      INTEGER,
  second_won     INTEGER,
  serve_games    INTEGER,
  bp_saved       INTEGER,
  bp_faced       INTEGER,
  rank_at_match  INTEGER,
  rank_points    INTEGER,
  seed           INTEGER,
  entry          TEXT,
  snapshot_id    INTEGER NOT NULL REFERENCES ingest_snapshot(snapshot_id),
  PRIMARY KEY (match_id, player_id, source)
) STRICT;

-- Weekly rankings. As-of lookup: latest ranking_date strictly before the match.
CREATE TABLE ranking (
  tour         TEXT NOT NULL CHECK (tour IN ('atp','wta')),
  ranking_date TEXT NOT NULL,
  player_id    INTEGER NOT NULL REFERENCES player(player_id),
  rank         INTEGER NOT NULL,
  points       INTEGER,
  snapshot_id  INTEGER NOT NULL REFERENCES ingest_snapshot(snapshot_id),
  PRIMARY KEY (tour, ranking_date, player_id)
) STRICT;
CREATE INDEX idx_ranking_player ON ranking (player_id, ranking_date);

------------------------------------------------------------------------
-- 4. Point-level data (MCP, slam point-by-point)
------------------------------------------------------------------------
CREATE TABLE point (
  point_id       INTEGER PRIMARY KEY,
  match_id       INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  data_source    TEXT NOT NULL REFERENCES source(source),
  point_idx      INTEGER NOT NULL,           -- 1-based order within the match
  set_idx        INTEGER,
  game_idx       INTEGER,
  server_id      INTEGER NOT NULL REFERENCES player(player_id),
  returner_id    INTEGER NOT NULL REFERENCES player(player_id),
  winner_id      INTEGER NOT NULL REFERENCES player(player_id),  -- point winner
  serve_no       INTEGER CHECK (serve_no IS NULL OR serve_no IN (1,2)),  -- serve it was decided on
  is_ace         INTEGER CHECK (is_ace IS NULL OR is_ace IN (0,1)),
  is_double_fault INTEGER CHECK (is_double_fault IS NULL OR is_double_fault IN (0,1)),
  rally_len      INTEGER,                    -- NULL where the source does not provide it
  end_type       TEXT,                       -- 'winner','unforced_error','forced_error','ace','double_fault'
  is_tiebreak    INTEGER CHECK (is_tiebreak IS NULL OR is_tiebreak IN (0,1)),
  is_break_point INTEGER CHECK (is_break_point IS NULL OR is_break_point IN (0,1)),
  score_before   TEXT,
  raw_first      TEXT,                       -- original MCP notation, kept for re-parsing
  raw_second     TEXT,
  snapshot_id    INTEGER NOT NULL REFERENCES ingest_snapshot(snapshot_id),
  UNIQUE (match_id, data_source, point_idx)
) STRICT;
CREATE INDEX idx_point_server ON point (server_id);

-- Phase 2: shot-by-shot (MCP only). Expect low millions of rows.
CREATE TABLE shot (
  shot_id    INTEGER PRIMARY KEY,
  point_id   INTEGER NOT NULL REFERENCES point(point_id) ON DELETE CASCADE,
  shot_idx   INTEGER NOT NULL,
  hitter_id  INTEGER NOT NULL REFERENCES player(player_id),
  shot_type  TEXT,
  direction  INTEGER,
  depth      INTEGER,
  outcome    TEXT,
  raw_code   TEXT,
  UNIQUE (point_id, shot_idx)
) STRICT;

------------------------------------------------------------------------
-- 5. Odds (benchmark; never a training feature unless timing is verified)
------------------------------------------------------------------------
CREATE TABLE odds (
  match_id     INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  source       TEXT NOT NULL REFERENCES source(source),
  bookmaker    TEXT NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('open','close','other')),
  player_id    INTEGER NOT NULL REFERENCES player(player_id),
  decimal_odds REAL NOT NULL CHECK (decimal_odds > 1),
  captured_at  TEXT,
  PRIMARY KEY (match_id, source, bookmaker, kind, player_id)
) STRICT;

------------------------------------------------------------------------
-- 6. Entity resolution support
------------------------------------------------------------------------
CREATE TABLE review_queue (
  review_id      INTEGER PRIMARY KEY,
  entity_type    TEXT NOT NULL CHECK (entity_type IN ('player','edition','match')),
  source         TEXT NOT NULL REFERENCES source(source),
  source_key     TEXT NOT NULL,
  candidate_ids  TEXT,                       -- JSON array of candidate entity IDs
  reason         TEXT NOT NULL,              -- 'ambiguous','low_confidence','no_candidate',
                                             -- 'player_unresolved','duplicate_key_conflict', ...
  detail         TEXT,                       -- JSON: names, dates, scores, for the reviewer
  snapshot_id    INTEGER REFERENCES ingest_snapshot(snapshot_id),
  status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','ignored')),
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  resolved_at    TEXT,
  UNIQUE (entity_type, source, source_key)
) STRICT;

-- Loaded from data/overrides.csv (version-controlled); always wins over automatic rules.
-- Targets are stable SOURCE keys (surrogate IDs change on every rebuild), resolved to
-- target_id at build time. An override whose target does not resolve stops the build.
CREATE TABLE override_link (
  entity_type   TEXT NOT NULL CHECK (entity_type IN ('player','edition','match')),
  source        TEXT NOT NULL REFERENCES source(source),
  source_key    TEXT NOT NULL,
  action        TEXT NOT NULL CHECK (action IN ('link','forbid')),
  target_source TEXT NOT NULL REFERENCES source(source),
  target_key    TEXT NOT NULL,
  target_id     INTEGER NOT NULL,
  note          TEXT,
  PRIMARY KEY (entity_type, source, source_key, action, target_id)
) STRICT;

------------------------------------------------------------------------
-- 7. Modeling support (phase 2+): as-of ratings and walk-forward predictions
------------------------------------------------------------------------
-- State of a rating AFTER all matches strictly before as_of_date. Never overwritten.
CREATE TABLE rating_snapshot (
  model_id      TEXT NOT NULL,
  player_id     INTEGER NOT NULL REFERENCES player(player_id),
  surface       TEXT NOT NULL CHECK (surface IN ('all','hard','clay','grass','carpet')),
  as_of_date    TEXT NOT NULL,
  rating        REAL NOT NULL,
  deviation     REAL,
  n_matches     INTEGER,
  PRIMARY KEY (model_id, player_id, surface, as_of_date)
) STRICT;

CREATE TABLE prediction (
  model_id        TEXT NOT NULL,
  match_id        INTEGER NOT NULL REFERENCES matches(match_id),
  predicted_at    TEXT NOT NULL,
  p_lo_wins       REAL NOT NULL CHECK (p_lo_wins BETWEEN 0 AND 1),  -- P(pair_lo_id player wins)
  PRIMARY KEY (model_id, match_id)
) STRICT;

------------------------------------------------------------------------
-- 8. Views
------------------------------------------------------------------------
-- Per-player match history, ordered the way rating updates must consume it.
CREATE VIEW v_player_match AS
SELECT m.match_id, m.winner_id AS player_id, m.loser_id  AS opponent_id, 1 AS won,
       e.start_date AS edition_start, m.round_order, m.match_date, e.surface, m.status
FROM matches m JOIN tournament_edition e USING (edition_id)
UNION ALL
SELECT m.match_id, m.loser_id,  m.winner_id, 0,
       e.start_date, m.round_order, m.match_date, e.surface, m.status
FROM matches m JOIN tournament_edition e USING (edition_id);

-- Invariant: no player appears in two matches on the same exact day. Must be empty.
CREATE VIEW v_inv_same_day_conflicts AS
SELECT a.player_id, a.match_date, a.match_id AS match_a, b.match_id AS match_b
FROM v_player_match a
JOIN v_player_match b
  ON a.player_id = b.player_id AND a.match_date = b.match_date AND a.match_id < b.match_id
WHERE a.match_date IS NOT NULL
  AND a.status NOT IN ('walkover') AND b.status NOT IN ('walkover');

-- Invariant: every registered source row is accounted for. Must be empty.
CREATE VIEW v_inv_load_counts AS
WITH runs AS (
  SELECT s.snapshot_id, s.source, s.file_relpath, s.loader, s.row_count,
         r.load_run_id, r.finished_at, r.rows_read, r.rows_inserted, r.rows_linked,
         r.rows_skipped, r.rows_queued,
         (SELECT COALESCE(SUM(k.n), 0) FROM load_skip k WHERE k.load_run_id = r.load_run_id)
           AS skip_reason_total
  FROM ingest_snapshot s
  LEFT JOIN load_run r ON r.snapshot_id = s.snapshot_id AND r.loader = s.loader
)
SELECT *,
  CASE
    WHEN load_run_id IS NULL THEN 'never_loaded'
    WHEN finished_at IS NULL THEN 'unfinished'
    WHEN rows_read <> row_count THEN 'rows_read_ne_row_count'
    WHEN rows_inserted + rows_linked + rows_skipped + rows_queued <> rows_read
      THEN 'outcomes_ne_rows_read'
    WHEN rows_skipped <> skip_reason_total THEN 'skips_without_reason'
  END AS problem
FROM runs
WHERE problem IS NOT NULL;

-- Invariant: point server and returner are the two players of the match. Must be empty.
CREATE VIEW v_inv_point_players AS
SELECT p.point_id, p.match_id, p.data_source
FROM point p JOIN matches m USING (match_id)
WHERE NOT (p.server_id   IN (m.winner_id, m.loser_id)
       AND p.returner_id IN (m.winner_id, m.loser_id)
       AND p.winner_id   IN (m.winner_id, m.loser_id)
       AND p.server_id <> p.returner_id);

CREATE VIEW v_open_review AS
SELECT * FROM review_queue WHERE status = 'open';
