-- Tennis data backbone: SQLite schema, v1
-- Apply with: sqlite3 tennis.db < schema.sql   (or executescript in Python)
-- Conventions: STRICT tables, ISO-8601 TEXT for dates/times, own surrogate IDs,
-- source IDs only ever live in *_xref tables.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

------------------------------------------------------------------------
-- 0. Meta, sources, raw snapshots
------------------------------------------------------------------------
CREATE TABLE schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;
INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1');

CREATE TABLE source (
  source      TEXT PRIMARY KEY,   -- 'sackmann_atp','sackmann_wta','mcp','sackmann_slam_pbp','tennis_data_uk'
  description TEXT,
  license     TEXT,               -- e.g. MCP: 'CC BY-NC-SA 4.0 (attribution, non-commercial)'
  url         TEXT
) STRICT;

-- One row per raw file ingested. Raw files are immutable on disk.
CREATE TABLE ingest_snapshot (
  snapshot_id INTEGER PRIMARY KEY,
  source      TEXT NOT NULL REFERENCES source(source),
  fetched_at  TEXT NOT NULL,
  source_ref  TEXT,               -- git commit, release tag, or URL
  sha256      TEXT NOT NULL,
  raw_path    TEXT NOT NULL,
  row_count   INTEGER,
  UNIQUE (source, sha256)
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
  level         TEXT,             -- 'G','M','500','250','F','C','I', ...
  surface       TEXT CHECK (surface IS NULL OR surface IN ('hard','clay','grass','carpet')),
  indoor        INTEGER CHECK (indoor IS NULL OR indoor IN (0,1)),
  draw_size     INTEGER,
  best_of       INTEGER CHECK (best_of IS NULL OR best_of IN (3,5)),
  UNIQUE (tournament_id, start_date)
) STRICT;

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
                   ('Q1','Q2','Q3','R128','R64','R32','R16','QF','SF','F','RR','BR','ER')),
  round_order    INTEGER NOT NULL,           -- chronological order within an edition
  match_date     TEXT,                       -- exact day if known, else NULL
  date_precision TEXT NOT NULL CHECK (date_precision IN ('day','week')),
  winner_id      INTEGER NOT NULL REFERENCES player(player_id),
  loser_id       INTEGER NOT NULL REFERENCES player(player_id),
  score          TEXT,
  status         TEXT NOT NULL DEFAULT 'completed'
                   CHECK (status IN ('completed','retired','walkover','default','unfinished')),
  best_of        INTEGER CHECK (best_of IS NULL OR best_of IN (3,5)),
  minutes        INTEGER,
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

CREATE TABLE match_xref (
  source      TEXT NOT NULL REFERENCES source(source),
  source_key  TEXT NOT NULL,      -- e.g. MCP match string, or Sackmann tourney_id + match_num
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
  PRIMARY KEY (match_id, player_id, source)
) STRICT;

-- Weekly rankings. As-of lookup: latest ranking_date strictly before the match.
CREATE TABLE ranking (
  tour         TEXT NOT NULL CHECK (tour IN ('atp','wta')),
  ranking_date TEXT NOT NULL,
  player_id    INTEGER NOT NULL REFERENCES player(player_id),
  rank         INTEGER NOT NULL,
  points       INTEGER,
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
  reason         TEXT NOT NULL,              -- 'ambiguous','low_confidence','no_candidate'
  status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','ignored')),
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  resolved_at    TEXT,
  UNIQUE (entity_type, source, source_key)
) STRICT;

-- Loaded from the version-controlled overrides file; applied last, always wins.
CREATE TABLE override_link (
  entity_type TEXT NOT NULL CHECK (entity_type IN ('player','edition','match')),
  source      TEXT NOT NULL REFERENCES source(source),
  source_key  TEXT NOT NULL,
  action      TEXT NOT NULL CHECK (action IN ('link','forbid')),
  target_id   INTEGER,                       -- required for 'link', the entity to forbid for 'forbid'
  note        TEXT,
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

-- Invariant: source rows that neither resolved nor were queued are the pipeline's job to
-- report; this view lists the open review items.
CREATE VIEW v_open_review AS
SELECT * FROM review_queue WHERE status = 'open';
