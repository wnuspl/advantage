-- models.db: persistent modeling results (Elo/Glicko runs, parameter sweeps, metrics).
-- Applied by src/tennis/models_db.py on every connect. Unlike tennis.db this file is
-- NEVER deleted or rebuilt: every statement is idempotent, and changes are additive only
-- (new tables, new nullable columns), recorded by bumping models_schema_version.
--
-- Stable keys. tennis.db surrogate IDs change on every rebuild, so rows here refer to
-- players and matches by the source key of their backbone (seed) xref:
--   player_key  '<source>:<source_key>' e.g. 'sackmann_atp:104925'
--   match_key   '<source>:<source_key>' e.g. 'sackmann_atp:2024-0339:300:F:105777:208029'
--               (if a match has several seed keys, the lexicographically first)
-- Join back to tennis.db through player_xref / match_xref; see models_db.py.
--
-- Volume policy: a parameter sweep writes model_run + run_metric only. Per-match rows
-- (prediction, pre_match_rating) are written only for runs explicitly saved, since one
-- run is ~1.8M predictions and ~3.5M ratings.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS models_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;
INSERT OR IGNORE INTO models_meta (key, value) VALUES ('models_schema_version', '1');

-- One row per model run (one parameter setting over the full history).
CREATE TABLE IF NOT EXISTS model_run (
  run_id        INTEGER PRIMARY KEY,
  model_id      TEXT NOT NULL,      -- family + version, e.g. 'elo_surface_v1'
  params_json   TEXT NOT NULL,      -- canonical JSON (sorted keys) of every parameter
  sweep_id      TEXT,               -- groups the runs of one parameter sweep
  data_built_at TEXT,               -- tennis.db schema_meta 'built_at' the run read
  data_sources  TEXT,               -- JSON {source: commit} from tennis.db ingest_snapshot
  code_commit   TEXT,               -- git HEAD at run time, '+dirty' if uncommitted changes
  started_at    TEXT NOT NULL,
  finished_at   TEXT,               -- NULL = crashed or still running
  notes         TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS idx_model_run_model ON model_run (model_id, sweep_id);

-- Scores of a run on one evaluation block and match scope.
CREATE TABLE IF NOT EXISTS run_metric (
  run_id           INTEGER NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
  block            TEXT NOT NULL,   -- label, e.g. 'val', 'test', '2023'
  scope            TEXT NOT NULL,   -- which matches were scored, e.g. 'tour_main', 'all'
  date_from        TEXT NOT NULL,   -- edition start range, inclusive
  date_to          TEXT NOT NULL,
  n                INTEGER NOT NULL,
  log_loss         REAL,
  brier            REAL,
  accuracy         REAL,            -- reported, never used to choose
  calibration_json TEXT,            -- [[bin_lo, bin_hi, n, mean_p, win_rate], ...]
  PRIMARY KEY (run_id, block, scope)
) STRICT;

-- Walk-forward prediction for one match, made from state strictly before it.
-- Player order is canonical (p1_key < p2_key), never winner-first.
CREATE TABLE IF NOT EXISTS prediction (
  run_id        INTEGER NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
  match_key     TEXT NOT NULL,
  edition_start TEXT NOT NULL,      -- copied for window filters without joining tennis.db
  p1_key        TEXT NOT NULL,
  p2_key        TEXT NOT NULL,
  p_p1_wins     REAL NOT NULL CHECK (p_p1_wins BETWEEN 0 AND 1),
  p1_won        INTEGER CHECK (p1_won IS NULL OR p1_won IN (0,1)),  -- NULL until known
  PRIMARY KEY (run_id, match_key),
  CHECK (p1_key < p2_key)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prediction_window ON prediction (run_id, edition_start);

-- Each player's rating going INTO a match (the feature later layers read).
CREATE TABLE IF NOT EXISTS pre_match_rating (
  run_id         INTEGER NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
  match_key      TEXT NOT NULL,
  player_key     TEXT NOT NULL,
  rating         REAL NOT NULL,     -- overall
  surface_rating REAL,              -- NULL when the edition has no surface
  deviation      REAL,              -- Glicko-style models only
  n_matches      INTEGER,           -- rated matches before this one
  PRIMARY KEY (run_id, match_key, player_key)
) STRICT;
