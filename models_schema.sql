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

------------------------------------------------------------------------
-- Heavy layer: PCA style vectors (schema version 2). Code: src/tennis/heavy.py.
-- One fit per (tour, side, fit_date); it sees only charted matches whose edition started
-- strictly before fit_date. Between refits, vectors come from the latest prior fit.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS style_fit (
  fit_id          TEXT PRIMARY KEY,   -- '<model_id>:<config hash>:<tour>:<side>:<fit_date>'
  model_id        TEXT NOT NULL,      -- e.g. 'style_pca_v1'
  config_hash     TEXT NOT NULL,      -- sha of config_json; groups fits of one configuration
  tour            TEXT NOT NULL CHECK (tour IN ('atp','wta')),
  side            TEXT NOT NULL CHECK (side IN ('serve','return','rally')),
  fit_date        TEXT NOT NULL,      -- uses charted matches with edition start < fit_date
  data_from       TEXT,               -- earliest / latest edition start actually used
  data_to         TEXT,
  n_players       INTEGER NOT NULL,   -- players in the fit population
  n_components    INTEGER NOT NULL,
  explained_json  TEXT NOT NULL,      -- explained-variance ratio of every kept component
  config_json     TEXT NOT NULL,      -- canonical JSON of the HeavyConfig
  feature_version TEXT NOT NULL,
  pipeline_path   TEXT,               -- joblib file, relative to the repo root
  data_built_at   TEXT,
  code_commit     TEXT,
  created_at      TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_style_fit_lookup ON style_fit (config_hash, tour, side, fit_date);

-- Per fit and feature: the shrinkage prior and the scaler, so a vector can be rebuilt
-- without the joblib file.
CREATE TABLE IF NOT EXISTS style_fit_feature (
  fit_id       TEXT NOT NULL REFERENCES style_fit(fit_id) ON DELETE CASCADE,
  feature      TEXT NOT NULL,
  position     INTEGER NOT NULL,
  tour_mean    REAL NOT NULL,
  prior_k      REAL NOT NULL,
  scaler_mean  REAL NOT NULL,
  scaler_scale REAL NOT NULL,
  PRIMARY KEY (fit_id, feature)
) STRICT;

CREATE TABLE IF NOT EXISTS style_loading (
  fit_id    TEXT NOT NULL REFERENCES style_fit(fit_id) ON DELETE CASCADE,
  feature   TEXT NOT NULL,
  component INTEGER NOT NULL,         -- 1-based
  loading   REAL NOT NULL,            -- after the sign convention (largest |loading| > 0)
  PRIMARY KEY (fit_id, feature, component)
) STRICT;

-- Materialized vectors (a cache: heavy.get_style can compute any date on demand).
-- player_id is the stable player key. as_of_date: uses only editions starting before it.
CREATE TABLE IF NOT EXISTS style_vectors (
  player_id   TEXT NOT NULL,
  as_of_date  TEXT NOT NULL,
  side        TEXT NOT NULL CHECK (side IN ('serve','return','rally')),
  fit_id      TEXT NOT NULL REFERENCES style_fit(fit_id) ON DELETE CASCADE,
  n_matches   INTEGER NOT NULL,
  n_points    INTEGER NOT NULL,
  c1 REAL, c2 REAL, c3 REAL, c4 REAL, c5 REAL, c6 REAL, c7 REAL, c8 REAL, c9 REAL, c10 REAL,
  PRIMARY KEY (player_id, as_of_date, side)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_style_vectors_fit ON style_vectors (fit_id);

UPDATE models_meta SET value = '2' WHERE key = 'models_schema_version' AND CAST(value AS INTEGER) < 2;
