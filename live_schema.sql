-- live.db: what the Kalshi watch saw, poll by poll (track B2, docs/PLAN.md).
-- Append-only and never rebuilt: additive changes only (CREATE ... IF NOT EXISTS).
-- Stable keys are Kalshi's (event ticker, market tickers, competitor ids) and player names;
-- player_id is only valid for the tennis.db build recorded on the watch row.

PRAGMA foreign_keys = ON;

-- Raw JSON, zlib-compressed, content-addressed: a part that did not change between polls
-- (events, milestones) is stored once.
CREATE TABLE IF NOT EXISTS blob (
  sha256 TEXT PRIMARY KEY,                 -- of the uncompressed UTF-8 JSON
  bytes  INTEGER NOT NULL,                 -- uncompressed size
  zlib   BLOB NOT NULL
) STRICT;

-- One run of scripts/watch.py.
CREATE TABLE IF NOT EXISTS watch (
  watch_id    INTEGER PRIMARY KEY,
  started_at  TEXT NOT NULL,
  predictor   TEXT NOT NULL,               -- Predictor.name, e.g. points_v1/run790
  policy      TEXT NOT NULL,               -- JSON of engine.decide.Policy
  series      TEXT NOT NULL,               -- comma-separated Kalshi series
  every_s     REAL NOT NULL,
  data_db     TEXT,                        -- tennis.db fingerprint (player_id validity)
  code_rev    TEXT                         -- git HEAD at start
) STRICT;

CREATE TABLE IF NOT EXISTS poll (
  poll_id         INTEGER PRIMARY KEY,
  watch_id        INTEGER NOT NULL REFERENCES watch(watch_id),
  as_of           TEXT NOT NULL,           -- when the pull started (UTC ISO)
  done_at         TEXT NOT NULL,           -- when it was recorded
  events_sha      TEXT REFERENCES blob(sha256),
  milestones_sha  TEXT REFERENCES blob(sha256),
  live_sha        TEXT REFERENCES blob(sha256),
  books_sha       TEXT REFERENCES blob(sha256),
  n_events        INTEGER NOT NULL DEFAULT 0,
  error           TEXT                     -- set when the poll failed (no parts then)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_poll_as_of ON poll (as_of);

-- Every match event ever seen: the stable identity used for settlement.
CREATE TABLE IF NOT EXISTS event (
  event       TEXT PRIMARY KEY,            -- Kalshi event ticker
  series      TEXT NOT NULL,
  tour        TEXT NOT NULL,
  title       TEXT NOT NULL,
  tournament  TEXT,
  start       TEXT,                        -- scheduled start (UTC ISO)
  a_ticker    TEXT NOT NULL,               -- side A = lower ticker (engine.live.parse_events)
  b_ticker    TEXT NOT NULL,
  a_name      TEXT NOT NULL,
  b_name      TEXT NOT NULL,
  a_comp      TEXT NOT NULL,               -- competitor ids (live data maps by these)
  b_comp      TEXT NOT NULL,
  milestone   TEXT,
  first_seen  TEXT NOT NULL,
  last_seen   TEXT NOT NULL                -- last poll that listed it as open
) STRICT;

-- One row per (poll, open match): price, score, prediction and the tags B3b splits by.
CREATE TABLE IF NOT EXISTS snapshot (
  poll_id       INTEGER NOT NULL REFERENCES poll(poll_id),
  event         TEXT NOT NULL REFERENCES event(event),
  status        TEXT NOT NULL,             -- not_started / live / closed / no_feed
  skip          TEXT NOT NULL,             -- why not priced ('' = priced)
  p_market      REAL,                      -- P(A), de-overrounded mid
  ask_a         REAL,                      -- cheapest way to buy A (YES A or NO B), before fee
  ask_b         REAL,
  score         TEXT,                      -- live score as shown, e.g. '6-2 4-2 0-0 *B serving'
  sets_a INTEGER, sets_b INTEGER, games_a INTEGER, games_b INTEGER, pts_a INTEGER, pts_b INTEGER,
  a_serving     INTEGER,
  serve_a_won INTEGER, serve_a_played INTEGER, serve_b_won INTEGER, serve_b_played INTEGER,
  live_problem  TEXT,
  p_model       REAL,                      -- P(A) from the predictor
  p_raw         REAL,
  in_play       INTEGER,                   -- the prediction used the live score
  stale         INTEGER,
  -- tags
  a_pid INTEGER, b_pid INTEGER,
  a_resolved    TEXT,                      -- exact / last+initial / ... (engine.players)
  b_resolved    TEXT,
  a_last_rated  TEXT,                      -- predictor detail last_a / last_b, when given
  b_last_rated  TEXT,
  a_stale_days  INTEGER,                   -- match day - last rated match
  b_stale_days  INTEGER,
  day           TEXT,
  surface       TEXT,
  surface_from  TEXT,                      -- edition / history: ... / fallback / unknown
  best_of       INTEGER,
  best_of_from  TEXT,
  final         TEXT,
  edition_id    INTEGER,
  p_sd          REAL,                      -- SD of p_model under the ratings' variances (added 2026-11-16)
  PRIMARY KEY (poll_id, event)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_snapshot_event ON snapshot (event, poll_id);

-- The B3a decision per (poll, match).
CREATE TABLE IF NOT EXISTS decision (
  poll_id    INTEGER NOT NULL,
  event      TEXT NOT NULL,
  side       TEXT CHECK (side IS NULL OR side IN ('a','b')),
  p_use      REAL,
  p_side     REAL,
  ask        REAL,
  contracts  INTEGER NOT NULL,
  cost       REAL NOT NULL,
  fees       REAL NOT NULL,
  ev         REAL NOT NULL,
  fills      TEXT NOT NULL,                -- JSON [{price, contracts, route}]
  reason     TEXT NOT NULL,                -- why no bet ('' = bet)
  PRIMARY KEY (poll_id, event),
  FOREIGN KEY (poll_id, event) REFERENCES snapshot(poll_id, event)
) STRICT;

-- How each match ended. One row per event once Kalshi has determined its markets.
CREATE TABLE IF NOT EXISTS settlement (
  event          TEXT PRIMARY KEY REFERENCES event(event),
  seen_at        TEXT NOT NULL,            -- when the watch found it settled
  winner         TEXT CHECK (winner IS NULL OR winner IN ('a','b')),   -- NULL: neither market resolved yes
  result_a       TEXT,                     -- Kalshi market result: yes / no / '' / other
  result_b       TEXT,
  value_a        REAL,                     -- settlement value, dollars
  value_b        REAL,
  market_status  TEXT,                     -- determined / finalized / ...
  match_status   TEXT,                     -- Kalshi live data at the end: ended / retired / walkover / ...
  final_score    TEXT,                     -- sets, A first, e.g. '7-6 6-4'
  raw_sha        TEXT REFERENCES blob(sha256)   -- the event + markets + last live details JSON
) STRICT;

-- Every decision change the watch detected (track B4, engine/alerts.py), pushed or not.
CREATE TABLE IF NOT EXISTS alert (
  alert_id   INTEGER PRIMARY KEY,
  poll_id    INTEGER NOT NULL REFERENCES poll(poll_id),
  event      TEXT NOT NULL REFERENCES event(event),
  kind       TEXT NOT NULL CHECK (kind IN ('new','side','grew','gone')),
  title      TEXT NOT NULL,
  message    TEXT NOT NULL,
  priority   INTEGER NOT NULL,
  status     TEXT NOT NULL CHECK (status IN ('sent','suppressed','failed','printed')),
  detail     TEXT NOT NULL DEFAULT ''          -- why suppressed / failed
) STRICT;
CREATE INDEX IF NOT EXISTS idx_alert_event ON alert (event, poll_id);

-- Each match's pre-match model block (added 2026-11-17): the predictor's pre-match detail
-- (serve / return ratings going into the match, serve-point logits, context offsets,
-- pre-match p and SD) from the last poll before the match went live, or from the first
-- poll that saw it if it was already live then. Written by the recorder, read by the
-- live page's match panel.
CREATE TABLE IF NOT EXISTS prematch (
  event    TEXT PRIMARY KEY REFERENCES event(event),
  poll_id  INTEGER NOT NULL REFERENCES poll(poll_id),
  model    TEXT NOT NULL,                  -- Prediction.model
  status   TEXT NOT NULL,                  -- the snapshot status it was taken at
  detail   TEXT NOT NULL                   -- JSON: recorder.PREMATCH_KEYS of Prediction.detail
) STRICT;
