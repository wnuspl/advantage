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
| AcesAPI (`acesapi`) | `matches`, `tournament_edition`, new `player` rows after the Sackmann cutoff; `match_seen` | Tour-level singles (ATP 250+, WTA incl. 125), qualifying, from 2007; refreshed daily. No serve stats, no Challenger/ITF. Licence: AcesAPI terms (research use and storage allowed; no redistribution as a dataset). See "AcesAPI" below. |
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
| 4b AcesAPI | Latest pull of each tournament: link to the backbone where it overlaps, extend it past the frontier; then `match_seen`. Runs before MCP so later charts resolve. | (as 3) |
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

## AcesAPI (results after the Sackmann cutoff)

The Sackmann match files stop at 2026-05-25 (tour level: Roland Garros 2026). AcesAPI supplies tour-level results from then on. Code: `src/tennis/acesapi.py` (parsing), `acesapi_fetch.py` (client, pull planning), `load_acesapi.py` (loader), `scripts/fetch_acesapi.py` (CLI). Response shapes were checked against live responses on 2026-11-14 (Roland Garros 2026, then the full first pull: 6,448 rows, every round label and outcome mapped).

**Pulls are snapshots.** `python scripts/fetch_acesapi.py --pin` writes one immutable folder `data/raw/acesapi-<YYYYMMDDTHHMMSSZ>/` (the pull id, UTC):
- `draws/<tour>/<tournament_id>.json`: one `get_tournament draw=all` response per tournament, saved byte for byte (2 credits each).
- `tournaments/*.json`: the `list_tournaments` pages used to plan the pull (1 credit each). Ignored by the build.
- `pull.json`: every request, its HTTP status, response sha256 and credits left. Ignored by the build.

`--pin` appends the folder to `data/build_manifest.json` (commit = pull id, `fetched_at` = pull time). From there it is an ordinary snapshot: files are hashed into `data/snapshot_lock.json`, and a JSON file's `row_count` is its number of match rows. The key comes from `ACESAPI_KEY` in the environment or in `.env`.

**What a pull fetches.** Tournaments that have started and overlap `[--since, today]`, except those an earlier pull already saw final more than `--refetch-days` (10) after they ended. AcesAPI rewrites the current season every day at 10:00 UTC, so recent tournaments are fetched again to pick up corrections. `--max-credits` caps a pull; whatever is left over is picked up by the next one. `--plan` spends only the list calls and prints what would be fetched. Steady state is about 20-25 tournaments a month, or about 50 credits plus lists; the free plan gives 500.

**Timestamps and supersession.**
- A tournament appears in many pulls. The build loads the **latest pull** of each tournament (pull ids sort as UTC timestamps). Its rows in earlier pulls are read and skipped as `superseded_by_later_pull`, so `v_inv_load_counts` still accounts for every row.
- `match_seen(first_seen_at, last_seen_at, n_pulls, revised)` keeps the knowledge time that supersession would otherwise lose. `first_seen_at` is the first pull that reported the match as final: the earliest moment this project could have known the result. `revised = 1` if winner, score or outcome changed between pulls. Use it for any "as known at time T" replay (live prediction, the odds benchmark). Backtests on results alone can ignore it.
- `matches.match_date` is the API's local match day, so these rows are `date_precision = 'day'` and are covered by `v_inv_same_day_conflicts`. That view allows two rounds of one event on one day: the first pull has 85 such rain/weekend doubleheaders, and none across events.
- Edition `start_date` is the first main-draw match day, the same convention as Sackmann's `tourney_date` (which is not always a Monday). AcesAPI's own `start_date` includes qualifying, so it is only a fallback.

**Overlap and the frontier.** The frontier per tour is the last backbone edition start minus 7 days (2026-05-25 for both tours on the current mirror). Every row is first matched to the backbone: unordered pair, edition start +-3 days, with round and tournament name as tiebreaks (`link_match`). Then:
- Candidates are backbone matches only: never a row AcesAPI itself inserted (its match ids are unique), never across the qualifying/main-draw line (a lucky loser can meet the same opponent in both). A candidate with the other winner is queued as `result_conflict`, never linked.
- Linked rows (`match_xref` method `pair_tournament_date`) are the validation sample against Sackmann.
- An unlinked row in an edition the backbone already has goes into that edition only if the backbone has **no** match in that round (e.g. slam qualifying, which Sackmann lacks). A partial round is never filled, so a row that merely failed to link can't duplicate a backbone match.
- Otherwise, before the frontier the row is queued (`no_backbone_match`). Past the frontier it is inserted as a new match (`match_xref` method `seed`, key `atp:<match_id>`) in a new edition (`edition_xref` key `atp:<tournament_id>`).
- Tournament identity: an existing tournament with an edition about a year earlier (+-21 days) whose alias equals the AcesAPI name, else the city, else shares a distinctive token (4+ letters, not "open"/"masters"/...: "Rolex Shanghai Masters" ~ "Shanghai Masters"). Exactly one must qualify at a step. The new edition inherits that tournament's Sackmann `level`. Otherwise it gets a new tournament with `level` `G` for majors, else `X` (tier unknown). `X` matches are rated but not in the "tour level" scoring scope (`elo.TOUR_LEVELS`), since many are WTA 125s. The AcesAPI name is added as an alias, so next year's edition finds it.

**Players.** Source key `atp:<id>` / `wta:<id>`, resolved with the same cascade as MCP: override, exact name, event context, opponent anchoring. An unresolved name becomes a **new player** only past the frontier, and only when the resolver found no candidate at all, or every fuzzy candidate has a different country. Its `review_queue` row is kept as `resolved`. Otherwise the match is queued as `player_unresolved`.

**Not covered.** Serve stats (the light layer uses its win/loss update for these matches), Challenger and ITF events, rankings (not fetched yet). Retirements and walkovers arrive as `outcome` and map to `matches.status`.

## models.db (modeling results)

`models.db` (schema `models_schema.sql`, code `src/tennis/models_db.py`) holds everything a model run produces. It is **never rebuilt**: the schema is applied idempotently on every connect and changes only additively. `build_db.py` does not touch it, so sweeps and chosen parameters survive any number of data rebuilds.

- **Stable keys, not surrogate IDs.** Rows refer to players and matches by the source key of their backbone (seed) xref, e.g. `sackmann_atp:104925` and `sackmann_atp:2024-0339:300:F:105777:208029`. `models_db.match_keys()` / `player_keys()` map the current `tennis.db` IDs to them. Analysis code attaches `tennis.db` read-only as `data`.
- **Tables:** `model_run` (model id, canonical parameter JSON, sweep id, the data build and source commits it read, git commit), `run_metric` (log loss / Brier / accuracy / calibration per evaluation block and match scope), `prediction` (canonical player order `p1_key < p2_key`, never winner-first), `pre_match_rating` (each player's rating going into each match: the feature later layers read).
- **Volume policy:** sweeps write only `model_run` + `run_metric`; per-match rows are written only for runs explicitly saved (one run is ~1.8M predictions and ~3.5M ratings). Deleting a run cascades to its rows.

## Elo baseline (`elo_v1`)

Code: `src/tennis/elo.py` (engine), `src/tennis/evaluate.py` (scoring), `src/tennis/tuning.py` + `scripts/elo_sweep.py` (sweeps). Elo with an overall rating plus one rating per surface (`blend = 0` is plain overall Elo). ATP and WTA are naturally separate because their players never meet.

- **Model:** P(a beats b) = 1 / (1 + 10^((R_b - R_a)/400)) with R = (1 - blend) * overall + blend * surface rating. After a match each rating moves by K(n) * weight * (result - its own expected score), with K(n) = k / (n + offset)^shape (shape = 0 gives constant K). For the overall rating n counts all rated matches; for a surface rating, only matches on that surface. New players start at `init`; a player's first rating on a surface starts at `init` (`surface_init = 0`) or at their current overall rating (`surface_init = 1`). Editions with no recorded surface update only the overall rating and are predicted from it.
- **blend is prediction-only:** ratings update identically whatever its value, so a sweep runs each update setting once and scores every blend from the stored pre-match ratings. It is still chosen on the validation block like every other parameter, never on test.
- **Order:** edition start date, then round order, then match_id. Sackmann has no match day, so matches in the same week are ordered by round only. Each prediction is made from the state strictly before the match (tested by truncation).
- **What updates ratings:** every level (ITF, challengers, qualifying, team events). Walkovers and unfinished matches never update ratings. Retirements update with `retired_weight` (swept: 0, 0.5, 1).
- **What is scored:** completed and retired matches only. Scopes: `tour` (ATP+WTA tour-level main draws: ATP levels G/M/A/F/O, WTA G/PM/P/I/W/T1-T5/F/O, rounds not Q*), `tour_atp`, `tour_wta`, `all`. Log loss is the selection metric; Brier, accuracy and a 10-bin calibration table are recorded. Calibration orients each match by stable player key, never winner-first.
- **Blocks:** every setting runs over the full history (warm ratings). `sweep` scores only the validation block (default 2019-2021); `finalize RUN_ID` scores the test block (default 2022-2026) for runs already chosen. Never choose by test score.
- **First sweep (`default-v1`, 2026-11-13, 40 settings, ~3 s per full-history pass):** best on validation was k=150, offset=5, shape=0.3, retired_weight=1 (run 12). Scores:

  | Block | Scope | n | Log loss | Brier | Accuracy |
  |---|---|---|---|---|---|
  | val 2019-21 | tour | 11,307 | 0.6103 | 0.2119 | 0.655 |
  | test 2022-26 | tour | 21,362 | 0.6110 | 0.2121 | 0.658 |
  | test 2022-26 | tour_atp | 11,090 | 0.6159 | 0.2140 | 0.653 |
  | test 2022-26 | tour_wta | 10,272 | 0.6058 | 0.2099 | 0.663 |

  Counting retirements fully beat ignoring them in every K setting (by ~0.001 log loss). Calibration (bins folded so order cannot matter): about 3 points overconfident throughout; e.g. predicted 0.65 -> won 0.62, predicted 0.84 -> won 0.81. Judged sufficient for a baseline; no post-hoc calibration applied.
- **Surface sweep (`surface-v1`, 90 settings in 18 passes, then `blend-fine`):** best on validation was k=150, offset=5, shape=0.25, retired_weight=1, surface_init=0, blend=0.35 (run 132). blend 0.25-0.5 all within 0.0004 of each other; starting surface ratings from the overall rating was slightly worse (~0.0003). blend = 0 reproduced run 12 exactly. Scores vs plain Elo (run 12):

  | Block | Scope | Plain Elo (12) | Surface Elo (132) | Gain |
  |---|---|---|---|---|
  | val 2019-21 | tour | 0.6103 | 0.6040 | 0.0063 |
  | test 2022-26 | tour | 0.6110 | **0.6076** | 0.0035 |
  | test 2022-26 | tour_atp | 0.6159 | 0.6116 | 0.0043 |
  | test 2022-26 | tour_wta | 0.6058 | 0.6032 | 0.0026 |

  Test accuracy 66.2% (tour), Brier 0.2106. **Run 132 is the current baseline** the heavy and light layers must beat; calibration unchanged (~3 points overconfident).
- **Standings:** `scripts/elo_ratings.py [--as-of DATE] [--run ID] [--tour] [--surface]` replays a run's parameters over editions starting strictly before the date and lists ratings after each player's last match, highest first (per tour; by default 10+ rated matches and active in the prior year).
- **Recording:** each setting is a `model_run` row in `models.db` (params, sweep id, data build, git commit) with `run_metric` rows per block x scope. `finalize --save-predictions` writes per-match `prediction` and `pre_match_rating` rows for that run.

## Glicko-2 (`glicko2_v1`)

Code: `src/tennis/glicko2.py`; registered with Elo in `src/tennis/ratingmodels.py`, so `scripts/sweep.py` (sweep/show/finalize) and `scripts/ratings.py` (standings) work for both, reading each run's model from `model_run.model_id`. `elo_sweep.py` / `elo_ratings.py` remain as Elo-default wrappers.

- **Model:** overall + per-surface states (rating, RD, volatility), same structure, retirement/walkover rules and prediction-only `blend` as Elo. Each match is its own rating period; RD grows with time since the player's last rated match (`period_days`, capped at `rd_max`), and surface RD with time since their last match on that surface. Predictions use both players' RD: P = 1/(1+exp(-g(sqrt(phi_a^2+phi_b^2))(mu_a-mu_b))). Volatility update is the Illinois algorithm (checked against Glickman's worked example); `tau = 0` keeps it fixed.
- **Sweeps (`glicko-v1`, then `glicko-edge`, 2026-11-14):** tau had no measurable effect (0 and 0.5 identical: with one match per period volatility barely moves), so tau = 0 (about 40% faster) is used. Shorter periods help up to an interior optimum at 4 days (2 and 7 both worse); starting RD 150-250 is flat. Best on validation: tau=0, rd_init=200, period_days=4, blend=0.3 (run 187).

  | Block | Scope | Surface Elo (132) | Glicko-2 (187) | Gain |
  |---|---|---|---|---|
  | val 2019-21 | tour | 0.6040 | 0.6027 | 0.0013 |
  | test 2022-26 | tour | 0.6076 | **0.6051** | 0.0024 |
  | test 2022-26 | tour_atp | 0.6116 | 0.6086 | 0.0030 |
  | test 2022-26 | tour_wta | 0.6032 | 0.6014 | 0.0018 |
  | test 2022-26 | all levels | 0.5751 | 0.5734 | 0.0017 |

  Brier 0.2097 vs 0.2106; accuracy unchanged (66.1% vs 66.2%). Calibration improves: folded bins at predicted 0.65 won 0.624 (Elo 0.617), at 0.85 won 0.831 (Elo 0.814), roughly halving Elo's overconfidence. **Run 187 is now the best baseline.** Without surfaces Glicko also beats plain Elo (val 0.6086 vs 0.6103).

## Heavy layer: style vectors (`style_pca_v1`)

Spec: `HEAVY_LAYER_SPEC.md`. Code: `src/tennis/mcp_notation.py` (parser), `src/tennis/style_features.py` (counts, point-in-time totals, shrinkage), `src/tennis/heavy.py` (fits, storage, `get_style_vector`), `scripts/style_fit.py` (`fit`, `report`, `show`). Dependencies: numpy + scikit-learn (`requirements.txt`).

- **Parser.** Follows the Instructions tab of `MatchChart 0.3.2.xlsm` in the MCP repo. Conventions found in the data but not in the key are accepted only where no other reading exists: depth digit on rally shots (`f39`, ~233k points), position marks after the direction, `!` as a shank mark, `f@` without an error letter, a bare fault letter. Everything else is skipped and counted: 27,985 of 1,838,989 points (1.5%), mostly S/R/P/Q whole-point codes (16,140) and malformed rally endings (10,202). The notation's implied point winner must agree with `PtWinner`, or the point is skipped.
- **Features.** 36 rates, each kept as numerator/denominator per (charted match, player): 16 serve, 10 return, 10 rally (rally = the player's own shots after serve and return, plus net play). The rally set is a **third vector**, not duplicated into serve/return (decided 2026-11-13). Rally length counts strokes including the serve and the error shot. Serve direction 4/5/6 is handedness-free; shot direction 1/2/3 is unused in v1 because it depends on handedness. Surface is kept on every count row but v1 sums across surfaces. Retired matches' points are included.
- **Timing.** A charted match is dated by its backbone edition's start date (the clock every layer uses). "As of D" = editions starting strictly before D, so charts from the event being predicted, including its qualifying, are never used.
- **Shrinkage.** Per feature, at fit time: prior mean = average raw rate of players with den >= 30 (each player counts once, so heavily charted stars don't drag it); prior strength k = beta-binomial method of moments, clipped to [1, 1e4]. Typical k is 10-250 events (`serve_volley` ~1: true between-player spread is huge across eras). Shrinkage priors are frozen per fit.
- **Fits.** Separate per tour (never pooled) and per side. StandardScaler then PCA (full SVD); fewest components reaching 80% explained variance, capped at 10; sign fixed so each component's largest-magnitude loading is positive. Fit population: players with >= 5 charted matches; no fit until 50 qualify (ATP from 1997-10-01, WTA from 2014-07-01). Quarterly refits; between refits, the latest fit with fit_date <= D transforms. 489 fits on the 2026-11-13 build; all fits take ~4 s, parsing the points ~90 s (cached per tennis.db build in `models/style/`, gitignored, along with the joblib pipelines).
- **Storage (models.db schema v2).** `style_fit`, `style_fit_feature` (priors and scaler per feature), `style_loading`, `style_vectors` (spec table; `player_id` holds the **stable player key**, not a surrogate ID, per the models.db rule). Vectors are materialized for each (charted player, edition start) of a backbone match: 527,574 rows. `heavy.get_style()` returns vector + n_matches + n_points + fit_id, reading the table or computing any other date on demand; `get_style_vector()` returns the vector only, and `None` when the player has no charted match before the date or no fit exists yet.
- **Reading the axes** (latest fits, 2026-04-01; see `docs/style_report.md`, regenerate with `scripts/style_fit.py report`). Guesses for human review:
  - ATP serve: c1 big serve / short points (aces, unreturned, short rallies; Isner, Opelka, Kyrgios high); c2 deuce-court T vs wide placement; c3 ad-court T with serve+1 forehand, against serve-and-volley; c5 safe first serve vs risky second.
  - ATP return: c1 return-in-play and long return points (Djokovic, Medvedev high; Isner low); c2 short/shallow returns vs deep; c3 slice/chip returns and chip-and-charge; c4 forehand return share.
  - ATP rally: c1 attacking all-court (winners, net, slice); c2 error-prone vs defensive lob; c3 drop shots and net success.
  - WTA serve: c1 big serve (Serena Williams, Osaka, Rybakina); c2/c3 placement patterns by court. WTA return: c1 short return points (fewer returns in); c2 slice/chip-and-charge. WTA rally: c1 high-risk (winners and unforced errors); c2 all-court (net, slice, drop).
- **Open (flag for review).** Whole-career aggregation (no recency window or weighting in v1); k by empirical Bayes vs tuned; quarterly vs monthly refits (`--refit-months`, a separate config hash); `style_vectors` keeps the spec's primary key, so it holds one configuration at a time per (player, date, side) and a refit with another config overwrites rows; `len_*` features appear in both serve and return sets (they describe different points). Whether any of this helps is judged by the middle layer, not here.

## Matchup layer (middle layer: `matchup_*_v1`)

Spec: `MIDDLE_LAYER_SPEC.md` ("middle layer" there; called the **matchup layer** from now on). Code: `src/tennis/matchup/` (`data.py` inputs, `models.py` models, `harness.py` walk-forward + scoring + persistence), `scripts/matchup.py run`. Dependencies: torch (CPU) and lightgbm (`requirements.txt`). Results: `docs/matchup_report.md`.

    logit P(p1 wins) = alpha * glicko_logit + delta(s_p1, s_p2, surface)

- **Inputs.** One row per scored match (completed or retired), p1 = smaller stable key. `glicko_logit` = logit of Glicko-2 run 187's pre-match probability (both RDs, surface blend 0.3), recomputed from the run's parameters. Never modified. Style vectors come from the heavy layer's fit objects, not its tables.
- **One fit per side per fold.** `heavy.get_style` uses the latest fit at each date, which would mix PCA axes inside a fold. `matchup.data.StyleSource.vectors(tour, players, dates, fit_ids)` (the spec's `get_style_vectors(requests, fit_ids)`) projects each player's charted totals before the match date through the fold's chosen fits. Candidate to move into `heavy.py`.
- **Folds** (per tour): test block [S, S+3 months), validation [S-6 months, S) (early stopping only), train before cutoff C = S-6 months. Fits are the latest per side with fit_date <= C. Alpha = scalar logistic regression of y on glicko_logit (no intercept) over the fold's both-vector training rows of the last 10 years, then frozen. Training rows are every scored match (any level) where both players have vectors; the match itself need not be charted. Test rows are every scored match in the block. Rows without both vectors get exactly sigmoid(alpha * glicko_logit).
- **Models**, all antisymmetric by construction (unit-tested on random weights):
  - `glicko`: delta = 0 (the benchmark); `glicko_raw` = run 187 unchanged (alpha = 1).
  - `bilinear`: a_s W_sr b_r - b_s W_sr a_r + a_l W_rr b_l, with W_sr = U V^T and W_rr = P Q^T - Q P^T (rank 3); V, Q start at zero. Ablations `bilinear_sr`, `bilinear_rr`. Full-batch Adam, L2 on the W matrices, early stopping (patience 40).
  - `siamese`: shared encoder g (2 x 32, dropout 0.2); d = h(e_a, e_b, surf) - h(e_b, e_a, surf). **Deviation from the spec:** a linear h (as written in §5.2) reduces to (w1 - w2)·(e_a - e_b), a per-player score difference with no interaction, and the surface term cancels exactly. So h is a 16-unit MLP; `head_hidden=0` restores the spec's form. h's last layer starts at zero, so the untrained model is exactly Glicko-only (tested).
  - `lgbm`: glicko_logit, six vectors, paired products (serve x return over the first min(K_serve, K_return) components), surface; offset as init_score; trained on both orderings; prediction averages p(a,b) and 1 - p(b,a).
- **Persistence (models.db schema v3).** One `model_run` per model (`matchup_<name>_v1`, sweep_id = run tag); `run_metric` block `wf_test` = pooled test metrics per scope, `test:<block>` = per block; `prediction` rows for every test match; `matchup_bundle` = per fold: fit_ids, alpha, train cutoff, sizes, training info and the model file (`models/matchup/<run_id>/<tour>_<block>.pt`, gitignored).
- **Scopes.** `tour` (tour-level main draws, every match), `tour_both` (both players have vectors: where the layer can act), `tour_well` (both >= 10 charted matches), plus per tour and per surface.
- **First results (run tag `matchup-v1-full`, runs 202-208, test 2022-01 to 2026-06, 36 folds, ~5 min).** `tour` log loss: raw Glicko 0.6051, alpha-calibrated Glicko 0.6045, bilinear 0.6049, bilinear_sr 0.6047, bilinear_rr 0.6047, Siamese 0.6042, LightGBM 0.6045. 93% of tour-level matches have both vectors. Alpha calibration is the only significant gain (t = 3 across 18 blocks). Siamese is best but within noise (t = -1.2), and bilinear is slightly worse. **No v1 interaction model beats Glicko-only by the spec §8 rule yet.** Details and ablation: `docs/matchup_report.md`.
- **Siamese on top of points_v1 (2026-11-14 check, stored run 207 vs 203):** the siamese deviation from calibrated Glicko is nearly uncorrelated with what points_v1 adds over Glicko (r = 0.07). Stacked on points_v1 run 273 (calibrated), with its weight cross-fitted across 2022-23 / 2024-26, it gains 0.0002 ± 0.00013 (coef t = 2.9 in the full sample). That is a weak signal chosen on test data. To adopt it, retrain the siamese model on a points_v1 base and judge it on validation.
- **Open (flag for review).** Bilinear rank 3, no per-surface W; `n_matches` not an input; well-charted N = 10; blocks 3 months, validation 6 months; alpha fit on both-vector training rows (`--alpha-rows scope` = all rows); Siamese head is an MLP, not linear; Phase 2 (point-level target + Markov roll-up) not started; per-fold style projection loops in Python and could be vectorized.

## Composite model (end-to-end predictions and accuracy)

Code: `src/tennis/composite.py` (predictor), `src/tennis/composite_eval.py` (scoring), `scripts/composite.py` (`predict`, `eval`, `check`). Tests: `tests/test_composite.py`. Report: `docs/composite_report.md`.

- **Chain.** P(A wins) = sigmoid(alpha * logit(Glicko-2 P) + delta). The Glicko-2 state is replayed from the matchup run's light-layer run (187) over editions starting before the date; vectors go through the bundle's fit_ids; alpha and the model come from `matchup_bundle` + the bundle file. Nothing is retrained. Bundle = the one whose test block contains the date; after the last block, the latest (a production retrain on all data is still to do).
- **`check`** rebuilds stored walk-forward predictions through this chain for each player's first match of chosen edition starts (later rounds of an event under way are not in a before-date state). On matchup-v1-full: 1,095 matches, max difference 7e-9 (siamese), 2e-16 (lgbm), light layer 2e-16 vs run 187.
- **`eval`** inner-joins stored predictions of the baselines (Elo 132, Glicko-2 187) and every model of one matchup sweep, so all models are scored on identical matches. Metrics: log loss, Brier, accuracy, ECE, recalibration (a, b), paired log-loss difference vs a reference with SE, quarterly blocks won. "Better" = lower log loss with z < -2 and ECE no more than 0.002 worse.

## Phase style correction (milestone 10, `phase_test.py`): negative

Code: `src/tennis/phase.py` (profiles), `scripts/phase_test.py` (all three steps), results in
`results/phase_test.json` (run 2026-11-14 on points_v1 run 273 and Glicko-2 run 187).

- **Phases** per charted point (server's view): short (<= 3 strokes, double faults included),
  medium (4-6), long (>= 7), server came to net, returner came to net. Same parser as the style
  features.
- **Profile** per player and side, from charted matches with edition start before the match:
  level d (win rate vs population), edge e_k (phase win rate beyond population phase rate + own
  level), steering t_k (phase share vs population). Shrinkage 300 / 150 / 150 points. Players need
  >= 300 charted serve and return points, else no correction. Population rates use years before
  the match's year.
- **Variants:** `pooled` (population per tour; the first test), `surface` (population per tour x
  surface, own level measured against it), `same` (only points on the match's surface).
- **Matchup term**, A serving to B: M_A = sum_k t^ret_B,k e^srv_A,k - sum_k t^srv_A,k e^ret_B,k
  (the cross terms of mix x win rate; all one-player terms are level, which ratings carry).

Results:
- **Surface control changes nothing.** Edge stability between periods is identical pooled vs
  surface-adjusted (e.g. serve short 0.64/0.64, serve long 0.54/0.54, 2015-18 -> 2019-22;
  101 and 160 players). Null bands (phase wins redrawn at population phase rate + own level)
  sit at about -0.15..+0.22; all edges clear them except return short and return long in
  2015-18 -> 2019-22.
- **Residual test** (33,662 matches with stats, both players profiled, 2011+; partial correlation
  with total point share beyond the reference): matchup part r = 0.032 (pooled), 0.031 (surface),
  0.031 (same) beyond Glicko-2; 0.042 / 0.039 / 0.033 beyond points_v1 (t 6-8). The player part
  (charted level) is stronger: r = 0.05 beyond Glicko-2, **0.107 beyond points_v1 (t ~ 20)**.
- **Correction** (serve logit += beta M / (p(1-p)), beta fitted on serve points before 2019:
  0.44-0.55, t ~ 5): tour-level log loss vs points_v1 run 273, paired on identical matches.
  Val 2019-21 calibrated: +0.00006 / +0.00013 / +0.00007 (worse). Test 2022-26 calibrated:
  -0.00007 / -0.00010 / 0.00000, SE 0.00011. Adding the level term: test raw +0.00026..+0.00028
  (worse), calibrated -0.00003..-0.00012. **Not kept**: the effect on serve points is real but
  too small to move match log loss.
- Reference scores from the same pass: points_v1 run 273 on tour-level test 2022-26 (n = 21,362):
  0.59843 raw, 0.59771 calibrated (run 203: 0.6045). Run 273 was chosen on validation only.
- **Integration check (same day):** the null result is not the point -> Markov integration.
  (1) Serve-side coefficients on M (~0.2-0.4) and the level term (~0.05-0.07) are the same with or
  without a control for points_v1's own logit, so step 3's beta and gamma were the right size (step
  2's 0.27 is a different unit: serve + return level, total point share). (2) Skipping the
  integration and fitting match outcomes directly (logistic on points_v1 logit + level + M, fitted
  before 2019 or on val) gives -0.0001..+0.0001 out of sample. (3) Oracle (coefficients fitted on
  the test matches themselves): -0.00007 / -0.00009 per tour-level test match. That is the ceiling
  for this form. On match outcomes the level term has t = 0.5-1.7 and M t <= 3.
- **Why point-level t does not carry to wins:** 33k matches x ~150 points make tiny effects
  significant. r = 0.04 explains 0.16% of point-share variance, worth ~0.0003 log loss at most.
  The level term's t ~ 20 is mostly points_v1's serve-point expectations being slightly compressed
  (controlling for its own logit halves the level term's t in 2022-26), and match-level
  calibration already fixes that. It is not a strong lead for the ratings layer.

## Point-in-time rules (leakage)

- Features and ratings for a match may use only matches that happened strictly before it. Order by `(edition start_date, round_order)` when `match_date` is NULL.
- `matches.date_precision` says whether `match_date` is exact (`day`) or only the week is known (`week`). Same-day logic applies to `day` rows only.
- Rankings: use the latest `ranking_date` strictly before the match, never the rank stored on the match row for features.
- Model outputs (pre-match ratings, predictions) live in `models.db`, not here. A prediction is made from state strictly before the match and evaluated walk-forward only.
- Odds are a benchmark. Using them as features needs verified capture times.
- Rolling sources (AcesAPI): `match_seen.first_seen_at` is when a result was first known. A live or "as of T" replay must use only matches with `first_seen_at <= T`. `ingest_snapshot.fetched_at` is the pull time of the version actually loaded.

## Table map

| Group | Tables |
|---|---|
| Plumbing | `schema_meta`, `source`, `ingest_snapshot`, `load_run`, `load_skip` |
| Entities | `player`, `player_alias`, `tournament`, `tournament_alias`, `tournament_edition` |
| Crosswalks | `player_xref`, `edition_xref`, `match_xref` |
| Matches | `matches`, `match_player_stats`, `ranking`, `match_seen` |
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
-- 1. No player in two matches on the same exact day in different events, or twice in one
--    round (must return 0 rows). Two rounds of one event on one day are real doubleheaders.
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
