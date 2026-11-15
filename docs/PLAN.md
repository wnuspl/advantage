# Reboot plan

Written 2026-11-15. Background and the run 790 model: `docs/REBOOT.md`.

The project restarts blank on top of the data layer. Everything else from v1 sits in `archive/v1/` and comes back only when a milestone needs it, usually rewritten against the engine rather than copied. This plan is the list of what gets added back, in which order, and why.

**Priority:** retrain the model with context (track A) and, at the same time, build the Kalshi retriever (track B), whose decisions are driven by a black-box predictor. The tracks share one interface, `engine.Predictor.predict(MatchQuery) -> Prediction`. Track B starts on the `FixedPredictor` placeholder and takes real models later without code changes.

## What the blank project has (2026-11-15)

| Part | Files | Status |
|---|---|---|
| Data layer | `schema.sql`, `src/tennis/` build, db, load_*, resolve, normalize, overrides, snapshots, loadrun, acesapi*, `scripts/` build_db, init_db, fetch_data, fetch_acesapi, export_review | Unchanged from v1; tested |
| Model results store | `models_schema.sql`, `models_db.py`, `models.db` | Unchanged; holds every v1 run (790 = baseline) |
| Match table | `matches.py` (loader + scoring scopes, extracted from v1 `elo.py`) | Added back |
| Model core | `points_rating.py`, `context.py`, `markov.py` | Kept as is: the form of the model survives, its parameters get re-tuned |
| Kalshi client | `kalshi.py` (endpoints, order books, live scores, fees, Kelly walk) | Kept; no model dependency |
| Engine seam | `engine/predictor.py` (`Predictor`, `MatchQuery`, `Prediction`, `FixedPredictor`) | New |
| Predictor (A1) | `calibrate.py`, `engine/predict.py` (`PointsPredictor`), `engine/players.py`, `engine/editions.py`, `scripts/predict.py` | Added 2026-11-15 |

What it cannot do yet: calibrate, score a run, record a run, resolve a name, price a market, run a CLI. The milestones add those back.

## Track A: retrain the model + context

### A1. A real predictor behind the seam
Adds back:
- **Calibration** (from archived `calibrate.py`): rewritten without its v1 dependencies (`composite_eval.block_of`, `matchup.data.fit_alpha` become local helpers). Per (tour, best-of) alpha, past data only. Its test comes back from `archive/v1/tests/test_calibrate.py`.
- **Player resolution** (from `kalshi_edges.py`): `engine/players.py`, shared with track B.
- **The canonical prediction path** (from `kalshi_edges.py` + `points_rating.matchup`): `engine/predict.py`, `PointsPredictor(run_id)`: final state, inactivity-grown variance, blend, **context offsets** (v1 `matchup()` lacked them), Markov roll-up, `unc` shrink, calibration, `in_play_probability` when `query.live` is set, `stale` past 120 days. Plus a pickled-state cache keyed on tennis.db + code + params (as `kalshi_edges` had).
- **CLI** (from `predict_match.py`): `scripts/predict.py A B --tournament ... --date ... [--score ... --serve-pts ...]`.

Done when: stored run-790 predictions are reproduced exactly on a sample, and in-play output matches v1 on the same inputs.

**Done 2026-11-15.** Run 790 has no stored per-match predictions (val metrics only), so: a fresh replay reproduces run 790's stored val log loss in every scope (tour 0.5896388009) and run 694's 1,768,019 stored predictions (max |diff| 6e-17); `PointsPredictor` built from the state just before 12 sampled matches (val, test, all levels; new editions and later rounds) reproduces runs 790 and 694's walk-forward predictions (max |diff| 6e-16); in play equals the v1 `Pricer` arithmetic on the same inputs (6 states, 0 mismatches). The rewritten calibration gives val tour 0.590071 per tour (v1: 0.5901) and 0.589950 per (tour, best-of). Live alphas for 2026 Q4: ATP bo3 0.900, bo5 1.082; WTA 0.930. Cold replay ~5 min, cached load seconds (`models/engine/run790.pkl`, 35 MB). Deviation from v1 `kalshi_edges`: a match in a known edition is rated on the edition's start date (as `run()` does), not the match day. Calibration lives in `src/tennis/calibrate.py`; editions lookup (`engine/editions.py`, incl. `shared_edition` for B1) added alongside.

### A2. Training harness
Adds back, as `src/tennis/train/` + `scripts/train.py`:
- **Run preparation and recording** (from `tuning.py`): points models only; the Elo/Glicko sweep machinery stays archived.
- **Scoring** (from `evaluate.py` as is, and `composite_eval.metrics`): log loss, Brier, calibration bins, paired difference with SE, quarters won.
- **Run comparison** (from `compare_runs.py`): `train.py compare A B`.
- **Context fit** (from `context_fit.py`): iterated residual WLS for the `cs_*`/`cr_*` coefficients, inside the training loop.

Done when: `train.py` reproduces run 790's val numbers (tour 0.5901 calibrated) from scratch.

### A3. Retrain: `points_v2` (pre-match, with context)
Nothing to add back; this is the work itself:
- Data through now (AcesAPI scores to 2026-11) so context features see recent layoffs and load.
- Refit context; re-tune `q`, `obs_var`, `v_init`, `unc`, `blend`, `score_weight` jointly on val.
- Per (tour, best-of) calibration; recheck WTA raw vs calibrated.
- Diagnostic: calibration binned by serve dominance (`x_A + x_B`), the v1 open question.

Done when: `points_v2` beats run 790 on val, test is scored once, and A1's predictor switches to it only if test holds.

### A4. In-play model
Adds back:
- **In-play evaluation** (from `inplay_form_test.py`): a harness scoring point/game/set-end log loss on slam point-by-point and MCP, walk-forward.
- **MCP point parsing** (`mcp_notation.py`), only if MCP matches are used beyond the loaded `point` rows.

Then: separate serve and return form (`f_s`, `f_r`), an in-play `unc`, and calibration by match stage.

Done when: in-play log loss beats form_var 0.02 on val, and test is scored once.

### A5. State persistence + incremental replay
New, nothing from v1: save state and context bookkeeping at cutoffs (monthly snapshots), and resume `run()` from them. Done when incremental replay equals full replay and a daily AcesAPI pull updates the predictor in seconds.

## Track B: Kalshi retriever on a black-box predictor

### B1. Retriever
Adds back (from `kalshi_edges.py`): event and order-book pulls, milestone/live-score mapping by competitor id, edition lookup for surface/best-of/rules, raw-pull saving, as `engine/live.py` + `scripts/live.py --predictor fixed|points`. Each market becomes a `MarketSnapshot` with a `MatchQuery` and the predictor's `Prediction`.

Done when: every open market is listed with model vs market, and tests run on saved raw fixtures with no network.

**Built 2026-11-15:** `engine/live.py` (`fetch_pull` / `save_pull` / `load_pull`, `snapshots()` -> `MarketSnapshot`), `scripts/live.py [--pull DIR] [--predictor fixed|points]`, `tests/test_live.py` on `tests/fixtures/kalshi/` (a real pull subset; network calls fail the tests). Notes:
- Pulls now also save `milestones.json` and `pull.json`. Older pulls (before 2026-11-15 23:00) load without them; live data maps by the competitor-id pair either way, and a milestone whose pair disagrees with the markets is refused.
- Best-of: milestone > running edition > men's slam by name > 3. Deciding set: the edition's inferred rule > 'tb'. Surface: edition > `--surface` > unknown (None). New events (ATP Finals before its first result in tennis.db, every Challenger: AcesAPI has none) have no edition, so surface is unknown there; the predictor must handle `surface=None`.
- Kalshi labels namesakes "Cezar Cretu (b. 2001)"; the suffix is stripped before resolving (the year is not yet checked against dob).
- `--predictor points` waits for A1's `engine/predict.py`.

### B2. Watch loop + recorder
Adds back (from `kalshi_edges.py --watch`): the polling loop, extended into an append-only store (proposed `live.db`, gitignored): snapshots, scores and predictions per poll, settled when markets finish. The v1 watch logs (`archive/v1/models/kalshi/live_*.jsonl`) can be imported as the first data.

Done when: a multi-hour watch runs unattended and answers "price, score and model at time t".

### B3. Decision layer + paper ledger
Adds back (from `kalshi_edges.py`): blending, fractional Kelly over the book, fees, the stale-player refusal. New: exposure caps, an in-play flag, and a paper ledger settled on results (P&L, closing-line value, log loss vs market).

Done when: a day of watching produces a ledger. As a sanity test, `FixedPredictor(0.5)` must place nothing.

### B4. Alerts
New: a push notification when the decision layer fires (channel to decide; ntfy is the low-effort default).

### B5. Scorecard
Adds back, optionally, the closing-odds benchmark (`odds.py`, `xlsx_lite.py`, `odds_benchmark.py`) for historical market log loss. New: live model vs market log loss pre-match and in play, per predictor, and the feed-lag study.

## Later (C)
Rankings at any date (from `eoy_serve_return.py`'s logic on A5 snapshots), the web app, Challenger coverage (a new results source; the shelved `ingest_daily.py` only if that source is a scraper), Kalshi live serve points as a stats source after 2026-05.

## Stays archived
Elo, Glicko-2 and their sweeps/standings (`elo`, `glicko2`, `ratingmodels`, `standings`, `sweep`, `ratings`, `eoy_ratings`, `elo_*`): their runs are in `models.db` as baselines. Style, matchup, phase, composite, heavy: measured negatives. One-off studies: `blackout_test`, `context_report`, `context_test`, `predict_finals`, `phase_test`.

## Order
A1 and B1 first, in parallel (A1 is what makes B's numbers mean something). Then A2 -> A3 (the retrain) alongside B2 -> B3. Then A4, A5, B4, B5. Each milestone is one commit with tests passing.

## Open decisions
- Alert channel (B4).
- Live store: `live.db` proposed (B2).
- Paper-ledger defaults (B3): 1/4 Kelly, $1000, as in v1, unless told otherwise.
