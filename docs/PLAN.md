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
| Training harness (A2) | `train/` (data, scoring, runs, context_fit, compare), `scripts/train.py` | Added 2026-11-15 |

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

**Done 2026-11-15.** Run 790 rebuilt from zero context, bit for bit, in two recorded v1 steps: `train.py context --base 694 --iters 11` (q 1.5e-5, unc 5) reproduces run 711's params exactly (max |diff| 0.0), and `train.py context --base 711 --set q=7e-6 --set unc=3 --iters 3` reproduces run 790's params exactly (max |diff| 0.0). Val tour: 0.5896 raw, 0.5899 calibrated per (tour, best-of) (the harness default, `tour_cal`), 0.5901 per tour; run 694 0.5937 calibrated. These match v1's records. Three iterations from zero alone reach only 0.5905: the age offsets converge slowly. Commands: `score`, `sweep`, `context`, `show`, `compare`, and `finalize` (test, refuses a second scoring without `--force`). Calibrated scopes are stored as `tour_cal`, `tour_atp_cal`, `tour_wta_cal` next to v1's raw scopes. `--model-id points_v2` names the retrain's runs.

### A3. Retrain: `points_v2` (pre-match, with context)
Nothing to add back; this is the work itself:
- Data through now (AcesAPI scores to 2026-11) so context features see recent layoffs and load.
- Refit context; re-tune `q`, `obs_var`, `v_init`, `unc`, `blend`, `score_weight` jointly on val.
- Per (tour, best-of) calibration; recheck WTA raw vs calibrated.
- Diagnostic: calibration binned by serve dominance (`x_A + x_B`), the v1 open question.

Done when: `points_v2` beats run 790 on val, test is scored once, and A1's predictor switches to it only if test holds.

**Done 2026-11-15: no gain, predictor stays on run 790.** Data as in tennis.db (results through editions starting 2026-11-09; no new AcesAPI pull). Val (2019-21, tour, calibrated per (tour, best-of)):
- Sweeps `v2-r1` (90 settings from run 790: `q`, `obs_var`, `v_init`, `score_weight` x `unc` x `blend`) and `v2-r2` (36 around the best): best is `score_weight` 1.0, `blend` 0.6, `unc` 1.5 (run 905), 0.5894 vs 0.5899. `q`, `obs_var`, `v_init` are flat around run 790's values.
- Context: 6 more residual iterations from run 790 (run 848) and 2 from run 905 (run 930) gain nothing (-0.0001). The remaining steps are the age serve/return split, which is weakly identified.
- Serve dominance (`train.py diagnose`): no trend in calibration along `x_A + x_B` (ATP slopes 0.88-1.18 across quintiles, no order). The one pattern is WTA: calibration helps the least serve-dominated fifth and hurts the middle.
- WTA raw beat calibrated on val (0.5901 vs 0.5907), so the candidate left WTA raw (`cal_wta` 0, new prediction-only params `cal_atp` / `cal_wta`): run 931, val 0.5892 vs 0.5899 (-0.0007).

Test 2022-26, scored once (run 931; 23,269 tour matches): **0.5942 vs run 790 0.5942, -0.0000 ± 0.0003** (9/20 quarters). ATP -0.0005 ± 0.0003 (10/20); WTA +0.0005 ± 0.0006: on test, calibration helps WTA (run 790 WTA calibrated 0.5929 vs raw 0.5933), the reverse of val, and run 931's lower `unc` leaves WTA overconfident (slope 0.895). Lessons: the val gains were val overfitting at the 0.0005 level; WTA raw vs calibrated flips between eras, so keep calibrating both tours; the parameters of run 790 are at a flat optimum, and further pre-match gains need new information (A4 in-play, or new features), not re-tuning.

### A4. In-play model
Adds back:
- **In-play evaluation** (from `inplay_form_test.py`): a harness scoring point/game/set-end log loss on slam point-by-point and MCP, walk-forward.
- **MCP point parsing** (`mcp_notation.py`), only if MCP matches are used beyond the loaded `point` rows.

Then: separate serve and return form (`f_s`, `f_r`), an in-play `unc`, and calibration by match stage.

Done when: in-play log loss beats form_var 0.02 on val, and test is scored once.

**Harness done, model unchanged (2026-11-15).** `train/inplay.py` + `scripts/inplay_eval.py`: 17,434 pbp matches 2011+ (slam pbp + MCP; 241 dropped because the points don't replay to the result) give 442,735 game-end checkpoints; run 790's walk-forward pre-match logits; variants paired against the live setting (form 0.02, independent streams), SE clustered by match. MCP parsing not needed (the loaded `point` rows suffice). Findings:
- The form update itself is confirmed on val: -0.0060 ± 0.0018 vs score only.
- **Serve vs return form.** Within a match they are only identified through the correlation of the two serve streams. It is real and negative (player day form, not conditions): whole-match cross covariance -0.011 ± 0.001 in every period (same-stream persistence +0.024); the predictive part (A's first half vs B's second half) is -0.005 ± 0.001, correlation about -0.25. Built as `points_rating.in_match_form2` (2-d update, `form_corr`). It does not help: on fit 2011-18 rho in [-0.5, 0.25] moves log loss by ±0.0004 at best and -0.5 hurts (+0.0017); on val rho -0.25 hurts (+0.0006 ± 0.0003). By the time it matters, B's own service games say more.
- **In-play `unc`**: the run's 3 beats 4.5 and 6 on fit.
- **Calibration by stage** (alpha per best-of and sets played, fitted on 2011-18): -0.0010 on fit (in-sample; best-of-5 early stages underconfident, alpha 1.13), +0.0003 on val. Not kept.
- **Point details** (aces, double faults, first serve in; the point-by-point question). First-half aces above the player's own point-in-time rate (from `match_player_stats`) predict second-half serve results beyond points won (z +9, both tours); double faults and first-serve % add little. As an ace bonus in the form update (k + gamma (aces - expected)): fit best gamma 0.5 with form 0.015, -0.0007 ± 0.0004; val +0.0003 ± 0.0006 (0.25: -0.0001 ± 0.0004). Not kept. The ace signal may be partly pre-match (big servers vs their rating): a candidate pre-match feature, not an in-play one.
- Form 0.015 vs 0.02: -0.0002 ± 0.0003 on val, noise; ATP and WTA swap preferences between fit and val.

Test not scored: nothing beat the live setting on val. In play stays form 0.02, independent streams, run's `unc`, uncalibrated.

### A5. State persistence + incremental replay
New, nothing from v1: save state and context bookkeeping at cutoffs (monthly snapshots), and resume `run()` from them. Done when incremental replay equals full replay and a daily AcesAPI pull updates the predictor in seconds.

**Done 2026-11-16.** `engine/state.py` + `scripts/state.py build|status|verify`. A snapshot at cutoff C (a month start; never splits an edition) holds the PointsState, context bookkeeping, tournament deciding-set rule history, edition rules and 11 years of tour-level predictions (for calibration), all keyed by **stable keys** (player and edition seed xref keys, tournament tour|name), because a tennis.db rebuild changes surrogate IDs. Validity: hash of overrides.csv + non-AcesAPI raw files + birth dates (old history), a stable-key hash of every match and its stats in the 400 days before C (late results; AcesAPI starts 2026-05-05), match count and score totals, and the model code + params. The latest valid snapshot is used; stale ones are skipped. `run(start_state=)`, `load_matches/attach(since=, before=, rule_last=, ctx_init=)` and copy-on-touch context resume make the window replay exact.
- Real data (run 790, snapshots 2026-09/10/11-01, 30 MB each): resume from 2026-11-01 replays 302 matches in 5.4 s and equals the full replay (state, context, every prediction); `PointsPredictor.load` after a data change takes ~20 s (state + validity check + name index) vs ~300 s for a full replay; unchanged data 2.4 s from the ModelState cache. Tests: `tests/test_state.py` (resume = full replay at two cutoffs, as_of, stale on a changed score, IDs shifted).
- Daily flow: `fetch_acesapi.py --pin` and `build_db.py` (~10 min, unchanged), then any predictor load resumes. Monthly: `state.py build` (~3 min). Snapshots go stale when a non-AcesAPI source or overrides.csv changes; `state.py status` shows which are valid.

## Track B: Kalshi retriever on a black-box predictor

### B1. Retriever
Adds back (from `kalshi_edges.py`): event and order-book pulls, milestone/live-score mapping by competitor id, edition lookup for surface/best-of/rules, raw-pull saving, as `engine/live.py` + `scripts/live.py --predictor fixed|points`. Each market becomes a `MarketSnapshot` with a `MatchQuery` and the predictor's `Prediction`.

Done when: every open market is listed with model vs market, and tests run on saved raw fixtures with no network.

**Built 2026-11-15:** `engine/live.py` (`fetch_pull` / `save_pull` / `load_pull`, `snapshots()` -> `MarketSnapshot`), `scripts/live.py [--pull DIR] [--predictor fixed|points]`, `tests/test_live.py` on `tests/fixtures/kalshi/` (a real pull subset; network calls fail the tests). Notes:
- Pulls now also save `milestones.json` and `pull.json`. Older pulls (before 2026-11-15 23:00) load without them; live data maps by the competitor-id pair either way, and a milestone whose pair disagrees with the markets is refused.
- Best-of: milestone > running edition > men's slam by name > 3. Deciding set: the edition's inferred rule > 'tb'. Surface: running edition > tournament history > `--surface` > unknown (None).
- Kalshi gives no surface anywhere (events, markets, milestones, live data, series and event metadata checked). `engine/surfaces.py` maps Kalshi's tournament name (milestone `tournament_name`, else the rules text) to tennis.db history: same-city Challenger/WTA 125 > same-city tour level (Metz 2026) > same-city ITF; season finals and tour events by name. The edition nearest in the calendar from the last 6 years wins. Every tournament in the 2026-11-15 pulls resolves.
- Kalshi labels namesakes "Cezar Cretu (b. 2001)"; the suffix is stripped before resolving (the year is not yet checked against dob).
- `--predictor points [--run 790] [--raw]` uses A1's `PointsPredictor` (and its name resolver); `DbEditions` reuses A1's `editions.shared_edition`. On the 2026-11-15 23:14 pull: 41 priced, every Challenger player flagged stale (Challenger results end 2026-05-25).

### B3a. Decisions on one snapshot
Adds back (from `kalshi_edges.py`): blending, fractional Kelly over the book, fees, the stale-player refusal. New: per-match and total exposure caps, an in-play flag, and refusal of placeholder predictions. As `engine/decide.py`: `MarketSnapshot` + its books + a `Policy` -> a `Decision` (the bet, or why none). Pure: no network, no state. Shown in `scripts/live.py`.

Why before B2: it needs only a B1 snapshot, and B2 records its output, so what B3b will need (full books, the decision and its reason) is fixed before recording starts.

Done when: every priced snapshot gets a decision or a reason, on the saved fixture. Sanity tests: market-only weight places nothing; a predictor that echoes the market's fair price places nothing (spread + fees); `FixedPredictor` is refused as a placeholder. (The original test, "`FixedPredictor(0.5)` must place nothing", fails as stated: fully trusted, 0.5 against a 0.985 favourite is a large edge on the underdog.)

**Built 2026-11-15:** `engine/decide.py` (`Policy`, `decide`, `decide_all`), decision column and policy flags in `scripts/live.py`, `tests/test_decide.py` (the three sanity tests, side choice, caps, refusals). `FixedPredictor` marks its predictions `detail['placeholder']`. On the 2026-11-15 23:14 pull with run 790 (defaults): 8 bets, $229 staked; 29 of 41 priced matches refused as stale (Challenger results end 2026-05-25). With `--allow-stale`, 27 bets and the total cap binds: the stale refusal is doing real work.

### B2. Watch loop + recorder
Adds back (from `kalshi_edges.py --watch`): the polling loop, extended into an append-only store (`live.db`, gitignored). Per poll:
- **raw**: events, milestones, live data and full order books (compressed, deduplicated), so any poll can be reloaded as a B1 `Pull` and repriced or re-decided;
- **snapshot**: price, score, prediction and B3a decision per match;
- **tags** for diagnosing misses: each player's days since last rated match, how each name resolved, where the surface came from, in play or not, series (tour / Challenger), model - market gap;
- **settlement**, when a market finishes: winner, Kalshi's market result, and how the match ended (`match_status`: completed / retired / walkover as Kalshi reports it) with the final score.

Market sides are ordered by ticker, so A/B stays the same across polls. `player_id` is only valid for the tennis.db build of the watch (recorded per watch); names and competitor ids are the stable keys. The v1 watch logs (`archive/v1/models/kalshi/live_*.jsonl`) can be imported as the first data.

Why before B3b: Kalshi's public API serves only the current book and score, so unrecorded days are lost; the sooner it records, the more history B3b and B5 have.

Done when: a multi-hour watch runs unattended and answers "price, score, model and decision at time t", and every match it saw is settled or still open.

**Built 2026-11-15:** `live_schema.sql`, `engine/recorder.py` (`Recorder`: `record_poll`, `load_poll`, `to_settle` / `settle`, `at`, `counts`), `scripts/watch.py run|at|status`, `tests/test_recorder.py` (round trip and replay, blob dedup, tags, decisions, time-t query, settlement, one full `poll_once` offline). Shared setup moved to `live.load_predictor` / `live.Retriever` and `decide.add_policy_args`. Markets now ordered by ticker (A/B stable across polls). First real watch 2026-11-15: 44 open matches, ~4 s and ~40 KB per poll (~60 MB/day at 60 s); a match closed in the live feed waits until Kalshi determines its markets. Not yet done: the multi-hour unattended run (watch 2, started 2026-11-15 23:50 UTC, 8 h), the v1 log import.

### B3b. Paper ledger
New: a paper ledger replayed from B2's recordings: decide per poll (B3a), fill against the recorded book, track open exposure across polls, settle on the recorded result. Because it replays, a policy or predictor change (e.g. `points_v2` after A3) is re-scored on the same matches without new watching.

Reports, built to say *what kind* of miss, not only *whether*:
- **Who was right**: log loss of model vs market on every priced match (far more data than bets), and P&L on bets.
- **Line movement** (closing-line value): the market price from each decision to the match start. Moving toward the model means the model saw something early; moving away means the market most likely had news (injury, withdrawal). Needs no result, so it reads per match.
- **Splits by tag**: stale days, loose name resolution, surface from history vs edition, in play, Challenger vs tour, gap size. Losses concentrated in one tag point at that cause (missing results, wrong player, wrong surface); spread evenly, at the model itself.
- **How it ended**: bets on the healthy-looking player lost to a retirement or walkover point at injuries the model cannot see.

A single match stays ambiguous; conclusions come from patterns across many.

Done when: a day of recordings produces a ledger with these reports, and replaying it twice gives the same ledger.

### B4. Alerts
New: a push notification when the decision layer fires, with ntfy (chosen 2026-11-16).

**Scaffold built 2026-11-16:** `engine/alerts.py`: `ChangeDetector` (per match: new bet, side flipped, stake grew x1.5 since the last alert, alerted bet gone), `Alerter` (per-match cooldown after a push, 'gone' bypasses it; hourly cap, most urgent / largest EV first; failures recorded, never stop the watch), `NtfyNotifier` (JSON publish, UTF-8 names, optional token). Messages carry the trust tags (last rated N days ago, loose name, surface from history / unknown, stale, in play). Every change goes to live.db's `alert` table as sent / suppressed (with why) / failed / printed. `scripts/watch.py run --ntfy` pushes (topic `NTFY_TOPIC` in `.env`; without `--ntfy` changes are printed and recorded only); `watch.py test-alert` sends one test push. Tests: `tests/test_alerts.py` (fake ntfy, nothing sent).

Defaults, to calibrate on recorded data (B3b and the in-play study): min stake $5, min EV $0.50, grow x1.5, cooldown 15 min, 12 pushes/hour, priority high at EV >= $10. Detector state is in memory: a restarted watch re-alerts current bets within the cap. Open: alert only on non-stale bets by default once Challenger data is current; an in-play alert path once feed lag is measured.

### B5. Scorecard
Adds back, optionally, the closing-odds benchmark (`odds.py`, `xlsx_lite.py`, `odds_benchmark.py`) for historical market log loss. New: live model vs market log loss pre-match and in play, per predictor, and the feed-lag study.

**Built 2026-11-16 (moved ahead of B3b: the historical half needs no watch data):**
- `src/tennis/odds.py` (Valuebetennis reader, linker, de-vig, blend, backtest; the tennis-data xlsx reader and the simulation stay archived) and `scripts/odds_benchmark.py --runs 790,...`: points runs are replayed in process and scored raw and calibrated, other runs read from `models.db`; nothing written to `models.db`. The blend `y ~ a*logit(close) + b*logit(model)` is also reported as `model_weight = b/(a+b)`, the number `engine.decide` uses.
- `src/tennis/engine/scorecard.py`, `scripts/watch.py scorecard [--json]`: per predictor, on settled matches in live.db, model vs market log loss pre-match (last snapshot before going live) and in play (each match weighted once), and the feed-lag correlation (model move vs market move k polls later; a peak at k < 0 = the market moves before the feed shows the score). Resolution is the poll spacing.
- Tests: `tests/test_odds.py` (hand-made Valuebetennis fixture), `tests/test_scorecard.py`.

**Historical result (2026-11-16, `odds_benchmark.py --runs 790,931,203`, `models/odds/benchmark_790_931_203.json`):** 26,308 rows linked; 86.4% of tour-level 2022-26 matches have a Pinnacle close; 18,268 identical matches (the same set as v1's benchmark).
- Pinnacle close 0.5864; run 790 calibrated **0.5949 (+0.0085 ± 0.0011)**, raw 0.5954; run 931 the same; Glicko-2 (203) 0.6065 (+0.0201). The gap is down from 0.0138 (run 602) and 0.020 (Glicko); steady by year (+0.007 to +0.012) and tour (ATP +0.0084, WTA +0.0086). The opening line is +0.0040.
- Blend: **b is now positive** (790 cal b = +0.15, z = 3.0; Glicko's was -0.22): the model carries information the close does not fully price, in sample. Walk-forward by year it still adds nothing usable: held-out gain +0.00004 ± 0.00026 nats (break-even against Pinnacle is about 0.0004). The implied `model_weight` is **0.15-0.19** in every year (all four points variants), against the live default of 0.3 (lowered to 0.15 on 2026-11-16).
- Backtest at the close: -1.4% to -2.2% ROI (± 1.1%), about the margin. **At the opening line: +2.3% to +3.8% ± 1.5% with a 5% EV threshold** (raw runs about 2.2-2.6 SE; eight correlated variants, so treat as a lead): the close moves toward the model, i.e. the model has closing-line value. Opening-odds capture times are not documented, so this is not a backtest anyone could have traded. For Kalshi it says where to look: early prices (before the line settles) and B3b's line-movement report.
- Pre-match verdict: about 0.0085 behind the sharp close and adding nothing out of sample; the edge, if any, is against early or soft prices, not the close.

## Later (C)
### C1. Ratings viewer (rankings at any date)
**Built 2026-11-16.** `engine/standings.py` (v1 `eoy_serve_return.py`'s standings, weekly): one replay with `run(on_snap=)` (a new uncopied-view callback, so ~880 weekly cutoffs cost no state copies) gives each player's serve, return, SDs, age offset, per-surface deviations and counts at every Monday; a week = editions starting before it, as the predictor and A5 snapshots cut. `scripts/ratings_export.py` writes `models/ratings/run<id>/` (index, players by stable key, one gzipped JSON per week; default weekly from 2010, top 500 per tour, 10+ matches, active 365 days; swapped in atomically). `scripts/ratings_server.py` (stdlib, read-only, default 127.0.0.1:8790) serves `web/ratings.html`: week stepping (buttons, dropdown, arrow keys), ATP/WTA, surface (effective rating = s + blend·ds, plus the deviations), age-adjusted or neutral, search, min matches, sort by any column, rank and rating change vs the previous week, serve/return points won vs the view's top 100; the view lives in the URL hash. Tests: `tests/test_ratings_view.py` (each week equals a replay of only the editions before it; surfaces, filters, export layout, server routes and path traversal). Daily: rerun the export after `build_db.py`; the server picks it up without a restart. Deployed on the Oracle server as `advantage-ratings.service` (127.0.0.1:8790) behind nginx (`advantage-ratings.nginx`: HTTPS on 150-136-121-153.sslip.io with a Let's Encrypt certificate (certbot webroot, renewed by certbot.timer, nginx reloaded by a deploy hook), basic auth from /etc/nginx/.htpasswd, GET/HEAD only, 20 req/s, noindex; password-only because AcesAPI data may not be redistributed): the export runs here (the replay needs more than the server's 1 GB) and `scripts/ratings_push.py` uploads it (host: `ADVANTAGE_SERVER` in .env).

### Not started
The web app (the ratings viewer is C1), Challenger coverage (a new results source; the shelved `ingest_daily.py` only if that source is a scraper), Kalshi live serve points as a stats source after 2026-05.

## Stays archived
Elo, Glicko-2 and their sweeps/standings (`elo`, `glicko2`, `ratingmodels`, `standings`, `sweep`, `ratings`, `eoy_ratings`, `elo_*`): their runs are in `models.db` as baselines. Style, matchup, phase, composite, heavy: measured negatives. One-off studies: `blackout_test`, `context_report`, `context_test`, `predict_finals`, `phase_test`.

## Order
A1 and B1 first, in parallel (A1 is what makes B's numbers mean something). Then A2 -> A3 (the retrain) alongside B3a -> B2 -> B3b. Then A4, A5, B4, B5. Each milestone is one commit with tests passing.

## Open decisions
- Alert channel (B4): ntfy (decided 2026-11-16).
- Live store: `live.db` (B2), gitignored, never rebuilt (additive schema).
- Paper-ledger defaults (B3a/B3b): 1/4 Kelly, $1000, as in v1, unless told otherwise.
