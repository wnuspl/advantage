# Reboot: from pre-match research to a live betting engine

Written 2026-11-15. This is the starting point for the next phase. It does three things:

> Paths: since the archive move (same day), v1 scripts and research modules named here live in `archive/v1/`; `points_rating`, `context`, `markov`, `kalshi` stay in `src/tennis/`. The milestone plan that supersedes section 7 is `docs/PLAN.md`.

1. Records how the best model so far (**run 790**, `points_v1` + context offsets) works, exactly as the code does it, so it can be rebuilt or re-tuned without re-reading the history.
2. Says what stays (the data layer) and what gets reset (model parameters, scripts, caching).
3. Lays out the target system: on-demand prediction, rankings, cached state, an HTML front end on a server, Kalshi live odds and alerts.

Sources for the model description: `src/tennis/points_rating.py`, `context.py`, `markov.py`, `calibrate.py`, `kalshi.py`, `scripts/kalshi_edges.py`, `scripts/predict_match.py`, `docs/DESIGN.md`, and run 790's stored parameters in `models.db`.

---

## 1. The new goal

**Be a live (in-play) betting engine.** The edge we want is in the middle of the match: the pre-match rating gives a prior on each player's serve and return, and what happens on court today updates it point by point. Pre-match and mid-match are separated cleanly:

- **Pre-match:** ratings (serve, return, surface, context) give each player's serve-point logit going into the match.
- **Mid-match:** a per-match *form* term on serve and on return, updated from the points played so far, plus the exact Markov roll-up from the current score.

The focus this time is a clean interface: one script per function, a shared engine underneath, and cached state so nothing replays 1.7M matches on every call.

---

## 2. What stays unchanged

The whole data layer stays: `schema.sql`, `scripts/build_db.py`, the snapshot/manifest/lock protocol, resolution and overrides, `load_run` accounting, AcesAPI fetching, and the invariants and their tests. `tennis.db` stays disposable and rebuildable; `models.db` stays additive and keyed by stable source keys.

Current contents (2026-11-15): 1,768,019 matches, 137,508 players, 58,095 editions (latest starts 2026-11-09), 3,740,808 charted/point-by-point points. Per-match serve stats end 2026-05-25 (mirror cutoff); AcesAPI results after that have scores but no stats.

The research code stays in the repo as history (style, matchup, phase, composite, Elo, Glicko). It is not part of the new engine.

---

## 3. How run 790 works

### 3.1 State

Everything is on the **logit scale of a serve point**.

| Per | Value | Variance | Notes |
|---|---|---|---|
| player | serve rating `s` | `vs` | grows `q`/day inactive, capped at `v_init` |
| player | return rating `r` | `vr` | same |
| (player, surface) | serve deviation `ds` | `dvs` | prior mean 0, prior var `surf_var`; grows `q_surf`/day off that surface |
| (player, surface) | return deviation `dr` | `dvr` | same |
| (tour, surface) | base logit `b` | none | average serve-point win rate; starts 0.53 ATP, 0.24 WTA; moves by `base_lr * z` |

The state stores no age, rust or form. Those are applied per match and never written into the ratings.

### 3.2 Pre-match serve-point logits

For A serving to B:

```
x_A = b(tour, surface) + s_A - r_B + blend * (ds_A - dr_B) + o_A
x_B = b(tour, surface) + s_B - r_A + blend * (ds_B - dr_A) + o_B
var_A = vs_A + vr_B + blend * (dvs_A + dvr_B)        (and likewise var_B)
```

`blend` applies only at prediction time. During the replay, updates use the full surface deviations.

**Context offset** (milestone 9): `o_A = beta_serve · c_A - beta_ret · c_B`. `c` holds point-in-time features for each player:

| Feature | Meaning | `cs_` (own serve) | `cr_` (own return) |
|---|---|---|---|
| `back1` | 1st match after a > 12-week gap between events | -0.063 | -0.047 |
| `back2_5` | 2nd-5th match after such a gap | -0.023 | -0.036 |
| `gap5_12` | event 5-12 weeks after the previous one | +0.015 | -0.060 |
| `gap1` | back-to-back weeks | +0.016 | +0.001 |
| `load28` | matches in the previous 4 weeks / 5 | +0.036 | -0.024 |
| `age_lt22` | max(0, 22 - age) | -0.099 | +0.037 |
| `age_lt26` | max(0, 26 - age) | +0.042 | +0.017 |
| `age_gt28` | max(0, age - 28) | -0.012 | -0.019 |
| `age_gt32` | max(0, age - 32) | -0.013 | +0.008 |

Gaps are measured edition start to edition start, because Sackmann dates matches only by event week. The age curve is piecewise linear and flat between 26 and 28, so nothing jumps on a birthday. The coefficients are shared across tours (a per-tour set was tried and didn't help).

### 3.3 Pre-match match probability

1. **Markov roll-up.** Independent points give P(hold), then tiebreak, set and match, with the match's best-of and deciding-set rule (`tb`, `adv`, `tb12`, `mtb`). Best-of comes from `matches.best_of`. The deciding-set rule is inferred per edition from its own scores; with no evidence, it falls back to the tournament's previous rule, then `tb`. The roll-up is tabulated per format on a 0.01 logit grid (bilinear lookup with gradients, error < 1e-4).
2. **Uncertainty shrink** (Glicko's g()): `logit P / sqrt(1 + unc * 3 * var(logit P) / pi^2)`, where `var(logit P)` comes from the Markov gradient and `var_A`, `var_B`. `unc` = 3 is large because the Kalman variances ignore cross-player correlation and so come out too small.
3. **Calibration** (scoring only): per quarter and tour, `P_cal = sigmoid(alpha * logit P)`, with alpha fitted on the previous 10 years of tour-level matches. It is roughly neutral: raw and calibrated differ by less than noise, and WTA prefers raw. Per-(tour, best-of) alpha gains ~0.0002.

### 3.4 Updates (after the match, from the pre-match state, context-adjusted x)

Matches are processed in walk-forward order. Walkovers and unfinished matches never update. Retirements are weighted by `retired_weight` (1). There are three tiers, and the first one that applies is used:

1. **Serve points** (both players' serve points and points won in `match_player_stats`). Each server's record is a noisy observation of their x:
   `z = (k/n - p) / (p(1-p))`, `R = 1/(n p(1-p)) + obs_var`.
   A scalar Kalman update spreads the innovation over the four terms of x (s, ds of the server; r, dr of the returner) in proportion to their variances. The two directions use disjoint terms. The base logit moves by `base_lr * (z_A + z_B)`. Then an extended-Kalman result update with weight `result_weight` (0.3).
2. **Set scores** (no stats, but a parseable score; `score_weight` 1.5). A Newton step on the log-likelihood of the set scores (6-0 ... 7-6, advantage sets, match tiebreaks), averaged over who served first, using the Fisher information of the set tables. The step is spread over the terms in proportion to their variances. It replaces the result update.
3. **Result only.** An extended-Kalman update on the Bernoulli result using the Markov gradient, weight `nostat_weight` (1).

Variances shrink with each update and grow by `q` per day inactive. Context offsets enter every tier through x, so a player who performs exactly as their context predicts is not re-rated.

### 3.5 Run 790 parameters

| Param | Value | Role |
|---|---|---|
| `v_init` | 0.02 | new-player variance, and the cap |
| `q` | 7e-6 | variance growth per inactive day (1.5e-5 before context) |
| `obs_var` | 0.003 | extra per-match noise in the points observation |
| `surf_var` | 0.003 | surface-deviation prior variance and cap |
| `q_surf` | 2e-6 | surface variance growth per day off that surface |
| `init_offset` | 0 | new players start at s = r = 0 (the all-levels average) |
| `result_weight` | 0.3 | extra result update when stats exist |
| `nostat_weight` | 1.0 | result-only update weight |
| `score_weight` | 1.5 | set-score tier weight |
| `retired_weight` | 1.0 | retirements count fully |
| `rust_weight` | 1.0 | comeback matches count fully (< 1 hurt) |
| `base_lr` | 0.0005 | base-logit learning rate |
| `blend` | 0.7 | prediction: weight on surface deviations |
| `unc` | 3 | prediction: uncertainty shrink |
| `cs_*`, `cr_*` | table in 3.2 | context offsets, fitted by iterated residual WLS on 2000-2018 stats |

One replay over all 1.77M matches takes ~45 s for the core loop and ~5 min end to end (data load, attach, context, calibration alphas).

### 3.6 Results (tour level, test 2022-26, scored once)

| Model | Log loss | vs previous |
|---|---|---|
| Glicko-2 calibrated (run 203) | 0.6045 | |
| points_v1 (run 605) | 0.5968 | -0.0077 ± 0.0010 |
| + score tier (run 694) | 0.5968 tour / 0.5554 all levels | tour ≈ 0; all levels -0.020 |
| **+ context offsets (run 790)** | **0.5944** (0.5942 with per-best-of calibration) | -0.0024 ± 0.0005 |

Closing-odds log loss on the same matches is in `docs/odds_benchmark_report.md`. If stats disappear (blackout test), about 60% of the edge over Glicko-2 survives on set scores alone.

### 3.7 In-play (built 2026-11-15, uncommitted)

- **Exact roll-up from any score** (`markov.match_from`): sets, games, points, tiebreak serve rotation, and all deciding-set formats. Tested against an independent state machine and by simulation (`tests/test_inplay.py`).
- **In-match form** (`points_rating.in_match_form`): for this match, the serve logit is `x + e`, with `e ~ N(0, form_var)` shared by every point. The prior is `N(x, v + form_var)`, the serve points so far are binomial, and the posterior mode is found by Newton. `IN_PLAY_FORM_VAR = 0.02`, fitted on 2011-18 set-end predictions (`scripts/inplay_form_test.py`). Set-end log loss vs score-only: -0.0106 ± 0.0024 val, **-0.0130 ± 0.0021 test**. Measured within-match shared serve deviation: 0.039 logit², ~90% persisting from the first half of the match to the second. That is ~4.5x what the pre-match model assumes.
- **In-play probability** (`in_play_probability`): `match_from` plus the same `unc` shrink. It is not calibrated in play.

**Serve form vs return form, as now built.** The form term is per *serve-point logit*, so `x_A` absorbs both A's serve form and B's return form. Within a single match these cannot be separated: there is one observation stream per direction. Separating them needs a variance split (`form_var_serve`, `form_var_return`) estimated across many matches, plus a cross-match prior for each player (some players' return varies more from day to day). This is reboot work item R3.

### 3.8 Kalshi (read-only, built 2026-11-15)

`src/tennis/kalshi.py` and `scripts/kalshi_edges.py` do the following:

- Pull open ATP/WTA/Challenger match-winner markets and order books from public endpoints.
- Resolve names to players.
- Price each match pre-match, or in play when live.
- Size a fractional-Kelly stake by walking the book, including the taker fee `0.07 P(1-P)`.
- Refuse players whose last rated match is more than 120 days old.

Live scores come from Kalshi's own milestone and live-data endpoints. They are mapped to markets by competitor id, never by name. `--watch` logs every poll to `models/kalshi/live_*.jsonl` for a feed-lag study. The full replay is pickled to `models/kalshi/replay_run790.pkl` (26 MB, ~10 s warm), keyed on tennis.db contents, model code and params. The script places no orders.

### 3.9 Known gaps in the current code

- `points_rating.matchup()` applies no context. `kalshi_edges.py` adds it by hand; `predict_match.py` does not, and still defaults to run 694. The three entry points compute "the prediction" in three slightly different ways.
- Every cold call replays everything (~5 min). The only cache is Kalshi's pickle, which is all-or-nothing.
- `run()` cannot resume from a saved state, so new results always force a full replay.
- Rankings (`scripts/ratings.py`) understand only Elo and Glicko. The serve/return equivalent (`eoy_serve_return.py`) works only at year ends.
- In play is uncalibrated, and the Kalshi feed lag is unknown.
- Players with no stats since 2026-05-25 are rated on scores only, so their uncertainty is growing.
- The in-play work, `kalshi.py`, its tests and `inplay_form_test.py` are all uncommitted. Untracked stray files at the repo root: `file`, `glicko-2`, `points_v1`.

---

## 4. Measured negatives (don't redo)

PCA style vectors and the matchup layer (bilinear, siamese, LightGBM); personal style edges (don't repeat, r ≈ 0.02); the phase-profile correction (-0.0001); siamese on a points_v1 base (+0.0000); per-tour context coefficients; `rust_weight` < 1; home advantage and in-event workload (nothing on val). Details: `CLAUDE.md` "Findings", `docs/DESIGN.md`.

---

## 5. What gets reset

The model *form* is kept: serve/return Kalman ratings, Markov roll-up, context offsets, in-match form. What is reset is the parameter set, the objective and the code around it.

| Item | Why | How |
|---|---|---|
| **R1. Objective** | Run 790 was tuned for pre-match log loss. A live engine is judged on in-play predictions. | Two scores: pre-match log loss (as now), and in-play log loss at point/game/set ends on slam point-by-point (2011-24) and MCP matches, walk-forward, same val/test years. |
| **R2. Joint re-tune** | `unc`, `q`, `obs_var`, `blend` interact with in-play form: the shrink that helps pre-match may over-shrink once today's points are in. | Sweep the pre-match params with the in-play score as a second criterion. Fit an in-play `unc` separately from the pre-match one. |
| **R3. Serve vs return form** | The user's priority: separate serve and return nuance in play. | `x_A,match = x_A + e_sA - e_rB`, with `e_s ~ N(0, f_s)` and `e_r ~ N(0, f_r)`. Fit `f_s`, `f_r` (per tour, maybe per surface) on pbp data. Then test player-level form volatility (a per-player multiplier with a shrunk prior). Report it only if it beats a single `form_var`. |
| **R4. In-play calibration** | Raw in-play probabilities are uncalibrated. | Calibrate by match stage (set/game state), past data only. Check against Kalshi live prices once enough `live_*.jsonl` exists. |
| **R5. One canonical prediction path** | Three entry points currently disagree slightly. | A single `engine.predict(...)` that applies blend, context, shrink, calibration and (optionally) live state. |

New runs are stored as new `model_run` rows (a new `model_id`, for example `points_v2`), so 790 stays the comparison baseline.

---

## 6. Target system

### 6.1 Layers

```
tennis.db (unchanged)          models.db (additive)
      │                              │
      ▼                              ▼
src/tennis/engine/   ── one package the scripts and web app share
   state.py     RatingState: load/save, resume, incremental update
   predict.py   pre-match + in-play probability (the one canonical path)
   rankings.py  standings from a state at any date
   players.py   name -> player resolution (shared with Kalshi)
   live.py      Kalshi markets + live scores -> priced rows, alerts
      │
      ├── scripts/predict.py     (CLI)
      ├── scripts/rankings.py    (CLI)
      ├── scripts/state.py       (build / refresh / inspect cache)
      ├── scripts/live.py        (Kalshi watch + alerts)
      └── web/                   (HTML server on top of the same engine)
```

### 6.2 Caching (the core of "efficient")

- **Persisted rating state.** After a replay, save the full `PointsState` plus the context bookkeeping (`context.player_states`) at the cutoff. Store it in a new `models.db` table `rating_state(run_id, as_of, data_hash, blob)`, or as a file under `models/state/` indexed by such a table. Use stable player keys, not surrogate IDs.
- **Snapshots at fixed dates.** `run(snap_before=...)` already produces states at many dates in one pass. Save monthly (or weekly) snapshots so historical rankings and "prediction as of date D" need at most a few weeks of replay, not 70 years.
- **Incremental update.** Teach `run()` to start from a saved state (both the state and the context bookkeeping) and process only matches after it. A daily AcesAPI pull then costs seconds. Key validity on the hash of the matches before the cutoff: if an older match changes (a re-resolution, an override), fall back to the nearest valid snapshot.
- **Precomputed tables.** Markov grids are already built once per format; keep them module-level. Calibration alphas are stored per (run, quarter, tour, best-of).

### 6.3 Functions

1. **On-demand prediction:** `predict.py A B --tournament "Paris" --date 2026-11-20 [--round] [--live-score ... --serve-pts ...]`. The tournament gives surface, tour, best-of and deciding-set rule from the edition (or the tournament's last edition), so they never need typing. Output: win probability (raw and calibrated), each player's serve/return/surface/context breakdown, set and scoreline distribution, and in-play probability when a score is given.
2. **Rankings from all measured ratings:** one table per tour with overall (s + r), serve, return, surface-adjusted values, age-adjusted "expected now", deviation, matches, last played, and points-won equivalents vs the top-100 reference. Filter by tour, surface, minimum matches, activity; sort by any column; available at any date (from the snapshots).
3. **Cached rankings:** rankings computed once per snapshot and stored (`ranking_cache(run_id, as_of, tour, player_key, ...)`), so listing and diffing across dates (risers/fallers) is a query, not a replay.

### 6.4 HTML server

A small web app over the engine with these pages:

- **Predict:** two player boxes, tournament, date, live score.
- **Rankings:** sortable table and date picker.
- **Player:** rating history from the snapshots, serve vs return, surfaces.
- **Live board:** open Kalshi markets with model vs market, in-play prices, refreshed every few seconds, deviation highlighted.

The engine stays loaded in memory, so requests hit the cached state.

**Alerts:** when model minus market crosses a threshold on a fresh, resolved player, send a notification. The channel is open (see §8). The engine keeps the current rule that it only reads: it never places orders.

---

## 7. Proposed order of work

1. **Commit the current in-play and Kalshi work** (tests passing) so the reboot starts from a clean tree. Tidy the stray root files.
2. **Engine package + one prediction path** (R5): `engine.predict` reproduces run 790's scored probabilities exactly (test against stored predictions).
3. **State persistence + incremental replay**, with tests that incremental equals full replay.
4. **Scripts:** `predict.py`, `rankings.py`, `state.py`, on top of the cache.
5. **In-play evaluation harness** (R1) on slam pbp + MCP; then serve/return form split (R3) and in-play calibration (R4); joint re-tune (R2) as `points_v2`.
6. **Web app:** read-only pages first, then the live board.
7. **Kalshi alerts** and the feed-lag study from the logged `live_*.jsonl`.

---

## 8. Open decisions

- **Web stack.** Standard library (`http.server` + plain HTML/JS, no new dependency) or FastAPI/Flask (cleaner, but a new dependency)? Where will it run (home machine, VPS), and does it need a login?
- **Alert channel.** Desktop/phone push (e.g. ntfy), email, Discord/Telegram webhook?
- **"All measured rankings."** Does this mean every player at every level the data covers (ITF and Challenger included, not just tour level), every rating dimension (serve, return, surface), or every past date? This document assumes all three.
- **Challenger live markets.** Kalshi lists them, but Challenger results end 2026-05-25 in our data (AcesAPI covers no Challengers). Restoring Challenger coverage needs a new source.
- **Stats after 2026-05-25.** Live serve points from Kalshi's feed could be stored and used as a points-tier update for live-era matches. Worth building, given that ~40% of the edge is lost without stats.
