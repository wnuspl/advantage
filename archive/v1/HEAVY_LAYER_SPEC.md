# Advantage — Heavy Layer Spec (Style Vectors)

Hand-off doc for agents building the heavy layer. Read the whole thing before writing code.

## 1. Where this fits

Advantage predicts tennis match outcomes with three layers combined in logit space:

```
Light  : Glicko-2 on match results only        -> r_A - r_B (+ RD)
Heavy  : PCA style vectors from charted points  -> s_A, s_B        <- THIS DOC
Middle : interaction model on (s_A, s_B, surf)  -> delta
Output : logit P(A wins) = alpha * (r_A - r_B) + delta
```

Hard boundaries:
- **The light layer (Glicko-2) is done and must not be touched.** It never sees style data. The heavy layer never reads or writes ratings.
- **The heavy layer only produces style vectors.** It does not predict outcomes. Whether a vector is "good" is judged later by the middle layer.
- The middle layer (a Siamese antisymmetric MLP with a Glicko offset, benchmarked against an antisymmetric bilinear model) is **out of scope** here. Its needs set the output contract in §6.

## 2. Stack

- Python, on Windows. Must run with plain CPU and no GPU.
- **SQLite** is the project data store. Read with `pd.read_sql` / `sqlite3`.
- **polars** (preferred for time-windowed aggregation) or pandas.
- **scikit-learn**: `StandardScaler` -> `PCA` inside a `Pipeline`. (`FactorAnalysis` is a possible later swap; do not use it now.)
- **joblib** to persist fitted pipelines.
- numpy for shrinkage math.

## 3. Data source

- Tennis Abstract **Match Charting Project (MCP)**: shot-by-shot charted matches.
- Parse the MCP shot notation according to the MCP's own published documentation/key. Do not guess codes. If a code is ambiguous, log it and skip that point rather than inventing a meaning.
- Coverage skews toward top players and big matches. Many players have few or no charted matches; the design must handle that (see §5).

## 4. Features

Build **two separate feature sets per player**: one for **serve** and one for **return**. They get separate scalers, separate PCA fits and separate vectors. (The middle layer pairs A's serve style with B's return style and vice versa.)

Features are **rates/proportions with known denominators**. Store the numerator and denominator for each, because the shrinkage in §5 needs the counts.

Candidate serve features:
- first-serve in %
- serve direction split (wide / body / T), by court side (deuce/ad), first and second serve separately
- serve-plus-one patterns (e.g. forehand on the third shot)
- unreturned-serve rate
- serve-point rally-length distribution (e.g. share of points ending ≤3 shots, 4–6, 7–9, 10+)

Candidate return features:
- return-in %, against first and second serves separately
- return depth distribution
- return-point rally-length distribution

Shared/rally features (include in both sets or in a third set, and flag the choice in a PR for review):
- forehand/backhand ratio
- slice frequency
- net approach rate and success rate
- winner and unforced-error rates by shot type

Start with a modest set (~15–30 features per vector). It's easy to add more later; noisy features hurt PCA.

Optional segmentation to keep in mind: surface. **v1 computes features across all surfaces**, but keep surface on every aggregated row so per-surface features can be added later.

## 5. Shrinkage (before PCA)

Players with little charted data must not get extreme vectors.

For each rate feature, blend the player's raw rate toward the tour mean, weighted by the player's denominator:

```
shrunk = (n * raw + k * tour_mean) / (n + k)
```

- `n` = player's denominator for that feature (e.g. number of first serves)
- `k` = prior strength, estimated per feature by empirical Bayes (method of moments on the beta-binomial is fine) or as a tunable hyperparameter
- tour mean is computed **from the same point-in-time window** (see §7)

Players with **zero** charted matches get **no vector** (null). Do not impute; the middle layer treats null as delta = 0.

## 6. PCA and output contract

- Pipeline: `StandardScaler` -> `PCA`. Scale first so high-variance features don't dominate.
- Keep enough components for **~80% explained variance**, capped at **10**. Expect 5–10. Make the threshold configurable.
- Fix the **sign convention** of each component (e.g. flip so the component's largest-magnitude loading is positive) so vectors are comparable across refits.

Persist to SQLite. Proposed table:

```
style_vectors(
  player_id       TEXT,
  as_of_date      DATE,     -- vector uses only data strictly before this date
  side            TEXT,     -- 'serve' | 'return'
  fit_id          TEXT,     -- which fitted pipeline produced it
  n_matches       INTEGER,  -- charted matches behind it
  n_points        INTEGER,
  c1 REAL, c2 REAL, ..., cK REAL,
  PRIMARY KEY (player_id, as_of_date, side)
)
```

Also persist per fit: the joblib pipeline, explained-variance ratios, the loadings table (feature × component) and the fit date range.

Required API (the middle layer and backtest harness call this):

```python
get_style_vector(player_id: str, as_of: date, side: Literal["serve", "return"]) -> np.ndarray | None
```

`n_matches` / `n_points` must be retrievable alongside the vector; the middle layer may use them.

## 7. Point-in-time correctness (critical)

Leakage here would silently inflate every later backtest.

- A vector "as of date D" uses **only matches played strictly before D**: features, tour means, shrinkage priors, scaler and PCA fit.
- **Refit the pipeline on a schedule** (default: quarterly; configurable) using only data before the refit date. Between refits, new vectors are produced by `transform` with the most recent prior fit.
- Never fit the scaler or PCA on the full dataset for anything used in evaluation.
- Write a test that fails if any input row's match date is ≥ `as_of`.

## 8. Acceptance criteria

1. MCP parsing: unit tests on a handful of known points/matches, with expected feature counts checked by hand.
2. Leakage test from §7 passes.
3. Shrinkage test: a player with very few points lands close to the tour mean; a player with many points stays close to their raw rate.
4. Refits are deterministic: same data and config give identical vectors (set `random_state`, fixed sign convention).
5. Sanity report (markdown or notebook) for the latest fit: explained variance per component, top loadings per component with a plain-language guess at what each axis means (e.g. "big-serve / short-rally"), and nearest neighbours for a few well-known players. Humans review this. It is not an automated gate.
6. `get_style_vector` returns `None` for players with no charted data before `as_of`.

## 9. Out of scope for this phase

- Any change to the Glicko-2 light layer.
- The middle layer (bilinear / Siamese MLP), the point-level Markov conversion and outcome evaluation.
- The daily results ingest feed.
- Learned end-to-end embeddings replacing PCA (possible future phase, only if the PCA-fed middle layer works).

## 10. Open choices (decide in PR, flag for review)

- Shared rally features: in both serve and return sets, or a third vector?
- Prior strength `k`: empirical Bayes vs. tuned.
- Refit cadence: quarterly vs. monthly.
- Recency weighting of matches within the window (none in v1 unless trivially cheap).
