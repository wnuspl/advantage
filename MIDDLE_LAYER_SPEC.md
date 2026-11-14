# Advantage — Middle Layer Spec (Matchup / Interaction Model)

Hand-off doc for agents building the middle layer. Read the whole thing, and `HEAVY_LAYER_SPEC.md`, before writing code.

## 1. Where this fits

```
Light  : Glicko-2 on match results only        -> r_A - r_B (+ RD)
Heavy  : PCA style vectors from charted points  -> s_A, s_B
Middle : interaction model on (s_A, s_B, surf)  -> delta            <- THIS DOC
Output : logit P(A wins) = alpha * (r_A - r_B) + delta
```

The middle layer learns **only the residual**: how a specific pairing of styles departs from what the ratings predict. Glicko-2 already captures each player's average strength, so the middle layer must not relearn it.

Hard boundaries:
- **Never modify Glicko-2.** The middle layer reads pre-match ratings as frozen inputs. Gradients never flow into the light layer.
- **Never modify the heavy layer.** Style vectors come only from the heavy-layer API (`get_style_vectors` for training, `get_style_vector` for debugging and live use), never from its tables directly. See `HEAVY_LAYER_SPEC.md` §6.5.
- **One fit per run.** PCA axes change at every refit, so all vectors in one walk-forward fold (train, val and test) come from the same fit per side: `fit_ids = {side: latest_fit(side, before=train_cutoff) for side in SIDES}`. Never mix fits within a fold.
- If either player has no vector (no charted data), **delta = 0** and the prediction is Glicko-only.

## 2. Stack

- Python, on Windows, CPU only.
- **PyTorch** for the bilinear and Siamese models. One framework for both, so the comparison is fair.
- **LightGBM** for the gradient-boosted benchmark.
- **scikit-learn** for metrics (`log_loss`, `brier_score_loss`, `calibration_curve`) and the alpha calibration.
- **Optuna** for hyperparameter tuning, inside the walk-forward harness only.
- SQLite for data. Persist trained models with `torch.save` / joblib, keyed by training-window end date.

## 3. Inputs (per match, A vs B)

| Input | Source | Notes |
|---|---|---|
| `glicko_diff` | light layer | Pre-match ratings only. Convert to a logit via the Glicko-2 expected-score formula (using RDs). |
| `s_A_serve, s_A_return, s_A_rally, s_B_serve, s_B_return, s_B_rally` | `get_style_vectors(requests, fit_ids)` with `as_of = match_date` | Strictly pre-match. `fit_ids` fixed per fold (§1). |
| `n_matches_A, n_matches_B` | heavy layer | Optional features; off in v1. |
| `surface` | results data | One-hot (hard / clay / grass; fold carpet into hard or drop). |
| `best_of` | results data | 3 or 5. Needed for the point-level path. |
| Label | results feed | Match: did A win. Point-level: A's serve points won / played, and B's. |

**Training set = every match where both players have a non-null style vector as of match date.** The training match itself does **not** need to be charted. Outcomes and serve stats come from the results feed. This is much larger than the charted set; use it.

## 4. The baseline offset (alpha)

1. Before any interaction model, fit a **single scalar alpha** by logistic regression of outcome on `glicko_logit` over the training window. This is the Glicko-only model and the benchmark everything must beat.
2. **Freeze alpha** while training the interaction models. The interaction model's output is added to `alpha * glicko_logit` as a fixed offset.
3. Initialize every interaction model's output layer **at or near zero**, so training starts as exactly Glicko-only.

## 5. Models (build in this order)

All models output `delta` in logit space and must be **antisymmetric**: swapping A and B flips the sign of delta exactly, so P(A beats B) = 1 − P(B beats A). Enforce this by construction, not by data. Add a unit test that checks it on random inputs.

Pairing rule: A's **serve** interacts with B's **return**, A's **return** with B's **serve**, and A's **rally** with B's **rally**.

### 5.1 Antisymmetric bilinear (baseline interaction)

```
delta = ( s_A_serve^T W_sr s_B_return  -  s_B_serve^T W_sr s_A_return )   # serve vs return
      + s_A_rally^T W_rr s_B_rally                                         # rally vs rally
```

- `W_sr` is a general low-rank matrix (U Vᵀ, rank 2–4). The subtraction makes the serve/return term antisymmetric.
- `W_rr` must itself be antisymmetric: W_rr = U Vᵀ − V Uᵀ (rank 2–4). Same side on both players, so this is what makes the term flip sign when the players swap.
- Strong weight decay. Surface-specific W variants are a later option.
- This is the "does style carry signal at all?" test. Ablate each term (serve/return only, rally only) to see where the signal is.

### 5.2 Siamese antisymmetric MLP (main experiment)

```python
class SiameseMatchup(nn.Module):
    def __init__(self, k_total, hidden=32, n_surf=3):
        # k_total = K_serve + K_return + K_rally for the fold's fits
        super().__init__()
        self.g = nn.Sequential(                    # shared encoder, both players
            nn.Linear(k_total, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.h = nn.Linear(2 * hidden + n_surf, 1)  # init weights ~0

    def forward(self, x_a, x_b, surf, offset):
        # x_* = concat(serve_vec, return_vec, rally_vec) for that player, always in this order
        e_a, e_b = self.g(x_a), self.g(x_b)
        d = self.h(torch.cat([e_a, e_b, surf], -1)) - self.h(torch.cat([e_b, e_a, surf], -1))
        return offset + d.squeeze(-1)               # logits
```

- K can differ per side and per fit, so build `k_total` from the fold's `fit_ids`. A model is only valid for the fits it was trained with; store the `fit_ids` with the saved model.
- 1–2 hidden layers, width 16–32, dropout, `AdamW` weight decay, early stopping on the time-based validation fold.
- Loss: `BCEWithLogitsLoss` (match level) or the binomial loss in §6 (point level).

### 5.3 LightGBM (benchmark)

Features: `glicko_logit`, all six vectors, the elementwise products of the paired vectors (A serve × B return, A return × B serve, A rally × B rally), surface. Train on **both orderings of every match** (A,B,y) and (B,A,1−y). At predict time, average p(A,B) and 1 − p(B,A) so the output is symmetric.

## 6. Two training targets (match level first, then point level)

### Phase 1: match level
Label = A won. Gets the pipeline and evaluation working end to end.

### Phase 2: point level (once Phase 1 works)
1. **Baseline serve-point probability.** Fit a mapping from `glicko_diff` to A's expected serve-point win rate against B (logistic regression on historical serve-point percentages, surface as a feature). Frozen, like alpha. This uses only the single Glicko-2 rating; **no separate serve/return ratings.**
2. **Interaction model** outputs a logit adjustment to that baseline, for A serving vs B returning (and separately B serving vs A returning; antisymmetry here means the same network with players swapped). Inputs: the server's serve vector, the returner's return vector, and **both players' rally vectors** (rallies happen on every point).
3. **Loss:** binomial log-likelihood on (serve points won, serve points played) for each server in each match. **Weight each match equally**, not each point, so long matches don't dominate.
4. **Match probability:** convert (p_A_serve, p_B_serve) to P(A wins) with the standard hierarchical Markov model (point → game → tiebreak → set → match). Must handle best-of-3 vs 5 and final-set tiebreak rules by tournament/era; get the rule from the results data, don't assume it.
5. Evaluate match-level metrics on the converted probabilities, so Phase 2 compares directly with Phase 1.

## 7. Walk-forward harness (build this first)

One harness that every model plugs into:

```python
class Model(Protocol):
    def fit(self, train: DataFrame, val: DataFrame) -> None: ...
    def predict_proba(self, test: DataFrame) -> np.ndarray: ...   # P(A wins)
```

- Rolling time windows: train on everything before T, validate on the next block, test on the block after; step T forward (e.g. quarterly). Configurable.
- **Splits are by time, never random.** For point-level training, never split points from one match across folds.
- Every model, including Glicko-only, runs through identical splits.
- All inputs are point-in-time: ratings, vectors, alpha and the baseline mapping use only data before the match. Add a leakage test that fails if any feature is built from data dated ≥ match date.

## 8. Evaluation

Report per model, per test block and pooled:
- log loss (primary), Brier score, accuracy
- calibration curve and expected calibration error
- **the same metrics on the subset where both players have ≥ N charted matches** (N configurable, default 10). This is where the middle layer should show value.
- the same metrics by surface
- if historical closing odds become available: log loss against the de-vigged market. Beating the market is the real goal.

A model is "better" only if it beats Glicko-only on pooled walk-forward log loss **and** isn't worse on calibration.

## 9. Acceptance criteria

1. Harness runs Glicko-only, bilinear, Siamese and LightGBM end to end on match level.
2. Antisymmetry unit test passes for bilinear and Siamese; LightGBM symmetrization is tested.
3. Leakage test passes.
4. Null-vector players get exactly the Glicko-only prediction (test it).
5. With the zero-initialized output layer and no training, the Siamese model reproduces Glicko-only predictions exactly.
6. Fit-consistency test: within every fold, all vectors come from one `fit_id` per side, and the saved model records those `fit_ids`.
7. Ablation in the report: serve/return terms only vs rally only vs all three.
8. A results report (markdown) with the metrics in §8 and a short written comparison.

## 10. Retraining policy

| Event | Action |
|---|---|
| Daily Glicko-2 rating updates | None. Ratings are a frozen input; new values flow through. |
| Glicko-2 **configuration** change (τ, initial rating/RD, scale) | Recalibrate alpha (and the point-level baseline mapping), then retrain the middle layer. |
| New weekly heavy-layer snapshots | None. Project through the current fits. |
| Heavy-layer **refit** (new `fit_id`s) | Retrain the middle layer on the new fits. |

- A trained middle-layer model is a **bundle**: model weights + the `fit_ids` it was trained with + alpha. Store and load them together.
- Deploy the new fits and the retrained model **together, in one switch**. Until the retrain passes its checks, keep serving the previous bundle. It stays valid because new snapshots can be projected through the old fits.
- Do not try to align new PCA axes to old ones (e.g. Procrustes). Retraining is cheap; just retrain.

## 11. Out of scope

- Any change to Glicko-2 or the heavy layer.
- Using RD to weight delta (possible later; off in v1).
- End-to-end learned embeddings replacing PCA (only if the PCA-fed Siamese model clearly beats bilinear).
- Live/daily prediction serving.

## 12. Open choices (decide in PR, flag for review)

- Bilinear rank and whether W varies by surface.
- Whether `n_matches` (data confidence) is an input to the Siamese model.
- Charted-match threshold N for the "well-charted" evaluation subset.
- Walk-forward block size and step.
