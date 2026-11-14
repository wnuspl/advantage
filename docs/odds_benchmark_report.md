# Odds benchmark: the closing market vs our models

Measured 2026-11-14 with `python scripts/odds_benchmark.py` (about 40 s; output also in `models/odds/benchmark.json`).

**Source:** Valuebetennis, *Tennis results and odds since 2021*, https://www.valuebetennis.com/en/donnees.htm, CC BY 4.0 (credit with a link). It has Pinnacle opening and closing odds, where closing is "last odds recorded before the match", captured live since 2021. The six season CSVs are pinned in `data/raw/valuebetennis-20261114/` (gitignored). Their sha256 values:

| file | sha256 |
|---|---|
| valuebetennis-matchs-2021.csv | ea30ae06fab7734ada518b974aacf55bea3cde41ff569f1135c9d05f79cc9aee |
| valuebetennis-matchs-2022.csv | e4bab3987820ecdded8895648ac7bcbdac2539410fc5d4a9901febb6f95b114c |
| valuebetennis-matchs-2023.csv | fa45622d678b17176c8b950d09872538dd765a684b16e1910a3ede4e75664bfb |
| valuebetennis-matchs-2024.csv | 09437d8208c937d9efb82af232edf6c6f8520be7ec2d6773dde71886b79c68ef |
| valuebetennis-matchs-2025.csv | 46bd9c25673228aeb279f653b508a6e1c8f0b5291645614de6912a581f217d21 |
| valuebetennis-matchs-2026.csv | ad16d85e9af1b0bcda91bff0834c9398e1dbf6397dce39f1558082e317f64652 |

tennis-data.co.uk blocked both networks we tried (Cloudflare 403). Its reader (`xlsx_lite.py`, `odds.read_td_file`) is kept and tested for when it becomes reachable.

The site sells betting tips. Only its raw odds are used here, never its predictions or published track record. Odds are a benchmark and never a feature.

## Results

### Linking

- **Rows:** 476,336 rows read. 47,741 are in tour-level categories ("Main tour", "Masters", "Grand Chelem" or blank).
- **Linked:** 24,408 linked to a backbone match by `odds.link` with the full-name scorer.
- **Not linked** (each row has its reason):
  - 23,053 with no candidate. These are mostly qualifying (round codes 1-3), juniors, Hopman Cup and matches after our 2026-05-25 cutoff.
  - 154 ambiguous.
  - 69 where the sources disagree on the winner. All of these are retirements, where Valuebetennis records the retiring player as winner.
  - 57 duplicate claims.
- **Spot check:** 10 of 10 random links were correct.
- **Coverage:** **85.5% of our tour-level 2022-26 matches (18,382 of 21,505) have Pinnacle closing odds.** The comparison below uses the 18,268 matches that every compared run also predicted.

### The market vs our models (identical matches, tour level, 2022-01 to 2026-06)

Market = Pinnacle close, de-vigged by the power method. Δ = log-loss difference vs the market, with standard error. Pinnacle's median margin is 2.85%.

| model | log loss | Δ vs market | SE | Brier | accuracy | ECE |
|---|---|---|---|---|---|---|
| **Pinnacle close** (power) | **0.5864** | 0 | | 0.2019 | 68.0% | 0.011 |
| Pinnacle close (multiplicative / Shin) | 0.5866 / 0.5864 | +0.0001 / -0.0000 | | 0.2019 | 68.0% | 0.010 |
| Pinnacle open (power, where available) | 0.5904 | +0.0040 | 0.0006 | 0.2035 | 67.7% | 0.013 |
| Surface Elo (run 132) | 0.6098 | +0.0234 | 0.0014 | 0.2116 | 65.9% | 0.034 |
| Glicko-2 (run 187) | 0.6072 | +0.0208 | 0.0013 | 0.2107 | 65.8% | 0.028 |
| Glicko-2 calibrated (run 203) | 0.6065 | +0.0201 | 0.0013 | 0.2104 | 65.8% | 0.027 |
| Siamese matchup (run 207) | 0.6061 | +0.0197 | 0.0013 | 0.2102 | 65.7% | 0.026 |

- **Gap to the closing line: 0.020 nats** (0.023 ATP, 0.017 WTA). The gap is steady every year: 0.016 to 0.024.
- **The < 0.60 target is reachable.** The closing market sits at 0.586, so 0.60 is about 70% of the way from our models to the market. The market itself scored **0.6006 in 2025**, so 2025 was a hard year for everyone. Always compare against the market on the same matches and years, never against a fixed number.
- On these matches our models score about 0.002 worse than on the full set (0.6065 vs 0.6045 for run 203). The 15% of matches without odds are easier ones.
- The market is close to calibrated (ECE 0.011; refitting its slope gives a = 0.97, no gain). Our models have 2-3 times its calibration error.
- Opening lines are 0.004 worse than closing lines. That is the information that arrives between the line opening and the start of the match.

### Do we add anything to the market?

Blend fitted walk-forward by year (fit on earlier years only, score the next year, 2023-26): `logit P = a*logit(market) + b*logit(model)`.

| model | in-sample b (z) | held-out gain over market, nats/match |
|---|---|---|
| Surface Elo | -0.18 (-4.5) | +0.00048 ± 0.00023 |
| Glicko-2 raw | -0.21 (-4.6) | +0.00050 ± 0.00026 |
| Glicko-2 calibrated | -0.22 (-4.6) | +0.00049 ± 0.00026 |
| Siamese | -0.20 (-4.2) | +0.00040 ± 0.00024 |

**The weight b on our models is negative.** None of these models adds information in the ordinary sense. When Glicko rates a player above the market, the player wins *less* than even the market says:

| model minus market (logit) | n | market p | model p | actual |
|---|---|---|---|---|
| below -0.5 | 1,099 | 0.688 | 0.566 | 0.684 |
| -0.5 to -0.2 | 2,802 | 0.590 | 0.524 | 0.606 |
| -0.2 to +0.2 | 7,349 | 0.515 | 0.517 | 0.512 |
| +0.2 to +0.5 | 4,501 | 0.453 | 0.521 | 0.431 |
| above +0.5 | 2,631 | 0.357 | 0.494 | 0.328 |

Reading: the closing price leans slightly toward what result-based ratings say, and the truth sits a little further from the ratings than the market does. The small held-out gain (about 2 SE) comes entirely from that fade, not from sharpening the market (sharpening gains 0.0000).

**Betting:**
- Betting our models directly at closing prices loses **-8% ROI** (±1.1-1.2%) on about 15,500 bets. That is much worse than the margin. The models disagree most on underdogs (average odds 3.4-3.7), which is exactly where they are wrong.
- The walk-forward "fade" blend makes **+2.0% ± 1.8%** on 4,171 bets (threshold 0) and +6.6% ± 3.2% on 1,740 bets (threshold 0.02).
- The fade blend is not significant and not stable: 2023 +0.2%, 2024 +7.2%, 2025 0.0%, 2026 -1.9%.
- **There is no edge.** The fade is a lead to recheck on the next season, not a result.

### Verdict

1. **The gap to beat:** 0.020 log loss to the Pinnacle close, on 18k identical tour-level matches.
2. **Target < 0.60:** reachable in principle; the market is at 0.586.
3. **Adding information:** the current models add nothing usable. Their disagreements with the market point the wrong way.
4. **Next test:** the points-based ratings (milestone 8). They use serve/return points won, which the market may underweight. The test is a **positive** b with z > 2 and a held-out gain of at least about 0.0004 nats/match. Rerun with `python scripts/odds_benchmark.py --runs <new run>,203`.

---

The sections below are the betting math and simulation written before the data arrived. Their market log losses (0.560 to 0.590) bracket the measured 0.586.

## 1. The exact math

### One bet

Decimal odds `o` pay `o - 1` on a win and lose 1 otherwise. The expected profit per unit staked is `p*o - 1`. A bet is worth making only when your probability `p` is above `1/o`.

A bookmaker builds its margin `m` into the odds: `1/o1 + 1/o2 = 1 + m`. Spread proportionally, the price of a side with fair probability `pi` becomes `pi*(1+m)`. You therefore need `p > pi*(1+m)`, which means beating the fair probability by `pi*m`.

| fair probability | Exchange, about 2% | Pinnacle, about 2.5% | Average bookmaker, about 5% | Soft bookmaker, about 7% |
|---|---|---|---|---|
| 0.30 | 0.306 | 0.308 | 0.315 | 0.321 |
| 0.50 | 0.510 | 0.513 | 0.525 | 0.535 |
| 0.70 | 0.714 | 0.718 | 0.735 | 0.749 |
| 0.85 | 0.867 | 0.871 | 0.893 | 0.910 |

Against Pinnacle you must be right by about 1.3 points on a 50/50 match. Against an average bookmaker you must be right by 2.5 points. In both cases you must be right *in the direction of your disagreement with the market*. Being 1.3 points off at random is easy. Being 1.3 points closer to the truth than the closing line is the whole game.

### Betting on every match

Suppose you spread your whole bankroll across both sides in proportion to your probabilities (Kelly betting with no option to sit out). Then your expected log growth per match is exactly:

    growth = LL_market - LL_model - ln(1 + m)

Here `LL_market` is the log loss of the market's fair probabilities. Against Pinnacle (m about 0.025) you would need a log loss about **0.025 below the market's**. That is unrealistic, which is why a profitable bettor only bets selectively.

### How many bets before you know

The standard error of the return on investment (ROI) on flat stakes at odds near 1.9 is about 0.95 per bet. Confirming a true +2% ROI at two standard errors takes about **9,000 bets**. A +1% ROI takes about **36,000 bets**, and +2% at odds of 2.5 takes about 15,000. The walk-forward test period has about 21,000 tour-level matches. One season of results proves nothing either way.

## 2. When does a model make money? (simulation)

Setup (`odds.simulate`):
- Each match has a true win probability.
- The market knows most of it and quotes a calibrated price. The part it doesn't know has a standard deviation of 0.3 or 0.6 on the logit scale.
- Market probabilities are resampled from the shape of run 187's predictions, scaled so the market's log loss hits each target.
- The model sees what the market sees, plus a share `kappa` of what the market misses, plus its own noise. It is then recalibrated.
- The model bets on positive expected value at the market price with margin `m`.
- 600,000 matches per setting.

### Result 1: a lower log loss than the market is neither necessary nor sufficient

- A model with **no information the market lacks** (`kappa = 0`) loses about the margin on every bet: -2% to -3% ROI at a 2.5% margin. That holds whatever its log loss, even when that log loss matches the market's. Nothing it knows is new.
- A model whose log loss is **0.02 to 0.03 worse** than the market's still makes a profit if part of its signal is independent of the market.

The number that decides profit is how much the model adds to the market, not how good it is alone.

### Result 2: the number to target

Fit the blend `logit P = a*logit(market) + b*logit(model)` and call `gain` the drop in log loss over the market alone. Then:
- `b > 0` means the model knows something the closing price doesn't.
- `gain` measures how much.

Break-even, the same with or without a 3% minimum-edge filter, and nearly the same for all three market log losses:

| bookmaker margin | gain needed to break even (nats per match) |
|---|---|
| 0 (fair market) | any positive gain |
| 2.5% (Pinnacle close) | about **0.0004** |
| 5% (average bookmaker) | about **0.0012 to 0.0014** |

Where that leads, at a 2.5% margin, hidden-information sd 0.6:

| model vs market log loss | blend weight b | gain | flat-stake ROI | quarter-Kelly growth |
|---|---|---|---|---|
| +0.025 (much worse) | 0.00 | 0.0000 | -3.0% | negative |
| +0.020 | 0.11 | 0.0003 | -0.9% | negative |
| +0.015 | 0.24 | 0.0013 | +1.6% | about 0 |
| +0.006 | 0.45 | 0.0053 | +6.9% | positive |
| -0.004 (beats market) | 0.61 | 0.0079 | +9.5% | positive |

Treat these ROIs as an **upper bound**. The simulation assumes:
- the model's extra information is exactly what the market lacks;
- the closing price is perfectly calibrated;
- you always get the closing price;
- your account is never limited.

Real edges against a closing line are usually a fraction of these figures. Quarter-Kelly growth lags flat-stake ROI because a noisy model bets on most matches.

**One sizing comparison:** the whole matchup layer improved on Glicko by 0.0006 nats. That is the same size as the Pinnacle break-even gain. The break-even amount is tiny, but it must be information the market *doesn't already price*. Glicko-style form and serve/return style are both public, so they are probably priced already.

## 3. What is possible, and good goals

Ordered from realistic to hard:

1. **Measure the gap.** Done: the closing market is 0.020 ahead of our best models (0.5864 vs 0.6061 on 18,268 identical matches).
2. **Halve the gap.** Get within about 0.01 of the closing line, meaning a log loss of about 0.596 on these matches. That is a strong result for a model built only on results and charting data, and a good research goal on its own.
3. **Add information to the market.** Get a walk-forward blend weight `b > 0` with z > 2, and a held-out `gain > 0` in most quarterly blocks. This is the honest definition of "beating the market". It does not require a better standalone log loss.
4. **Break even against Pinnacle's closing line.** Held-out gain of at least about **0.0004 nats per match**, stable across blocks, confirmed by a flat-stake backtest of 10,000 or more bets. Clearing it against average bookmakers needs about 0.0013. Odds better than the close (opening lines, best price across bookmakers) lower the bar but need verified capture times.
5. **Beat the close outright** (standalone log loss below Pinnacle's). Rare even for professional syndicates. Not a sensible target.

Reasonable target: goals 1 to 3 this phase, with goal 4 as the stretch bar. Treat any backtest ROI above about 5% against the close as a bug (leakage, stale odds or a matching error) until proven otherwise.

## Rerunning

```bash
python scripts/odds_benchmark.py --runs 132,187,203,207
```

- Add a new model's run ID to `--runs`. The script scores every run on the same matches and refits the blend for each one.
- New seasons: download a new pinned folder (raw files are immutable) and pass `--raw`.
- Not yet done: loading the odds into `tennis.db`'s `odds` table through the build (snapshot registration, `load_run` accounting). For now the link is recomputed on each run, which takes about 30 seconds.
