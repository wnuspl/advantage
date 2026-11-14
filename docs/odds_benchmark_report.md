# Odds benchmark: how good the model has to be

Status (2026-11-14): **the measured comparison against closing odds is not done yet.** tennis-data.co.uk returns a Cloudflare 403 from both networks tried, so no odds file has been downloaded. This report covers what can be worked out without the files:

1. the betting math, which is exact;
2. a simulation of when a model makes money against a closing market;
3. targets to measure once the files arrive.

Every number below that depends on the market's own accuracy treats that accuracy as an assumption and varies it. None of it is a measurement.

Code is ready for when the data arrives. It has been checked by hand but has no unit tests yet:
- `src/tennis/xlsx_lite.py`: reads `.xlsx` with the standard library only.
- `src/tennis/odds.py`: parses tennis-data rows, links them to backbone matches with every row accounted for, removes the bookmaker margin (multiplicative, additive, power or Shin), fits a blend of model and market, runs a betting backtest and simulates.

## Where we stand

| | tour-level log loss, test 2022-26 | Brier | accuracy |
|---|---|---|---|
| Surface Elo (run 132) | 0.6076 | 0.2106 | 66.2% |
| Glicko-2 (run 187) | 0.6051 | 0.2097 | 66.1% |
| Glicko-2 + alpha calibration (run 203) | 0.6045 | 0.2095 | 66.1% |
| Siamese matchup (run 207) | 0.6042 | 0.2094 | 65.9% |
| Pinnacle closing odds | **not measured yet** | | |

The run 187 predictions alone hint at how much is learnable. If outcomes were drawn from run 187's own probabilities, its log loss would be 0.589. Its actual 0.605 means its probabilities carry noise. The closing market is expected to sit below 0.605. How far below is the first thing to measure. The simulation covers markets at 0.560, 0.575 and 0.590, and the conclusions below hold across all three.

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

1. **Measure the gap.** Find the log loss, Brier and calibration of Pinnacle's closing odds on the same 21k tour matches. Our expectation, to be confirmed: the market is 0.02 to 0.04 ahead of run 187.
2. **Halve the gap.** Get within about 0.01 of the closing line. That is a strong result for a model built only on results and charting data, and a good research goal on its own.
3. **Add information to the market.** Get a walk-forward blend weight `b > 0` with z > 2, and a held-out `gain > 0` in most quarterly blocks. This is the honest definition of "beating the market". It does not require a better standalone log loss.
4. **Break even against Pinnacle's closing line.** Held-out gain of at least about **0.0004 nats per match**, stable across blocks, confirmed by a flat-stake backtest of 10,000 or more bets. Clearing it against average bookmakers needs about 0.0013. Odds better than the close (opening lines, best price across bookmakers) lower the bar but need verified capture times.
5. **Beat the close outright** (standalone log loss below Pinnacle's). Rare even for professional syndicates. Not a sensible target.

Reasonable target: goals 1 to 3 this phase, with goal 4 as the stretch bar. Treat any backtest ROI above about 5% against the close as a bug (leakage, stale odds or a matching error) until proven otherwise.

## To run once the files are available

1. Put the ATP and WTA yearly `.xlsx` files for 2019-2026, plus `notes.txt`, in their own folder under `data/raw/`.
2. Confirm from `notes.txt` which columns are *closing* odds before calling anything "closing". Odds are never a feature.
3. Link the rows with `odds.link`. Report the linked share and the reasons for rows left unlinked.
4. Score the market (devig by power and Shin as well as multiplicative) next to runs 187, 203 and 207 on identical matches.
5. Fit the blend walk-forward by year and report `b`, its z-score and the held-out gain.
6. Backtest with `odds.backtest` against PS (Pinnacle), Avg and Max prices.
