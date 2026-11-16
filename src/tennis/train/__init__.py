"""Training harness for points models (milestone A2): replay, score, fit context, record.

    data.py      Prepared: the match table + canonical player order + per-match best-of
    scoring.py   log loss, Brier, calibration bins, ECE, recalibration slope, paired
                 differences with SE and quarters won (from v1 evaluate.py / composite_eval)
    runs.py      replay a parameter setting, score val (raw + calibrated), record model_run +
                 run_metric rows; sweeps; the one-time test scoring; saved predictions
    context_fit.py  iterated residual WLS for the cs_*/cr_* context coefficients
    compare.py   two or more runs on identical matches, raw and calibrated, paired

CLI: scripts/train.py. Never choose on the test block; test is scored once per finished model.
"""
