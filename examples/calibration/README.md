# Calibration examples

End-to-end demonstrations of the calibration diagnostics in `jurebes.benchmark.calibration`.

| file | purpose |
|---|---|
| [`compare_baselines.py`](compare_baselines.py) | Run `compare()` with `ece` and `brier` among the scoring metrics; rank baselines by calibration quality. |
| [`reliability_diagram.py`](reliability_diagram.py) | Side-by-side reliability diagrams for four baselines on the same data (requires `jurebes[bench-plot]`). |
| [`threshold_tuning.py`](threshold_tuning.py) | Sweep confidence thresholds against a service-level accuracy target; recommends a value for `OPM conf_high` / active-learning `hard_conf_max`. |

## Why these scripts exist

Jurebes auto-wraps non-probabilistic baselines (`LinearSVC`, hinge `SGDClassifier`, …) in `CalibratedClassifierCV`. The wrap is necessary for `predict_proba` to exist at all, but the *quality* of those calibrations varies by estimator family and dataset size.

- **Tree-family classifiers** (RandomForest, ExtraTrees, GradientBoosting) expose native `predict_proba` that is often poorly calibrated.
- **Logistic regression** is well-calibrated by construction (its loss IS the cross-entropy of the predicted probabilities).
- **Calibrated LinearSVC** depends on the calibration method (sigmoid / isotonic) and the available CV folds.

If you depend on confidence values downstream (active-learning gates, OPM thresholds, OOD reject rules), measure these. See [`docs/theory/calibration.md`](../../docs/theory/calibration.md).

## Run

```bash
python examples/calibration/compare_baselines.py        # ranks 6 baselines by ECE
python examples/calibration/threshold_tuning.py         # recommends a confidence cut
pip install 'jurebes[bench-plot]'
python examples/calibration/reliability_diagram.py      # saves reliability_diagram.png
```

## Related

- [`docs/theory/calibration.md`](../../docs/theory/calibration.md) — the conceptual side.
- [`docs/reference/active-learning.md`](../../docs/reference/active-learning.md) — `bucket_paraphrases` uses calibrated confidences for the HARD bucket boundary.
- [`docs/opm.md`](../../docs/opm.md) — `conf_high`/`conf_med`/`conf_low` are the production consumers of calibrated probabilities.
- [`examples/tutorials/104_calibration_diagnostics.py`](../tutorials/104_calibration_diagnostics.py) — minimal sigmoid-vs-isotonic comparison.
