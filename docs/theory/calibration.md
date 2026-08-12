# Calibration

A probabilistic classifier outputs $\hat{p}(y \mid x) \in [0, 1]$. It is *calibrated* when those probabilities match empirical frequencies: if the model outputs 0.8 on a set of inputs, roughly 80 % of them should actually be class $y$.

Calibration matters whenever the absolute probability matters — for thresholding, ensembling, or downstream cost-sensitive decision-making.

## What goes wrong without calibration

- **SVM** outputs decision-function values, not probabilities. `LinearSVC` does not expose `predict_proba` at all.
- **Tree ensembles** (Random Forest, Gradient Boosting) output probabilities, but they are systematically over-confident in the middle range and concentrate mass near 0 and 1.
- **Naive Bayes** outputs probabilities but the independence assumption distorts them — typically over-confident.
- **Logistic regression** is well-calibrated by construction *when the model is correctly specified*. On real text data it is mildly miscalibrated but the closest to calibrated of the linear family.

## Reliability diagrams

Bin predictions by confidence; plot the mean predicted probability vs the empirical accuracy in each bin. Perfect calibration is the diagonal.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

probs = np.array([clf.predict(x).confidence for x in X_test])
correct = np.array([clf.predict(x).intent == y for x, y in zip(X_test, y_test)])
prob_true, prob_pred = calibration_curve(correct, probs, n_bins=10, strategy="quantile")

plt.plot([0, 1], [0, 1], "--")
plt.plot(prob_pred, prob_true, marker="o")
plt.xlabel("mean predicted")
plt.ylabel("empirical")
```

Curves below the diagonal indicate over-confidence; above, under-confidence.

## Brier score

A scalar summary of calibration plus accuracy:

$$\mathrm{BS} = \frac{1}{N} \sum_i (\hat{p}_i - y_i)^2$$

Lower is better. Decomposes into reliability, resolution, and uncertainty terms.

## ECE: Expected Calibration Error

The average gap between mean predicted confidence and accuracy across confidence bins:

$$\mathrm{ECE} = \sum_b \frac{|B_b|}{N} \left| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \right|$$

Lower is better. ECE is sensitive to the binning scheme; report both equal-width and equal-mass binning.

## Post-hoc calibration

Two standard methods to fix a miscalibrated classifier:

### Platt scaling

Fit a sigmoid on top of the classifier's decision function:

$$\hat{p}_{\text{cal}}(y = 1 \mid s) = \sigma(a \cdot s + b)$$

Parameters $a, b$ are fitted by maximum likelihood on a held-out set. Originally proposed by Platt (1999) for SVMs.

### Isotonic regression

Fit a non-decreasing step function from the raw scores to empirical frequencies. More flexible than Platt; needs more data to fit reliably.

`sklearn.calibration.CalibratedClassifierCV(method="sigmoid")` does Platt; `method="isotonic"` does isotonic.

## jurebes' policy

`IntentClassifier.__init__` accepts a `calibrate` argument:

| value | behaviour |
| --- | --- |
| `"if_missing"` (default, alias `True`) | Wrap with `CalibratedClassifierCV(cv=3)` only if the estimator has no `predict_proba`. |
| `"always"` | Always wrap, even when `predict_proba` exists. Useful for tree ensembles and kNN. |
| `False` | Never wrap. Raises `ValueError` if the estimator has no `predict_proba`. |

`CalibratedClassifierCV(cv=3)` uses 3-fold internal CV: splits the training data into 3 folds, fits the base estimator on each train-fold, fits a sigmoid on the held-out fold, then averages.

### When to override the default

- **Tree models** for thresholding: `calibrate="always"`.

```python
from sklearn.ensemble import RandomForestClassifier
IntentClassifier(RandomForestClassifier(), calibrate="always")
```

- **Tiny datasets** (<3 samples per class): `calibrate=False` to avoid `cv=3` failing. Pair with a natively probabilistic estimator (`LogisticRegression`, `MultinomialNB`).

## Verifying calibration

Run a 5-fold CV with `log_loss` as a scoring metric:

```python
from jurebes.benchmark import cross_validate
r = cross_validate("logreg", X, y, k=5, scoring=("f1_macro", "log_loss"))
print(r.extra_scores["log_loss"])
```

Lower log-loss correlates with better calibration on the same dataset; cross-baseline comparisons are messier because the optimum log-loss depends on the class distribution.

For real diagnosis use reliability diagrams + ECE on a held-out set.

## Measuring calibration in jurebes

`jurebes.benchmark.calibration` exposes three numbers and one diagnostic plot:

```python
from jurebes.benchmark import compare

# add ece and brier alongside any other scoring metrics
result = compare(["logreg", "linear_svc", "nb_multinomial"],
                 X, y, k=5, scoring=("f1_macro", "ece", "brier"))
```

Or call them directly on (y_true, y_pred_proba, classes):

```python
from jurebes.benchmark import (
    expected_calibration_error,
    brier_score,
    reliability_curve,
)

proba = clf.predict_proba_matrix(X_test)  # n × len(classes)
ece = expected_calibration_error(y_test, proba, classes, n_bins=15)
brier = brier_score(y_test, proba, classes)
centers, acc, conf, counts = reliability_curve(y_test, proba, classes, n_bins=15)
```

A perfect classifier has `ece=0` and `brier=0`. ECE close to zero means predicted confidence matches empirical accuracy *on average*; Brier penalises both miscalibration and incorrect prediction.

Render the reliability diagram (requires `jurebes[bench-plot]`):

```python
from jurebes.benchmark.calibration import plot_reliability
plot_reliability(y_test, proba, classes, n_bins=15)
plt.savefig("reliability.png")
```

## Why this matters for `IntentClassifier(calibrate="if_missing")`

Jurebes auto-wraps non-probabilistic estimators (LinearSVC, hinge SGD, etc.) in `CalibratedClassifierCV`. The wrap is necessary for `predict_proba` to exist at all — but the *quality* of those calibrations varies by estimator family and dataset size. Tree-family classifiers (RandomForest, ExtraTrees) expose native `predict_proba` that is often poorly calibrated (frequency of leaf labels, frequently spiking at 0/1). Use ECE / Brier on a held-out set to verify the calibration is good enough for downstream consumers (confidence-threshold gates, active-learning loops, OPM intent matchers).

---
- Back to [docs index](../index.md)
