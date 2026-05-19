# Confidence thresholds

`IntentResult.confidence` is the predicted probability of the chosen class. Choosing thresholds for accept / ask-clarification / reject is downstream of how well-calibrated those probabilities are.

## What the number means

For pipelines wrapped with `CalibratedClassifierCV` — every `*_svc`, `*_hinge`, `perceptron`, `ridge`, `passive_aggressive` baseline, plus any estimator configured with `calibrate="always"` — `confidence` approximates the empirical frequency of correctness at that score. A confidence of 0.8 should be right about 80 % of the time *on data drawn from the training distribution*.

For natively probabilistic baselines without an extra calibration wrap (`nb_*`, `logreg`, `sgd_log`, `sgd_modified_huber`, tree ensembles, `mlp_shallow`), the score is the model's raw `predict_proba` output. Tree-based proba is well-known to be poorly calibrated (overconfident at the extremes); wrap with `calibrate="always"` if you intend to threshold:

```python
from sklearn.ensemble import RandomForestClassifier
from jurebes import IntentClassifier
clf = IntentClassifier(RandomForestClassifier(), calibrate="always")
```

See [theory/calibration.md](../theory/calibration.md) for the underlying math and reliability diagrams.

## The OVOS three-level convention

The OVOS pipeline plugin maps three thresholds onto the `ConfidenceMatcherPipeline` buckets:

```json
{
  "jurebes": {
    "conf_high": 0.8,
    "conf_med":  0.6,
    "conf_low":  0.4
  }
}
```

- `match_high` accepts only when `confidence > conf_high`.
- `match_medium` accepts when `confidence > conf_med`.
- `match_low` accepts when `confidence > conf_low`.

OVOS runs higher-confidence pipelines first; only utterances no pipeline accepted at "high" fall through to "medium", and so on.

## Tuning on a held-out set

For an asymmetric cost (false-accept worse than false-reject, or vice versa) tune the threshold empirically.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from jurebes import IntentClassifier, BASELINES

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = IntentClassifier(BASELINES.build("logreg"))
for label in sorted(set(y_train)):
    clf.add_intent(label, [x for x, yy in zip(X_train, y_train) if yy == label])
clf.fit()

confidences = np.array([clf.predict(x).confidence for x in X_test])
correct     = np.array([clf.predict(x).intent == y for x, y in zip(X_test, y_test)])

precision, recall, thresholds = precision_recall_curve(correct, confidences)
# Find lowest threshold meeting a precision target.
target_precision = 0.95
mask = precision[:-1] >= target_precision
if mask.any():
    chosen = thresholds[mask].min()
    print(f"threshold {chosen:.3f} → precision ≥ {target_precision}")
```

The resulting `chosen` value is a sensible `conf_high` setting.

## Per-intent thresholds

Sometimes the cost is asymmetric per intent — a wrong `shutdown` is worse than a wrong `tell_joke`. The OVOS pipeline plugin uses a single global threshold; for per-intent gating, wrap your own dispatcher around `clf.predict_proba`:

```python
PER_INTENT_MIN = {"shutdown": 0.95, "tell_joke": 0.5}

def dispatch(utt):
    ranked = clf.predict_proba(utt)
    for r in ranked:
        thr = PER_INTENT_MIN.get(r.intent, 0.6)
        if r.confidence >= thr:
            return r
    return None
```

## Relation to out-of-domain detection

A confidence threshold can also serve as a coarse out-of-domain filter: if no class clears the bar, the utterance is rejected as "did not understand". This is weak compared to a dedicated OOD detector — see [guides/out-of-domain-detection.md](out-of-domain-detection.md) for the autoencoder-based approach.

---
- Back to [docs index](../index.md)
