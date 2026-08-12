# Out-of-domain detection

An intent classifier always returns *some* intent for any input. To reject utterances unrelated to the training inventory, layer an out-of-domain (OOD) detector on top.

The CLINC150 benchmark (`oos` label) is the canonical OOD evaluation. Four scoring strategies were compared on it; results inform the recommendation below.

| OOD scoring method | ROC AUC | TPR @ FPR=0.05 | TPR @ FPR=0.10 |
| --- | ---: | ---: | ---: |
| `bm25_logreg` top-1 confidence (1 − conf) | **0.9254** | 0.6290 | 0.7910 |
| `bm25_logreg` top1 − top2 margin | 0.9096 | 0.4690 | 0.7290 |
| Autoencoder reconstruction error | 0.6004 | 0.0810 | 0.1600 |
| One-class SVM (rbf) on TF-IDF | 0.5529 | 0.0840 | 0.1550 |

See `examples/trained_models/reports/clinc_ood_alternatives.md` for the full bench.

## Recommended approach: calibrated confidence

A well-calibrated multi-class classifier already exposes the OOD signal directly: when the top class probability is low, the input is unlike anything in the training distribution. `bm25_logreg` calibrated with `CalibratedClassifierCV` is the strongest detector measured here.

```python
from collections import defaultdict
from jurebes import IntentClassifier
from jurebes.baselines import BASELINES

clf = IntentClassifier(BASELINES.build("bm25_logreg"))
grouped = defaultdict(list)
for x, lbl in zip(X_train, y_train):
    grouped[lbl].append(x)
for lbl, samples in grouped.items():
    clf.add_intent(lbl, samples)
clf.fit()

def ood_score(utt: str) -> float:
    proba = clf.estimator.predict_proba([utt])[0]
    return float(1.0 - proba.max())          # higher = more OOD
```

## Choosing the threshold

Hold out a labelled mix of in-domain and OOD utterances and pick the threshold that hits your target false-positive rate:

```python
import numpy as np
from sklearn.metrics import roc_curve

scores = np.array([ood_score(x) for x in X_eval])
labels = np.array([0 if y != "oos" else 1 for y in y_eval])
fpr, tpr, thr = roc_curve(labels, scores)

target_fpr = 0.05
idx = int(np.argmax(fpr >= target_fpr))
threshold = float(thr[idx])
```

Top-1 minus top-2 *margin* is a near-equivalent variant. `1 − top-1` had a slight edge in the bench.

## Combining with the intent classifier

```python
def predict_with_ood(utt: str, threshold: float):
    proba = clf.estimator.predict_proba([utt])[0]
    if (1.0 - proba.max()) > threshold:
        return None                          # reject as OOD
    return clf.estimator.classes_[proba.argmax()]
```

Hard gate (reject) vs soft signal (surface the score alongside the prediction and decide downstream) are both valid — pick based on whether downstream code can handle `None`.

## Alternative: autoencoder reconstruction error

A bottleneck autoencoder trained to reconstruct in-domain TF-IDF vectors can be used as an OOD score via `SklearnAutoencoder.reconstruction_error(X)`. On CLINC150 it reached AUC 0.60 — barely above chance — because short utterances produce sparse TF-IDF vectors that reconstruct near-trivially in both regimes. Use it as a method demonstration; for production, prefer calibrated confidence. See [cookbook/ood-with-autoencoder.md](../cookbook/ood-with-autoencoder.md).

## Caveats

- The classifier must be calibrated. `IntentClassifier` wraps non-probabilistic baselines in `CalibratedClassifierCV` by default; verify your baseline exposes `predict_proba` before relying on probability magnitudes.
- Confidence-based OOD detection inherits the classifier's biases — if a class is under-represented in training, low-confidence in-domain utterances will be wrongly rejected. Stratify your OOD threshold evaluation across all intents.
- For deeper discussion of low-confidence prediction, see [confidence-thresholds.md](confidence-thresholds.md).
- For dimensionality-reduction-based novelty detection, see [theory/dimensionality-reduction.md](../theory/dimensionality-reduction.md).

---
- Back to [docs index](../index.md)
