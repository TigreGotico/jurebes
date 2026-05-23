# Cookbook: out-of-domain detection on CLINC150

CLINC150's `plus` configuration includes an `oos` (out-of-scope) label specifically for benchmarking OOD detectors. This cookbook compares four OOD scoring strategies on it: calibrated `bm25_logreg` top-1 confidence, top-1 − top-2 margin, an autoencoder reconstruction-error detector, and a one-class SVM.

Measured results:

| OOD scoring method | ROC AUC | TPR @ FPR=0.05 | TPR @ FPR=0.10 |
| --- | ---: | ---: | ---: |
| `bm25_logreg` top-1 confidence (1 − conf) | **0.9254** | 0.6290 | 0.7910 |
| `bm25_logreg` top1 − top2 margin | 0.9096 | 0.4690 | 0.7290 |
| `SklearnAutoencoder` reconstruction error | 0.6004 | 0.0810 | 0.1600 |
| One-class SVM (rbf) on TF-IDF | 0.5529 | 0.0840 | 0.1550 |

Calibrated confidence wins by a large margin. Short TF-IDF inputs reconstruct near-trivially through any bottleneck, so the AE's reconstruction-error gap lives in the fifth decimal place and barely separates the two regimes. The AE recipe below is kept as a method demonstration; for production OOD detection, use the calibrated-confidence recipe.

Bench script: `examples/trained_models/bench_clinc_ood_alternatives.py`. Report: `examples/trained_models/reports/clinc_ood_alternatives.md`.

## Prerequisites

```bash
pip install jurebes[hf]
```

## Recommended: calibrated `bm25_logreg` confidence

```python
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.datasets.canonical import load_clinc

X_train, y_train = load_clinc("train", include_ood=False)
X_test,  y_test  = load_clinc("test",  include_ood=True)
is_ood = np.array([lbl == "oos" for lbl in y_test], dtype=int)

clf = IntentClassifier(BASELINES.build("bm25_logreg"))
grouped = defaultdict(list)
for x, lbl in zip(X_train, y_train):
    grouped[lbl].append(x)
for lbl, samples in grouped.items():
    clf.add_intent(lbl, samples)
clf.fit()

proba = clf.estimator.predict_proba(list(X_test))
scores = 1.0 - proba.max(axis=1)               # higher = more OOD
print(f"AUC = {roc_auc_score(is_ood, scores):.4f}")

fpr, tpr, thr = roc_curve(is_ood, scores)
idx = int(np.argmax(fpr >= 0.05))
threshold = float(thr[idx])
print(f"threshold at 5% FPR: {threshold:.4f}  -> OOD recall {tpr[idx]:.3f}")
```

The threshold plugs into a deployment-side filter:

```python
def predict_with_ood(utt: str):
    p = clf.estimator.predict_proba([utt])[0]
    if (1.0 - p.max()) > threshold:
        return None                            # reject as OOD
    return clf.estimator.classes_[p.argmax()]
```

## Method demonstration: autoencoder reconstruction error

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from jurebes.datasets.canonical import load_clinc
from jurebes.featurizers import SklearnAutoencoder, tfidf_word


X, y = load_clinc()
in_dom_mask = np.array([label != "oos" for label in y])
X_in  = [x for x, m in zip(X, in_dom_mask) if m]
X_oos = [x for x, m in zip(X, in_dom_mask) if not m]

pipe = Pipeline([
    ("feat", tfidf_word()),
    ("ae",   SklearnAutoencoder(hidden_layer_sizes=(128, 32, 128),
                                max_iter=200, random_state=0)),
])
pipe.fit(X_in)
feat = pipe.named_steps["feat"]
ae   = pipe.named_steps["ae"]

in_scores  = ae.reconstruction_error(feat.transform(X_in))
oos_scores = ae.reconstruction_error(feat.transform(X_oos))
all_scores = np.concatenate([in_scores, oos_scores])
all_labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(oos_scores))])
print(f"AUC = {roc_auc_score(all_labels, all_scores):.4f}")     # ≈0.60
```

The AE pipeline is useful for novelty detection on domains with longer, vocabulary-rich inputs (long-form text, queries with proper nouns). On the short utterances typical of intent corpora it under-performs the confidence-based detector by ~0.3 AUC.

## Reading the output

- **AUC** measures rank-quality: 0.5 is random, 1.0 is perfect. Above 0.8 is a working OOD detector; above 0.9 is strong.
- The ROC curve trades in-domain false-rejects (FPR) against OOD recall (TPR). At `FPR=0.05`, 5 % of legitimate in-domain utterances are incorrectly flagged.

## Caveats

- AUC depends on the OOD test distribution. CLINC150's `oos` split is curated; results on in-house OOD data may differ — re-bench on a representative held-out mix before picking a threshold.
- Confidence-based OOD inherits the classifier's biases. Under-represented intents produce low-confidence in-domain predictions that get rejected. Stratify threshold evaluation across all intents.
- Autoencoder reconstruction error tracks vocabulary novelty more than semantic OOD-ness. Inputs full of unseen words flag as OOD even when intent-relevant.

## Related reading

- [../guides/out-of-domain-detection.md](../guides/out-of-domain-detection.md) — conceptual guide.
- [../theory/dimensionality-reduction.md](../theory/dimensionality-reduction.md) — where autoencoders sit in the reduced-dim taxonomy.
- [../guides/confidence-thresholds.md](../guides/confidence-thresholds.md) — using calibrated confidence for non-OOD rejection (low-information utterances, ambiguous intents).

---
- Back to [docs index](../index.md)
