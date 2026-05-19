# Cookbook: out-of-domain detection with an autoencoder

CLINC150's `plus` configuration includes an `oos` (out-of-scope) label specifically for benchmarking OOD detectors. This cookbook trains an autoencoder on in-domain CLINC utterances and evaluates reconstruction-error-based OOD detection via ROC/AUC.

## Prerequisites

```bash
pip install jurebes[hf]
```

## Script

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from jurebes.datasets.canonical import load_clinc
from jurebes.featurizers import SklearnAutoencoder, tfidf_word


# Load CLINC150 (151 labels including `oos`)
X, y = load_clinc()
print(f"total {len(X)} samples, {len(set(y))} labels")

# Split: in-domain training, mixed eval
in_dom_mask = np.array([label != "oos" for label in y])
X_in = [x for x, m in zip(X, in_dom_mask) if m]
X_oos = [x for x, m in zip(X, in_dom_mask) if not m]
print(f"in-domain {len(X_in)}, OOS {len(X_oos)}")

# Train an autoencoder on in-domain TF-IDF vectors.
pipe = Pipeline([
    ("feat", tfidf_word()),
    ("ae",   SklearnAutoencoder(hidden_layer_sizes=(128, 32, 128), random_state=0,
                                max_iter=200)),
])
pipe.fit(X_in)
feat = pipe.named_steps["feat"]
ae   = pipe.named_steps["ae"]

# Score every sample by reconstruction error.
in_scores  = ae.reconstruction_error(feat.transform(X_in))
oos_scores = ae.reconstruction_error(feat.transform(X_oos))

# ROC/AUC: label 1 = OOS, label 0 = in-domain.
all_scores = np.concatenate([in_scores, oos_scores])
all_labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(oos_scores))])
auc = roc_auc_score(all_labels, all_scores)
print(f"AUC = {auc:.4f}")

# Pick a threshold at 5% in-domain false-reject rate.
fpr, tpr, thr = roc_curve(all_labels, all_scores)
idx = int(np.argmax(fpr >= 0.05))
chosen_thr = float(thr[idx])
print(f"threshold at 5% FPR: {chosen_thr:.4f}  -> OOS recall {tpr[idx]:.3f}")
```

## Reading the output

- **AUC** measures rank-quality: 0.5 is random, 1.0 is perfect. Anything above 0.8 is a working OOD detector; above 0.9 is strong.
- The ROC curve trades off in-domain false-rejects (FPR) against OOS recall (TPR). A `5% FPR` threshold means 5 % of legitimate in-domain utterances get incorrectly flagged as OOS.
- The chosen threshold value plugs into a deployment-side filter:

```python
def is_ood(utt: str) -> bool:
    err = float(ae.reconstruction_error(feat.transform([utt]))[0])
    return err > chosen_thr
```

## Tuning the autoencoder

Levers to pull:

- **Hidden-layer sizes.** `(128, 32, 128)` is a reasonable default for CLINC. Increase the outer layers for larger vocabularies; decrease the bottleneck for sharper compression.
- **`max_iter`.** 200 is a fast default. 500+ improves reconstruction but slows training.
- **Activation.** `"relu"` (default) usually works; `"tanh"` is sometimes better when the input has been standardised.
- **`early_stopping=True`** with a validation split prevents overfitting on larger corpora.

```python
ae = SklearnAutoencoder(
    hidden_layer_sizes=(128, 32, 128),
    activation="relu",
    max_iter=400,
    early_stopping=True,
    random_state=0,
)
```

## Combining with the intent classifier

Wrap both:

```python
from jurebes import IntentClassifier, BASELINES

clf = IntentClassifier(BASELINES.build("linear_svc"))
for label in sorted(set(y) - {"oos"}):
    clf.add_intent(label, [x for x, yy in zip(X, y) if yy == label])
clf.fit()

def safe_predict(utt: str):
    if is_ood(utt):
        return None
    return clf.predict(utt)
```

## Caveats

- The autoencoder's reconstruction error tracks vocabulary novelty more than semantic OOD-ness. Inputs full of unseen words flag as OOD even when intent-relevant.
- AUC depends on the OOS test distribution. CLINC150's `oos` split is curated; results on your in-house OOS data may differ.
- For very small in-domain corpora (<100 samples per class) the autoencoder overfits and gives poor OOD discrimination. Use a simpler confidence-threshold filter instead.

## Related reading

- [../guides/out-of-domain-detection.md](../guides/out-of-domain-detection.md) — the conceptual guide.
- [../theory/dimensionality-reduction.md](../theory/dimensionality-reduction.md) — where autoencoders sit in the reduced-dim taxonomy.

---
- Back to [docs index](../index.md)
