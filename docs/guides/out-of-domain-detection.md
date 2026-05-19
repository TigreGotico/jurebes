# Out-of-domain detection

An intent classifier always returns *some* intent for any input. To reject utterances unrelated to the training inventory, layer an out-of-domain (OOD) detector on top.

## Approach: autoencoder reconstruction error

A bottleneck autoencoder trained to reconstruct in-domain TF-IDF vectors will reconstruct in-domain inputs well and OOD inputs poorly. The reconstruction error becomes an OOD score.

`jurebes.featurizers.SklearnAutoencoder` exposes `reconstruction_error(X)` returning per-row MSE:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from jurebes.featurizers import SklearnAutoencoder, tfidf_word

# 1. Fit the AE on in-domain TF-IDF vectors.
pipe = Pipeline([
    ("feat", tfidf_word()),
    ("ae",   SklearnAutoencoder(hidden_layer_sizes=(64, 16, 64), random_state=0)),
])
pipe.fit(X_train)

# 2. Pick a threshold from the in-domain reconstruction-error distribution.
feat = pipe.named_steps["feat"]
ae   = pipe.named_steps["ae"]

train_errs = ae.reconstruction_error(feat.transform(X_train))
threshold  = float(np.percentile(train_errs, 95))   # 95th-percentile of in-domain errors
print(f"OOD threshold = {threshold:.4f}")

# 3. Score new utterances.
def is_ood(utt: str) -> bool:
    err = float(ae.reconstruction_error(feat.transform([utt]))[0])
    return err > threshold
```

The threshold percentile controls the in-domain false-reject rate: 95 % means 5 % of legitimate in-domain inputs will be incorrectly flagged as OOD on the training distribution.

## Choosing the threshold

The 95th percentile is a starting point. For a principled choice, hold out a labelled mix of in-domain and OOD utterances and compute ROC/AUC:

```python
from sklearn.metrics import roc_auc_score, roc_curve

scores = ae.reconstruction_error(feat.transform(X_eval))
labels = np.array([0 if y == "in_domain" else 1 for y in y_eval])
print(f"AUC = {roc_auc_score(labels, scores):.3f}")

fpr, tpr, thr = roc_curve(labels, scores)
# pick the threshold that hits a target FPR, e.g. 5 %
target_fpr = 0.05
idx = np.argmax(fpr >= target_fpr)
threshold = float(thr[idx])
```

The CLINC150 dataset includes an `oos` (out-of-scope) label, making it the canonical OOD benchmark. See [cookbook/ood-with-autoencoder.md](../cookbook/ood-with-autoencoder.md) for an end-to-end CLINC150 worked example.

## Combining with the intent classifier

Two policies:

1. **Hard gate.** Score every incoming utterance with the AE before the intent classifier; reject when over threshold.
2. **Soft signal.** Run both, surface the AE score alongside the confidence, and decide downstream.

The hard gate is simplest:

```python
def predict_with_ood(utt: str):
    if is_ood(utt):
        return None
    return clf.predict(utt)
```

## Alternative: confidence thresholding

When the intent classifier is well-calibrated, a low confidence is itself a coarse OOD signal — see [confidence-thresholds.md](confidence-thresholds.md). The autoencoder approach is sharper because it does not rely on the multi-class probability simplex summing to 1.

## Caveats

- The autoencoder has its own hidden-layer hyperparameters. The defaults `(64, 16, 64)` work on small corpora; widen the hidden layers for richer vocabularies.
- Reconstruction error grows with the *novelty of vocabulary*, not necessarily with semantic OOD-ness. Inputs full of unseen words flag as OOD even if intent-relevant.
- For deeper OOD discussion, see [theory/dimensionality-reduction.md](../theory/dimensionality-reduction.md).

---
- Back to [docs index](../index.md)
