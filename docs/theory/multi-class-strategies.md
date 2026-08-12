# Multi-class strategies

Some classifiers are intrinsically binary (linear SVM in its classical form, vanilla perceptron). Multi-class problems require a strategy to combine binary classifiers — or a model that is natively multinomial.

## Native multinomial

Some classifiers handle $K$ classes directly:

- **Logistic regression** with softmax (sklearn's `multi_class="multinomial"`, default for solver `lbfgs`).
- **Multinomial / Bernoulli / Complement naive Bayes**.
- **Decision trees and tree ensembles**.
- **MLP** with $K$ output units and softmax.

For these the strategy question does not apply; the model handles all classes in a single fit.

## One-vs-Rest (OvR)

Train $K$ binary classifiers; the $k$-th classifier predicts class $k$ vs all others. At prediction time, run all $K$ classifiers and pick the highest-scoring one.

- Train cost: $K \cdot \text{(binary cost)}$.
- Prediction cost: $K \cdot \text{(binary cost)}$.
- Simple, well-suited to high-dimensional sparse text data.

Baseline `ovr_linear_svc` uses `sklearn.multiclass.OneVsRestClassifier(LinearSVC())`.

## One-vs-One (OvO)

Train $\binom{K}{2}$ binary classifiers, one per class pair. At prediction time, run all pairwise classifiers and vote.

- Train cost: $O(K^2) \cdot \text{(binary cost)}$ — but each classifier sees only data from two classes, so each fit is cheaper.
- Prediction cost: $O(K^2) \cdot \text{(binary cost per prediction)}$.

Baseline `ovo_linear_svc` uses `sklearn.multiclass.OneVsOneClassifier(LinearSVC())`.

For $K$ around 5–10, OvO is competitive. Beyond ~50 classes, the $K^2$ classifier count becomes a memory and latency burden.

## Error-correcting output codes (ECOC)

A generalisation: assign each class a binary code (a row of a binary matrix), train one classifier per column. At prediction time, predict the code vector and decode to the nearest class. Robust to individual classifier errors. Implemented in sklearn as `OutputCodeClassifier`. Not bundled in jurebes; build via `Pipeline` if needed.

## Which strategy wins on text

For sparse high-dimensional text features, the strategies typically rank:

```
native multinomial  ≥  OvR  ≥  OvO  ≥  ECOC
```

with diminishing returns at the top. The OvR wrapper is a reasonable default when the underlying binary classifier is strong (LinearSVC). On well-tuned LogReg / LinearSVC the gap to native multinomial is usually under one macro-F1 point.

## Calibration and multi-class

`CalibratedClassifierCV` handles multi-class via OvR internally — even on a multinomially-fit base estimator. The output probabilities are normalised across classes but the underlying per-class calibrators were trained binary.

For sharp calibration on more than a handful of classes, isotonic calibration with a held-out set is the cleanest approach. See [calibration.md](calibration.md).

## Picking a strategy

| situation | strategy |
| --- | --- |
| natively multinomial classifier available | use it |
| many classes (>50) and LinearSVC needed | OvR |
| few classes (<10), high accuracy needed | OvO |
| extreme class count, latency-critical | OvR with hashing featurizer |

The jurebes registry exposes both `ovr_linear_svc` and `ovo_linear_svc` for direct comparison.

---
- Back to [docs index](../index.md)
