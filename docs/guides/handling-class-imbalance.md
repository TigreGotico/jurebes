# Handling class imbalance

When some intents have many more samples than others, accuracy stops being a meaningful metric and many classifiers drift toward predicting the majority class. jurebes does not impose a single fix; this guide collects the standard remedies.

## Diagnose first

```python
from collections import Counter
counts = Counter(y)
for label, n in counts.most_common():
    print(f"  {label:30s} {n}")
print("imbalance ratio:", max(counts.values()) / min(counts.values()))
```

Ratios over 10:1 are worth treating explicitly. Ratios over 100:1 require both reweighting and metric changes.

## Metric: macro-F1 over accuracy

Macro-F1 is the unweighted mean of per-class F1. A classifier that ignores small classes gets a low macro-F1 even with high accuracy.

```python
from jurebes.benchmark import compare, to_markdown
result = compare(["logreg", "linear_svc", "nb_complement"], X, y, k=5,
                 scoring=("accuracy", "f1_macro"))
print(to_markdown(result, sort_by="macro_f1"))
```

The harness always computes both `macro_f1` and `micro_f1` (= accuracy in single-label multi-class). Sort by `macro_f1` when classes are imbalanced.

For per-class diagnostics:

```python
from jurebes.benchmark import cross_validate
r = cross_validate("logreg", X, y, k=5)
for label, f1 in sorted(r.per_class_f1.items(), key=lambda kv: kv[1]):
    print(f"  {label:30s} {f1:.3f}")
```

The weakest per-class F1 highlights the class your model is dropping.

## Estimator reweighting

Most sklearn classifiers accept `class_weight="balanced"`, which scales each class's loss by the inverse of its frequency. To inject this in jurebes, build the estimator yourself:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from jurebes import IntentClassifier

clf = IntentClassifier(Pipeline([
    ("feat", TfidfVectorizer()),
    ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced")),
]))
```

`class_weight="balanced"` is supported by `LogisticRegression`, `LinearSVC`, `SVC`, `RidgeClassifier`, `Perceptron`, `PassiveAggressiveClassifier`, `SGDClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`. It is *not* supported by the naive-Bayes family or by `HistGradientBoostingClassifier`.

## Complement Naive Bayes

`ComplementNB` was designed specifically for imbalanced text corpora (Rennie et al. 2003). It computes parameters from the *complement* of each class, which mutes the dominance of majority-class word counts. Two registered baselines use it:

- `nb_complement` (TF-IDF features)
- `complement_nb_count` (raw counts)

Either is a strong default on heavily skewed data.

## Resampling

When reweighting is insufficient, oversample minority classes or undersample majority ones before training:

```python
from sklearn.utils import resample
import numpy as np

X_arr, y_arr = np.array(X), np.array(y)
classes, counts = np.unique(y_arr, return_counts=True)
target = counts.max()                        # oversample to majority count

X_balanced, y_balanced = [], []
for c in classes:
    mask = y_arr == c
    Xc, yc = resample(X_arr[mask], y_arr[mask],
                      replace=True, n_samples=target, random_state=0)
    X_balanced.extend(Xc.tolist())
    y_balanced.extend(yc.tolist())

# X_balanced / y_balanced are now class-balanced; feed into IntentClassifier as usual.
```

Random oversampling is the simplest baseline; SMOTE-style synthesis from `imbalanced-learn` is an option but adds a dependency and rarely helps on sparse TF-IDF.

## Stratified CV is the default

`jurebes.benchmark.cross_validate` uses `StratifiedKFold` so class proportions are preserved per fold. Without stratification, an imbalanced run can produce a fold containing zero samples of a minority class, which crashes the per-class metric. No action required — this is the default.

## Threshold-by-intent

When accept thresholds matter (see [confidence-thresholds.md](confidence-thresholds.md)), lowering the threshold for minority intents trades precision for recall on those classes. Tune per-class on a held-out set.

---
- Back to [docs index](../index.md)
