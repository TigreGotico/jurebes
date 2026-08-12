"""Brier score and ECE via jurebes.benchmark.calibration.

Compares sigmoid-calibrated vs isotonic-calibrated LinearSVC using the
public calibration API. Lower Brier and ECE are better.
"""

# %%
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from jurebes.benchmark.calibration import (
    brier_score,
    expected_calibration_error,
)

X = (
    ["hello", "hi", "hey there", "good morning"] * 6
    + ["goodbye", "bye", "see you", "later"] * 6
    + ["thanks", "thank you", "much appreciated", "ta"] * 6
)
y = ["greet"] * 24 + ["bye"] * 24 + ["thanks"] * 24

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def _train(method):
    p = Pipeline([
        ("v", TfidfVectorizer()),
        ("c", CalibratedClassifierCV(LinearSVC(), cv=3, method=method)),
    ])
    p.fit(X_tr, y_tr)
    return p


for method in ("sigmoid", "isotonic"):
    clf = _train(method)
    proba = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    ece = expected_calibration_error(y_te, proba, classes, n_bins=10)
    brier = brier_score(y_te, proba, classes)
    print(f"{method:8s} Brier={brier:.4f}  ECE={ece:.4f}")
