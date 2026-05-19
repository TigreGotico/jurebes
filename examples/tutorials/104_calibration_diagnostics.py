"""Brier score and ECE on calibrated vs uncalibrated LinearSVC.

Computes both metrics by hand on a small held-out split. LinearSVC's
default IntentClassifier uses CalibratedClassifierCV under the hood;
contrast it with a raw calibration wrapper using a finer grain.
"""

# %%
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def ece(probs, labels, n_bins=10):
    """Expected calibration error over n_bins equally-spaced confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.any():
            total += abs(accuracies[m].mean() - confidences[m].mean()) * m.sum() / n
    return total


def brier(probs, labels, n_classes):
    onehot = np.eye(n_classes)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


X = (
    ["hello", "hi", "hey there", "good morning"] * 6
    + ["goodbye", "bye", "see you", "later"] * 6
    + ["thanks", "thank you", "much appreciated", "ta"] * 6
)
y = np.array([0] * 24 + [1] * 24 + [2] * 24)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Calibrated (sigmoid) LinearSVC
cal = Pipeline([("v", TfidfVectorizer()), ("c", CalibratedClassifierCV(LinearSVC(), cv=3, method="sigmoid"))])
cal.fit(X_tr, y_tr)
P_cal = cal.predict_proba(X_te)

# Isotonic version for contrast
iso = Pipeline([("v", TfidfVectorizer()), ("c", CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic"))])
iso.fit(X_tr, y_tr)
P_iso = iso.predict_proba(X_te)

n_classes = len(set(y))
print(f"sigmoid  Brier={brier(P_cal, y_te, n_classes):.4f} ECE={ece(P_cal, y_te):.4f}")
print(f"isotonic Brier={brier(P_iso, y_te, n_classes):.4f} ECE={ece(P_iso, y_te):.4f}")
