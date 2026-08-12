"""Calibration modes: if_missing, always, False.

LinearSVC has no native predict_proba — the three calibrate modes route
this case differently. The script also demonstrates the ValueError raised
when calibrate=False is combined with a non-proba estimator.
"""

# %% imports
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from jurebes import IntentClassifier


def raw_svc():
    return Pipeline([("feat", TfidfVectorizer()), ("clf", LinearSVC())])


# %% if_missing (default) wraps because LinearSVC lacks proba
a = IntentClassifier(raw_svc(), calibrate="if_missing")
print(f"if_missing wrapped: {type(a.estimator).__name__}")

# %% always wraps even if proba already exists
b = IntentClassifier(estimator=None, calibrate="always")
print(f"always wrapped: {type(b.estimator).__name__}")

# %% False with non-proba estimator must raise
try:
    IntentClassifier(raw_svc(), calibrate=False)
except ValueError as e:
    print(f"raised as expected: {e}")
