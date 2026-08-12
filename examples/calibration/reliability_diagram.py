"""Render a reliability diagram for a fitted IntentClassifier.

Requires ``jurebes[bench-plot]``. Saves a PNG; useful as a one-glance
sanity check that predicted confidence matches empirical accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.benchmark.calibration import (
    brier_score,
    expected_calibration_error,
    plot_reliability,
)


X = (
    ["play africa", "put on hey jude", "queue bohemian rhapsody",
     "start smells like teen spirit", "spin africa", "throw on hey jude"] * 4
    + ["set a timer for five minutes", "wake me in ten",
       "remind me in twenty minutes", "timer for half an hour",
       "set alarm for fifteen minutes", "wake me up in an hour"] * 4
    + ["how is the weather", "is it raining outside", "weather forecast",
       "current temperature", "is it sunny", "weather in lisbon"] * 4
)
y = ["play_song"] * 24 + ["set_timer"] * 24 + ["weather"] * 24

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y,
)


def _train(name):
    clf = IntentClassifier(BASELINES.build(name))
    for label in sorted(set(y_tr)):
        clf.add_intent(label, [x for x, yy in zip(X_tr, y_tr) if yy == label])
    clf.fit()
    return clf


def _proba_matrix(clf, X_te):
    classes = sorted(set(y_tr))
    cls_idx = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(X_te), len(classes)))
    for row, utt in enumerate(X_te):
        for r in clf.predict_proba(utt):
            j = cls_idx.get(r.intent)
            if j is not None:
                proba[row, j] = r.confidence
        if proba[row].sum() > 0:
            proba[row] /= proba[row].sum()
    return proba, classes


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("install jurebes[bench-plot] to render the reliability diagram")
        sys.exit(0)

    baselines = ["nb_multinomial", "logreg", "linear_svc", "random_forest"]
    fig, axes = plt.subplots(1, len(baselines), figsize=(5 * len(baselines), 5))
    for ax, name in zip(axes, baselines):
        clf = _train(name)
        proba, classes = _proba_matrix(clf, X_te)
        ece = expected_calibration_error(y_te, proba, classes, n_bins=10)
        brier = brier_score(y_te, proba, classes)
        plot_reliability(y_te, proba, classes, n_bins=10,
                         title=f"{name}\nECE={ece:.3f}  Brier={brier:.3f}", ax=ax)

    out = Path(__file__).parent / "reliability_diagram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
