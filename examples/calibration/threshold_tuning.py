"""Find the right confidence threshold for OPM / active-learning gates.

The active-learning loop and the OPM pipeline both rely on confidence
thresholds (``hard_conf_max``, ``conf_high``, ``conf_med``, ``conf_low``).
A well-chosen threshold isolates the band where the classifier is
unreliable from the band where it can be trusted.

This script trains a classifier, computes the reliability curve on a
held-out set, and reports for each confidence threshold:

- ``coverage``  — fraction of test set predicted above the threshold.
- ``accuracy``  — accuracy on predictions above the threshold.
- ``ece_above`` — calibration error on the above-threshold subset.

Pick the lowest threshold at which accuracy clears your service-level
target. The default target is 95% accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.benchmark.calibration import (
    expected_calibration_error,
    reliability_curve,
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


def main(target_accuracy: float = 0.95):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y,
    )

    clf = IntentClassifier(BASELINES.build("logreg"))
    for label in sorted(set(y_tr)):
        clf.add_intent(label, [x for x, yy in zip(X_tr, y_tr) if yy == label])
    clf.fit()

    preds = [clf.predict(x) for x in X_te]
    classes = sorted(set(y_tr))
    cls_idx = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(X_te), len(classes)))
    for i, p in enumerate(preds):
        for r in clf.predict_proba(X_te[i]):
            j = cls_idx.get(r.intent)
            if j is not None:
                proba[i, j] = r.confidence
        if proba[i].sum() > 0:
            proba[i] /= proba[i].sum()

    top_conf = proba.max(axis=1)
    correct = np.array([p.intent == g for p, g in zip(preds, y_te)])

    print(f"{'threshold':>10} {'coverage':>10} {'accuracy':>10} {'ece_above':>10}")
    print("-" * 44)
    chosen = None
    for t in np.linspace(0.0, 0.9, 10):
        mask = top_conf >= t
        cov = mask.mean()
        if mask.sum() == 0:
            print(f"{t:10.2f} {cov:10.2f} {'-':>10} {'-':>10}")
            continue
        acc = correct[mask].mean()
        ece_above = expected_calibration_error(
            [y_te[i] for i in np.where(mask)[0]],
            proba[mask], classes, n_bins=10,
        )
        flag = "" if chosen else ""
        if acc >= target_accuracy and chosen is None:
            chosen = t
            flag = " <- recommended"
        print(f"{t:10.2f} {cov:10.2f} {acc:10.4f} {ece_above:10.4f}{flag}")

    print()
    if chosen is not None:
        print(f"recommended threshold for accuracy ≥ {target_accuracy}: {chosen:.2f}")
        print(f"  use as OPM conf_high or active-learning hard_conf_max")
    else:
        print("no threshold achieved the target accuracy — train on more data")


if __name__ == "__main__":
    main()
