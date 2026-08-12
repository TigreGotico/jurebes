"""Side-by-side comparison of self_train, co_train and label_propagation.

Runs all three strategies on the SAME tiny inline seed set and
unlabeled pool, then prints a comparison table of final coverage,
eval macro-F1 and wall time.

This script does NOT hit the network — the meteocat run lives in
`run_meteocat.py`.
"""

from __future__ import annotations

import time
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.semi_supervised import (
    co_train,
    label_propagation,
    self_train,
)


SEED_X = [
    "hello there", "hi friend", "hey", "good morning",
    "goodbye", "see you later", "bye now", "have a good night",
    "play some music", "start the playlist", "play a song", "put on music",
    "stop the music", "pause playback", "turn off the music", "silence please",
]
SEED_Y = (
    ["greet"] * 4 + ["bye"] * 4 + ["play"] * 4 + ["stop"] * 4
)

UNLABELED = [
    "hi", "good evening", "morning",
    "bye bye", "see you tomorrow", "later",
    "play another track", "start a tune", "music please",
    "pause it", "stop playing", "quiet now",
    "yo", "afternoon", "ciao", "playback off",
]

EVAL_X = [
    "hello", "goodbye", "play music", "stop the song",
    "hey there", "see ya", "start a song", "mute it",
]
EVAL_Y = ["greet", "bye", "play", "stop", "greet", "bye", "play", "stop"]


def _word_view() -> IntentClassifier:
    return IntentClassifier(
        Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression(max_iter=500))]),
    )


def _char_view() -> IntentClassifier:
    return IntentClassifier(BASELINES.build("linear_svc_char"))


def _macro_f1(clf, X, y) -> float:
    preds = [clf.predict(u).intent for u in X]
    return float(f1_score(list(y), preds, average="macro", zero_division=0))


def main() -> None:
    rows: List[tuple] = []

    # self_train
    t0 = time.monotonic()
    res = self_train(
        _word_view(), SEED_X, SEED_Y, UNLABELED,
        confidence_threshold=0.4, k_per_round=3, max_rounds=5,
        eval_X=EVAL_X, eval_y=EVAL_Y,
    )
    rows.append((
        "self_train",
        len(res.labeled_X),
        _macro_f1(res.classifier, EVAL_X, EVAL_Y),
        time.monotonic() - t0,
    ))

    # co_train
    t0 = time.monotonic()
    co = co_train(
        _word_view, _char_view, SEED_X, SEED_Y, UNLABELED,
        confidence_threshold=0.4, k_per_round=3, max_rounds=5,
        eval_X=EVAL_X, eval_y=EVAL_Y,
    )
    rows.append((
        "co_train",
        len(co.labeled_X),
        _macro_f1(co.view_a, EVAL_X, EVAL_Y),
        time.monotonic() - t0,
    ))

    # label_propagation
    t0 = time.monotonic()
    lp = label_propagation(SEED_X, SEED_Y, UNLABELED, method="propagation")
    # Build a quick classifier on seed + propagated labels to score eval.
    final = _word_view()
    by_label = {}
    for x, y in zip(SEED_X + list(UNLABELED), SEED_Y + lp.predicted_labels):
        by_label.setdefault(y, []).append(x)
    for label, samples in by_label.items():
        final.add_intent(label, samples)
    final.fit()
    rows.append((
        "label_propagation",
        len(SEED_X) + len(UNLABELED),
        _macro_f1(final, EVAL_X, EVAL_Y),
        time.monotonic() - t0,
    ))

    print(f"{'strategy':<20} {'coverage':>10} {'eval_f1':>10} {'wall_s':>10}")
    for name, cov, f1, wall in rows:
        print(f"{name:<20} {cov:>10d} {f1:>10.3f} {wall:>10.2f}")


if __name__ == "__main__":
    main()
