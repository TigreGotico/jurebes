"""Two-view co-training (Blum & Mitchell 1998) for IntentClassifier.

Maintains two classifiers built from two different feature ``view``
factories. Each round, every view pseudo-labels the unlabeled pool;
the high-confidence picks from view A are promoted into view B's
training set and vice versa. Both views' final predictions can be
combined externally (e.g. averaged) — this routine returns both
classifiers and lets the caller decide.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from sklearn.metrics import f1_score

from jurebes.core import IntentClassifier
from jurebes.semi_supervised.pseudo_label import (
    pseudo_label,
    select_high_confidence,
)


@dataclass
class CoTrainResult:
    """Outcome of a :func:`co_train` run.

    Attributes:
        view_a / view_b: the final fitted classifiers.
        labeled_X / labeled_y: the (shared) final labeled pool.
        added_per_round_view_a: pseudo-labels view A contributed each round.
        added_per_round_view_b: pseudo-labels view B contributed each round.
        eval_f1_per_round_view_a / _view_b: per-view eval macro-F1 per
            round (empty if no eval set was provided).
        wall_time_s: total wall-clock time.
    """

    view_a: IntentClassifier
    view_b: IntentClassifier
    labeled_X: List[str]
    labeled_y: List[str]
    added_per_round_view_a: List[int] = field(default_factory=list)
    added_per_round_view_b: List[int] = field(default_factory=list)
    eval_f1_per_round_view_a: List[float] = field(default_factory=list)
    eval_f1_per_round_view_b: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0


def _fit_view(view_factory: Callable[[], IntentClassifier],
              labeled_X: Sequence[str], labeled_y: Sequence[str]) -> IntentClassifier:
    clf = view_factory()
    by_label: Dict[str, List[str]] = {}
    for x, y in zip(labeled_X, labeled_y):
        by_label.setdefault(y, []).append(x)
    for label, samples in by_label.items():
        clf.add_intent(label, samples)
    clf.fit()
    return clf


def co_train(
    view_a_factory: Callable[[], IntentClassifier],
    view_b_factory: Callable[[], IntentClassifier],
    labeled_X: Sequence[str],
    labeled_y: Sequence[str],
    unlabeled_X: Sequence[str],
    *,
    confidence_threshold: float = 0.8,
    selection: str = "global_top_k",
    k_per_round: int = 10,
    max_rounds: int = 10,
    eval_X: Optional[Sequence[str]] = None,
    eval_y: Optional[Sequence[str]] = None,
) -> CoTrainResult:
    """Run a two-view co-training loop.

    Args:
        view_a_factory / view_b_factory: zero-arg callables returning a
            fresh, unfitted :class:`jurebes.IntentClassifier`. The two
            views should use genuinely different feature spaces
            (e.g. word TF-IDF vs char n-gram) for co-training to help.
        labeled_X / labeled_y: shared seed labeled pool.
        unlabeled_X: shared pool to draw pseudo-labels from.
        confidence_threshold, selection, k_per_round, max_rounds: as
            in :func:`self_train`.
        eval_X / eval_y: optional frozen held-out evaluation set.
    """
    labeled_X = list(labeled_X)
    labeled_y = list(labeled_y)
    pool = list(unlabeled_X)
    pool_idx = list(range(len(pool)))

    added_a: List[int] = []
    added_b: List[int] = []
    eval_a: List[float] = []
    eval_b: List[float] = []
    start = time.monotonic()

    view_a = _fit_view(view_a_factory, labeled_X, labeled_y)
    view_b = _fit_view(view_b_factory, labeled_X, labeled_y)

    for r in range(max_rounds):
        if not pool_idx:
            break

        remaining = [pool[i] for i in pool_idx]

        scored_a = pseudo_label(view_a, remaining)
        picks_a_local = select_high_confidence(
            scored_a, strategy=selection, k=k_per_round,
            threshold=confidence_threshold,
        )
        scored_b = pseudo_label(view_b, remaining)
        picks_b_local = select_high_confidence(
            scored_b, strategy=selection, k=k_per_round,
            threshold=confidence_threshold,
        )

        # Avoid double-promoting the same global index.
        promoted: Dict[int, str] = {}
        for li in picks_a_local:
            gi = pool_idx[li]
            promoted[gi] = scored_a[li][0]
        for li in picks_b_local:
            gi = pool_idx[li]
            promoted.setdefault(gi, scored_b[li][0])

        if not promoted:
            added_a.append(0)
            added_b.append(0)
            if eval_X is not None and eval_y is not None:
                eval_a.append(_macro_f1(view_a, eval_X, eval_y))
                eval_b.append(_macro_f1(view_b, eval_X, eval_y))
            break

        for gi, lbl in promoted.items():
            labeled_X.append(pool[gi])
            labeled_y.append(lbl)
        promoted_set = set(promoted)
        pool_idx = [i for i in pool_idx if i not in promoted_set]

        added_a.append(len(picks_a_local))
        added_b.append(len(picks_b_local))

        view_a = _fit_view(view_a_factory, labeled_X, labeled_y)
        view_b = _fit_view(view_b_factory, labeled_X, labeled_y)

        if eval_X is not None and eval_y is not None:
            eval_a.append(_macro_f1(view_a, eval_X, eval_y))
            eval_b.append(_macro_f1(view_b, eval_X, eval_y))

    return CoTrainResult(
        view_a=view_a,
        view_b=view_b,
        labeled_X=labeled_X,
        labeled_y=labeled_y,
        added_per_round_view_a=added_a,
        added_per_round_view_b=added_b,
        eval_f1_per_round_view_a=eval_a,
        eval_f1_per_round_view_b=eval_b,
        wall_time_s=time.monotonic() - start,
    )


def _macro_f1(clf, eval_X, eval_y) -> float:
    preds = [clf.predict(u).intent for u in eval_X]
    return float(f1_score(list(eval_y), preds, average="macro", zero_division=0))


__all__ = ["CoTrainResult", "co_train"]
