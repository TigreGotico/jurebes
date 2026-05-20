"""Single-view self-training (Yarowsky 1995) for IntentClassifier.

Bootstraps a classifier from a labeled seed set by repeatedly:

1. fitting on the current labeled pool,
2. pseudo-labeling the unlabeled pool,
3. promoting the highest-confidence pseudo-labels into the labeled
   pool, and
4. (optionally) evaluating on a frozen held-out set to drive early
   stopping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from sklearn.base import clone
from sklearn.metrics import f1_score

from jurebes.core import IntentClassifier

from jurebes.semi_supervised.pseudo_label import (
    pseudo_label,
    select_high_confidence,
)


@dataclass
class SelfTrainResult:
    """Outcome of a :func:`self_train` run.

    Attributes:
        classifier: the final fitted :class:`jurebes.IntentClassifier`.
        labeled_X: utterances in the final labeled pool (seed + promoted).
        labeled_y: labels parallel to ``labeled_X``.
        added_per_round: number of pseudo-labels promoted in each round.
        eval_f1_per_round: held-out macro-F1 per round (empty if no
            eval set was provided).
        stopped_early: True iff training stopped before ``max_rounds``
            due to plateau detection.
        wall_time_s: total wall-clock time in seconds.
    """

    classifier: object
    labeled_X: List[str]
    labeled_y: List[str]
    added_per_round: List[int] = field(default_factory=list)
    eval_f1_per_round: List[float] = field(default_factory=list)
    stopped_early: bool = False
    wall_time_s: float = 0.0


def _refit(template, labeled_X: Sequence[str], labeled_y: Sequence[str]):
    """Build a fresh :class:`IntentClassifier` from ``template`` and fit it.

    Uses :func:`sklearn.base.clone` so the estimator state from prior
    rounds does not leak in.
    """
    # template.estimator already has any calibration wrapper baked in;
    # clone preserves structure but resets fitted state.
    new = IntentClassifier(
        clone(template.estimator), tagger=None, calibrate="if_missing",
    )
    by_label: Dict[str, List[str]] = {}
    for x, y in zip(labeled_X, labeled_y):
        by_label.setdefault(y, []).append(x)
    for label, samples in by_label.items():
        new.add_intent(label, samples)
    new.fit()
    return new


def self_train(
    clf,
    labeled_X: Sequence[str],
    labeled_y: Sequence[str],
    unlabeled_X: Sequence[str],
    *,
    confidence_threshold: float = 0.8,
    selection: str = "global_top_k",
    k_per_round: int = 10,
    max_rounds: int = 10,
    threshold_schedule: Optional[Callable[[int, float], float]] = None,
    eval_X: Optional[Sequence[str]] = None,
    eval_y: Optional[Sequence[str]] = None,
    early_stop_patience: int = 3,
) -> SelfTrainResult:
    """Run a self-training loop.

    Args:
        clf: an unfitted :class:`jurebes.IntentClassifier` (template).
        labeled_X / labeled_y: seed labeled pool.
        unlabeled_X: pool to draw pseudo-labels from.
        confidence_threshold: minimum confidence for promotion.
        selection: key into
            :data:`jurebes.semi_supervised.SELECTION_STRATEGIES`.
        k_per_round: budget per round (interpretation depends on
            selection strategy).
        max_rounds: cap on iterations.
        threshold_schedule: optional ``(round_idx, current_threshold)
            -> new_threshold`` callable invoked at the start of each
            round (round_idx is 0-based).
        eval_X / eval_y: optional frozen held-out evaluation set.
        early_stop_patience: stop if macro-F1 does not improve for
            this many consecutive rounds.
    """
    labeled_X = list(labeled_X)
    labeled_y = list(labeled_y)
    pool = list(unlabeled_X)
    pool_idx = list(range(len(pool)))

    added_per_round: List[int] = []
    eval_f1_per_round: List[float] = []
    stopped_early = False
    threshold = confidence_threshold
    best_f1 = -1.0
    no_improve = 0
    start = time.monotonic()

    current = _refit(clf, labeled_X, labeled_y)

    for r in range(max_rounds):
        if threshold_schedule is not None:
            threshold = float(threshold_schedule(r, threshold))

        if not pool_idx:
            break

        remaining = [pool[i] for i in pool_idx]
        scored = pseudo_label(current, remaining)
        picks_local = select_high_confidence(
            scored, strategy=selection, k=k_per_round, threshold=threshold,
        )
        if not picks_local:
            added_per_round.append(0)
            if eval_X is not None and eval_y is not None:
                preds = [current.predict(u).intent for u in eval_X]
                f1 = float(f1_score(list(eval_y), preds, average="macro", zero_division=0))
                eval_f1_per_round.append(f1)
            break

        promoted_global = [pool_idx[i] for i in picks_local]
        for local_i, global_i in zip(picks_local, promoted_global):
            label, _ = scored[local_i]
            labeled_X.append(pool[global_i])
            labeled_y.append(label)
        picks_set = set(promoted_global)
        pool_idx = [i for i in pool_idx if i not in picks_set]
        added_per_round.append(len(promoted_global))

        current = _refit(clf, labeled_X, labeled_y)

        if eval_X is not None and eval_y is not None:
            preds = [current.predict(u).intent for u in eval_X]
            f1 = float(f1_score(list(eval_y), preds, average="macro", zero_division=0))
            eval_f1_per_round.append(f1)
            if f1 > best_f1 + 1e-9:
                best_f1 = f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    stopped_early = True
                    break

    return SelfTrainResult(
        classifier=current,
        labeled_X=labeled_X,
        labeled_y=labeled_y,
        added_per_round=added_per_round,
        eval_f1_per_round=eval_f1_per_round,
        stopped_early=stopped_early,
        wall_time_s=time.monotonic() - start,
    )


__all__ = ["SelfTrainResult", "self_train"]
