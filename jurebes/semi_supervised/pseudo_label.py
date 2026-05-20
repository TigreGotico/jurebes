"""Pseudo-labeling primitives.

Wraps a fitted :class:`jurebes.IntentClassifier` and produces ranked
``(label, confidence)`` pairs for unlabeled utterances, plus selection
strategies that pick which pseudo-labels to promote into the training
set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from jurebes.active_learning import uncertainty_scores


def pseudo_label(clf, X_unlabeled: Sequence[str]) -> List[Tuple[str, float]]:
    """Return ``[(top_label, top_confidence), ...]`` parallel to ``X_unlabeled``.

    ``clf`` must be a fitted :class:`jurebes.IntentClassifier`. The
    confidence is whatever ``clf.predict_proba`` reports for the top-1
    class; pair the classifier with ``calibrate="always"`` for
    well-calibrated values.
    """
    scored = uncertainty_scores(clf, list(X_unlabeled))
    return [(s.top_pred, s.top_conf) for s in scored]


def _global_top_k(
    scored: Sequence[Tuple[str, float]],
    *,
    k: int,
    threshold: float,
) -> List[int]:
    """Pick up to ``k`` indices with the highest confidence ≥ threshold."""
    candidates = [
        (i, conf) for i, (_, conf) in enumerate(scored) if conf >= threshold
    ]
    candidates.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in candidates[:k]]


def _per_class_quota(
    scored: Sequence[Tuple[str, float]],
    *,
    k: int,
    threshold: float,
) -> List[int]:
    """Pick up to ``k`` indices per class with the highest confidence ≥ threshold.

    Mitigates class drift: the standard global top-K rule concentrates
    pseudo-labels on whichever class the seed classifier is already
    most confident about, accelerating systematic bias.
    """
    by_class: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, (label, conf) in enumerate(scored):
        if conf >= threshold:
            by_class[label].append((i, conf))
    picked: List[int] = []
    for label, items in by_class.items():
        items.sort(key=lambda t: t[1], reverse=True)
        picked.extend(i for i, _ in items[:k])
    return picked


SELECTION_STRATEGIES: Dict[
    str,
    Callable[..., List[int]],
] = {
    "global_top_k": _global_top_k,
    "per_class_quota": _per_class_quota,
}


def select_high_confidence(
    scored: Sequence[Tuple[str, float]],
    *,
    strategy: str = "global_top_k",
    k: int = 10,
    threshold: float = 0.0,
) -> List[int]:
    """Select indices of pseudo-labels to promote into training.

    Args:
        scored: output of :func:`pseudo_label`.
        strategy: key into :data:`SELECTION_STRATEGIES`.
        k: budget — interpretation depends on strategy
            (``global_top_k``: total; ``per_class_quota``: per class).
        threshold: minimum confidence to consider.

    Raises:
        KeyError: if ``strategy`` is not a registered key.
    """
    if strategy not in SELECTION_STRATEGIES:
        raise KeyError(
            f"unknown selection strategy: {strategy!r}; "
            f"available: {sorted(SELECTION_STRATEGIES)}"
        )
    fn = SELECTION_STRATEGIES[strategy]
    return fn(scored, k=k, threshold=threshold)


__all__ = [
    "SELECTION_STRATEGIES",
    "pseudo_label",
    "select_high_confidence",
]
