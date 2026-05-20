"""Semi-supervised learning primitives.

Pure-sklearn / pure-jurebes helpers for bootstrapping an intent
classifier from a small labeled seed set plus a large unlabeled pool.

The :func:`pseudo_label` and :func:`select_high_confidence` primitives
underpin self-training, co-training and label-propagation flows; the
:data:`SELECTION_STRATEGIES` registry exposes named selection
callables.
"""

from __future__ import annotations

from jurebes.semi_supervised.pseudo_label import (
    SELECTION_STRATEGIES,
    pseudo_label,
    select_high_confidence,
)
from jurebes.semi_supervised.self_training import SelfTrainResult, self_train
from jurebes.semi_supervised.co_training import CoTrainResult, co_train

__all__ = [
    "SELECTION_STRATEGIES",
    "pseudo_label",
    "select_high_confidence",
    "SelfTrainResult",
    "self_train",
    "CoTrainResult",
    "co_train",
]
