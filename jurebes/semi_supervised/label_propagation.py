"""Graph-based label propagation / label spreading (scikit-learn wrapper).

Vectorises labeled + unlabeled utterances with a shared featurizer,
densifies the matrix (the sklearn implementations require dense
input), and runs either :class:`sklearn.semi_supervised.LabelPropagation`
or :class:`sklearn.semi_supervised.LabelSpreading` over the
combined graph. Returns predicted labels for every unlabeled sample.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from jurebes.featurizers import tfidf_word


@dataclass
class LabelPropResult:
    """Outcome of a :func:`label_propagation` run.

    Attributes:
        predicted_labels: predicted label for each unlabeled sample.
        predicted_confidences: max class probability per unlabeled sample.
        method: ``"propagation"`` or ``"spreading"``.
        wall_time_s: total wall-clock time.
        model: the fitted sklearn estimator.
    """

    predicted_labels: List[str]
    predicted_confidences: List[float] = field(default_factory=list)
    method: str = "propagation"
    wall_time_s: float = 0.0
    model: Any = None


def label_propagation(
    labeled_X: Sequence[str],
    labeled_y: Sequence[str],
    unlabeled_X: Sequence[str],
    *,
    featurizer: Optional[Any] = None,
    method: Literal["propagation", "spreading"] = "propagation",
    **kwargs,
) -> LabelPropResult:
    """Propagate labels from ``labeled_X`` over a graph that also includes ``unlabeled_X``.

    Args:
        labeled_X / labeled_y: seed labeled pool.
        unlabeled_X: pool to predict labels for.
        featurizer: any sklearn-style vectorizer with
            ``fit_transform``; defaults to :func:`jurebes.featurizers.tfidf_word`.
        method: ``"propagation"`` uses :class:`LabelPropagation`,
            ``"spreading"`` uses the more noise-robust
            :class:`LabelSpreading`.
        **kwargs: forwarded to the sklearn estimator constructor.
    """
    if method not in {"propagation", "spreading"}:
        raise ValueError(f"unknown method {method!r}; pick 'propagation' or 'spreading'")

    start = time.monotonic()
    vec = featurizer if featurizer is not None else tfidf_word()
    all_text = list(labeled_X) + list(unlabeled_X)
    X = vec.fit_transform(all_text)
    # LabelPropagation/Spreading require dense input.
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    le = LabelEncoder()
    le.fit(list(labeled_y))
    y_lab = le.transform(list(labeled_y))
    # -1 marks unlabeled in sklearn's semi-supervised API.
    y = np.concatenate([y_lab, np.full(len(unlabeled_X), -1, dtype=y_lab.dtype)])

    cls = LabelPropagation if method == "propagation" else LabelSpreading
    model = cls(**kwargs)
    model.fit(X, y)

    pred_idx = model.transduction_[len(labeled_X):]
    pred_labels = list(le.inverse_transform(pred_idx))

    confidences: List[float] = []
    if hasattr(model, "label_distributions_"):
        dist = model.label_distributions_[len(labeled_X):]
        confidences = [float(np.max(row)) for row in dist]

    return LabelPropResult(
        predicted_labels=pred_labels,
        predicted_confidences=confidences,
        method=method,
        wall_time_s=time.monotonic() - start,
        model=model,
    )


__all__ = ["LabelPropResult", "label_propagation"]
