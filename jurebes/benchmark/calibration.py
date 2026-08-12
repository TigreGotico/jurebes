"""Calibration diagnostics — ECE, Brier, reliability curves.

Pure sklearn / numpy. Plugs into the scoring registry as ``ece`` and
``brier``; expose :func:`reliability_curve` for users who want the raw
bin-by-bin numbers, and :func:`plot_reliability` (matplotlib-gated)
for a one-line visual.

Multi-class definitions:

- **Expected Calibration Error** (Naeini et al. 2015) — bin top-1
  probabilities, compare bin accuracy vs. bin mean confidence,
  weighted by bin size. Returns a float in ``[0, 1]``; lower is better.
- **Brier score (multi-class)** — mean squared error between the
  one-hot true labels and the predicted probabilities. Lower is better.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def expected_calibration_error(
    y_true: Sequence,
    y_pred_proba: Optional[np.ndarray],
    classes: Optional[Sequence] = None,
    *,
    n_bins: int = 15,
) -> float:
    """Top-1 ECE over ``n_bins`` equal-width confidence bins.

    Returns NaN when probabilities are not available.
    """
    if y_pred_proba is None or classes is None:
        return float("nan")
    proba = np.asarray(y_pred_proba, dtype=float)
    classes_arr = list(classes)
    top_idx = proba.argmax(axis=1)
    top_conf = proba.max(axis=1)
    top_pred = np.asarray([classes_arr[i] for i in top_idx])
    correct = (top_pred == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (top_conf > lo) & (top_conf <= hi)
        if lo == 0.0:
            mask |= top_conf == 0.0
        bin_n = mask.sum()
        if bin_n == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = top_conf[mask].mean()
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(
    y_true: Sequence,
    y_pred_proba: Optional[np.ndarray],
    classes: Optional[Sequence] = None,
) -> float:
    """Multi-class Brier score: mean squared error between one-hot
    true labels and the predicted probability vectors. Lower = better.
    """
    if y_pred_proba is None or classes is None:
        return float("nan")
    proba = np.asarray(y_pred_proba, dtype=float)
    classes_arr = list(classes)
    n, k = proba.shape
    one_hot = np.zeros((n, k), dtype=float)
    cls_to_idx = {c: i for i, c in enumerate(classes_arr)}
    for row, label in enumerate(y_true):
        idx = cls_to_idx.get(label)
        if idx is not None:
            one_hot[row, idx] = 1.0
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def reliability_curve(
    y_true: Sequence,
    y_pred_proba: Optional[np.ndarray],
    classes: Optional[Sequence] = None,
    *,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin-by-bin reliability curve over top-1 confidence.

    Returns four arrays of length ``n_bins`` (bins with zero count are
    NaN in the accuracy / mean-confidence columns):

    - ``bin_centers``     — equal-width bin midpoints in ``[0, 1]``.
    - ``bin_accuracy``    — fraction of correct predictions in each bin.
    - ``bin_confidence``  — mean predicted top-1 confidence in each bin.
    - ``bin_counts``      — number of predictions falling in each bin.
    """
    if y_pred_proba is None or classes is None:
        empty = np.full(n_bins, np.nan)
        return empty, empty, empty, np.zeros(n_bins, dtype=int)
    proba = np.asarray(y_pred_proba, dtype=float)
    classes_arr = list(classes)
    top_idx = proba.argmax(axis=1)
    top_conf = proba.max(axis=1)
    top_pred = np.asarray([classes_arr[i] for i in top_idx])
    correct = (top_pred == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (top_conf > lo) & (top_conf <= hi)
        if lo == 0.0:
            mask |= top_conf == 0.0
        n_bin = mask.sum()
        counts[i] = n_bin
        if n_bin:
            acc[i] = correct[mask].mean()
            conf[i] = top_conf[mask].mean()
    return centers, acc, conf, counts


def plot_reliability(
    y_true: Sequence,
    y_pred_proba: np.ndarray,
    classes: Sequence,
    *,
    n_bins: int = 15,
    title: str = "Reliability diagram",
    ax: Optional[Any] = None,
):
    """Render a reliability diagram. Requires ``jurebes[bench-plot]``."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError("install jurebes[bench-plot] to use plot_reliability") from exc
    centers, acc, conf, counts = reliability_curve(
        y_true, y_pred_proba, classes, n_bins=n_bins,
    )
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    valid = ~np.isnan(acc)
    ax.plot(centers[valid], acc[valid], "o-", label="model")
    ax.bar(centers, counts / max(counts.max(), 1), width=1.0 / n_bins,
           alpha=0.2, color="gray", label="bin weight")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


__all__ = [
    "expected_calibration_error",
    "brier_score",
    "reliability_curve",
    "plot_reliability",
]
