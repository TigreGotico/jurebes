"""Optional plot hooks — requires matplotlib (extra: bench-plot)."""

from __future__ import annotations


def plot_comparison(comparison, metric: str = "macro_f1", *, ax=None, title: str = None):
    """Render a horizontal bar chart of one metric across baselines.

    Raises ``ImportError`` if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "install jurebes[bench-plot] to use plot_comparison"
        ) from e
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(2, 0.3 * len(comparison.rows))))
    names = [r.name for r in comparison.rows]
    values = [getattr(r, metric, None) if hasattr(r, metric) else r.extra_scores.get(metric, 0.0)
              for r in comparison.rows]
    ax.barh(names, values)
    ax.set_xlabel(metric)
    if title:
        ax.set_title(title)
    return ax
