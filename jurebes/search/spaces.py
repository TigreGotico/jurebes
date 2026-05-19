"""Predefined search spaces for the most-used baselines."""

from __future__ import annotations

from typing import Any, Dict


_SPACES: Dict[str, Dict[str, Any]] = {
    "logreg": {
        "feat__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "feat__min_df": [1, 2, 3],
        "clf__C": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0],
    },
    "linear_svc": {
        "feat__ngram_range": [(1, 1), (1, 2)],
        "feat__min_df": [1, 2],
    },
    "nb_multinomial": {
        "feat__ngram_range": [(1, 1), (1, 2)],
        "feat__min_df": [1, 2],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0],
    },
    "sgd_log": {
        "feat__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    },
}


def for_baseline(name: str) -> Dict[str, Any]:
    """Return the canonical search space for a given baseline name."""
    if name not in _SPACES:
        raise KeyError(f"no predefined search space for baseline {name!r}")
    # return a shallow copy so callers can mutate freely
    return {k: list(v) if isinstance(v, list) else v for k, v in _SPACES[name].items()}


def available() -> list:
    """Return the list of baseline names with predefined spaces."""
    return sorted(_SPACES.keys())
