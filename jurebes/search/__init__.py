"""Hyperparameter search subsystem.

Lightweight wrappers around sklearn's search CV classes plus optional
``skopt`` (Bayesian) and ``sklearn-genetic-opt`` (genetic) backends.

The public surface is :func:`search` returning a :class:`SearchResult`.
"""

from jurebes.search.api import SearchResult, search
from jurebes.search import spaces

__all__ = ["search", "SearchResult", "spaces"]
