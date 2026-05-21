"""🐾 jurebes — Just-sklearn Utility for Reproducible Evaluation of Baselines, Estimators and Solvers.

A classical-ML text classification research framework. Intent
classification is the flagship application; the core is domain-agnostic
and works for spam, sentiment, topic, language-ID and any other
single-label text-classification task.

In memory of Jurebes.
"""

from jurebes.core import (
    IntentClassifier,
    IntentResult,
    TextClassifier,
    TextResult,
)
from jurebes.baselines import BASELINES
from jurebes import active_learning
from jurebes import semi_supervised

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "TextClassifier",
    "TextResult",
    "BASELINES",
    "active_learning",
    "semi_supervised",
]
