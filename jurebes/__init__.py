"""🐾 jurebes — Just-sklearn Utility for Reproducible Evaluation of Baselines, Estimators and Solvers.

A classical-ML intent classification research framework.

In memory of Jurebes.
"""

from jurebes.core import IntentClassifier, IntentResult
from jurebes.baselines import BASELINES
from jurebes import active_learning
from jurebes import semi_supervised

__all__ = [
    "IntentClassifier",
    "IntentResult",
    "BASELINES",
    "active_learning",
    "semi_supervised",
]
