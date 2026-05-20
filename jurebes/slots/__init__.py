"""Slot tagging — pluggable tagger strategies."""

from jurebes.slots.iob import SklearnIOBTagger, tokenize
from jurebes.slots.features import token_features
from jurebes.slots.dictionary import DictionaryTagger
from jurebes.slots.template import TemplateTagger
from jurebes.slots.knn import KNNTagger
from jurebes.slots.hybrid import HybridCascadeTagger
from jurebes.slots.registry import TAGGERS

__all__ = [
    "SklearnIOBTagger",
    "DictionaryTagger",
    "TemplateTagger",
    "KNNTagger",
    "HybridCascadeTagger",
    "TAGGERS",
    "token_features",
    "tokenize",
]
