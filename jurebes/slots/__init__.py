"""Slot tagging — pluggable tagger strategies."""

from jurebes.slots.iob import SklearnIOBTagger, tokenize
from jurebes.slots.features import token_features
from jurebes.slots.dictionary import DictionaryTagger
from jurebes.slots.template import TemplateTagger

__all__ = [
    "SklearnIOBTagger",
    "DictionaryTagger",
    "TemplateTagger",
    "token_features",
    "tokenize",
]
