"""MASSIVE multilingual intent benchmark (FitzGerald et al., AmazonScience)."""

from __future__ import annotations

from typing import List, Tuple


def load_massive(split: str = "train", lang: str = "en-US") -> Tuple[List[str], List[str]]:
    """Load MASSIVE intent benchmark via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("install jurebes[hf] to fetch massive") from e
    ds = load_dataset("AmazonScience/massive", lang, split=split)
    label_field = "intent"
    text_field = "utt"
    labels = getattr(ds.features.get(label_field, None), "names", None)
    X = [str(r[text_field]) for r in ds]
    raw = [r[label_field] for r in ds]
    y = [labels[i] for i in raw] if labels else [str(v) for v in raw]
    return X, y
