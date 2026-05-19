"""ATIS Air Travel Information System intent benchmark (Hemphill et al., 1990)."""

from __future__ import annotations

from typing import List, Tuple


def load_atis(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load ATIS intent benchmark via HuggingFace datasets."""
    from jurebes.datasets.huggingface import load_hf
    return load_hf("tuetschek/atis", split=split, text_field="text", label_field="intent")
