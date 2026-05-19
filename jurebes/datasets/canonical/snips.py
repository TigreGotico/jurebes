"""SNIPS intent benchmark loader (Coucke et al., SNIPS NLU benchmark)."""

from __future__ import annotations

from typing import List, Tuple


def load_snips(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load SNIPS intent benchmark via HuggingFace datasets."""
    from jurebes.datasets.huggingface import load_hf
    try:
        return load_hf("benayas/snips", split=split, text_field="text", label_field="category")
    except Exception:
        return load_hf("DeepPavlov/snips", split=split, text_field="text", label_field="category")
