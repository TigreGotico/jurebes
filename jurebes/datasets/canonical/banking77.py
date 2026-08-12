"""BANKING77 fine-grained banking intent benchmark (Casanueva et al., 2020)."""

from __future__ import annotations

from typing import List, Tuple


def load_banking77(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load BANKING77 intent benchmark via HuggingFace datasets."""
    from jurebes.datasets.huggingface import load_hf
    return load_hf("banking77", split=split, text_field="text", label_field="label")
