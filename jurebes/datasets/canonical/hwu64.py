"""HWU64 home-assistant intent benchmark (Liu et al., 2019)."""

from __future__ import annotations

from typing import List, Tuple


def load_hwu64(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load HWU64 intent benchmark via HuggingFace datasets."""
    from jurebes.datasets.huggingface import load_hf
    try:
        return load_hf("DeepPavlov/hwu64", split=split, text_field="text", label_field="category")
    except Exception:
        return load_hf("liyucheng/hwu64", split=split, text_field="text", label_field="category")
