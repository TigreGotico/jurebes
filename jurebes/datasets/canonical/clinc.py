"""CLINC150 OOS intent benchmark loader (Larson et al., 2019)."""

from __future__ import annotations

from typing import List, Tuple


def load_clinc(split: str = "train", include_ood: bool = False) -> Tuple[List[str], List[str]]:
    """Load CLINC150 (clinc_oos) intent benchmark via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("install jurebes[hf] to fetch clinc") from e
    ds = load_dataset("clinc_oos", "plus", split=split)
    label_field = "intent"
    text_field = "text"
    labels = getattr(ds.features.get(label_field, None), "names", None)
    X: List[str] = []
    y: List[str] = []
    for row in ds:
        raw = row[label_field]
        name = labels[raw] if labels else str(raw)
        if not include_ood and name == "oos":
            continue
        X.append(str(row[text_field]))
        y.append(name)
    return X, y
