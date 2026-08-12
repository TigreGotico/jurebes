"""HuggingFace datasets loader — lazy optional dep."""

from __future__ import annotations

from typing import List, Tuple


def load_hf(name: str, split: str = "train", text_field: str = "text", label_field: str = "label") -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("install jurebes[hf] to use load_hf") from e
    ds = load_dataset(name, split=split)
    X = [str(r[text_field]) for r in ds]
    y_raw = [r[label_field] for r in ds]
    labels = getattr(ds.features.get(label_field, None), "names", None)
    if labels:
        y = [labels[i] for i in y_raw]
    else:
        y = [str(v) for v in y_raw]
    return X, y
