"""HWU64 home-assistant intent benchmark (Liu et al., 2019)."""

from __future__ import annotations

from typing import List, Tuple


def load_hwu64(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load HWU64 intent benchmark via HuggingFace datasets.

    Source: ``DeepPavlov/hwu64`` (text field ``utterance``, integer
    label field ``label``). Integer labels are stringified so downstream
    baselines treat them as discrete classes.
    """
    from jurebes.datasets.huggingface import load_hf
    X, y = load_hf("DeepPavlov/hwu64", split=split,
                   text_field="utterance", label_field="label")
    return X, [str(lbl) for lbl in y]
