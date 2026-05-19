"""Token-level feature dicts for IOB tagging."""

from __future__ import annotations

from typing import Dict, List, Union


def token_features(tokens: List[str], i: int) -> Dict[str, Union[str, int, bool]]:
    tok = tokens[i]
    feats: Dict[str, Union[str, int, bool]] = {
        "word": tok,
        "lower": tok.lower(),
        "suffix2": tok[-2:].lower(),
        "suffix3": tok[-3:].lower(),
        "prefix2": tok[:2].lower(),
        "prefix3": tok[:3].lower(),
        "is_upper": tok.isupper(),
        "is_title": tok.istitle(),
        "is_digit": tok.isdigit(),
        "has_digit": any(c.isdigit() for c in tok),
        "bos": i == 0,
        "eos": i == len(tokens) - 1,
    }
    feats["prev_word"] = tokens[i - 1].lower() if i > 0 else "<BOS>"
    feats["next_word"] = tokens[i + 1].lower() if i < len(tokens) - 1 else "<EOS>"
    return feats
