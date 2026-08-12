"""JSONL dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union


def load_jsonl(path: Union[str, Path], text: str = "text", label: str = "intent") -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            X.append(obj[text])
            y.append(obj[label])
    return X, y
