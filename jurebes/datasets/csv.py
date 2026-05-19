"""CSV dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Union


def load_csv(path: Union[str, Path], text: str = "text", label: str = "intent") -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append(row[text])
            y.append(row[label])
    return X, y
