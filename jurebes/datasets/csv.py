"""CSV dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Union

from ovos_utils.log import LOG


def load_csv(path: Union[str, Path], text: str = "text", label: str = "intent") -> Tuple[List[str], List[str]]:
    """Load a CSV file with text and label columns.

    Opens with ``utf-8-sig`` to transparently strip a UTF-8 BOM that would
    otherwise corrupt the first column name. Rows with empty text or empty
    label are skipped, with a single warning summarising the skip count.
    """
    X: List[str] = []
    y: List[str] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get(text) or "").strip()
            lbl = (row.get(label) or "").strip()
            if not t or not lbl:
                skipped += 1
                continue
            X.append(t)
            y.append(lbl)
    if skipped:
        LOG.warning(f"load_csv: skipped {skipped} row(s) with empty text/label")
    return X, y
