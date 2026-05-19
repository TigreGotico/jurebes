"""OVOS .intent/.voc/.entity directory loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union


def _read_samples(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


def load_ovos_intents(directory: Union[str, Path]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    root = Path(directory)
    X: List[str] = []
    y: List[str] = []
    entities: Dict[str, List[str]] = {}
    for f in root.rglob("*.intent"):
        for s in _read_samples(f):
            X.append(s)
            y.append(f.stem)
    for f in root.rglob("*.voc"):
        for s in _read_samples(f):
            X.append(s)
            y.append(f.stem)
    for f in root.rglob("*.entity"):
        entities[f.stem] = _read_samples(f)
    return X, y, entities
