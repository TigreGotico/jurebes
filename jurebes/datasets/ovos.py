"""OVOS .intent/.voc/.entity directory loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Union


# Matches the leftmost innermost (...) or [...] group.
_GROUP_RE = re.compile(r"\(([^()\[\]]*)\)|\[([^()\[\]]*)\]")


def expand_parentheses(line: str) -> List[str]:
    """Expand Padatious-style alternations.

    Handles ``(a|b)`` alternation and ``[optional]`` constructs, recursively
    for nesting. Returns a list of fully-expanded variants.

    Examples:
        >>> expand_parentheses("(hi|hello) friend")
        ['hi friend', 'hello friend']
        >>> expand_parentheses("hello [there] friend")
        ['hello  friend', 'hello there friend']
    """
    m = _GROUP_RE.search(line)
    if not m:
        return [re.sub(r"\s+", " ", line).strip()]
    pre = line[: m.start()]
    post = line[m.end():]
    if m.group(1) is not None:
        options = m.group(1).split("|")
    else:
        # [optional] → "" or the content
        options = ["", m.group(2)]
    results: List[str] = []
    for opt in options:
        for variant in expand_parentheses(pre + opt + post):
            results.append(variant)
    # de-dupe preserving order
    seen = set()
    out = []
    for v in results:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


_ENTITY_RE = re.compile(r"\{([^{}]+)\}")


def _read_samples(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


def load_ovos_intents(
    directory: Union[str, Path],
    *,
    expand_alternations: bool = True,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Load OVOS-style intent directory.

    Reads ``*.intent`` and ``*.voc`` as labelled samples, ``*.entity`` files
    as entity vocabularies. When ``expand_alternations`` is true (default),
    Padatious-style ``(a|b)`` alternation and ``[optional]`` constructs are
    expanded into multiple training samples. ``{entity}`` placeholder names
    are collected into the returned entities dict; lines containing them are
    still emitted as training samples (with placeholders left in place).

    Returns:
        ``(X, y, entities)`` where ``entities`` maps entity-name → samples
        (from .entity files plus discovered placeholder names with empty
        sample lists if no .entity file exists for them).
    """
    root = Path(directory)
    X: List[str] = []
    y: List[str] = []
    entities: Dict[str, List[str]] = {}

    def _process(raw_line: str, label: str) -> None:
        for slot in _ENTITY_RE.findall(raw_line):
            entities.setdefault(slot, [])
        variants = expand_parentheses(raw_line) if expand_alternations else [raw_line]
        for v in variants:
            if v:
                X.append(v)
                y.append(label)

    for f in root.rglob("*.intent"):
        for s in _read_samples(f):
            _process(s, f.stem)
    for f in root.rglob("*.voc"):
        for s in _read_samples(f):
            _process(s, f.stem)
    for f in root.rglob("*.entity"):
        entities[f.stem] = _read_samples(f)
    return X, y, entities
