"""Bracket-expansion utilities for training data preprocessing.

Two helpers for Padatious-style template strings:

- :func:`expand_template` — expand ``[optional]`` parts and ``(a|b|c)``
  alternations into the full cartesian set of realised utterances.
- :func:`expand_slots`    — also substitute ``{slot}`` placeholders with
  values drawn from a dictionary.

Pure stdlib. Useful when a user wants to materialise extra training
samples from a small template grammar before handing the data to
:class:`jurebes.IntentClassifier`.

Ported from `ovos_utils.bracket_expansion` (Apache-2.0).
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List


def expand_template(template: str) -> List[str]:
    """Expand ``[optional]`` and ``(a|b)`` constructs in a template.

    ``"hello [there] (friend|world)"`` →
    ``["hello friend", "hello there friend", "hello there world", "hello world"]``.

    Returns a sorted, de-duplicated list of all realised strings.
    """
    def _expand_optional(text: str) -> str:
        return re.sub(r"\[([^\[\]]+)\]", lambda m: f"({m.group(1)}|)", text)

    def _expand_alternatives(text: str):
        parts = []
        for segment in re.split(r"(\([^\(\)]+\))", text):
            if segment.startswith("(") and segment.endswith(")"):
                options = segment[1:-1].split("|")
                parts.append(options)
            else:
                parts.append([segment])
        return itertools.product(*parts)

    def _fully_expand(texts):
        result = set(texts)
        while True:
            expanded = set()
            for text in result:
                options = list(_expand_alternatives(text))
                expanded.update(
                    re.sub(r"\s+", " ", "".join(opt)).strip() for opt in options
                )
            if expanded == result:
                break
            result = expanded
        return sorted(result)

    return _fully_expand([_expand_optional(template)])


def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Expand alternatives, optionals, then substitute ``{slot}`` placeholders.

    ``slots`` maps slot name → list of replacement values. Each placeholder
    in the template is replaced with every value from its list and the full
    cartesian product across placeholders is returned.

    Unknown ``{slots}`` (not in the dict) are left intact.
    """
    base = expand_template(template)
    out: List[str] = []
    for sentence in base:
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if not matches:
            out.append(sentence)
            continue
        slot_options = [slots.get(m, ["{" + m + "}"]) for m in matches]
        for combo in itertools.product(*slot_options):
            filled = sentence
            for name, value in zip(matches, combo):
                filled = filled.replace("{" + name + "}", value)
            out.append(filled)
    return out
