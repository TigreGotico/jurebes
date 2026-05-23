"""Bracket-expansion utilities — deprecated shim over ``ovos_spec_tools``.

Use :func:`ovos_spec_tools.expand` directly. This module is preserved for
backward compatibility and will be removed in a future major release.
"""

from __future__ import annotations

import itertools
import re
import warnings
from typing import Dict, List

from ovos_spec_tools import expand as _expand
from ovos_utils.log import deprecated

from jurebes.version import VERSION_MAJOR

_REMOVAL = f"{VERSION_MAJOR + 1}.0.0"


@deprecated(
    "use 'ovos_spec_tools.expand' instead",
    _REMOVAL,
)
def expand_template(template: str) -> List[str]:
    """Deprecated — delegate to :func:`ovos_spec_tools.expand`."""
    warnings.warn(
        "jurebes.datasets.expansion.expand_template is deprecated; "
        "use ovos_spec_tools.expand instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sorted(set(_expand(template)))


@deprecated(
    "use 'ovos_spec_tools.expand' with slot substitution instead",
    _REMOVAL,
)
def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Deprecated — expand brackets via ovos_spec_tools then substitute slots."""
    warnings.warn(
        "jurebes.datasets.expansion.expand_slots is deprecated; "
        "use ovos_spec_tools.expand instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base = sorted(set(_expand(template)))
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
