"""Canonical intent benchmark loaders — SNIPS, CLINC150, BANKING77, HWU64, ATIS, MASSIVE."""

from __future__ import annotations

from typing import Callable, Dict

from jurebes.datasets.canonical.atis import load_atis
from jurebes.datasets.canonical.banking77 import load_banking77
from jurebes.datasets.canonical.clinc import load_clinc
from jurebes.datasets.canonical.hwu64 import load_hwu64
from jurebes.datasets.canonical.massive import load_massive
from jurebes.datasets.canonical.snips import load_snips

CANONICAL: Dict[str, Callable] = {
    "snips": load_snips,
    "clinc": load_clinc,
    "banking77": load_banking77,
    "hwu64": load_hwu64,
    "atis": load_atis,
    "massive": load_massive,
}

__all__ = [
    "load_snips",
    "load_clinc",
    "load_banking77",
    "load_hwu64",
    "load_atis",
    "load_massive",
    "CANONICAL",
]
