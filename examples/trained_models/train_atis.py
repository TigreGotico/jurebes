"""Train and evaluate the canonical baselines portfolio on ATIS.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

from collections import Counter

from _common import run_pipeline, write_report


def _atis_filtered(split):
    """ATIS has long-tail classes with <5 samples — drop them so 5-fold
    stratified CV + inner CalibratedClassifierCV(cv=3) can both fit."""
    from jurebes.datasets.canonical import load_atis
    X, y = load_atis(split)
    counts = Counter(y)
    keep = {lbl for lbl, n in counts.items() if n >= 5}
    Xf = [x for x, lbl in zip(X, y) if lbl in keep]
    yf = [lbl for lbl in y if lbl in keep]
    return Xf, yf


def main():
    report = run_pipeline("atis", _atis_filtered, cv=5, n_iter=10)
    path = write_report("atis", report)
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
