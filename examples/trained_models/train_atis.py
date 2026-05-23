"""Train and evaluate the canonical baselines portfolio on ATIS.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

from _common import run_pipeline, write_report


def main():
    from jurebes.datasets.canonical import load_atis

    report = run_pipeline("atis", load_atis, cv=5, n_iter=10)
    path = write_report("atis", report)
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
