"""Train and evaluate the canonical baselines portfolio on SNIPS.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

from _common import run_pipeline, write_report


def main():
    from jurebes.datasets.canonical import load_snips

    report = run_pipeline("snips", load_snips, cv=5, n_iter=10)
    path = write_report("snips", report)
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
