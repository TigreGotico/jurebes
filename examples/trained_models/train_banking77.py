"""Train and evaluate the canonical baselines portfolio on BANKING77.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

from _common import run_pipeline, write_report


def main():
    from jurebes.datasets.canonical import load_banking77

    report = run_pipeline("banking77", load_banking77, cv=5, n_iter=10)
    path = write_report("banking77", report)
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
