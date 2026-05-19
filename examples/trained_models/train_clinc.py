"""Train and evaluate the canonical baselines portfolio on CLINC-150 (no OOD).

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

from _common import run_pipeline, write_report


def main():
    from jurebes.datasets.canonical import load_clinc

    def loader(split: str):
        return load_clinc(split=split, include_ood=False)

    report = run_pipeline("clinc", loader, cv=5, n_iter=10)
    path = write_report("clinc", report)
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
