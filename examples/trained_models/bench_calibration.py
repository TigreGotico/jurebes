"""Calibration analysis on the canonical datasets.

`compare()` with the calibration scorers (`ece`, `brier`, `log_loss`)
alongside `accuracy` and `f1_macro` for the default portfolio on
SNIPS / BANKING77 / CLINC. Answers whether the `CalibratedClassifierCV`
auto-wrap on `linear_svc_char` (and friends) produces honest
probabilities, or whether the confidences are systematically off.
"""

from __future__ import annotations

from pathlib import Path

from jurebes.benchmark import compare, to_markdown
from jurebes.datasets.canonical import (
    load_banking77, load_clinc, load_snips,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "calibration.md"

LOADERS = {
    "snips":     load_snips,
    "banking77": load_banking77,
    "clinc":     load_clinc,
}

# Compact portfolio: top intent baselines + the ones whose calibrated
# probabilities are interesting (label_guided uses MLP; tree-free linear
# baselines are the ones with the CalibratedClassifierCV wrap that we
# want to scrutinise).
PORTFOLIO = [
    "linear_svc_char",
    "linear_svc",
    "bm25_logreg",
    "logreg",
    "union_bm25_pos_logreg",
    "union_skipgram_tfidf_logreg",
    "nb_multinomial",
    "label_guided_logreg",
]


def main():
    sections = ["# calibration analysis — canonical datasets\n"]
    sections.append("ECE = expected calibration error (lower is better, 0 = perfect).")
    sections.append("Brier = mean squared error of probability vs one-hot ground truth.")
    sections.append("`log_loss` = cross-entropy of predicted probabilities.")
    sections.append("Friedman+Nemenyi disabled — calibration is dataset-specific.\n")
    for ds, loader in LOADERS.items():
        print(f"[{ds}] running compare with calibration scoring...", flush=True)
        X, y = loader("train")
        result = compare(PORTFOLIO, X, y, k=3,
                         scoring=("accuracy", "f1_macro", "ece", "brier", "log_loss"))
        sections.append(f"## {ds}\n")
        sections.append(to_markdown(result, sort_by="ece", precision=4))
        sections.append("")
        # bottom-line: best vs worst calibrated baselines by ECE
        rows = result.rows
        by_ece = sorted(rows, key=lambda r: r.extra_scores.get("ece", 1.0))
        best = by_ece[0]
        worst = by_ece[-1]
        sections.append(
            f"Best-calibrated: **`{best.name}`** (ECE {best.extra_scores['ece']:.4f}). "
            f"Worst-calibrated: **`{worst.name}`** (ECE {worst.extra_scores['ece']:.4f}).\n"
        )
    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    main()
