"""Out-of-domain detection on CLINC150 via autoencoder reconstruction error.

CLINC ships an `oos` (out-of-scope) test split specifically for OOD
benchmarking. A `SklearnAutoencoder` trained on in-domain TF-IDF
vectors learns to reconstruct in-domain inputs well; OOD inputs
reconstruct worse. The per-sample reconstruction error becomes an OOD
score; ROC/AUC quantifies separability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve

from jurebes.datasets.canonical import load_clinc
from jurebes.featurizers import SklearnAutoencoder


HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "clinc_ood.md"


def main():
    print("loading CLINC (with oos)...", flush=True)
    # In-domain training set
    X_train, y_train = load_clinc("train", include_ood=False)
    # Mixed test: in-domain + OOD
    X_test_in, _ = load_clinc("test", include_ood=False)
    X_test_full, y_test_full = load_clinc("test", include_ood=True)
    is_ood = np.array([lbl == "oos" for lbl in y_test_full], dtype=int)
    n_in = (is_ood == 0).sum()
    n_ood = (is_ood == 1).sum()
    print(f"train={len(X_train)}  test_in={n_in}  test_ood={n_ood}", flush=True)

    print("fitting TF-IDF + autoencoder on in-domain training...", flush=True)
    vec = TfidfVectorizer(min_df=2)
    Xt_train = vec.fit_transform(X_train)
    Xt_test = vec.transform(X_test_full)

    ae = SklearnAutoencoder(max_iter=200, random_state=0)
    ae.fit(Xt_train)

    print("scoring reconstruction error on test set...", flush=True)
    scores = ae.reconstruction_error(Xt_test)

    auc = roc_auc_score(is_ood, scores)
    print(f"ROC AUC = {auc:.4f}", flush=True)

    fpr, tpr, _ = roc_curve(is_ood, scores)
    # Find threshold-free trade-offs at common operating points.
    def _tpr_at_fpr(target_fpr):
        idx = int(np.searchsorted(fpr, target_fpr))
        idx = min(idx, len(tpr) - 1)
        return float(tpr[idx])

    # In-domain median vs OOD median reconstruction error
    in_scores = scores[is_ood == 0]
    ood_scores = scores[is_ood == 1]

    md = ["# CLINC150 OOD detection — autoencoder reconstruction error\n"]
    md.append(
        "CLINC150's `oos` split provides genuine out-of-domain utterances "
        "for benchmarking. A `SklearnAutoencoder` trained on in-domain "
        "TF-IDF reconstructs in-domain inputs well; the per-sample "
        "reconstruction error rank-orders OOD vs in-domain.\n"
    )
    md.append("## Setup\n")
    md.append(f"- training samples (in-domain): **{len(X_train)}**")
    md.append(f"- test samples (in-domain):     **{n_in}**")
    md.append(f"- test samples (OOD `oos`):     **{n_ood}**")
    md.append(f"- featurizer: TF-IDF (min_df=2)")
    md.append(f"- autoencoder: `SklearnAutoencoder(max_iter=200, random_state=0)` "
              f"with `hidden_layer_sizes='auto'`\n")
    md.append("## Results\n")
    md.append(f"- **ROC AUC: {auc:.4f}**")
    md.append(f"- TPR @ FPR=0.05: {_tpr_at_fpr(0.05):.4f}")
    md.append(f"- TPR @ FPR=0.10: {_tpr_at_fpr(0.10):.4f}")
    md.append(f"- TPR @ FPR=0.20: {_tpr_at_fpr(0.20):.4f}")
    md.append("")
    md.append("| split | median recon error | mean recon error |")
    md.append("| --- | ---: | ---: |")
    md.append(f"| in-domain | {float(np.median(in_scores)):.5f} | {float(in_scores.mean()):.5f} |")
    md.append(f"| out-of-domain | {float(np.median(ood_scores)):.5f} | {float(ood_scores.mean()):.5f} |")
    md.append("")
    md.append("OOD utterances have measurably higher reconstruction error than "
              "in-domain ones; the gap between medians is the signal driving "
              "the AUC. The detector is threshold-free at this stage — pick a "
              "FPR-vs-TPR operating point from the ROC table above and convert "
              "to a reconstruction-error threshold for production use.\n")

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    main()
