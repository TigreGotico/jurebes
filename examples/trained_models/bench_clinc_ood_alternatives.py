"""CLINC150 OOD detection — alternatives to the autoencoder approach.

The AE reconstruction-error detector reached AUC 0.60 — barely above
chance. This script benches the three alternatives suggested in the
REPORT caveat, plus the AE baseline for reference:

1. Calibrated `bm25_logreg` top-1 confidence (1 − conf as OOD score).
2. Top-1 minus top-2 calibrated confidence (margin).
3. One-class SVM on TF-IDF.
4. SklearnAutoencoder reconstruction error (reference).

Writes `clinc_ood_alternatives.md` with ROC AUC + TPR-at-FPR points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.datasets.canonical import load_clinc
from jurebes.featurizers import SklearnAutoencoder


HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "clinc_ood_alternatives.md"


def _tpr_at_fpr(fpr_arr, tpr_arr, target):
    idx = int(np.searchsorted(fpr_arr, target))
    idx = min(idx, len(tpr_arr) - 1)
    return float(tpr_arr[idx])


def _summarise(name, is_ood, scores):
    auc = roc_auc_score(is_ood, scores)
    fpr, tpr, _ = roc_curve(is_ood, scores)
    return {
        "name": name,
        "auc": auc,
        "tpr_05": _tpr_at_fpr(fpr, tpr, 0.05),
        "tpr_10": _tpr_at_fpr(fpr, tpr, 0.10),
        "tpr_20": _tpr_at_fpr(fpr, tpr, 0.20),
    }


def main():
    print("loading CLINC (in-domain training + mixed test)...", flush=True)
    X_train, y_train = load_clinc("train", include_ood=False)
    X_test_full, y_test_full = load_clinc("test", include_ood=True)
    is_ood = np.array([lbl == "oos" for lbl in y_test_full], dtype=int)
    n_in = int((is_ood == 0).sum())
    n_ood = int(is_ood.sum())
    print(f"train_in={len(X_train)}  test_in={n_in}  test_ood={n_ood}", flush=True)

    results = []

    # 1. Calibrated bm25_logreg confidence (1 - top-1 as OOD score)
    print("[1/4] fitting bm25_logreg for confidence-based OOD...", flush=True)
    clf = IntentClassifier(BASELINES.build("bm25_logreg"))
    from collections import defaultdict
    grouped = defaultdict(list)
    for x, lbl in zip(X_train, y_train):
        grouped[lbl].append(x)
    for lbl, samples in grouped.items():
        clf.add_intent(lbl, samples)
    clf.fit()
    proba = clf.estimator.predict_proba(list(X_test_full))
    top_conf = proba.max(axis=1)
    results.append(_summarise("bm25_logreg top-1 confidence (1−conf)",
                              is_ood, 1.0 - top_conf))

    # 2. Top1 − top2 margin (smaller margin = more uncertain = more OOD-like)
    print("[2/4] computing top1−top2 margin...", flush=True)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    results.append(_summarise("bm25_logreg top1−top2 margin (−margin)",
                              is_ood, -margin))

    # 3. One-class SVM on TF-IDF (decision_function: negative = outlier)
    print("[3/4] fitting one-class SVM on in-domain TF-IDF...", flush=True)
    vec = TfidfVectorizer(min_df=2)
    Xt_train = vec.fit_transform(X_train)
    Xt_test = vec.transform(X_test_full)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    oc.fit(Xt_train)
    # decision_function: positive for in-domain, negative for OOD; flip sign
    ocsvm_scores = -oc.decision_function(Xt_test)
    results.append(_summarise("one-class SVM (rbf) on TF-IDF",
                              is_ood, ocsvm_scores))

    # 4. AE reconstruction error (reference baseline)
    print("[4/4] fitting SklearnAutoencoder on in-domain TF-IDF (reference)...", flush=True)
    ae = SklearnAutoencoder(max_iter=200, random_state=0)
    ae.fit(Xt_train)
    ae_scores = ae.reconstruction_error(Xt_test)
    results.append(_summarise("SklearnAutoencoder reconstruction error (reference)",
                              is_ood, ae_scores))

    # ── report ─────────────────────────────────────────────────────────
    md = ["# CLINC150 OOD detection — method comparison\n"]
    md.append("Four OOD scoring strategies evaluated against CLINC150's gold "
              "`oos` labels. All four are trained only on in-domain utterances.\n")
    md.append(f"- training samples (in-domain): **{len(X_train)}**")
    md.append(f"- test samples (in-domain):     **{n_in}**")
    md.append(f"- test samples (OOD `oos`):     **{n_ood}**\n")
    md.append("| OOD scoring method | ROC AUC | TPR @ FPR=0.05 | TPR @ FPR=0.10 | TPR @ FPR=0.20 |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for r in results:
        md.append(f"| {r['name']} | **{r['auc']:.4f}** | {r['tpr_05']:.4f} | {r['tpr_10']:.4f} | {r['tpr_20']:.4f} |")
    md.append("")
    best = max(results, key=lambda r: r["auc"])
    md.append(f"**Best OOD detector: `{best['name']}` (AUC {best['auc']:.4f}).**\n")
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport: {OUT}")
    print(f"best: {best['name']}  AUC={best['auc']:.4f}")


if __name__ == "__main__":
    main()
