# CLINC150 OOD detection — method comparison

Four OOD scoring strategies evaluated against CLINC150's gold `oos` labels. All four are trained only on in-domain utterances.

- training samples (in-domain): **15000**
- test samples (in-domain):     **4500**
- test samples (OOD `oos`):     **1000**

| OOD scoring method | ROC AUC | TPR @ FPR=0.05 | TPR @ FPR=0.10 | TPR @ FPR=0.20 |
| --- | ---: | ---: | ---: | ---: |
| bm25_logreg top-1 confidence (1−conf) | **0.9254** | 0.6290 | 0.7910 | 0.9030 |
| bm25_logreg top1−top2 margin (−margin) | **0.9096** | 0.4690 | 0.7290 | 0.8930 |
| one-class SVM (rbf) on TF-IDF | **0.5529** | 0.0840 | 0.1550 | 0.2780 |
| SklearnAutoencoder reconstruction error (reference) | **0.6004** | 0.0810 | 0.1600 | 0.3040 |

**Best OOD detector: `bm25_logreg top-1 confidence (1−conf)` (AUC 0.9254).**
