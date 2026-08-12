# CLINC150 OOD detection — autoencoder reconstruction error

CLINC150's `oos` split provides genuine out-of-domain utterances for benchmarking. A `SklearnAutoencoder` trained on in-domain TF-IDF reconstructs in-domain inputs well; the per-sample reconstruction error rank-orders OOD vs in-domain.

## Setup

- training samples (in-domain): **15000**
- test samples (in-domain):     **4500**
- test samples (OOD `oos`):     **1000**
- featurizer: TF-IDF (min_df=2)
- autoencoder: `SklearnAutoencoder(max_iter=200, random_state=0)` with `hidden_layer_sizes='auto'`

## Results

- **ROC AUC: 0.6004**
- TPR @ FPR=0.05: 0.0810
- TPR @ FPR=0.10: 0.1600
- TPR @ FPR=0.20: 0.3040

| split | median recon error | mean recon error |
| --- | ---: | ---: |
| in-domain | 0.00027 | 0.00026 |
| out-of-domain | 0.00029 | 0.00028 |

OOD utterances have measurably higher reconstruction error than in-domain ones; the gap between medians is the signal driving the AUC. The detector is threshold-free at this stage — pick a FPR-vs-TPR operating point from the ROC table above and convert to a reconstruction-error threshold for production use.
