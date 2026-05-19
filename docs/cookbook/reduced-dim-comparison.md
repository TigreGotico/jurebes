# Cookbook: compare reduced-dim featurizers

Three reduced-dim featurizers — LSA, NMF, and a neural autoencoder — feed into a common downstream classifier. Which preserves intent separability best on your data? Benchmark them.

## Prerequisites

```bash
pip install jurebes[hf]
```

## Script

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes.baselines import BASELINES
from jurebes.benchmark import compare, to_markdown
from jurebes.datasets.canonical import load_snips
from jurebes.featurizers import autoencoder, lsa, nmf, tfidf_word


# Register a controlled trio: same downstream LogReg, varying featurizer.
def _register(name, feat_builder):
    BASELINES.register(
        name,
        lambda: Pipeline([
            ("feat", feat_builder()),
            ("clf",  LogisticRegression(max_iter=1000)),
        ]),
        group="reduced_dim",
    )

_register("rdc_lsa_32", lambda: lsa(32, tfidf_word()))
_register("rdc_nmf_32", lambda: nmf(32, tfidf_word()))
_register("rdc_ae_32",  lambda: autoencoder(hidden_layer_sizes=(64, 32, 64), base=tfidf_word()))

X, y = load_snips()
result = compare(["rdc_lsa_32", "rdc_nmf_32", "rdc_ae_32"], X, y, k=5,
                 scoring=("accuracy", "f1_macro"))
print(to_markdown(result, sort_by="macro_f1", with_significance=True))
```

## Why control the downstream classifier

Different reduced-dim methods produce vectors with different geometries (sign, scale, sparsity). Pairing each with its "most-compatible" downstream classifier muddies the comparison. Holding the downstream classifier fixed isolates the featurizer's contribution.

`LogisticRegression` is a good fixed downstream: it handles both dense and sparse, both positive and signed input, and is well-calibrated.

## Choosing `n_components`

The `32` in the script is a placeholder. The right value depends on:

- The vocabulary size — `n_components` cannot exceed the TF-IDF rank.
- The intent count — fewer classes need fewer latent dimensions to separate.
- The corpus size — too many components on too few samples overfits.

A pragmatic sweep:

```python
for n in (8, 16, 32, 64, 128):
    _register(f"rdc_lsa_{n}", lambda n=n: lsa(n, tfidf_word()))
result = compare([f"rdc_lsa_{n}" for n in (8, 16, 32, 64, 128)], X, y, k=5)
print(to_markdown(result, sort_by="macro_f1"))
```

The macro-F1 curve typically plateaus or dips after the optimal $n$.

## Interpreting the result

Typical pattern on small intent corpora:

- **LSA** wins on speed and is a strong default.
- **NMF** sometimes edges out LSA when the topics have a parts-based structure (named-entity-style features in the vocabulary).
- **Autoencoder** can help when there is non-linear structure but needs enough data — on SNIPS-size corpora (~13k samples) it is competitive; on tiny corpora (<1k) it usually underperforms.

The Friedman+Nemenyi block confirms whether the gap is statistically meaningful or noise.

## When reduced-dim is worth it at all

For sparse-TF-IDF + linear classifier on intent text, reduced-dim rarely beats the raw TF-IDF baseline (`logreg`, `linear_svc`). It helps mainly:

- For non-linear downstream classifiers (RBF SVM, MLP, QDA).
- When the deployment needs a dense fixed-size representation (e.g. for an external indexer).
- For out-of-domain detection via reconstruction error (see [../guides/out-of-domain-detection.md](../guides/out-of-domain-detection.md)).

Verify empirically — `compare(["logreg"] + reduced_dim_names, ...)` reveals whether the dense detour helps.

## Related cookbooks

- [ood-with-autoencoder.md](ood-with-autoencoder.md) — using the autoencoder's reconstruction error as an OOD score.
- [compare-all-linear.md](compare-all-linear.md) — full-family comparison protocol.

---
- Back to [docs index](../index.md)
