# Dimensionality reduction

Bag-of-words featurization produces sparse vectors with one column per vocabulary entry — tens of thousands of dimensions are typical. Reducing this to a small dense space can help downstream classifiers that struggle with high-dimensional sparse input.

## The curse of dimensionality

In high dimensions:

- Distance becomes uninformative — every point is roughly equidistant from every other.
- Density estimation requires exponentially more samples.
- Some classifiers (kernel SVM with non-linear kernels, kNN, QDA) degrade.

Linear classifiers and naive Bayes are largely immune — they treat each dimension independently. Reduced-dim featurizers exist for the classifiers that are not immune.

## LSA: Latent Semantic Analysis

LSA is truncated SVD applied to the TF-IDF matrix $X \in \mathbb{R}^{N \times d}$:

$$X \approx U \Sigma V^\top$$

Keep the top $k$ singular values and use $X V_k$ as the new representation. Each row is a length-$k$ embedding of the document in a *latent topic* space.

`jurebes.featurizers.lsa(n_components, base=None)`. Baselines `lsa_logreg`, `lsa_linear_svc`, `lsa_rbf_svc`.

Properties:

- Linear, fast.
- Components can be negative — no direct probabilistic interpretation.
- Captures synonymy implicitly (synonyms point in the same direction).
- The $k$-th component captures less variance than the $(k-1)$-th; pick $k$ via the elbow of the singular-value curve.

## NMF: Non-negative Matrix Factorisation

Factorises $X \approx W H$ with both factors element-wise non-negative:

$$\min_{W, H \geq 0} \|X - W H\|_F^2$$

Non-negativity yields *additive parts-based* components — each component is a non-negative combination of vocabulary entries, often interpretable as topic-like word clusters.

`jurebes.featurizers.nmf(n_components, base=None)`. Baseline `nmf_logreg`.

Properties:

- Slower than LSA (iterative optimisation).
- Components are directly interpretable as topics.
- No analytic ordering of components — they are exchangeable.
- Sensitive to initialisation; the jurebes wrapper pins `init="nndsvd"` and `random_state=0`.

## LDA: Latent Dirichlet Allocation

A fully Bayesian topic model: documents are mixtures of topics; topics are distributions over words. Latent variables drawn from Dirichlet priors. Fit by variational inference.

`jurebes.featurizers.lda_topics(n_topics, base=None)`. Baseline `lda_logreg`.

Properties:

- Probabilistic interpretation of topics.
- Per-document topic distribution is a normalised vector — sums to 1.
- Expensive to fit relative to LSA / NMF.
- Best on long documents; on short utterances the topic distribution is noisy.

## Autoencoders

A neural network trained to reconstruct its input through a bottleneck. The bottleneck activation is the compressed representation.

`jurebes.featurizers.SklearnAutoencoder(hidden_layer_sizes, ...)` builds the autoencoder on top of `sklearn.neural_network.MLPRegressor` fit to `y = X`. The `transform()` method runs the forward pass through the encoder layers up to and including the bottleneck.

```python
from jurebes.featurizers import SklearnAutoencoder

ae = SklearnAutoencoder(hidden_layer_sizes=(64, 16, 64), random_state=0)
ae.fit(dense_X)
Z = ae.transform(dense_X)                   # latent representation
errs = ae.reconstruction_error(dense_X)     # per-row MSE — useful for OOD
```

Baselines: `autoencoder_logreg`, `autoencoder_linear_svc`, `autoencoder_rbf_svc`.

Properties:

- Non-linear — can learn structure LSA / NMF miss.
- More hyperparameters; harder to tune.
- Reconstruction error is directly usable for out-of-domain detection. See [../guides/out-of-domain-detection.md](../guides/out-of-domain-detection.md).

## Comparison

| method | linearity | interpretable? | speed | use when |
| --- | --- | --- | --- | --- |
| LSA | linear | partial | fast | general-purpose latent representation |
| NMF | linear, non-negative | yes (topic-like) | medium | interpretable topics needed |
| LDA | probabilistic | yes (topic-like) | slow | document length supports topic estimation |
| autoencoder | non-linear | no | medium | non-linear structure + OOD detection |

## How many components

The conventional approach:

1. Plot the cumulative singular-value variance for LSA. Pick $k$ where the curve flattens (typically 50–200 for intent corpora).
2. For NMF / LDA, search $k$ over $\{10, 20, 50, 100\}$ on a held-out set, scoring by downstream classifier accuracy.
3. For autoencoders, pick the bottleneck size by the same downstream-accuracy procedure.

Avoid letting $k$ exceed the rank of the TF-IDF matrix — sklearn will raise `n_components must be <= n_features` (see [../getting-started/05-troubleshooting.md](../getting-started/05-troubleshooting.md)).

---
- Back to [docs index](../index.md)
