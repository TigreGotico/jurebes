# Kernel methods

Kernel methods let linear algorithms operate in a non-linear feature space via the *kernel trick*: replace every inner product $\mathbf{x}^\top \mathbf{z}$ with a kernel function $k(\mathbf{x}, \mathbf{z})$ that implicitly evaluates an inner product in some (possibly infinite-dimensional) feature space.

## The kernel trick

The dual formulation of SVM expresses the decision function as:

$$f(\mathbf{x}) = \sum_i \alpha_i y_i \langle \mathbf{x}_i, \mathbf{x} \rangle + b$$

Replace $\langle \cdot, \cdot \rangle$ with $k(\cdot, \cdot)$:

$$f(\mathbf{x}) = \sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b$$

If $k$ corresponds to an inner product in a higher-dimensional space, the SVM operates in that space without ever materialising the lifted features.

## Common kernels

**RBF (Radial Basis Function / Gaussian):**

$$k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$$

The $\gamma$ parameter controls the kernel width. Large $\gamma$ → tight, local decision regions (risk of overfitting). Small $\gamma$ → smooth, broad regions (risk of underfitting). sklearn's default `gamma="scale"` sets $\gamma = 1 / (d \cdot \mathrm{Var}(X))$.

**Polynomial:**

$$k(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^\top \mathbf{z} + c)^d$$

Degree $d$ controls the order of interactions captured. Rarely competitive on text vs RBF.

**Linear:**

$$k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$$

Equivalent to a vanilla linear SVM. The dual is more expensive than the primal at high $d$, so prefer `LinearSVC` over `SVC(kernel="linear")` for text.

## Computational cost

Kernel SVMs solve a quadratic programme in the dual:

- Training: roughly $O(N^2)$ to $O(N^3)$ depending on implementation. The kernel matrix is $N \times N$.
- Prediction: $O(N_{SV})$ per query, where $N_{SV}$ is the number of support vectors (often a substantial fraction of $N$).

For $N \gtrsim 10\,000$, kernel SVM training is impractical without approximations (Nyström, random Fourier features). On smaller corpora it remains a useful baseline.

## kNN

k-Nearest-Neighbours is not strictly a kernel method but shares the "compare to training examples" structure: classify a query by majority vote among its $k$ nearest training points under some distance metric. No training cost, but prediction requires a search through the training set; index structures (KD-tree, ball-tree) help for low-to-medium dimensions but degrade in high-dimensional sparse text. Baseline `knn`.

## Bundled baselines

| baseline | kernel | featurizer | notes |
| --- | --- | --- | --- |
| `rbf_svc` | RBF | `tfidf_word` | the canonical non-linear text SVM |
| `nusvc` | RBF | `tfidf_word` | nu-SVM variant; bounds the support-vector fraction |
| `lsa_rbf_svc` | RBF | `lsa(50)` | dense reduced-dim input — friendlier to RBF than sparse TF-IDF |
| `autoencoder_rbf_svc` | RBF | `autoencoder(base=tfidf_word)` | neural bottleneck before RBF |
| `knn` | implicit (distance-based) | `tfidf_word` | k-Nearest-Neighbours |

## Practical guidance

- Use `rbf_svc` only on corpora of a few thousand samples or fewer.
- Pair RBF with reduced-dim featurizers (`lsa_*`, `autoencoder_*`). Sparse TF-IDF makes the RBF distance noisy.
- Tune `C` and `gamma` jointly via random or Bayesian search.
- Default first guess: `C=1.0`, `gamma="scale"`. Then sweep $C$ over $\{0.1, 1, 10, 100\}$ and $\gamma$ over $\{0.01, 0.1, 1, 10\}$ in log-space.
- If the corpus is large and you must keep an RBF, switch to an approximation: `Nystroem` + `LinearSVC` or `RBFSampler` + `SGDClassifier`. Neither is in the jurebes registry; build them manually.

---
- Back to [docs index](../index.md)
