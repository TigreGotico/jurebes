# Linear classifiers

A linear classifier predicts a label from $\text{sign}(\mathbf{w}^\top \mathbf{x} + b)$ (binary) or $\arg\max_k (\mathbf{w}_k^\top \mathbf{x} + b_k)$ (multi-class). On sparse high-dimensional text features, linear models are usually the right default.

## Logistic regression

Models the conditional class probability as a softmax over linear scores:

$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}{\sum_{j} \exp(\mathbf{w}_j^\top \mathbf{x} + b_j)}$$

For binary classification this reduces to the sigmoid:

$$P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^\top \mathbf{x} + b))}$$

Parameters are estimated by minimising the (regularised) negative log-likelihood — equivalently, cross-entropy:

$$\min_{\mathbf{w}, b}\; \frac{1}{N} \sum_i -\log P(y_i \mid \mathbf{x}_i) + \lambda \|\mathbf{w}\|^2$$

The $\lambda \|\mathbf{w}\|^2$ term is L2 regularisation. Two other regularisers are common:

- **L1**: $\lambda \|\mathbf{w}\|_1$ — induces sparsity (feature selection). Baseline `logreg_l1` uses solver `saga` for this.
- **Elastic net**: convex combination of L1 and L2. Baseline `logreg_elasticnet` uses `l1_ratio=0.5`.

The sklearn `C` hyperparameter equals $1/\lambda$ — smaller `C` means stronger regularisation.

## Linear SVM

Trades the cross-entropy loss for the *hinge* loss:

$$\mathcal{L}_{\text{hinge}}(\mathbf{w}; \mathbf{x}, y) = \max(0, 1 - y \cdot (\mathbf{w}^\top \mathbf{x} + b))$$

A correct prediction with margin $\geq 1$ incurs zero loss; smaller margins are penalised linearly. The decision function is non-probabilistic — `LinearSVC` does not expose `predict_proba`. jurebes wraps it with `CalibratedClassifierCV` so the `IntentClassifier` interface remains uniform.

`LinearSVC` solves the primal optimisation directly; this is fast on high-dimensional sparse text features. The "dual vs primal" distinction matters for kernel SVMs (see [kernel-methods.md](kernel-methods.md)), not for the linear case here.

Bundled baselines: `linear_svc`, `linear_svc_char`, `linear_svc_hinge`, `ovr_linear_svc`, `ovo_linear_svc`.

## Ridge classifier

`RidgeClassifier` solves a regularised least-squares problem on $\{-1, +1\}$ targets. Closed-form solution via $(X^\top X + \lambda I)^{-1} X^\top y$. Fast, deterministic, and surprisingly competitive on small intent corpora. Baseline `ridge`.

## Perceptron

The Rosenblatt perceptron: update weights by $\mathbf{w} \leftarrow \mathbf{w} + y \mathbf{x}$ on misclassified examples. Converges in finite steps when the data is linearly separable. Less common in modern intent stacks but available as `perceptron`.

## Passive-aggressive

A family of online linear learners. Updates aggressively on misclassified examples while remaining passive (no update) when the prediction is correct with sufficient margin. Useful for streaming data. Baseline `passive_aggressive`.

## SGD with various losses

`SGDClassifier` is a generic stochastic-gradient-descent solver supporting many losses:

| `loss=` | equivalent model | jurebes baseline |
| --- | --- | --- |
| `"log_loss"` | logistic regression | `sgd_log`, `hashing_sgd_log` |
| `"hinge"` | linear SVM (non-probabilistic) | `sgd_hinge`, `hashing_sgd_hinge` |
| `"modified_huber"` | smoothed hinge (probabilistic) | `sgd_modified_huber` |

SGD trades exact convergence for speed and online-update support. Each `partial_fit` accepts a mini-batch, making it ideal for very large or streaming corpora.

## When linear is good enough

For sparse high-dimensional text features, linear models are typically the right default because:

- High-dimensional sparse data is almost always linearly separable (or close to it).
- Linear models fit and predict in milliseconds.
- They produce small, interpretable models.
- They calibrate well after a `CalibratedClassifierCV` wrap.

The case for non-linear models is strongest when:

- Feature interactions matter and cannot be encoded as bigrams.
- The corpus is large and dense (e.g. dense embeddings, not sparse TF-IDF).
- The decision surface is genuinely curved in feature space.

For most intent classifiers on tens to thousands of samples, `logreg` and `linear_svc` are the strongest defaults. Confirm empirically with [../cookbook/compare-all-linear.md](../cookbook/compare-all-linear.md).

---
- Back to [docs index](../index.md)
