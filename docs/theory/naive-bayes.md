# Naive Bayes

The classical generative approach to text classification. Cheap to train, cheap to predict, and competitive on small corpora.

## Bayes' theorem applied

For class $y$ and document features $\mathbf{x} = (x_1, \ldots, x_d)$:

$$P(Y = y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid Y = y) \cdot P(Y = y)}{P(\mathbf{x})}$$

The denominator is constant across $y$, so prediction reduces to $\arg\max_y P(\mathbf{x} \mid Y = y) \cdot P(Y = y)$.

## The independence assumption

Naive Bayes assumes the features are conditionally independent given the class:

$$P(\mathbf{x} \mid Y = y) = \prod_j P(x_j \mid Y = y)$$

This is "naive" because in text it is obviously false (the word "york" depends heavily on "new"). The assumption fails as a generative model but often holds well enough for classification because misranking requires the *product* to flip — and ranking is more robust to small per-feature distortions than density estimation is.

## Multinomial NB

Models $P(x_j \mid y)$ as a multinomial probability over the vocabulary, with smoothing parameter $\alpha$ (Laplace / additive):

$$P(w_j \mid y) = \frac{\text{count}(w_j, y) + \alpha}{\sum_k \text{count}(w_k, y) + \alpha \cdot |V|}$$

Best on count-vector input. With TF-IDF input the math is technically incorrect (the multinomial likelihood expects counts), but it works fine in practice because the relative magnitudes still rank classes correctly. Baseline `nb_multinomial`.

## Bernoulli NB

Models each feature as a binary indicator: present or absent. Best on binary count-vector input (`CountVectorizer(binary=True)`). Useful for very short documents where token counts are all 0 or 1. Baseline `nb_bernoulli`.

## Gaussian NB

Models $P(x_j \mid y)$ as a per-class Gaussian. Best on dense continuous features, not sparse text. Not registered as a jurebes baseline; build manually if needed.

## Complement NB (Rennie et al. 2003)

Modifies Multinomial NB to estimate parameters from the *complement* of each class — every document not in $y$:

$$\hat{\theta}_{y, j} = \log \frac{\sum_{i : y_i \neq y} x_{i,j} + \alpha}{\sum_{i : y_i \neq y} \sum_k x_{i,k} + \alpha \cdot |V|}$$

This compensates for the bias Multinomial NB has on imbalanced corpora: when one class has many more documents than the others, its parameter estimates dominate. Complement NB inverts this so the *minority* class's parameters are estimated from the abundant complement, giving smoother estimates.

In practice Complement NB outperforms Multinomial NB on imbalanced text — sometimes by large margins. Baselines `nb_complement` (TF-IDF input) and `complement_nb_count` (count input).

## Why TF-IDF can "hurt" Multinomial NB

The multinomial likelihood assumes integer counts. TF-IDF produces real-valued weighted counts that have lost the strict count interpretation. Strictly speaking the resulting model is no longer the principled multinomial-NB, but the predicted class rankings are usually still correct. Reported accuracies on TF-IDF inputs with `MultinomialNB` are sometimes higher than with raw counts because the IDF weighting downweights common words.

## When NB shines

- **Very small corpora.** With 5–20 samples per class, NB's strong prior often beats more flexible models that overfit.
- **Latency-critical settings.** A trained NB classifier is a few dictionaries; prediction is a handful of dictionary lookups and adds.
- **Imbalanced classes (with Complement NB).** Designed for this case.
- **Sanity baseline.** Always run `nb_complement` alongside `logreg` and `linear_svc` in a benchmark; if NB wins, that is a strong signal the corpus is small.

## When NB falls short

- Heavy feature interactions (the independence assumption bites when phrases matter more than words).
- Long documents with rich syntax.
- Multi-modal feature spaces.

---
- Back to [docs index](../index.md)
