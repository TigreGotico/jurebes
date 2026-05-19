# Intent classification

## Problem statement

Given a finite set of intent labels $\mathcal{Y} = \{y_1, \ldots, y_K\}$ and an utterance $u$ drawn from natural-language strings $\mathcal{U}$, an intent classifier is a function $f : \mathcal{U} \to \mathcal{Y}$. The classifier is *learned*: given a training set $\{(u_i, y_i)\}_{i=1}^N$, the goal is to recover $f$ that minimises expected misclassification on future inputs from the same distribution.

A probabilistic classifier outputs $P(Y = y \mid U = u)$ for every $y$; the deterministic prediction is $\arg\max_y P(Y = y \mid U = u)$.

## Why classical ML on small intent corpora

Intent inventories in deployed voice assistants typically contain dozens to a few hundred intents, each with tens to hundreds of training utterances. At this scale, classical pipelines (TF-IDF + linear classifier) consistently match or beat transformer fine-tuning along several axes:

- **Data efficiency.** Linear models fit with high statistical efficiency on tens of samples per class; transformers need orders of magnitude more before specialised features beat sparse-bag-of-words generalisation.
- **Latency.** Linear and naive-Bayes predict in microseconds per utterance on CPU; transformers take milliseconds even with quantisation.
- **Memory.** A linear classifier over a 5 000-word vocabulary occupies tens of kB; a small transformer is hundreds of MB.
- **Interpretability.** The coefficient of "joke" under the `tell_joke` class is directly readable in a `LogisticRegression`. Attention maps are not directly interpretable.

These trade-offs flip on larger corpora and richer paraphrase distributions, which is when transformer-based pipelines win. jurebes intentionally occupies the classical-ML regime.

## Adjacent tasks

Intent classification is distinct from:

- **Slot filling.** Per-token labelling that extracts named arguments from an utterance. jurebes pairs intent classification with `SklearnIOBTagger` for slot filling.
- **Dialogue act classification.** Tags an utterance by its conversational role (question / statement / acknowledgement). Output space is a different ontology.
- **Semantic parsing.** Maps an utterance to a structured logical form (a tree of operators). Output space is unbounded.
- **Open-ended text classification.** Topic classification, sentiment analysis, spam detection. The techniques transfer, but the typical class count and corpus shape differ.

## Evaluation

For multi-class single-label intent classification:

- **Accuracy** = fraction of correctly labelled utterances. Equals micro-F1 in the single-label case.
- **Macro-F1** = unweighted mean of per-class F1. Penalises poor minority-class performance.
- **Per-class F1** = $\frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$ per class. Surfaces which classes the model handles poorly.
- **Log-loss** = $-\frac{1}{N}\sum_i \log P(y_i \mid u_i)$. Sensitive to mis-calibrated probabilities.
- **Top-k accuracy** = fraction of utterances where the true label is in the top-k predicted ranks. Useful when downstream disambiguation is available.

`jurebes.benchmark` reports macro-F1 and micro-F1 by default and exposes `accuracy`, `balanced_accuracy`, `log_loss`, and `top_k_accuracy` via the `scoring=` parameter.

For confidence thresholds and accept / reject trade-offs, see [calibration.md](calibration.md) and [../guides/confidence-thresholds.md](../guides/confidence-thresholds.md).

## Generalisation

The training distribution rarely covers the deployment distribution:

- **Vocabulary drift.** Users invent phrasings the training corpus never saw. Character n-grams and hashing featurizers handle this gracefully (see [text-featurization.md](text-featurization.md)).
- **Out-of-domain inputs.** Utterances unrelated to any intent. Pure classifiers always return *some* label; reject them with a confidence threshold or a dedicated OOD detector ([../guides/out-of-domain-detection.md](../guides/out-of-domain-detection.md)).
- **Asymmetric costs.** A wrong `shutdown` is worse than a wrong `tell_joke`. Tune per-class thresholds rather than a single global one.

## Practical pipeline

```
1. Define intents and gather samples.
2. Pick a metric (typically macro-F1 for imbalanced inventories).
3. Run a k-fold benchmark across baseline families.
4. Apply Friedman + Nemenyi to pick a winner with statistical justification.
5. Tune the winner with random or Bayesian search.
6. Evaluate on a held-out test set.
7. Calibrate (verify) and set thresholds.
8. Deploy.
```

Steps 3–4 are the core of jurebes; steps 5–6 sit in the search subsystem; step 7 sits in [calibration.md](calibration.md) plus [../guides/confidence-thresholds.md](../guides/confidence-thresholds.md). Step 8 is application-specific and partly handled by the OVOS plugin in `jurebes.opm`.

---
- Back to [docs index](../index.md)
