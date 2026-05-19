# Choosing a baseline

A decision tree for picking among the 48 registered baselines without running a full benchmark first. When in doubt, run a comparison (see [cookbook/compare-all-linear.md](../cookbook/compare-all-linear.md)).

## Quick decision table

| situation | first try | second try | avoid |
| --- | --- | --- | --- |
| <100 samples total | `logreg`, `linear_svc`, `nb_complement` | `linear_svc_char` for typo tolerance | tree ensembles, MLP, RBF SVM |
| 100–10 000 samples | `linear_svc`, `logreg` | `rbf_svc`, `hist_gbm`, `voting_soft` | full grid search before a baseline picked |
| >10 000 samples | `hashing_sgd_log` (memory), `linear_svc` | `hist_gbm`, online SGD variants | dense kernel SVM, full LSA over a huge vocab |
| many intents (>50) | `linear_svc_char`, `ovr_linear_svc` | `linear_svc` | one-vs-one strategies (quadratic in classes) |
| latency-critical (<5 ms) | `linear_svc`, `nb_multinomial`, `hashing_sgd_log` | `nb_complement` | `rbf_svc`, `stacking`, `mlp_shallow` |
| imbalanced classes | `nb_complement`, `linear_svc` (`class_weight="balanced"`) | `complement_nb_count` | accuracy as primary metric |
| need calibrated probas | `logreg`, `nb_*`, any `_cal()`-wrapped baseline | `voting_soft` | raw RF/GBM (use `calibrate="always"`) |
| typo tolerance | `linear_svc_char`, `logreg_char`, `union_logreg` | `hashing_char` (custom build) | word-only TF-IDF |
| dict-of-string features | `categorical_logreg`, `categorical_random_forest` | custom pipeline w/ `CategoricalVectorizer` | text-only baselines |

## Decision flow

```
                      ┌─── start ───┐
                      │             │
                      ▼             ▼
            text input?         dict-of-string input?
                  │                 │
                  ▼                 ▼
       ┌──────────────────┐    categorical_logreg
       │   how much data? │    categorical_random_forest
       └────────┬─────────┘
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
    < 100   100-10k    > 10k
       │        │         │
       │        │         ▼
       │        │   hashing_sgd_log
       │        │   linear_svc
       │        ▼
       │   linear_svc, logreg, rbf_svc, hist_gbm
       │
       ▼
   logreg, linear_svc, nb_complement
       │
       ▼
   short / typo-prone utterances?
       │
       ▼
   linear_svc_char, logreg_char, union_logreg
```

## Family characteristics

| family | strengths | weaknesses |
| --- | --- | --- |
| Naive Bayes (`nb_*`, `complement_nb_count`) | extremely fast train+predict, tiny model size, strong on small data | independence assumption ignores word order; calibration mediocre |
| Linear (`logreg`, `linear_svc`, `sgd_*`, `ridge`) | excellent default for sparse text; calibrated probabilities cheap; small models | shallow decision surface |
| Linear with char features (`*_char`, `union_logreg`) | typo tolerance and OOV robustness | larger vocabulary → larger model |
| Kernel (`rbf_svc`, `nusvc`, `knn`, `lsa_rbf_svc`) | non-linear decision surface; sometimes meaningful gains on 100–10k corpora | O(n²) training cost; slow predict |
| Tree (`random_forest`, `extra_trees`, `gradient_boosting`, `hist_gbm`, `decision_tree`, `bagging_logreg`) | non-linear without kernel cost; `hist_gbm` scales well | raw probabilities poorly calibrated; large model size |
| Reduced-dim (`lsa_*`, `nmf_*`, `lda_*`, `autoencoder_*`) | compresses sparse vocab into dense latent factors; sometimes generalises better | extra hyperparameter (`n_components`); needs enough data to estimate the basis |
| Ensemble (`voting_soft`, `stacking`, `union_logreg`, `bagging_logreg`) | small accuracy gains by combining diverse learners | slowest train+predict; largest model size |
| Strategy (`ovr_linear_svc`, `ovo_linear_svc`) | controls the multi-class decomposition explicitly | rarely beats native multinomial on text |
| Discriminant (`lda_classifier`, `qda_classifier`) | closed-form fit; QDA captures class-specific covariances | requires dense input; LDA assumes shared covariance |
| Neural (`mlp_shallow`) | universal approximator | small data is the wrong regime — usually loses to linear |
| Categorical (`categorical_logreg`, `categorical_random_forest`) | accepts `list[dict[str, str]]` not text | only useful for non-text features |

## Cost trade-offs

A rough ordering of train cost (n = samples, v = vocabulary):

```
nb_*  ≈  hashing_sgd_*  <  linear_svc  ≈  logreg  <  reduced_dim  <  trees  <  rbf_svc  ≈  mlp_shallow  <  stacking
```

And of predict latency (per utterance):

```
linear_svc  ≈  logreg  ≈  nb_*  <  hashing_sgd_*  <  reduced_dim  <  trees  <  rbf_svc  <  voting_soft  ≈  stacking
```

These are orders of magnitude, not measurements. For your dataset run `compare(..., scoring=("accuracy", "f1_macro"))` and sort by `predict_ms_p95_pooled` to settle the trade-off empirically.

## A pragmatic default

If you have no other information about the dataset: start with `linear_svc` and use `compare` to verify against `logreg`, `nb_complement`, and `linear_svc_char` over five folds.

See also: [theory/linear-classifiers.md](../theory/linear-classifiers.md), [theory/text-featurization.md](../theory/text-featurization.md), [reference/baselines.md](../reference/baselines.md).

---
- Back to [docs index](../index.md)
