# `BASELINES` registry

Module: `jurebes.baselines`. Re-exported at `from jurebes import BASELINES`.

## The registry object

`BASELINES` is an instance of `_Registry`. Public methods:

| method | signature | meaning |
| --- | --- | --- |
| `names()` | `() -> list[str]` | all registered baseline names |
| `build(name)` | `(str) -> Pipeline` | construct a fresh pipeline by name |
| `register(name, factory, *, group=None)` | `(str, Callable[[], Pipeline], str?) -> None` | add a new baseline (optionally tag a group) |
| `add_to_group(group, name)` | `(str, str) -> None` | tag an existing baseline with another group |
| `groups()` | `() -> dict[str, set[str]]` | group-name → baseline-names map |
| `in_group(name)` | `(str) -> set[str]` | groups this baseline belongs to |
| `resolve(selector)` | `(str) -> list[str]` | resolve a name or `@group` selector |

Magic methods: `__contains__(name)`, `__iter__()`, `__len__()`.

## Selectors

`resolve("name")` returns `["name"]`. `resolve("@group")` returns sorted member names. `resolve("@all")` returns every registered name.

The CLI's `--baselines` flag accepts comma-separated selectors and de-duplicates the resolved set.

## The 48 baselines

| name | featurizer | classifier | groups |
| --- | --- | --- | --- |
| `nb_multinomial` | `tfidf_word` | `MultinomialNB` | naive_bayes |
| `nb_complement` | `tfidf_word` | `ComplementNB` | naive_bayes |
| `nb_bernoulli` | `count_word(binary=True)` | `BernoulliNB` | naive_bayes |
| `complement_nb_count` | `count_word` | `ComplementNB` | naive_bayes |
| `logreg` | `tfidf_word` | `LogisticRegression(max_iter=1000)` | linear |
| `logreg_char` | `tfidf_char` | `LogisticRegression(max_iter=1000)` | linear |
| `logreg_l1` | `tfidf_word` | `LogisticRegression(penalty="l1", solver="saga")` | linear |
| `logreg_elasticnet` | `tfidf_word` | `LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)` | linear |
| `linear_svc` | `tfidf_word` | calibrated `LinearSVC` | linear |
| `linear_svc_char` | `tfidf_char` | calibrated `LinearSVC` | linear |
| `linear_svc_hinge` | `tfidf_word` | calibrated `LinearSVC(loss="hinge")` | linear |
| `sgd_log` | `tfidf_word` | `SGDClassifier(loss="log_loss")` | linear, online |
| `sgd_hinge` | `tfidf_word` | calibrated `SGDClassifier(loss="hinge")` | linear, online |
| `sgd_modified_huber` | `tfidf_word` | `SGDClassifier(loss="modified_huber")` | linear, online |
| `passive_aggressive` | `tfidf_word` | calibrated `PassiveAggressiveClassifier` | linear |
| `perceptron` | `tfidf_word` | calibrated `Perceptron` | linear |
| `ridge` | `tfidf_word` | calibrated `RidgeClassifier` | linear |
| `hashing_sgd_log` | `hashing_word` | `SGDClassifier(loss="log_loss")` | linear, online |
| `hashing_sgd_hinge` | `hashing_word` | calibrated `SGDClassifier(loss="hinge")` | linear, online |
| `rbf_svc` | `tfidf_word` | `SVC(kernel="rbf", probability=True)` | kernel |
| `nusvc` | `tfidf_word` | `NuSVC(probability=True)` | kernel |
| `knn` | `tfidf_word` | `KNeighborsClassifier()` | kernel |
| `lsa_rbf_svc` | `lsa(50)` | `SVC(kernel="rbf", probability=True)` | kernel, reduced_dim |
| `random_forest` | `tfidf_word` | `RandomForestClassifier()` | tree |
| `extra_trees` | `tfidf_word` | `ExtraTreesClassifier()` | tree |
| `gradient_boosting` | `tfidf_word` | `GradientBoostingClassifier()` | tree |
| `hist_gbm` | `tfidf_word` + dense | `HistGradientBoostingClassifier(min_samples_leaf=1)` | tree |
| `decision_tree` | `tfidf_word` | `DecisionTreeClassifier()` | tree |
| `bagging_logreg` | `tfidf_word` | `BaggingClassifier(LogisticRegression)` | tree, ensemble |
| `mlp_shallow` | `tfidf_word` | `MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)` | neural |
| `voting_soft` | `tfidf_word` | `VotingClassifier([LR, cal(LinearSVC), MNB], voting="soft")` | ensemble |
| `stacking` | `tfidf_word` | `StackingClassifier([LR, cal(LinearSVC), MNB], final=LR)` | ensemble |
| `union_logreg` | `char_word_union` | `LogisticRegression(max_iter=1000)` | ensemble |
| `lsa_logreg` | `lsa(50, tfidf_word())` | `LogisticRegression(max_iter=1000)` | reduced_dim |
| `lsa_linear_svc` | `lsa(50)` | calibrated `LinearSVC` | reduced_dim |
| `nmf_logreg` | `nmf(50)` | `LogisticRegression(max_iter=1000)` | reduced_dim |
| `lda_logreg` | `lda_topics(20)` | `LogisticRegression(max_iter=1000)` | reduced_dim |
| `autoencoder_logreg` | `autoencoder(base=tfidf_word)` | `LogisticRegression(max_iter=1000)` | reduced_dim |
| `autoencoder_linear_svc` | `autoencoder(base=tfidf_word)` | calibrated `LinearSVC` | reduced_dim |
| `autoencoder_rbf_svc` | `autoencoder(base=tfidf_word)` | `SVC(kernel="rbf", probability=True)` | reduced_dim |
| `ovr_linear_svc` | `tfidf_word` | calibrated `OneVsRestClassifier(LinearSVC)` | strategy |
| `ovo_linear_svc` | `tfidf_word` | calibrated `OneVsOneClassifier(LinearSVC)` | strategy |
| `lda_classifier` | `tfidf_word` + dense | `LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")` | discriminant |
| `qda_classifier` | `lsa(20)` | `QuadraticDiscriminantAnalysis(reg_param=0.5)` | discriminant |
| `text_stats_logreg` | `text_stats` | `LogisticRegression(max_iter=1000)` | feature_engineering |
| `union_text_stats_logreg` | `feature_union(tfidf_word, text_stats)` | `LogisticRegression(max_iter=1000)` | feature_engineering |
| `categorical_logreg` | `categorical` | `LogisticRegression(max_iter=1000)` | categorical |
| `categorical_random_forest` | `categorical` | `RandomForestClassifier()` | categorical |

`calibrated X` = `CalibratedClassifierCV(X, cv=3)`. `categorical_*` baselines accept `list[dict[str, str]]` input rather than raw text.

## Group summary

| group | members |
| --- | --- |
| `naive_bayes` | nb_multinomial, nb_complement, nb_bernoulli, complement_nb_count |
| `linear` | logreg, logreg_char, logreg_l1, logreg_elasticnet, linear_svc, linear_svc_char, linear_svc_hinge, sgd_log, sgd_hinge, sgd_modified_huber, passive_aggressive, perceptron, ridge, hashing_sgd_log, hashing_sgd_hinge |
| `kernel` | rbf_svc, nusvc, knn, lsa_rbf_svc |
| `tree` | random_forest, extra_trees, gradient_boosting, hist_gbm, decision_tree, bagging_logreg |
| `neural` | mlp_shallow |
| `ensemble` | voting_soft, stacking, union_logreg, bagging_logreg |
| `reduced_dim` | lsa_logreg, lsa_linear_svc, lsa_rbf_svc, nmf_logreg, lda_logreg, autoencoder_logreg, autoencoder_linear_svc, autoencoder_rbf_svc |
| `online` | sgd_log, sgd_hinge, sgd_modified_huber, hashing_sgd_log, hashing_sgd_hinge |
| `strategy` | ovr_linear_svc, ovo_linear_svc |
| `feature_engineering` | text_stats_logreg, union_text_stats_logreg |
| `discriminant` | lda_classifier, qda_classifier |
| `categorical` | categorical_logreg, categorical_random_forest |

Baselines can belong to multiple groups (e.g. `sgd_log` is both `linear` and `online`); the union of all group rows therefore exceeds the unique-name count of 48.

## Default

```python
from jurebes.baselines import default
default()                # → BASELINES.build("linear_svc")
```

Used by `IntentClassifier()` when no estimator is passed.

---
- Back to [reference index](index.md)
