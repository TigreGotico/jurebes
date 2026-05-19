# Ensemble methods

An ensemble combines several base classifiers into a stronger meta-classifier. The hope is that errors made by individual learners are uncorrelated and cancel out in aggregate.

## Bagging

*Bootstrap Aggregating.* Sample $B$ bootstrap replicates of the training data; fit a base classifier on each; predict by majority vote (or averaged probabilities).

The variance of an averaged predictor is roughly $\rho \sigma^2 + (1-\rho) \sigma^2 / B$, where $\rho$ is the correlation between base learners. Bagging works when the base learner is high-variance (decision trees, kNN) and the bootstrap samples produce decorrelated models.

**Random Forest** (`random_forest`): bagging of decision trees with an extra trick — at each split, only a random subset of features is considered, further decorrelating the trees.

**Extra Trees** (`extra_trees`): like Random Forest but with random thresholds, not optimal ones. More variance reduction, slightly higher bias.

**`bagging_logreg`**: bagging of logistic regressions. Less common because LogReg is already low-variance; usually negligible improvement.

## Boosting

Sequential ensembles: each new base learner focuses on the examples the previous ones got wrong.

**Gradient Boosting** (`gradient_boosting`): fit each tree to the negative gradient of the loss w.r.t. the current model's predictions. With cross-entropy loss, this approximates the residuals on the probability simplex.

**HistGradientBoosting** (`hist_gbm`): a histogram-based gradient-boosted-trees implementation. Bins continuous features into 256 buckets, then trains on the binned data. Dramatically faster than `GradientBoostingClassifier` on large datasets. Requires dense input — the jurebes baseline densifies via `FunctionTransformer(lambda X: X.toarray())`.

Boosting reduces bias rather than variance. Works well when the base learners are weak (shallow trees) — strong base learners overfit when boosted.

## Voting

A meta-classifier that takes votes from several base classifiers:

- **Hard voting.** Majority vote on predicted labels.
- **Soft voting.** Average predicted probabilities; predict the argmax.

`voting_soft` averages `LogisticRegression`, calibrated `LinearSVC`, and `MultinomialNB`. The three errors are weakly correlated; their average usually beats any single member by 0.5–2 macro-F1 points.

## Stacking

A meta-learner trained on the *out-of-fold predictions* of the base learners:

1. Split the training data into $k$ folds.
2. For each fold, hold out and have each base learner predict on it after training on the rest.
3. Concatenate the held-out predictions into an $(N, K)$ matrix.
4. Train the meta-learner (typically logistic regression) on this matrix.

`stacking` uses `LogisticRegression`, calibrated `LinearSVC`, and `MultinomialNB` as base learners and `LogisticRegression` as the meta-learner.

Stacking is strictly more expressive than voting but more expensive — adds an extra layer of cross-validation at training time.

## When ensembles help

- Base learners' errors are *diverse*. Ensembling identical models gains nothing.
- Base learners are individually decent but each makes different mistakes.
- The variance-reduction benefit exceeds the latency cost.

## When ensembles do not help

- The base learner is already low-variance (linear model on tons of data).
- The dataset is tiny — ensemble variance reduction is dominated by base-model bias.
- Latency or memory budgets are tight — voting and stacking pay linear cost in the number of base learners.

## Baselines summary

| baseline | type | base learners | meta |
| --- | --- | --- | --- |
| `random_forest` | bagging | decision trees w/ random feature subsets | majority vote |
| `extra_trees` | bagging | decision trees w/ random thresholds | majority vote |
| `bagging_logreg` | bagging | logistic regressions | majority vote |
| `gradient_boosting` | boosting | shallow trees | weighted sum |
| `hist_gbm` | boosting (histogram) | shallow trees on binned features | weighted sum |
| `voting_soft` | voting | LR, LinearSVC(cal), MNB | averaged proba |
| `stacking` | stacking | LR, LinearSVC(cal), MNB | LR meta |
| `union_logreg` | feature union | char+word TF-IDF | LR on union |

`union_logreg` is technically not an ensemble — it concatenates *features* not *models* — but is grouped with `ensemble` in the registry because of its similar role as a "stronger default".

---
- Back to [docs index](../index.md)
