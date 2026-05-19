"""BANKING77 full research flow — compare, test, tune, evaluate, save.

Requires: pip install jurebes[hf,bench-plot]
Not run in CI (depends on HuggingFace download).
"""

# %% [markdown]
# # BANKING77 research flow
# Compare linear baselines, run Friedman+Nemenyi, tune the winner, and
# evaluate on the test holdout.

# %%
from sklearn.metrics import accuracy_score, f1_score

from jurebes.baselines import BASELINES
from jurebes.benchmark import compare, to_markdown
from jurebes.benchmark.stats import critical_difference, friedman_nemenyi
from jurebes.datasets.canonical import load_banking77
from jurebes.search import search, spaces

# %%
X_train, y_train = load_banking77("train")
X_test, y_test = load_banking77("test")
print(f"train={len(X_train)} test={len(X_test)} intents={len(set(y_train))}")

# %% [markdown]
# ## 1. Compare linear baselines with 5-fold CV

# %%
linear = BASELINES.resolve("@linear")
result = compare(linear, X_train, y_train, k=5)
print(to_markdown(result, sort_by="macro_f1"))

# %% [markdown]
# ## 2. Friedman + Nemenyi critical-difference test

# %%
fold_scores = {name: scores["f1_macro"]
               for name, scores in result.fold_scores_by_baseline.items()}
fr = friedman_nemenyi(fold_scores)
print(f"Friedman p={fr.pvalue:.4f}  reject_null={fr.reject_null}")
cd = critical_difference(fold_scores)
print(cd.to_ascii())

# %% [markdown]
# ## 3. Tune the winning baseline with random search

# %%
winner = min(fold_scores, key=lambda n: cd.mean_ranks[n])
print(f"winner = {winner}")
tuned = search(
    winner, spaces.for_baseline(winner), X_train, y_train,
    backend="random", n_iter=20, cv=5, scoring="f1_macro",
)
print(f"best_score={tuned.best_score:.4f}  params={tuned.best_params}")

# %% [markdown]
# ## 4. Evaluate tuned model on the test holdout

# %%
est = tuned.best_estimator
preds = [est.predict(x).intent for x in X_test]
print(f"test accuracy = {accuracy_score(y_test, preds):.4f}")
print(f"test macro F1 = {f1_score(y_test, preds, average='macro'):.4f}")

# %% [markdown]
# ## 5. Save the tuned model

# %%
est.save("banking77_tuned.joblib")
print("saved -> banking77_tuned.joblib")
