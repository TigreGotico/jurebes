"""SNIPS quickstart — fetch, train, evaluate.

Requires: pip install jurebes[hf]
Not run in CI (depends on HuggingFace download).
"""

# %% [markdown]
# # SNIPS quickstart
# Train a default IntentClassifier on SNIPS and evaluate on the test split.

# %%
from sklearn.metrics import accuracy_score, f1_score

from jurebes import IntentClassifier
from jurebes.datasets.canonical import load_snips

# %%
X_train, y_train = load_snips("train")
X_test, y_test = load_snips("test")
print(f"train={len(X_train)}  test={len(X_test)}  intents={len(set(y_train))}")

# %%
clf = IntentClassifier()
for label in sorted(set(y_train)):
    samples = [x for x, y in zip(X_train, y_train) if y == label]
    clf.add_intent(label, samples)
clf.fit()

# %%
preds = [clf.predict(x).intent for x in X_test]
print(f"accuracy   = {accuracy_score(y_test, preds):.4f}")
print(f"macro F1   = {f1_score(y_test, preds, average='macro'):.4f}")
