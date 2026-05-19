"""Pass a hand-built sklearn Pipeline to IntentClassifier.

IntentClassifier accepts any sklearn-compatible estimator; you are not
limited to the BASELINES registry.
"""

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes import IntentClassifier

custom = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(C=2.0, max_iter=500)),
])

clf = IntentClassifier(custom)
clf.add_intent("greet", ["hello", "hi there", "hey"])
clf.add_intent("bye", ["goodbye", "bye", "see you"])
clf.fit()

r = clf.predict("hi there friend")
print(f"custom pipeline -> {r.intent} ({r.confidence:.3f})")
