"""Character n-gram TF-IDF featurizer.

Char-level TF-IDF is robust to small typos and morphology — useful for
short queries with spelling variance.
"""

# %%
from jurebes.featurizers import tfidf_char

vec = tfidf_char(ngram_range=(3, 5))
docs = ["hello", "helo", "world", "wrld"]
X = vec.fit_transform(docs)
print(f"shape={X.shape}")
print(f"first 5 char-ngrams: {sorted(vec.vocabulary_)[:5]}")
