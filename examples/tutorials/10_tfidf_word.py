"""Word-level TF-IDF featurizer.

Demonstrates the canonical word TF-IDF and how to vary ngram_range and
min_df.
"""

# %% basic word TF-IDF
from jurebes.featurizers import tfidf_word

vec = tfidf_word(ngram_range=(1, 2), min_df=1)
docs = ["hello world", "hello there", "world peace", "say hello world"]
X = vec.fit_transform(docs)
print(f"shape={X.shape} vocab_size={len(vec.vocabulary_)}")
print(f"sample features: {sorted(vec.vocabulary_)[:5]}")
