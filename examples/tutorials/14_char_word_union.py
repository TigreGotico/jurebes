"""Char + word FeatureUnion.

Combines word and char-ngram TF-IDF into a single feature space.
"""

# %%
from jurebes.featurizers import char_word_union

vec = char_word_union()
docs = ["hello world", "helo world", "see you later"]
X = vec.fit_transform(docs)
print(f"combined shape={X.shape}")
