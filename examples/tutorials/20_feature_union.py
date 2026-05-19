"""Generic FeatureUnion of featurizer builders.

Combines arbitrary builders — here TF-IDF word, char and text_stats —
into one feature space.
"""

# %%
from jurebes.featurizers import feature_union, tfidf_word, tfidf_char, text_stats

vec = feature_union(tfidf_word(), tfidf_char(), text_stats())
docs = ["hello world", "tell me the time", "play some jazz"]
X = vec.fit_transform(docs)
print(f"combined shape={X.shape}")
