"""Sublinear TF-IDF.

Uses log(1+tf) instead of raw tf — helps when some words occur many
times in a single document.
"""

# %%
from jurebes.featurizers import tfidf_word_sublinear, tfidf_word

docs = ["the the the cat", "dog cat", "cat cat cat dog"]
raw = tfidf_word().fit_transform(docs).toarray()
sub = tfidf_word_sublinear().fit_transform(docs).toarray()
print(f"raw max  ={raw.max():.3f}")
print(f"sub max  ={sub.max():.3f}  (compressed)")
