"""CountVectorizer wrapper.

Raw count features (or binary presence) — useful for naive-Bayes
variants and quick baselines.
"""

# %%
from jurebes.featurizers import count_word

vec = count_word(binary=False, ngram_range=(1, 1))
docs = ["the cat sat", "the dog ran", "cat dog cat"]
X = vec.fit_transform(docs).toarray()
print(f"vocab={vec.get_feature_names_out().tolist()}")
print(f"counts=\n{X}")
