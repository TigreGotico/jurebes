"""Latent Semantic Analysis via Truncated SVD.

Reduces a TF-IDF feature space to a small dense embedding.
"""

# %%
from jurebes.featurizers import lsa

docs = [
    "weather is sunny", "rain today", "snow tomorrow",
    "what time is it", "tell me the time", "current hour please",
]
vec = lsa(n_components=4)
X = vec.fit_transform(docs)
print(f"reduced shape={X.shape}")
print(f"first row: {X[0].round(3).tolist()}")
