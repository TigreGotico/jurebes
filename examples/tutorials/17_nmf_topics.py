"""Non-negative matrix factorisation topics.

NMF on TF-IDF yields non-negative latent factors that can be read as
topics.
"""

# %%
from jurebes.featurizers import nmf

docs = [
    "weather is sunny today", "rain forecast tomorrow", "snow this week",
    "play some jazz", "stop the music", "next song please",
]
vec = nmf(n_components=2)
X = vec.fit_transform(docs)
print(f"topic activations shape={X.shape}")
for i, row in enumerate(X.round(3)):
    print(f"  doc{i}: {row.tolist()}")
