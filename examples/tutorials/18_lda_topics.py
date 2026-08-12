"""Latent Dirichlet Allocation topics.

LDA on count features assigns a topic-distribution to each document.
"""

# %%
from jurebes.featurizers import lda_topics

docs = [
    "weather is sunny today", "rain forecast tomorrow", "snow this week",
    "play some jazz", "stop the music", "next song please",
]
vec = lda_topics(n_topics=2)
X = vec.fit_transform(docs)
print(f"topic distribution shape={X.shape}")
print(f"doc0 topic mix: {X[0].round(3).tolist()}")
