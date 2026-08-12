"""HashingVectorizer wrapper.

Fixed-dim feature space, no vocabulary fit — ideal for streaming or
very large corpora.
"""

# %%
from jurebes.featurizers import hashing_word

vec = hashing_word(n_features=2 ** 10)
docs = ["hello world", "hello there", "world peace"]
X = vec.transform(docs)
print(f"shape={X.shape} nnz={X.nnz}")
