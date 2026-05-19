"""Compare reduced-dimensionality methods: LSA vs NMF vs autoencoder.

Each pipeline ends with the same LogisticRegression head — only the
bottleneck featurizer differs.
"""

# %%
from jurebes.benchmark import compare

X = (
    ["hello", "hi", "hey there", "good morning", "howdy"] * 4
    + ["goodbye", "bye", "see you", "later", "farewell"] * 4
    + ["thanks", "thank you", "much appreciated", "ta", "cheers"] * 4
)
y = ["greet"] * 20 + ["bye"] * 20 + ["thanks"] * 20

names = ["lsa_logreg", "nmf_logreg", "lda_logreg", "autoencoder_logreg"]
result = compare(names, X, y, k=2)
for r in result.rows:
    print(f"{r.name:22s} macro_f1={r.macro_f1:.3f} train={r.train_seconds:.2f}s")
