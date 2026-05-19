"""Text statistics featurizer.

Per-utterance dense vector of [n_chars, n_words, mean_word_len,
digit_ratio, upper_ratio, punct_ratio, oov_rate].
"""

# %%
from jurebes.featurizers import text_stats

vec = text_stats()
train = ["hello world", "tell me the time"]
vec.fit(train)
out = vec.transform(["HELLO 123!!!", "tell me"])
print(f"feature shape={out.shape}")
print(f"row0 (uppercase + digits + punct): {out[0].round(3).tolist()}")
print(f"row1 (in-vocab plain): {out[1].round(3).tolist()}")
