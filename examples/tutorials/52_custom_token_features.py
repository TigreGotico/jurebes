"""Custom token-feature function for the IOB tagger.

The tagger's vectorizer accepts any callable producing a dict per token.
Here a minimal feature set is used: word, prefix, suffix, position
flags.
"""

# %%
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes.slots import SklearnIOBTagger
from jurebes.slots import iob as iob_module


def my_token_features(tokens, i):
    tok = tokens[i]
    return {
        "lower": tok.lower(),
        "prefix2": tok[:2].lower(),
        "suffix2": tok[-2:].lower(),
        "bos": i == 0,
        "eos": i == len(tokens) - 1,
    }


# Wire the custom feature function into the iob module before fitting.
iob_module.token_features = my_token_features

est = Pipeline([("vec", DictVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
tagger = SklearnIOBTagger(estimator=est)
tagger.add_entity("city", ["paris", "lisbon"])
tagger.fit({"weather": ["weather in {city}", "forecast for {city}"]})
print(f"fitted={tagger.fitted}")
print(f"predict={tagger.predict('weather in paris')}")
