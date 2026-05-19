"""Register a custom baseline in the BASELINES registry.

Custom factories slot into the registry and can be addressed by name or
group selector.
"""

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from jurebes.baselines import BASELINES


def my_baseline():
    return Pipeline([
        ("feat", TfidfVectorizer(ngram_range=(1, 3))),
        ("clf", LogisticRegression(C=4.0, max_iter=1000)),
    ])


BASELINES.register("my_baseline", my_baseline, group="linear")
print(f"'my_baseline' in registry: {'my_baseline' in BASELINES}")
print(f"groups for my_baseline: {sorted(BASELINES.in_group('my_baseline'))}")
print(f"@linear now contains it: {'my_baseline' in BASELINES.resolve('@linear')}")
