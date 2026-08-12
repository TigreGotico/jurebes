"""BASELINES registry tour.

Lists names, resolves group selectors, and inspects group membership.
"""

# %%
from jurebes.baselines import BASELINES

print(f"total baselines registered: {len(BASELINES)}")
print(f"first 5 names: {BASELINES.names()[:5]}")

# %% group selectors
linear = BASELINES.resolve("@linear")
all_names = BASELINES.resolve("@all")
print(f"@linear ({len(linear)}): {linear[:5]}...")
print(f"@all   ({len(all_names)}) baselines")

# %% groups mapping
groups = BASELINES.groups()
print(f"groups: {sorted(groups)}")
print(f"logreg belongs to: {sorted(BASELINES.in_group('logreg'))}")
