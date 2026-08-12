"""Load any HuggingFace text-classification dataset.

Guarded import — install jurebes[hf] to actually fetch. This script
prints the call signature instead of touching the network.
"""

# %%
try:
    from jurebes.datasets import load_hf  # noqa: F401
except ImportError:
    print("install jurebes[hf] to use load_hf")
    raise SystemExit(0)

# What we would call (NOT executed to avoid a network fetch):
print("load_hf('emotion', split='train', text_field='text', label_field='label')")
print("# returns (X: list[str], y: list[str]) using the dataset's class names if available.")
