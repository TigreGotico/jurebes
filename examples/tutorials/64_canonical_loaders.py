"""Canonical intent benchmark loaders.

SNIPS, CLINC150, BANKING77, HWU64, ATIS, MASSIVE — each wraps a
HuggingFace dataset id. This script lists them and the HF ids; it
does not actually fetch (would need jurebes[hf]).
"""

# %%
try:
    from jurebes.datasets.canonical import CANONICAL
except ImportError:
    print("install jurebes[hf] to use the canonical loaders")
    raise SystemExit(0)

print(f"canonical loaders available: {sorted(CANONICAL)}")
# HF dataset ids each loader pulls from
hf_ids = {
    "snips": "benayas/snips (fallback DeepPavlov/snips)",
    "clinc": "clinc_oos (config 'plus')",
    "banking77": "PolyAI/banking77",
    "hwu64": "DeepPavlov/hwu64",
    "atis": "tuetschek/atis",
    "massive": "AmazonScience/massive",
}
for name, hf in hf_ids.items():
    print(f"  {name:10s} -> {hf}")
# Would call e.g. load_clinc(split='train') — not executed to avoid network.
