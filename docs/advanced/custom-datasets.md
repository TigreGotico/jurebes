# Custom datasets

Every dataset loader in jurebes returns `tuple[list[str], list[str]]` — a list of utterances and a parallel list of labels.

## The contract

```python
def load_my_dataset(path: str | Path, **kwargs) -> tuple[list[str], list[str]]:
    ...
```

- `X`: list of utterance strings.
- `y`: list of intent-label strings (same length).
- No `None`, no empty strings — drop rows with missing data.
- Encoding: UTF-8.

## Built-in loaders

| function | format | location |
| --- | --- | --- |
| `load_csv(path, text="text", label="intent")` | CSV with header | `jurebes/datasets/csv.py` |
| `load_jsonl(path, text="text", label="intent")` | JSON-lines | `jurebes/datasets/jsonl.py` |
| `load_ovos_intents(directory)` | OVOS `.intent`/`.voc`/`.entity` files | `jurebes/datasets/ovos.py` |
| `load_hf(name, split, ...)` | HuggingFace `datasets` | `jurebes/datasets/huggingface.py` |

## Writing a custom loader

```python
# my_loader.py
from pathlib import Path

def load_my_format(path: str | Path) -> tuple[list[str], list[str]]:
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, _, text = line.partition("\t")
            if not text:
                continue
            X.append(text)
            y.append(label)
    if not X:
        raise ValueError(f"empty dataset at {path}")
    return X, y
```

Use it directly with the harness:

```python
from jurebes.benchmark import compare
X, y = load_my_format("data.tsv")
compare(["logreg", "linear_svc"], X, y, k=5)
```

## Error handling

Conventional error cases:

- Missing file → `FileNotFoundError`.
- Empty result → `ValueError("empty dataset")`.
- Format mismatch (e.g. CSV without expected column) → `ValueError(...)` with a clear message naming the expected and observed columns.
- Optional dependency missing (e.g. `datasets` for HF) → `ImportError("install jurebes[<extra>] to use ...")`.

## Adding a canonical loader

For datasets that should be reachable via the CLI's `@<name>` selector:

1. Write the loader under `jurebes/datasets/canonical/<name>.py` exposing `load_<name>()`.
2. Add it to the `CANONICAL` dict in `jurebes/datasets/canonical/__init__.py`:

```python
CANONICAL: Dict[str, Callable] = {
    "snips": load_snips,
    # …
    "my_dataset": load_my_dataset,
}
```

3. The CLI's `--dataset @my_dataset` selector picks it up automatically.

## CLI integration without modifying jurebes

For ad-hoc canonical-style datasets, wrap the harness in a small script:

```python
# benchmark_my_dataset.py
from jurebes.benchmark import compare, to_markdown
from my_loader import load_my_format

X, y = load_my_format("data.tsv")
result = compare(["@linear"], X, y, k=5)   # — see note below
print(to_markdown(result))
```

Note: `compare()` requires a flat list of baseline names. To use the `@group` selector, expand via `BASELINES.resolve("@linear")`:

```python
from jurebes.baselines import BASELINES
result = compare(BASELINES.resolve("@linear"), X, y, k=5)
```

## Sample order and stratification

`compare()` and `cross_validate()` use `StratifiedKFold`, which preserves class proportions regardless of input order. You may shuffle the dataset yourself before passing it in for reproducibility, but it is not required.

## Dataset hygiene

Before training, check for:

- Duplicate utterances under different labels (unresolvable noise).
- Empty or whitespace-only utterances.
- Labels with fewer than 3 samples (breaks `CalibratedClassifierCV` defaults).
- Heavy imbalance (>10:1) — see [../guides/handling-class-imbalance.md](../guides/handling-class-imbalance.md).

A small auditor:

```python
from collections import Counter

counts = Counter(y)
print(f"{len(set(X))} unique utterances, {len(X)} rows")
print(f"{len(counts)} classes, min={min(counts.values())} max={max(counts.values())}")

dup_labels = {x: {yy for xx, yy in zip(X, y) if xx == x} for x in set(X)}
multi = {x: ls for x, ls in dup_labels.items() if len(ls) > 1}
print(f"{len(multi)} utterances under multiple labels")
```

---
- Back to [docs index](../index.md)
