# Cookbook: compare every linear baseline

End-to-end: load a canonical dataset, run a 5-fold CV across every linear baseline, render the Markdown report, interpret the result.

## Prerequisites

```bash
pip install jurebes[hf]
```

The `hf` extra installs `datasets` for the canonical loaders.

## Script

```python
from jurebes.benchmark import compare, to_markdown
from jurebes.baselines import BASELINES
from jurebes.datasets.canonical import load_banking77

X, y = load_banking77()
print(f"{len(X)} samples, {len(set(y))} intents")

linear_names = BASELINES.resolve("@linear")
print(f"comparing {len(linear_names)} linear baselines")

result = compare(linear_names, X, y, k=5,
                 scoring=("accuracy", "f1_macro", "log_loss"))

print(to_markdown(result, sort_by="macro_f1",
                  with_significance=True, precision=4))
```

## What happens

1. `load_banking77()` downloads the BANKING77 dataset from HuggingFace (caches under `~/.cache/huggingface/`) and returns `(X, y)`.
2. `BASELINES.resolve("@linear")` expands the `@linear` selector to the 15 linear baselines.
3. `compare(...)` runs stratified 5-fold CV per baseline. Each fold trains the pipeline factory's output on 4/5 of the data and evaluates on the holdout fold. Per-fold latencies are pooled for tail-percentile reporting.
4. `to_markdown(..., with_significance=True)` ranks the baselines by macro-F1, prints the table, and appends a Friedman+Nemenyi critical-difference block.

## Reading the output

The Markdown table includes columns for accuracy, macro-F1, micro-F1, per-class F1, training seconds, latency percentiles, model size, and the extra scoring metrics (`accuracy`, `log_loss`).

Below the table, the **Critical Difference** section ranks the baselines by mean fold-rank with the Nemenyi CD threshold. Baselines whose mean ranks are within CD of each other are grouped — for those pairs, the macro-F1 ordering is *not* statistically significant.

## Picking a winner

Two questions decide the winner:

1. **Top-rank within CD.** If exactly one baseline is at the top of the CD ranking and not tied with any other, it is the unambiguous winner.
2. **Tie-break.** Among baselines within CD of the top, prefer the one with the lower `predict_ms_p95_pooled` (latency) or smaller `model_size_bytes` (memory) — whichever constraint matters for your deployment.

Example interpretation:

```
Critical Difference = 0.823  (n=5)

rank  baseline
2.000  linear_svc_char
2.400  linear_svc
2.600  union_logreg
6.800  logreg

statistically indistinguishable groups:
  {linear_svc_char, linear_svc, union_logreg}
  {logreg}
```

`linear_svc_char`, `linear_svc`, and `union_logreg` are within CD. Pick the one with the smallest model or fastest inference; the macro-F1 difference between them is noise.

## Variations

Replace `load_banking77` with any canonical loader:

```python
from jurebes.datasets.canonical import (
    load_snips, load_clinc, load_banking77, load_hwu64, load_atis, load_massive,
)
```

Or substitute a local CSV:

```python
from jurebes.datasets import load_csv
X, y = load_csv("data.csv")
```

To compare a different family, swap the selector:

```python
result = compare(BASELINES.resolve("@naive_bayes"), X, y, k=5)
result = compare(BASELINES.resolve("@reduced_dim"), X, y, k=5)
result = compare(BASELINES.resolve("@tree"), X, y, k=5)
```

## CLI equivalent

```bash
jurebes benchmark --dataset @banking77 --baselines @linear \
  --cv 5 --scoring accuracy,f1_macro,log_loss \
  --sort-by macro_f1 --with-significance \
  --out banking_linear.md
```

## Next steps

- Tune the winner: [tune-with-random-search.md](tune-with-random-search.md).
- Run a full pipeline end-to-end: [full-research-pipeline.md](full-research-pipeline.md).
- Read about the stats: [../theory/statistical-comparison.md](../theory/statistical-comparison.md).

---
- Back to [docs index](../index.md)
