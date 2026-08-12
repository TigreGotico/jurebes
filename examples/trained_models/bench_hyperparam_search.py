"""Hyperparameter search vs default — how much does jurebes.search buy?

For each (dataset, baseline) pair: compare default-hyperparam CV macro-F1
against random-search-tuned CV macro-F1 using ``spaces.for_baseline()``.

Writes a single ``hyperparam_search.md`` with the tuned-vs-default table.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from jurebes.benchmark import compare
from jurebes.datasets.canonical import (
    load_banking77, load_clinc, load_snips,
)
from jurebes.search import search, spaces

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports" / "hyperparam_search.md"

LOADERS = {
    "snips":     load_snips,
    "banking77": load_banking77,
    "clinc":     load_clinc,
}
BASELINES_UNDER_TEST = ["linear_svc_char", "bm25_logreg"]


def _default_score(name, X, y, cv=3):
    """5-fold-CV macro-F1 with default hyperparameters."""
    r = compare([name], X, y, k=cv).rows[0]
    return float(np.mean(r.fold_scores["f1_macro"]))


def _tuned_score(name, X, y, cv=3, n_iter=15):
    space = spaces.for_baseline(name)
    t0 = time.perf_counter()
    res = search(name, space, X, y, backend="random",
                  n_iter=n_iter, cv=cv, scoring="f1_macro")
    return res.best_score, res.best_params, time.perf_counter() - t0


def main():
    rows = []
    for ds, loader in LOADERS.items():
        X, y = loader("train")
        for name in BASELINES_UNDER_TEST:
            print(f"[{ds}/{name}] default...", flush=True)
            default = _default_score(name, X, y)
            print(f"  default macro-F1 = {default:.4f}", flush=True)
            print(f"[{ds}/{name}] random search (n_iter=15, cv=3)...", flush=True)
            tuned, params, wall = _tuned_score(name, X, y)
            print(f"  tuned   macro-F1 = {tuned:.4f}  delta = {tuned-default:+.4f}",
                  f"({wall:.0f}s)", flush=True)
            rows.append({
                "dataset": ds, "baseline": name,
                "default": default, "tuned": tuned,
                "delta": tuned - default, "wall_s": wall,
                "best_params": params,
            })

    md = ["# hyperparameter search vs default\n"]
    md.append("Random search over `spaces.for_baseline()` for two top baselines "
              "on three canonical datasets. 15 iterations, 3-fold CV per iteration, "
              "`f1_macro` as the search target.\n")
    md.append("| dataset | baseline | default macro-F1 | tuned macro-F1 | Δ | wall (s) |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        md.append(
            f"| {r['dataset']} | `{r['baseline']}` | {r['default']:.4f} | "
            f"{r['tuned']:.4f} | {r['delta']:+.4f} | {r['wall_s']:.0f} |"
        )
    md.append("")
    md.append("## Best params per run\n")
    for r in rows:
        md.append(f"- **{r['dataset']} / `{r['baseline']}`** → `{r['best_params']}`")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    main()
