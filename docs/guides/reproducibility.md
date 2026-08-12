# Reproducibility

A run is reproducible when the same dataset, code, and seeds produce the same numbers across machines. This guide enumerates the knobs.

## Seeds

- **Featurizer seeds.** `lsa(n_components, base=...)` wraps `TruncatedSVD(random_state=...)` indirectly; for full control, build the SVD step yourself with an explicit `random_state`. `nmf(n_components)` and `lda_topics(n_topics)` already pin `random_state=0` internally. `SklearnAutoencoder(random_state=0)` pins by default.

- **Benchmark seeds.** `train_test(..., seed=0)`, `cross_validate(..., seed=0)`, `compare(..., seed=0)` all forward `random_state` to `train_test_split` or `StratifiedKFold`. Change the value to vary the split; keep it fixed for reproducibility.

- **Search seeds.** `search(..., seed=0)` forwards a seed to the backend (e.g. `RandomizedSearchCV.random_state`). The Bayesian and genetic backends also accept a `seed`.

A pragmatic recipe: set `seed=0` everywhere during development; vary it to estimate variance.

## Pinning dependencies

scikit-learn does not guarantee bit-identical outputs across versions. Pin the working version:

```
scikit-learn==1.4.2
```

joblib pickles are version-sensitive (see [saving-and-loading.md](saving-and-loading.md)). The `_jurebes_version` tag stored alongside models helps diagnose drift after upgrade.

## Recording the model provenance

Capture the git revision next to every benchmark report:

```python
import subprocess, json
from pathlib import Path
from jurebes.benchmark import compare, to_json

result = compare(["logreg", "linear_svc"], X, y, k=5)
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

Path("report.json").write_text(json.dumps({
    "git_sha": sha,
    "result": json.loads(to_json(result)),
}, indent=2))
```

When you later need to understand a number, the git SHA points back to the exact code.

## Floating-point determinism

Even with seeds pinned, threaded BLAS operations can introduce non-determinism in the lowest decimal places. To force serial execution:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python my_bench.py
```

This is rarely needed for the metric ranges jurebes reports (4-decimal precision), but matters for paired statistical tests where tiny score differences matter.

## Reporting the protocol

A reproducible benchmark report states:

1. The exact dataset version (HuggingFace dataset commit hash, or CSV checksum).
2. The k-fold count and stratification choice.
3. The seed.
4. The list of baselines and their factory definitions.
5. The scoring metrics.
6. The scikit-learn and jurebes versions.
7. The git SHA of the calling script.

The `to_markdown` and `to_json` outputs of `compare` cover points 4–5; the rest belongs in the surrounding text or a sidecar JSON.

---
- Back to [docs index](../index.md)
