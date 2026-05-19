# Trained models on canonical intent benchmarks

Reproducible training scripts on canonical intent benchmarks. Each
`train_<dataset>.py` fetches the dataset, fits a small portfolio of
baselines, runs Friedman + Nemenyi on the per-fold scores, tunes the
winning baseline with random search, evaluates it on the test split,
and writes a markdown report to `reports/`.

Run with:

```bash
pip install jurebes[hf]
python examples/trained_models/train_snips.py
python examples/trained_models/train_banking77.py
python examples/trained_models/train_clinc.py
```

Trained model artifacts are written under `models/` and committed only
if they fit under a 5 MB ceiling — larger artifacts are reported in the
markdown but excluded from the repo via `.gitignore`.
