# Semi-supervised examples

Runnable scripts that exercise `jurebes.semi_supervised` on real
unlabeled pools.

## Install

```
pip install jurebes[hf]
```

The `[hf]` extra pulls in `datasets`, which `run_meteocat.py` uses to
fetch the upstream pool.

## Scripts

- `run_meteocat.py` — fetches `crodri/meteocat` via Hugging Face
  `datasets`, hand-labels ~5 utterances each for six seed intents, and
  bootstraps `self_train` from those seeds on the remaining unlabeled
  questions. Prints per-round metrics and predicted labels for ~20
  holdout utterances.
- `compare_strategies.py` — runs `self_train`, `co_train`
  (`logreg` word view + `linear_svc_char` char view) and
  `label_propagation` on the SAME seed set. Prints a comparison
  table of final labeled-pool coverage, eval macro-F1 and wall time.
  Uses a tiny inline dataset so it runs without network access.

## Run

```
python examples/semi_supervised/run_meteocat.py
python examples/semi_supervised/compare_strategies.py
```
