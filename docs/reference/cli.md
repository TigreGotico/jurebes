# CLI reference

The `jurebes` command is registered as a console-script entry point. Run `jurebes --help` for the top-level help and `jurebes <subcommand> --help` for subcommand-specific help.

## Subcommands

- `list-baselines` — enumerate registered baselines
- `benchmark` — run a CV comparison of baselines
- `train` — train a model and save it
- `predict` — load a model and predict an utterance
- `search` — hyperparameter search for a baseline
- `stats` — statistical comparison across saved benchmark runs

All subcommands exit 0 on success and non-zero on argument errors.

## `jurebes list-baselines`

```
jurebes list-baselines [--group GROUP]
```

| flag | meaning |
| --- | --- |
| `--group GROUP` | filter to a single group name (e.g. `linear`, `kernel`, `tree`) |

Prints one baseline name per line, alphabetically.

## `jurebes benchmark`

```
jurebes benchmark --dataset PATH --baselines SPEC [--cv K]
                  [--scoring METRIC,...] [--sort-by FIELD] [--precision N]
                  [--format {markdown,json}] [--out PATH]
                  [--with-significance] [--save-run PATH]
```

| flag | default | meaning |
| --- | --- | --- |
| `--dataset PATH` | required | CSV, JSONL, or `@<canonical_name>` (e.g. `@banking77`) |
| `--baselines SPEC` | required | comma-separated names; `@<group>` selects all members; `@all` selects every baseline |
| `--cv K` | `5` | number of stratified folds |
| `--scoring METRIC,...` | `accuracy,f1_macro` | comma-separated metric names |
| `--sort-by FIELD` | None | column name to sort by in the markdown table |
| `--precision N` | `4` | floating-point precision in the markdown table |
| `--format {markdown,json}` | `markdown` | output format |
| `--out PATH` | stdout | write to file instead of stdout |
| `--with-significance` | false | append Friedman+Nemenyi (≥3 baselines) or paired-t+Wilcoxon (=2) block |
| `--save-run PATH` | None | dump the full `ComparisonResult` JSON for downstream `stats` use |

### Examples

```bash
# Compare three linear baselines on a local CSV
jurebes benchmark --dataset data.csv --baselines logreg,linear_svc,nb_complement --cv 5

# Compare every linear baseline on BANKING77, sort by macro-F1, include significance
jurebes benchmark --dataset @banking77 --baselines @linear \
  --sort-by macro_f1 --with-significance --out banking_linear.md

# Save full JSON for offline statistical analysis
jurebes benchmark --dataset data.csv --baselines @linear --save-run linear.json
```

## `jurebes train`

```
jurebes train --dataset PATH [--baseline NAME] [--tagger] --out PATH
```

| flag | default | meaning |
| --- | --- | --- |
| `--dataset PATH` | required | CSV, JSONL, or `@<canonical_name>` |
| `--baseline NAME` | `linear_svc` | any registered baseline name |
| `--tagger` | false | attach a `SklearnIOBTagger` |
| `--out PATH` | required | joblib output path |

### Example

```bash
jurebes train --dataset data.csv --baseline logreg --tagger --out model.joblib
```

## `jurebes predict`

```
jurebes predict --model PATH --text TEXT
```

| flag | meaning |
| --- | --- |
| `--model PATH` | path to a joblib model |
| `--text TEXT` | utterance to classify |

Prints `<intent>\t<confidence>\t<entities>` to stdout.

### Example

```bash
jurebes predict --model model.joblib --text "hello there"
```

## `jurebes search`

```
jurebes search --dataset PATH --baseline NAME
               [--backend {grid,halving_grid,random,halving_random,bayes,genetic}]
               [--cv K] [--n-iter N] [--scoring METRIC]
               [--space DICT_LITERAL] [--out PATH]
```

| flag | default | meaning |
| --- | --- | --- |
| `--dataset PATH` | required | dataset spec |
| `--baseline NAME` | required | baseline to tune |
| `--backend` | `random` | search backend |
| `--cv K` | `5` | inner CV folds |
| `--n-iter N` | `50` | candidate evaluations (random/bayes/genetic); ignored for `grid` |
| `--scoring METRIC` | `f1_macro` | first metric only |
| `--space DICT_LITERAL` | None | Python-literal dict overriding `spaces.for_baseline()` |
| `--out PATH` | None | save the best model |

### Example

```bash
jurebes search --dataset @banking77 --baseline logreg \
  --backend random --n-iter 40 --out best_logreg.joblib

jurebes search --dataset data.csv --baseline nb_multinomial \
  --backend grid --space "{'clf__alpha': [0.1, 0.5, 1.0]}"
```

## `jurebes stats`

```
jurebes stats (--runs FILE [FILE ...] | --pair A B)
              [--metric METRIC] [--alpha FLOAT]
```

| flag | default | meaning |
| --- | --- | --- |
| `--runs FILE ...` | None | paths to `ComparisonResult` JSON files (gathered series across runs) |
| `--pair A B` | None | paired statistical test between exactly two saved runs |
| `--metric METRIC` | `f1_macro` | metric name to compare on |
| `--alpha FLOAT` | `0.05` | significance threshold |

Exactly one of `--runs` and `--pair` is required.

### Examples

```bash
# Friedman+Nemenyi across several runs
jurebes stats --runs run_a.json run_b.json run_c.json --metric f1_macro

# Paired t / Wilcoxon between two runs
jurebes stats --pair run_a.json run_b.json --metric f1_macro
```

## Exit codes

- `0` — success.
- `1` — runtime error (e.g. fewer than 2 baselines in `--runs`).
- `2` — argument error (missing required flag, unknown subcommand, neither `--runs` nor `--pair` specified for `stats`).

## Dataset selector resolution

`--dataset` accepts:

- A path with `.csv` suffix → `load_csv(path)`.
- A path with `.jsonl` suffix → `load_jsonl(path)`.
- A token starting with `@` → looked up in `jurebes.datasets.canonical.CANONICAL`. Canonical loaders require the `hf` extra.

Any other input raises `ValueError("unsupported dataset suffix: …")`.

## Baseline selector resolution

`--baselines` accepts a comma-separated list of tokens. Each token is either a bare name (a single baseline) or `@<group>` (expanded to all members). `@all` expands to every registered name. Duplicates after expansion are deduplicated while preserving first-appearance order.

---
- Back to [reference index](index.md)
