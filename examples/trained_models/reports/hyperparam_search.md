# hyperparameter search vs default

Random search over `spaces.for_baseline()` for two top baselines on three canonical datasets. 15 iterations, 3-fold CV per iteration, `f1_macro` as the search target.

| dataset | baseline | default macro-F1 | tuned macro-F1 | Δ | wall (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| snips | `linear_svc_char` | 0.9842 | 0.9858 | +0.0016 | 11 |
| snips | `bm25_logreg` | 0.9840 | 0.9847 | +0.0007 | 3 |
| banking77 | `linear_svc_char` | 0.8811 | 0.8860 | +0.0048 | 41 |
| banking77 | `bm25_logreg` | 0.8690 | 0.8710 | +0.0020 | 23 |
| clinc | `linear_svc_char` | 0.9344 | 0.9166 | -0.0178 | 84 |
| clinc | `bm25_logreg` | 0.9307 | 0.9065 | -0.0241 | 72 |

## Best params per run

- **snips / `linear_svc_char`** → `{'feat__ngram_range': (3, 6), 'feat__min_df': 2, 'clf__estimator__C': 2.0}`
- **snips / `bm25_logreg`** → `{'feat__bm25__k1': 2.0, 'feat__bm25__b': 0.5, 'clf__C': 4.0}`
- **banking77 / `linear_svc_char`** → `{'feat__ngram_range': (2, 4), 'feat__min_df': 1, 'clf__estimator__C': 2.0}`
- **banking77 / `bm25_logreg`** → `{'feat__bm25__k1': 2.0, 'feat__bm25__b': 0.75, 'clf__C': 0.1}`
- **clinc / `linear_svc_char`** → `{'feat__ngram_range': (2, 4), 'feat__min_df': 2, 'clf__estimator__C': 2.0}`
- **clinc / `bm25_logreg`** → `{'feat__bm25__k1': 2.0, 'feat__bm25__b': 0.75, 'clf__C': 0.1}`
