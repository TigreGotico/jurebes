# jurebes — empirical benchmark report

This report consolidates every benchmark run in
[`reports/`](reports/) into a single narrative. The numbers come from
the actual training runs; reproduce any of them with the corresponding
script under [`examples/trained_models/`](.). Generation pipeline:

1. Per-dataset trainer (`train_<dataset>.py`) writes a Markdown report.
2. `_build_report.py` parses every report and emits the figures under
   [`reports/figures/`](reports/figures/) plus `_report_data.json`.
3. This file is hand-written from the structured data.

All numbers come from 5-fold stratified CV unless noted otherwise.

## Datasets

| dataset | intents | train | test | source |
| --- | ---: | ---: | ---: | --- |
| SNIPS                       |   7 | 13 084 | 1 400 | `benayas/snips` |
| BANKING77                   |  77 | 10 003 | 3 080 | `banking77` |
| CLINC-150                   | 150 | 15 250 | 5 500 | `clinc_oos` (`plus` config, OOD dropped) |
| intents-for-eval (12 langs) |  50 |  ~2.2k | 1 700 in-domain | `OpenVoiceOS/intents-for-eval` |

intents-for-eval ships with 1 000 Padatious-style templates per language
plus a per-slot `examples` list; the loader expands each template into
~2.2 k realised utterances via `jurebes.datasets.expand_slots` before
feeding them to the classifier. The 1 700 in-domain test utterances are
held out; the additional out-of-domain rows are dropped here because
the baselines have no OOD reject behaviour.

## Headline

![Portfolio accuracy across canonical NLU benchmarks](reports/figures/01_portfolio_across_datasets.png)

**`linear_svc_char` is the dominant baseline on every real-text-classification
benchmark we tested:**

| dataset | best CV baseline | CV accuracy | held-out test accuracy | held-out macro-F1 |
| --- | --- | ---: | ---: | ---: |
| SNIPS                | `linear_svc_char` | 0.9856 | **0.9850** | 0.9850 |
| BANKING77            | `linear_svc_char` | 0.8880 | **0.9029** | 0.9027 |
| CLINC-150            | `linear_svc_char` | 0.9401 | **0.9082** | 0.9071 |
| intents-for-eval avg | `linear_svc_char` | 0.7592 | — | — |

Friedman+Nemenyi rejects the null hypothesis that the six portfolio
baselines are equivalent on every dataset (p ≤ 0.0003). Character n-grams
+ calibrated linear SVM win the ranking decisively when training data
is genuinely-spelled text.

## Per-baseline behaviour

Read from `BANKING77` because it has enough classes (77) to differentiate
the baselines without being so large that ensemble / boosting methods are
prohibitively slow. The full table:

| baseline | accuracy | macro-F1 | train (s) | p95 latency (ms) | model size (KB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear_svc_char`   | 0.8880 | 0.8857 |  78.2 | 117.84 | 33 464 |
| `linear_svc`        | 0.8823 | 0.8805 |  36.5 |  69.01 |  4 021 |
| `logreg`            | 0.8600 | 0.8551 |  34.4 |   8.04 |  1 345 |
| `nb_multinomial`    | 0.8034 | 0.7653 |   0.4 |   8.15 |  2 631 |
| `lsa_logreg`        | 0.6455 | 0.6186 |  45.0 |  12.23 |    926 |
| `autoencoder_logreg`| 0.2711 | 0.2052 | 572.8 |  29.07 |  8 762 |

Three patterns generalise across the canonical datasets:

- **Char n-grams help.** `linear_svc_char` consistently outperforms
  `linear_svc` (word features). The gap is largest on CLINC-150
  (`0.9401` vs `0.9382` CV accuracy with 150 intents) — char features
  handle the long tail of rare-word intents better.
- **Naive Bayes is the fast-mode floor.** `nb_multinomial` fits in
  under a second on every dataset and lands within 5–10 accuracy points
  of the linear-SVM winners. For latency-sensitive applications it is
  the right baseline to deploy.
- **Autoencoders underperform on high-class-count data.** An
  autoencoder (`SklearnAutoencoder`: encoder + decoder, reconstruction
  loss, `y = X`, unsupervised) compresses whatever carries
  reconstruction error — not whatever discriminates classes. On
  7-class SNIPS `autoencoder_logreg` lands at 0.97; on 77-class
  BANKING77 it falls to 0.56 with the auto-sized bottleneck (it was
  0.27 with the old fixed `(64,16,64)`). `autoencoder_logreg_wide`
  reaches 0.71. `denoising_autoencoder_logreg` is *worse* — 0.19 on
  BANKING77 — because Gaussian noise on sparse TF-IDF destroys signal
  rather than regularising it; denoising autoencoders suit dense
  continuous inputs, not bag-of-words.

- **Label-guided embeddings are not autoencoders, and they win the
  reduced-dim group.** `label_guided_logreg` / `label_guided_linear_svc`
  (ported from `TigreGotico/guided-categorical-embeddings-sklearn`,
  Apache-2.0) train an `MLPClassifier` end-to-end on `(X, y)` and tap a
  hidden layer — **supervised bottleneck features**, no decoder and no
  reconstruction objective. They are a distinct method family that
  happens to be a `reduced_dim` featurizer. On BANKING77 they reach
  ~0.85 (vs the autoencoder's 0.56) — within 3-4 points of the
  `linear_svc_char` leader, and on SNIPS effectively tied (0.982 vs
  0.986). The lesson is not "fix the autoencoder" but "for intent
  classification, supervise the bottleneck — that is a classifier
  feature extractor, not an autoencoder."

### Latency vs accuracy

![Latency vs accuracy on BANKING77](reports/figures/04_latency_vs_accuracy.png)

The Pareto front has three operating points:

- **Ultra-low latency (≤10 ms p95):** `logreg` (0.86 acc) or
  `nb_multinomial` (0.80 acc).
- **Mid-tier (~70 ms p95):** `linear_svc` (0.88).
- **Maximum quality (~118 ms p95):** `linear_svc_char` (0.89).

The 14× latency jump from `logreg` to `linear_svc_char` buys 3 points
of accuracy. For OVOS-pipeline confidence thresholds, the cheaper
baseline is plenty when the use case can tolerate the gap.

## Multilingual intent classification

intents-for-eval covers 12 languages with the same 50-intent inventory
and 1 000 templates per language. The grid below shows test accuracy
for every portfolio baseline × language combination after the loader
expands `{slot}` placeholders into realised utterances.

![intents-for-eval — baseline × language](reports/figures/05_ife_baseline_by_language.png)

### Per-language winners

![intents-for-eval — best baseline per language](reports/figures/02_ife_per_language_intent.png)

| lang  | best baseline       | best accuracy |
| ---   | ---                 | ---: |
| en-US | `linear_svc_char`   | 0.8306 |
| pt-PT | `linear_svc_char`   | 0.8388 |
| pt-BR | `linear_svc_char`   | 0.8453 |
| es-ES | `linear_svc_char`   | 0.8335 |
| fr-FR | `linear_svc_char`   | 0.8376 |
| de-DE | `linear_svc_char`   | 0.8388 |
| it-IT | `linear_svc_char`   | 0.8394 |
| nl-NL | `linear_svc_char`   | 0.8394 |
| ca-ES | `linear_svc_char`   | 0.8471 |
| gl-ES | `linear_svc_char`   | 0.8441 |
| da-DK | `linear_svc_char`   | **0.8576** |
| eu-ES | `linear_svc_char`   | 0.8382 |

The cross-language band is tight — every winner falls between 0.83 and
0.86. Two observations:

- **Basque (eu-ES) and Catalan (ca-ES) do not underperform** despite
  smaller training corpora in the public eye. The character-n-gram
  representation is morphology-agnostic; it handles the agglutinative
  case-marking in Basque and the article-fronting in Catalan without
  any language-specific tuning.
- **Danish (da-DK) tops the table.** Likely because the template
  surface forms in this dataset are tighter and there is less
  inflectional variance to learn — not a claim about the language
  being intrinsically easier.

## Slot extraction

The same intents-for-eval dataset ships gold slot annotations, so we
can benchmark the six slot taggers in `jurebes.slots` head-to-head.

![Slot-tagger exact-match by language](reports/figures/03_slot_tagger_by_language.png)

| tagger        | mean exact-match | min | max | typical training cost |
| ---           | ---:    | ---:    | ---:    | --- |
| `crf`         | **0.8323** | 0.7747 | 0.8829 | minutes (`sklearn-crfsuite`) |
| `sklearn_iob` | 0.8132     | 0.7394 | 0.8647 | seconds (`LogisticRegression`) |
| `dictionary`  | 0.7938     | 0.7418 | 0.8176 | none |
| `hybrid`      | 0.7095     | 0.6600 | 0.7500 | sum of constituents |
| `knn`         | 0.6344     | 0.5876 | 0.6835 | seconds (`NearestNeighbors`) |
| `template`    | 0.6066     | 0.5529 | 0.6294 | none |

Three findings worth their own line items:

- **CRF wins by a sustained margin on 11 of 12 languages.** Conditional
  random fields are the canonical classical-ML slot-tagging baseline
  (Lafferty, McCallum & Pereira, 2001); the gap to the second-place
  `sklearn_iob` per-token classifier is consistently ~1.7 points of
  exact-match. The optional `[slots-crf]` extra is worth the install
  for any deployment that depends on slot quality.
- **Dictionary lookup is competitive.** Pure regex over the registered
  entity gazetteer (`DictionaryTagger`) lands within 4 points of CRF on
  average and beats every ML approach except CRF and `sklearn_iob`. For
  closed-set entities (city names, intent-bound vocabulary), the
  zero-training option is the right one.
- **`HybridCascadeTagger` is not the strongest tagger.** Cascading
  dictionary → template → IOB delivers lower exact-match than running
  CRF in isolation; the hybrid's value is *operational* (graceful
  fallback when training data is thin), not predictive. The
  recommendation in `docs/slots.md` should reflect this.

The `template` tagger's poor showing reflects the test set's
distribution: most utterances do NOT match a literal template surface
form (they are realised after slot substitution), so the template
regex captures only the unmodified parts.

## Statistical comparison

Friedman + Nemenyi (`jurebes.benchmark.stats.friedman_nemenyi`) on the
five-fold scores rejects the equal-baselines null on every dataset:

| dataset | Friedman p | n folds | best mean rank | worst mean rank |
| --- | ---: | ---: | --- | --- |
| SNIPS     | 0.0002 | 5 | `linear_svc_char` (1.0) | `autoencoder_logreg` (6.0) |
| BANKING77 | 0.0002 | 5 | `linear_svc_char` (1.4) | `autoencoder_logreg` (6.0) |
| CLINC-150 | 0.0003 | 5 | `linear_svc_char` (1.4) | `autoencoder_logreg` (6.0) |

`linear_svc_char` and `linear_svc` form the top statistically-
indistinguishable group on all three datasets (their CD-threshold
overlap is consistent across runs). `nb_multinomial` and `logreg` form
the next clique; `lsa_logreg` and `autoencoder_logreg` are clearly
distinguishable as worse.

## Recommendations by use case

- **Voice-assistant intent recognition (OVOS pipeline plugin).** Train
  with `linear_svc_char` if slot extraction also matters (the same
  baseline produces calibrated probabilities for the `conf_high`/
  `conf_med`/`conf_low` thresholds via `CalibratedClassifierCV`). Pair
  with `slots/crf` if the `[slots-crf]` extra is acceptable; fall back
  to `slots/sklearn_iob` otherwise.
- **Latency-critical embedded deployment.** `nb_multinomial` or `logreg`
  with the default `tfidf_word()` featurizer; p95 ≤ 10 ms on
  commodity CPU even at 77-class scale.
- **Hot domain bootstrap (~6 hand-labeled samples per intent).** Use
  the active-learning loop in `examples/llm_augmentation/` with
  `linear_svc_char` as the oracle baseline, or — once it lands — the
  semi-supervised pipeline in `examples/semi_supervised/` for the
  unlabeled-pool case.
- **Multilingual research bench.** The 12-language intents-for-eval
  results suggest `linear_svc_char` generalises well across morphology
  classes without per-language tuning. Use it as the cross-language
  control baseline; reach for `voting_soft` only when ensembling
  buys ≥ 1 point on the specific language of interest.

## Reproducing the numbers

```bash
pip install jurebes[hf,slots-crf,bench-plot]
python examples/trained_models/train_snips.py
python examples/trained_models/train_banking77.py
python examples/trained_models/train_clinc.py
python examples/trained_models/train_intents_for_eval_all_langs.py
python examples/trained_models/_build_report.py
```

Each script writes its Markdown report under `reports/`; the orchestrator
also writes the consolidated summary. The figures in this REPORT are
regenerated by `_build_report.py` from the latest report files.

## Caveats

- The autoencoder portfolio entries in the table above predate the
  `hidden_layer_sizes="auto"` default and the new `label_guided_*` /
  `denoising_autoencoder_logreg` / `autoencoder_logreg_wide` / `_deep`
  baselines. The next portfolio re-run will include those entries and
  the numbers here will be re-measured. Search-tuned AE variants are
  expected to recover most of the gap against the linear leader.
- intents-for-eval test rows tagged as out-of-domain (no
  `expected_intent`) are excluded. Adding OOD reject behaviour is on
  the framework roadmap.
- BANKING77 and CLINC-150 trained models exceed the 5 MB commit
  threshold; the .joblib artefacts are not in the repo but the scripts
  are reproducible end-to-end.

---
- Back to [examples index](README.md)
- [docs/research.md](../../docs/research.md) — research-framework deep dive
- [docs/theory/statistical-comparison.md](../../docs/theory/statistical-comparison.md) — Demšar protocol details
