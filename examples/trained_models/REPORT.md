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
| HWU64                       |  64 |  8 954 | 1 076 | `DeepPavlov/hwu64` |
| ATIS                        |  17 |  4 972 |   884 | `tuetschek/atis` (rare classes <5 samples dropped) |
| intents-for-eval (12 langs) |  50 |  ~2.2k | 1 700 in-domain | `OpenVoiceOS/intents-for-eval` |
| massive-templates (51 langs) |  60 | ~13.8k | 2 974 | `OpenVoiceOS/massive-templates` |

intents-for-eval ships with 1 000 Padatious-style templates per language
plus a per-slot `examples` list; the loader expands each template into
~2.2 k realised utterances via `jurebes.datasets.expand_slots` before
feeding them to the classifier. The 1 700 in-domain test utterances are
held out; the additional out-of-domain rows are dropped here because
the baselines have no OOD reject behaviour.

## Headline

![Portfolio accuracy across canonical NLU benchmarks](reports/figures/01_portfolio_across_datasets.png)

Two baselines split the canonical-dataset wins between them:

| dataset | best CV baseline | CV accuracy | held-out test accuracy | held-out macro-F1 |
| --- | --- | ---: | ---: | ---: |
| SNIPS                | `linear_svc_char`         | 0.9856 | **0.9843** | 0.9843 |
| BANKING77            | `linear_svc_char`         | 0.8880 | **0.9062** | 0.9060 |
| CLINC-150            | `linear_svc_char`         | 0.9401 | **0.9164** | 0.9157 |
| HWU64                | `union_bm25_pos_logreg`   | 0.8683 | **0.8690** | 0.8681 |
| ATIS                 | `bm25_logreg` (macro-F1)  | 0.7475 | 0.9446 acc / **0.6629** macro-F1 | — |
| intents-for-eval     | `linear_svc_char` (10/12 langs), `union_bm25_pos_logreg` (en-US, es-ES) | ~0.842 | — | — |
| massive-templates    | `linear_svc_char` (51/51 langs) | ~0.832 | — | — |

Friedman+Nemenyi rejects the null hypothesis that the portfolio
baselines are equivalent on every dataset (p ≤ 0.0003). Two findings:

- **`linear_svc_char` wins SNIPS, BANKING77, CLINC, and the 51-language
  massive-templates sweep** — character n-grams + calibrated linear SVM
  are the strongest on long-vocabulary supervised data.
- **The BM25 family wins where data is sparser or class counts are
  long-tailed.** `union_bm25_pos_logreg` takes HWU64 outright;
  `bm25_logreg` wins ATIS on macro-F1 (the meaningful metric — ATIS has
  one mega-class that dominates accuracy). `union_bm25_pos_logreg` also
  wins 2 of 12 intents-for-eval languages (en-US, es-ES).

`bm25_logreg` is consistently the **best-calibrated** baseline across
every dataset measured (see the Calibration section below) and produces
the **strongest CLINC-150 OOD detector** (calibrated top-1 confidence
reaches ROC AUC 0.93 — vastly better than autoencoder reconstruction
error at 0.60; see the OOD section). The production-deployment
recommendation throughout this report is now BM25-led.

## Per-baseline behaviour

Read from `BANKING77` because it has enough classes (77) to differentiate
the baselines without being so large that ensemble / boosting methods are
prohibitively slow. The full table:

| baseline | accuracy | macro-F1 | train (s) | p95 latency (ms) | model size (KB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear_svc_char`              | 0.8880 | 0.8857 |   2.6 |   4.25 | 33 464 |
| `linear_svc`                   | 0.8823 | 0.8805 |   1.0 |   4.12 |  4 021 |
| `union_skipgram_tfidf_logreg`  | 0.8775 | 0.8761 |  22.4 |  18.07 | 34 132 |
| `union_bm25_pos_logreg`        | 0.8740 | 0.8723 |   8.9 |   1.50 |  1 699 |
| `bm25_logreg`                  | 0.8729 | 0.8712 |   6.3 |   0.49 |  1 345 |
| `logreg`                       | 0.8600 | 0.8551 |   6.2 |   0.45 |  1 345 |
| `label_guided_linear_svc`      | 0.8549 | 0.8524 | 123.5 |   4.49 |  7 080 |
| `label_guided_logreg`          | 0.8494 | 0.8475 | 120.5 |   0.54 |  6 865 |
| `nb_multinomial`               | 0.8034 | 0.7653 |   0.1 |   0.38 |  2 631 |
| `autoencoder_logreg_wide`      | 0.7112 | 0.6907 | 638.7 |   6.67 | 27 394 |
| `lsa_logreg`                   | 0.6413 | 0.6129 |   3.3 |   0.57 |    926 |
| `autoencoder_logreg`           | 0.5595 | 0.5191 | 248.7 |   0.51 |  9 569 |
| `denoising_autoencoder_logreg` | 0.1902 | 0.1154 |  62.6 |   0.46 |  9 566 |

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
  BANKING77 it falls to 0.56. `autoencoder_logreg_wide` reaches 0.71.
  `denoising_autoencoder_logreg` is *worse* — 0.19 on BANKING77 —
  because Gaussian noise on sparse TF-IDF destroys signal rather than
  regularising it; denoising autoencoders suit dense continuous inputs,
  not bag-of-words.

- **Label-guided embeddings are not autoencoders, and they win the
  reduced-dim group.** `label_guided_logreg` / `label_guided_linear_svc`
  (ported from `TigreGotico/guided-categorical-embeddings-sklearn`,
  Apache-2.0) train an `MLPClassifier` end-to-end on `(X, y)` and tap a
  hidden layer — **supervised bottleneck features**, no decoder and no
  reconstruction objective. They are a distinct method family that
  happens to be a `reduced_dim` featurizer. On BANKING77 they reach
  ~0.85 (vs the autoencoder's 0.56) — within 3-4 points of the
  `linear_svc_char` leader, and on SNIPS effectively tied (0.982 vs
  0.986). For intent classification, a supervised bottleneck (classifier
  feature extractor) outperforms an unsupervised reconstruction bottleneck.

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

## Featurizers beyond TF-IDF

Eight featurizers extend the bag-of-words TF-IDF baseline — three
pure-sklearn (skip-grams, Okapi BM25, sparse random projection) and
four behind optional dependencies (POS-sequence and word⊕POS via
`brill_postagger`, Snowball-stemmed TF-IDF, simplemma-lemmatised
TF-IDF). Each is benched against `linear_svc_char` / `logreg` /
`nb_multinomial` on SNIPS, BANKING77 and CLINC-150 with 3-fold CV +
Friedman+Nemenyi.

### Mean rank across the three datasets (lower is better)

| baseline                   | SNIPS rank | BANKING77 rank | CLINC rank | mean |
| ---                        | ---: | ---: | ---: | ---: |
| `linear_svc_char` *(ref)*  | 1.67 | 1.00 | 1.00 | **1.22** |
| `bm25_logreg`              | 2.00 | 2.00 | 2.00 | **2.00** |
| `bm25_linear_svc`          | 2.33 | 6.00 | 4.33 | 4.22 |
| `stemmed_logreg`           | 6.67 | 3.00 | 4.00 | 4.56 |
| `lemmatized_logreg`        | 7.00 | 4.00 | 5.00 | 5.33 |
| `logreg` *(ref)*           | 4.33 | 5.67 | 5.67 | 5.22 |
| `word_pos_logreg`          | 5.00 | 6.33 | 7.33 | 6.22 |
| `nb_multinomial` *(ref)*   | 7.00 | 9.00 | 6.67 | 7.56 |
| `skipgram_logreg`          | 9.00 | 8.00 | 9.00 | 8.67 |
| `random_projection_logreg` | 10.00 | 10.00 | 10.00 | 10.00 |
| `pos_sequence_logreg`      | 11.00 | 11.00 | 11.00 | 11.00 |

Friedman p < 0.003 on every dataset — the differences are real.

### Findings

- **BM25 ranks second on every dataset.** `bm25_logreg` is statistically
  indistinguishable from `linear_svc_char` on SNIPS. CV macro-F1:
  SNIPS 0.984 (vs 0.984), BANKING77 0.869 (vs 0.881), CLINC 0.931 (vs
  0.934). The model is **~50× smaller** (647 KB vs 9.7 MB on SNIPS) and
  inference is **~3× faster** (p50 1.12 ms vs 3.53 ms). Okapi
  saturation (`tf*(k1+1)/(tf+k1*(...))` instead of raw TF) outperforms
  TF-IDF on bag-of-word features.

- **Stemming pays off at high class counts.** `stemmed_logreg` places
  3rd on BANKING77 (0.865) and 3rd on CLINC (0.915), ahead of plain
  `logreg` on both — but only mid-pack on 7-class SNIPS. The
  morphological collapse of Snowball stemming helps exactly when
  vocabulary sparsity is the bottleneck (many intents, few samples per
  rare-word variant). `lemmatized_logreg` follows the same pattern one
  step weaker.

- **Standalone skip-grams underperform contiguous n-grams.**
  `skipgram_logreg` ranks 8-9 on every dataset, 5-9 points behind
  contiguous n-grams; on short utterances they add noise faster than
  signal. They contribute in a `feature_union` with regular n-grams
  (see the union ablation below).

- **Random projection is consistently bad.** Rank 10 on all three.
  Johnson-Lindenstrauss preserves *distance*, but text classification
  needs to preserve *discriminative directions* — the random basis
  doesn't. Matches the LSA/NMF pattern.

- **Pure POS-sequence collapses.** `pos_sequence_logreg`: 0.60 on
  SNIPS, **0.15 on BANKING77, 0.17 on CLINC**. Pure syntactic structure
  with no lexical content is hopeless on high-class-count intent data.
  Use it only inside a `feature_union(tfidf_word(), pos_sequence())`.

- **`word_pos` (lexical+POS hybrid tokens) is middling.** Adds nothing
  over plain word TF-IDF when the lexical signal already discriminates.
  POS disambiguation matters more on noun-vs-verb-overloaded vocab
  than on these datasets.

### Recommendation

- **Production text-classification deployment:** prefer `bm25_logreg`
  over `linear_svc_char` when model size or inference latency matter
  — accuracy is within ~1 point and you save ~50× on disk and 3× on
  inference. Reach for `linear_svc_char` only when the last 1-2 points
  of accuracy buy back the size/latency cost.
- **High-class-count datasets (>50 intents):** include `stemmed_logreg`
  in the comparison portfolio.
- **POS, random-projection and standalone-skipgram:** comparison
  baselines, not defaults.

Full per-dataset tables and the Friedman+Nemenyi cliques are in
[`reports/featurizer_bench_{snips,banking77,clinc}.md`](reports/).
The bench script is [`train_featurizer_bench.py`](train_featurizer_bench.py).

### Union ablation — do the weak featurizers contribute in combination?

POS-sequence and skip-grams underperform in isolation. The
complementary-signal question is whether they contribute *as channels
in a `feature_union` with a strong lexical channel*. Four union
baselines exercise that:

- `union_skipgram_tfidf_logreg` — `tfidf_word + skipgram_word`
- `union_pos_tfidf_logreg` — `tfidf_word + pos_sequence`
- `union_pos_char_logreg` — `tfidf_char + pos_sequence`
- `union_bm25_pos_logreg` — `bm25_word  + pos_sequence`

Re-benched against `linear_svc_char` / `bm25_logreg` / `logreg` /
solo-skipgram / solo-POS on SNIPS / BANKING77 / CLINC, 3-fold CV:

| baseline | SNIPS rank | BANKING77 rank | CLINC rank | mean |
| --- | ---: | ---: | ---: | ---: |
| `linear_svc_char` *(ref)*        | 1.67 | 1.00 | 1.00 | **1.22** |
| `union_bm25_pos_logreg`          | 2.00 | 2.00 | 2.67 | **2.22** |
| `bm25_logreg` *(ref)*            | 2.67 | 3.33 | 3.00 | 3.00 |
| `union_skipgram_tfidf_logreg`    | 3.67 | 3.67 | 3.33 | **3.56** |
| `union_pos_char_logreg`          | 6.00 | 5.00 | 5.67 | 5.56 |
| `logreg` *(ref)*                 | 5.33 | 6.33 | 5.33 | 5.67 |
| `union_pos_tfidf_logreg`         | 6.67 | 6.67 | 7.00 | 6.78 |
| `skipgram_logreg` (solo, ref)    | 8.00 | 8.00 | 8.00 | 8.00 |
| `pos_sequence_logreg` (solo)     | 9.00 | 9.00 | 9.00 | 9.00 |

### Findings

- **Skip-grams contribute in union with TF-IDF.** Standalone they rank
  8-9; paired with `tfidf_word` they rank ~3.5, lifting accuracy over
  plain `logreg` by **+0.0145 on BANKING77 and +0.0179 on CLINC**
  (negligible on SNIPS). Contiguous n-grams + skip-grams beat either
  alone.

- **POS in union with plain TF-IDF reduces accuracy.**
  `union_pos_tfidf_logreg` is *worse* than plain `logreg` on every
  dataset: **−0.0017 (SNIPS), −0.0050 (BANKING77), −0.0039 (CLINC)**.
  The POS channel adds noise faster than signal when the lexical
  channel already discriminates well.

- **POS in union with BM25 gives a marginal lift.**
  `union_bm25_pos_logreg` beats plain `bm25_logreg` by **+0.0002 /
  +0.0011 / +0.0005** — statistically tied on SNIPS, tiny real lift
  on BANKING77 / CLINC. The BM25-saturated lexical channel leaves
  enough room for POS to contribute marginally. It carries the same
  cost advantage as `bm25_logreg`: ~50× smaller than `linear_svc_char`
  on CLINC (5.4 MB vs 99 MB) at within ~0.4 points of accuracy.

- **POS in union with char n-grams** sits between the two — marginal,
  never beats `bm25_logreg`.

- **No union overtakes `linear_svc_char`** on any dataset. Char n-grams
  + calibrated linear SVM wins raw accuracy; unions displace it only on
  production-deployment cost.

### Recommendation (union ablation)

- **Highest accuracy:** `linear_svc_char`.
- **Production deployment** (latency / size matters): `bm25_logreg`
  is the best simple choice; `union_bm25_pos_logreg` adds a marginal
  lift if the `[postag]` extra is acceptable.
- **High-class-count datasets:** include `union_skipgram_tfidf_logreg`
  in the portfolio — its +1.5 / +1.8 point lift over plain `logreg`
  on BANKING77 / CLINC is the most consistent ablation gain measured.
- **POS combined with plain word TF-IDF reduces accuracy** — avoid.

## Multilingual intent classification

intents-for-eval covers 12 languages with the same 50-intent inventory
and 1 000 templates per language. The grid below shows test accuracy
for every portfolio baseline × language combination after the loader
expands `{slot}` placeholders into realised utterances.

![intents-for-eval — baseline × language](reports/figures/05_ife_baseline_by_language.png)

### Per-language winners

![intents-for-eval — best baseline per language](reports/figures/02_ife_per_language_intent.png)

| lang  | best baseline             | best accuracy |
| ---   | ---                       | ---: |
| en-US | `union_bm25_pos_logreg`   | 0.8318 |
| pt-PT | `linear_svc_char`         | 0.8394 |
| pt-BR | `linear_svc_char`         | 0.8441 |
| es-ES | `union_bm25_pos_logreg`   | 0.8394 |
| fr-FR | `linear_svc_char`         | 0.8394 |
| de-DE | `linear_svc_char`         | 0.8406 |
| it-IT | `linear_svc_char`         | 0.8388 |
| nl-NL | `linear_svc_char`         | 0.8359 |
| ca-ES | `linear_svc_char`         | 0.8488 |
| gl-ES | `linear_svc_char`         | 0.8447 |
| da-DK | `linear_svc_char`         | **0.8535** |
| eu-ES | `linear_svc_char`         | 0.8376 |

`linear_svc_char` wins 10 of 12 languages; `union_bm25_pos_logreg`
takes en-US (0.8318) and es-ES (0.8394) by a fraction of a point. The
cross-language band is tight — every winner falls between 0.83 and
0.86. Two observations:

- **Basque (eu-ES) and Catalan (ca-ES) do not underperform** despite
  smaller training corpora in the public eye. The character-n-gram
  representation is morphology-agnostic; it handles the agglutinative
  case-marking in Basque and the article-fronting in Catalan without
  any language-specific tuning. The eu-ES loader drops 57 templates
  that use the `(letter)` Basque case-marker convention as informal
  optional brackets — the convention is incompatible with the parser's
  strict alternation requirement, so the bad rows are dropped at load
  time and the remaining 943 templates train cleanly.
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
  with `linear_svc_char` if peak accuracy matters; prefer `bm25_logreg`
  when the OPM `conf_high`/`conf_med`/`conf_low` thresholds need
  reliable probabilities (it is the best-calibrated baseline measured —
  see the calibration section). Pair with `slots/crf` if the
  `[slots-crf]` extra is acceptable; fall back to `slots/sklearn_iob`
  otherwise.
- **Latency-critical embedded deployment.** `bm25_logreg` for the best
  accuracy/latency/size trade-off (≤10 ms p95 at every dataset scale,
  ~50× smaller than `linear_svc_char`, best calibrated probabilities).
  `nb_multinomial` only when sub-millisecond inference is required and
  poor probability quality is acceptable.
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

## Calibration analysis

Reports `ece` (expected calibration error) and `brier` (mean squared
error of probability vs one-hot ground truth) alongside accuracy for
the eight headline baselines on each canonical dataset. Lower is
better; 0 = perfect calibration.

| dataset   | best-calibrated baseline    | ECE     | worst-calibrated baseline | ECE     |
| ---       | ---                         | ---:    | ---                        | ---:    |
| SNIPS     | `bm25_logreg`               | 0.0044  | `nb_multinomial`           | 0.1337  |
| BANKING77 | `union_bm25_pos_logreg`     | 0.0157  | `nb_multinomial`           | 0.5879  |
| CLINC-150 | `bm25_logreg`               | 0.0305  | `nb_multinomial`           | 0.7540  |

Three findings:

- **BM25 is consistently the best-calibrated baseline** across all
  three datasets, with ECE between 0.004 and 0.031. Combined with the
  size/latency advantages reported earlier, this strengthens the
  production-deployment recommendation: `bm25_logreg` has the most
  honest probabilities of any baseline tested.
- **`nb_multinomial` is consistently the worst-calibrated** — ECE
  0.59 on BANKING77 and 0.75 on CLINC despite ~0.90 accuracy. The
  independence assumption produces wildly overconfident probabilities
  that do not reflect empirical accuracy; never use NB confidence as a
  reject signal.
- **`linear_svc_char` is well-calibrated on small data and only
  partly-calibrated on large data.** ECE rises from 0.016 (SNIPS) to
  0.155 (CLINC). `CalibratedClassifierCV` does useful work but
  100-150-class problems saturate the simple sigmoid wrap. For
  confidence-threshold gates on CLINC-scale inventories, prefer BM25
  + LogReg's native calibrated probabilities over the wrap.

## Hyperparameter search vs default

Random search (`n_iter=15`, `cv=3`, `scoring="f1_macro"`) over
`spaces.for_baseline()` for the two top baselines across the canonical
datasets, compared against default-hyperparameter macro-F1.

| dataset   | baseline           | default | tuned   | Δ        | wall (s) |
| ---       | ---                | ---:    | ---:    | ---:     | ---:     |
| SNIPS     | `linear_svc_char`  | 0.9842  | 0.9858  | **+0.0016** | 11 |
| SNIPS     | `bm25_logreg`      | 0.9840  | 0.9847  | +0.0007  | 3        |
| BANKING77 | `linear_svc_char`  | 0.8811  | 0.8860  | **+0.0048** | 41 |
| BANKING77 | `bm25_logreg`      | 0.8690  | 0.8710  | +0.0020  | 23       |
| CLINC-150 | `linear_svc_char`  | 0.9344  | 0.9166  | **−0.0178** | 84 |
| CLINC-150 | `bm25_logreg`      | 0.9307  | 0.9065  | **−0.0241** | 72 |

Two findings:

- **Search helps on small and medium datasets** — small wins on SNIPS
  (~0.001–0.002 points), real wins on BANKING77 (+0.0048 for
  `linear_svc_char`). Search tends to land on `feat__ngram_range=(2,4)`
  and `clf__C` in the [2.0, 4.0] band.
- **15 iterations is not enough for 150-class CLINC** — tuned
  configurations *regress* by 1.8–2.4 macro-F1 points. Random search
  picks parameters whose 3-fold CV looks good but generalises worse
  than the default. Practical guidance: scale `n_iter` with the search
  space and class count. The framework's defaults are well-suited to
  high-class-count problems; speculative search needs more budget than
  15 iterations to challenge them.

Per-dataset best-parameter dumps are in
[`reports/hyperparam_search.md`](reports/hyperparam_search.md).

## Out-of-domain detection on CLINC150

CLINC150's `oos` split provides 1 000 genuine out-of-domain utterances.
Four scoring strategies trained on in-domain CLINC, evaluated on the
4 500-in + 1 000-OOD test set:

| OOD scoring method | ROC AUC | TPR @ FPR=0.05 | TPR @ FPR=0.10 | TPR @ FPR=0.20 |
| --- | ---: | ---: | ---: | ---: |
| **`bm25_logreg` top-1 confidence (1−conf)**     | **0.9254** | 0.6290 | 0.7910 | 0.9030 |
| `bm25_logreg` top1 − top2 margin                | 0.9096     | 0.4690 | 0.7290 | 0.8930 |
| `SklearnAutoencoder` reconstruction error       | 0.6004     | 0.0810 | 0.1600 | 0.3040 |
| One-class SVM (rbf) on TF-IDF                   | 0.5529     | 0.0840 | 0.1550 | 0.2780 |

Three findings:

- **The calibration result pays off directly.** `bm25_logreg`'s
  calibrated top-1 confidence is the strongest OOD detector tested by
  a wide margin — AUC 0.9254, catching 79 % of OOD utterances at a
  10 % false-positive rate. The same property that made BM25 the
  best-calibrated baseline (calibration section above) makes its
  confidence a meaningful reject signal.
- **The top1 − top2 margin is nearly as good** (AUC 0.9096), useful as
  a secondary score when raw confidence is uncalibrated for a given
  baseline.
- **Unsupervised approaches do not work on TF-IDF intent data.** The
  autoencoder (AUC 0.6004) and one-class SVM (0.5529) are barely
  above chance. TF-IDF vectors are sparse and short — reconstruction
  loss saturates near zero for both in-domain and OOD inputs alike
  (the median-error gap lives in the fifth decimal place), and an
  RBF-kernel one-class SVM cannot separate the two distributions in
  high-dimensional sparse space.

**Practical recommendation: use the calibrated top-1 confidence of a
production `bm25_logreg` as the OOD reject signal.** Pick a threshold
from the ROC table above (FPR ≤ 0.10 catches ~80 % of OOD with ~10 %
false positives on in-domain).

## MASSIVE-templates — 51-language breadth

`OpenVoiceOS/massive-templates` carries the same Padatious-style
template + test-split shape as intents-for-eval, but across **51
languages** with the 60-intent MASSIVE inventory and ~13.5k templates
per language (expanded to ~13.8k realised utterances). It is the
widest multilingual intent-classification test in the suite.

Intent classification was run with the seven fast linear / NB /
ensemble baselines (the MLP autoencoder and label-guided baselines are
characterised on the canonical datasets and intents-for-eval; they are
memory-bound on a 13.5k-template corpus). Slot extraction used the two
zero-memory regex taggers — `dictionary` and `template` — for the same
reason; the ML slot taggers are benchmarked on intents-for-eval.

### Intent classification

**`linear_svc_char` wins all 51 languages.** Test-set accuracy:

| statistic | value |
| --- | ---: |
| mean over 51 languages | 0.8321 |
| highest | pt-PT 0.8517, nl-NL 0.8514, az-AZ 0.8510 |
| lowest  | zh-TW 0.7441, km-KH 0.7552, zh-CN 0.7586 |

The cross-language band is tight — 0.74 to 0.85, ~11 points end to
end, with no per-language tuning. The three lowest are zh-TW, zh-CN and
km-KH: Chinese and Khmer are the languages where character n-grams over
whitespace-tokenised text help least (Chinese has no whitespace word
boundaries; the `char_wb` analyser still extracts useful sub-sequences,
which is why accuracy holds at ~0.75 rather than collapsing). Every
other language — including agglutinative (Turkish, Finnish, Hungarian),
Semitic (Arabic, Hebrew) and Indic scripts — lands in the 0.80-0.85
band. `linear_svc_char` is a genuinely language-agnostic default.

### Slot extraction

The regex taggers do poorly here: `dictionary` averages ~0.17
exact-match, `template` near zero. This is expected and not a tagger
defect — MASSIVE slots are open-vocabulary (song names, place names,
person names) drawn from a fixed-schema annotation, so gazetteer
lookup and literal-template matching have little to match. The
sklearn / CRF taggers on intents-for-eval (CRF exact-match 0.77-0.88)
are the meaningful slot-extraction numbers; the MASSIVE regex figures
are a memory-bounded sanity check, not a verdict on slot tagging.

Full per-language numbers: `reports/massive_templates_summary.md` and
`reports/massive_templates_<lang>.md`.

## Reproducing the numbers

```bash
pip install jurebes[hf,slots-crf,bench-plot]
python examples/trained_models/run_full_sweep.py            # canonical + intents-for-eval
python examples/trained_models/train_massive_templates_all_langs.py  # 51-language MASSIVE
python examples/trained_models/_build_report.py
```

`run_full_sweep.py` chains SNIPS → BANKING77 → CLINC → intents-for-eval ×12.
`train_massive_templates_all_langs.py` runs one subprocess per language
(memory reclaimed between languages) and is resumable — a completed
language is skipped, so an interrupted sweep can simply be relaunched.

Each script writes its Markdown report under `reports/`; the orchestrator
also writes the consolidated summary. The figures in this REPORT are
regenerated by `_build_report.py` from the latest report files.

## Caveats

- Search-tuned autoencoder variants would likely recover some of the
  gap against `linear_svc_char`; only default-hyperparameter
  configurations are reported here.
- intents-for-eval test rows tagged as out-of-domain (no
  `expected_intent`) are excluded — the baselines have no OOD reject
  behaviour.
- BANKING77 and CLINC-150 trained models exceed the 5 MB commit
  threshold; the .joblib artefacts are not in the repo but the scripts
  are reproducible end-to-end.

---
- Back to [examples index](README.md)
- [docs/research.md](../../docs/research.md) — research-framework deep dive
- [docs/theory/statistical-comparison.md](../../docs/theory/statistical-comparison.md) — Demšar protocol details
