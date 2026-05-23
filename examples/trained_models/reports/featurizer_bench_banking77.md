# featurizer benchmark — banking77

- train size: **10003**
- intents: **77**
- portfolio: **9** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.8836 | 0.8811 | 0.8836 | 2.236 | 4.04 | 4.49 | 31593.6 |
| feature_engineering | union_bm25_pos_logreg | 0.8714 | 0.8704 | 0.8714 | 6.215 | 1.08 | 1.40 | 1609.9 |
| linear | bm25_logreg | 0.8703 | 0.8690 | 0.8703 | 6.328 | 0.39 | 0.48 | 1255.8 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.8680 | 0.8656 | 0.8680 | 19.373 | 6.37 | 7.58 | 30744.8 |
| feature_engineering | union_pos_char_logreg | 0.8579 | 0.8556 | 0.8579 | 16.082 | 2.12 | 2.66 | 11066.5 |
| linear | logreg | 0.8535 | 0.8459 | 0.8535 | 5.605 | 0.34 | 0.38 | 1255.6 |
| feature_engineering | union_pos_tfidf_logreg | 0.8485 | 0.8452 | 0.8485 | 9.517 | 1.11 | 1.44 | 1609.6 |
| linear | skipgram_logreg | 0.7936 | 0.7792 | 0.7936 | 22.579 | 4.23 | 5.35 | 29505.5 |
| linguistic | pos_sequence_logreg | 0.1528 | 0.1275 | 0.1528 | 2.046 | 0.55 | 0.86 | 369.6 |

## Friedman + Nemenyi

- Friedman p-value: **0.0026** (reject H0: **True**)

```
Critical Difference = 6.9363  (n=3)

rank  baseline
1.000  linear_svc_char
2.000  union_bm25_pos_logreg
3.333  bm25_logreg
3.667  union_skipgram_tfidf_logreg
5.000  union_pos_char_logreg
6.333  logreg
6.667  union_pos_tfidf_logreg
8.000  skipgram_logreg
9.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_logreg, linear_svc_char, logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, skipgram_logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg}
  {pos_sequence_logreg}
```