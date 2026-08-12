# featurizer benchmark — clinc

- train size: **15000**
- intents: **150**
- portfolio: **9** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9345 | 0.9344 | 0.9345 | 3.720 | 5.71 | 5.99 | 99247.4 |
| feature_engineering | union_bm25_pos_logreg | 0.9310 | 0.9312 | 0.9310 | 8.408 | 1.31 | 1.60 | 5368.9 |
| linear | bm25_logreg | 0.9305 | 0.9307 | 0.9305 | 6.318 | 0.58 | 0.70 | 5005.6 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9281 | 0.9281 | 0.9281 | 21.391 | 31.87 | 36.85 | 71540.6 |
| linear | logreg | 0.9102 | 0.9101 | 0.9102 | 4.616 | 0.52 | 0.65 | 5005.4 |
| feature_engineering | union_pos_char_logreg | 0.9095 | 0.9096 | 0.9095 | 21.149 | 7.87 | 9.93 | 33750.2 |
| feature_engineering | union_pos_tfidf_logreg | 0.9063 | 0.9063 | 0.9063 | 7.697 | 1.22 | 1.56 | 5368.6 |
| linear | skipgram_logreg | 0.8734 | 0.8743 | 0.8734 | 19.864 | 29.85 | 34.62 | 66552.2 |
| linguistic | pos_sequence_logreg | 0.1680 | 0.1382 | 0.1680 | 2.210 | 0.45 | 0.53 | 379.5 |

## Friedman + Nemenyi

- Friedman p-value: **0.0032** (reject H0: **True**)

```
Critical Difference = 6.9363  (n=3)

rank  baseline
1.000  linear_svc_char
2.667  union_bm25_pos_logreg
3.000  bm25_logreg
3.333  union_skipgram_tfidf_logreg
5.333  logreg
5.667  union_pos_char_logreg
7.000  union_pos_tfidf_logreg
8.000  skipgram_logreg
9.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_logreg, linear_svc_char, logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, pos_sequence_logreg, skipgram_logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg}
  {pos_sequence_logreg}
```