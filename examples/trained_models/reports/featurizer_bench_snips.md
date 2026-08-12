# featurizer benchmark — snips

- train size: **13084**
- intents: **7**
- portfolio: **9** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9843 | 0.9842 | 0.9843 | 0.715 | 1.19 | 1.38 | 9663.5 |
| feature_engineering | union_bm25_pos_logreg | 0.9842 | 0.9841 | 0.9842 | 3.380 | 1.17 | 1.37 | 992.4 |
| linear | bm25_logreg | 0.9840 | 0.9840 | 0.9840 | 1.508 | 0.35 | 0.43 | 647.0 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9833 | 0.9832 | 0.9833 | 8.735 | 1.20 | 1.34 | 6464.4 |
| linear | logreg | 0.9810 | 0.9809 | 0.9810 | 6.867 | 0.29 | 0.35 | 646.8 |
| feature_engineering | union_pos_char_logreg | 0.9799 | 0.9799 | 0.9799 | 13.972 | 1.39 | 1.60 | 4201.1 |
| feature_engineering | union_pos_tfidf_logreg | 0.9793 | 0.9792 | 0.9793 | 9.282 | 1.01 | 1.18 | 992.0 |
| linear | skipgram_logreg | 0.9720 | 0.9719 | 0.9720 | 10.546 | 0.71 | 0.81 | 5819.4 |
| linguistic | pos_sequence_logreg | 0.5959 | 0.5904 | 0.5959 | 1.621 | 0.50 | 0.60 | 346.4 |

## Friedman + Nemenyi

- Friedman p-value: **0.0041** (reject H0: **True**)

```
Critical Difference = 6.9363  (n=3)

rank  baseline
1.667  linear_svc_char
2.000  union_bm25_pos_logreg
2.667  bm25_logreg
3.667  union_skipgram_tfidf_logreg
5.333  logreg
6.000  union_pos_char_logreg
6.667  union_pos_tfidf_logreg
8.000  skipgram_logreg
9.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_logreg, linear_svc_char, logreg, skipgram_logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, skipgram_logreg, union_bm25_pos_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {bm25_logreg, logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg, union_skipgram_tfidf_logreg}
  {logreg, pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg, union_pos_char_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg, union_pos_tfidf_logreg}
  {pos_sequence_logreg, skipgram_logreg}
  {pos_sequence_logreg}
```