# featurizer benchmark — snips

- train size: **13084**
- intents: **7**
- portfolio: **11** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9843 | 0.9842 | 0.9843 | 2.047 | 3.53 | 4.39 | 9663.5 |
| linear | bm25_logreg | 0.9840 | 0.9840 | 0.9840 | 2.670 | 1.12 | 1.56 | 647.0 |
| linear | bm25_linear_svc | 0.9839 | 0.9839 | 0.9839 | 1.591 | 3.81 | 5.00 | 1600.0 |
| linear | logreg | 0.9810 | 0.9809 | 0.9810 | 14.611 | 0.97 | 1.40 | 646.8 |
| linguistic | word_pos_logreg | 0.9799 | 0.9798 | 0.9799 | 11.076 | 1.63 | 2.00 | 1096.7 |
| linguistic | stemmed_logreg | 0.9781 | 0.9780 | 0.9781 | 8.300 | 1.13 | 1.37 | 584.4 |
| linguistic | lemmatized_logreg | 0.9775 | 0.9774 | 0.9775 | 7.958 | 1.06 | 1.34 | 600.2 |
| naive_bayes | nb_multinomial | 0.9757 | 0.9755 | 0.9757 | 0.148 | 0.92 | 1.22 | 1119.9 |
| linear | skipgram_logreg | 0.9720 | 0.9719 | 0.9720 | 24.850 | 2.36 | 3.30 | 5819.4 |
| reduced_dim | random_projection_logreg | 0.8818 | 0.8818 | 0.8818 | 0.497 | 1.61 | 2.10 | 401.5 |
| linguistic | pos_sequence_logreg | 0.5959 | 0.5904 | 0.5959 | 4.547 | 1.49 | 2.27 | 346.4 |

## Friedman + Nemenyi

- Friedman p-value: **0.0015** (reject H0: **True**)

```
Critical Difference = 8.7171  (n=3)

rank  baseline
1.667  linear_svc_char
2.000  bm25_logreg
2.333  bm25_linear_svc
4.333  logreg
5.000  word_pos_logreg
6.667  stemmed_logreg
7.000  nb_multinomial
7.000  lemmatized_logreg
9.000  skipgram_logreg
10.000  random_projection_logreg
11.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, linear_svc_char, logreg, nb_multinomial, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, logreg, nb_multinomial, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {lemmatized_logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {lemmatized_logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg}
  {lemmatized_logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg}
  {lemmatized_logreg, pos_sequence_logreg, random_projection_logreg, skipgram_logreg}
  {pos_sequence_logreg, random_projection_logreg, skipgram_logreg}
  {pos_sequence_logreg, random_projection_logreg}
  {pos_sequence_logreg}
```