# featurizer benchmark — clinc

- train size: **15000**
- intents: **150**
- portfolio: **11** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9345 | 0.9344 | 0.9345 | 12.168 | 18.20 | 23.55 | 99247.4 |
| linear | bm25_logreg | 0.9305 | 0.9307 | 0.9305 | 13.550 | 2.32 | 3.42 | 5005.6 |
| linguistic | stemmed_logreg | 0.9149 | 0.9147 | 0.9149 | 7.537 | 1.85 | 2.21 | 4107.6 |
| linear | bm25_linear_svc | 0.9128 | 0.9127 | 0.9128 | 158.757 | 17.86 | 20.95 | 14946.4 |
| linguistic | lemmatized_logreg | 0.9127 | 0.9123 | 0.9127 | 6.956 | 1.74 | 2.19 | 4236.7 |
| linear | logreg | 0.9102 | 0.9101 | 0.9102 | 7.916 | 1.81 | 2.34 | 5005.4 |
| linguistic | word_pos_logreg | 0.9086 | 0.9085 | 0.9086 | 12.990 | 3.26 | 4.18 | 5837.8 |
| naive_bayes | nb_multinomial | 0.9085 | 0.9079 | 0.9085 | 0.171 | 1.90 | 2.38 | 9912.8 |
| linear | skipgram_logreg | 0.8734 | 0.8743 | 0.8734 | 43.090 | 81.51 | 92.13 | 66552.2 |
| reduced_dim | random_projection_logreg | 0.8010 | 0.7990 | 0.8010 | 7.868 | 1.55 | 1.91 | 485.5 |
| linguistic | pos_sequence_logreg | 0.1680 | 0.1382 | 0.1680 | 5.970 | 1.42 | 1.80 | 379.5 |

## Friedman + Nemenyi

- Friedman p-value: **0.0021** (reject H0: **True**)

```
Critical Difference = 8.7171  (n=3)

rank  baseline
1.000  linear_svc_char
2.000  bm25_logreg
4.000  stemmed_logreg
4.333  bm25_linear_svc
5.000  lemmatized_logreg
5.667  logreg
6.667  nb_multinomial
7.333  word_pos_logreg
9.000  skipgram_logreg
10.000  random_projection_logreg
11.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, linear_svc_char, logreg, nb_multinomial, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, logreg, nb_multinomial, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {pos_sequence_logreg, random_projection_logreg, skipgram_logreg}
  {pos_sequence_logreg, random_projection_logreg}
  {pos_sequence_logreg}
```