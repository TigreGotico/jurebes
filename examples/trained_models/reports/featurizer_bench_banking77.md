# featurizer benchmark — banking77

- train size: **10003**
- intents: **77**
- portfolio: **11** baselines, 3-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.8836 | 0.8811 | 0.8836 | 5.684 | 10.79 | 13.35 | 31593.6 |
| linear | bm25_logreg | 0.8703 | 0.8690 | 0.8703 | 9.538 | 1.12 | 1.40 | 1255.8 |
| linguistic | stemmed_logreg | 0.8650 | 0.8592 | 0.8650 | 8.077 | 1.23 | 1.62 | 898.0 |
| linguistic | lemmatized_logreg | 0.8590 | 0.8538 | 0.8590 | 8.559 | 1.16 | 1.45 | 948.9 |
| linear | logreg | 0.8535 | 0.8459 | 0.8535 | 9.492 | 1.01 | 1.35 | 1255.6 |
| linear | bm25_linear_svc | 0.8441 | 0.8432 | 0.8441 | 34.249 | 10.80 | 13.10 | 3762.5 |
| linguistic | word_pos_logreg | 0.8485 | 0.8413 | 0.8485 | 11.674 | 1.78 | 2.52 | 1827.8 |
| linear | skipgram_logreg | 0.7936 | 0.7792 | 0.7936 | 34.300 | 7.76 | 8.90 | 29505.5 |
| naive_bayes | nb_multinomial | 0.7939 | 0.7548 | 0.7939 | 0.144 | 0.93 | 1.12 | 2455.4 |
| reduced_dim | random_projection_logreg | 0.7375 | 0.7305 | 0.7375 | 9.380 | 1.56 | 2.00 | 281.6 |
| linguistic | pos_sequence_logreg | 0.1528 | 0.1275 | 0.1528 | 4.966 | 1.56 | 2.32 | 369.6 |

## Friedman + Nemenyi

- Friedman p-value: **0.0010** (reject H0: **True**)

```
Critical Difference = 8.7171  (n=3)

rank  baseline
1.000  linear_svc_char
2.000  bm25_logreg
3.000  stemmed_logreg
4.000  lemmatized_logreg
5.667  logreg
6.000  bm25_linear_svc
6.333  word_pos_logreg
8.000  skipgram_logreg
9.000  nb_multinomial
10.000  random_projection_logreg
11.000  pos_sequence_logreg

statistically indistinguishable groups:
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, linear_svc_char, logreg, nb_multinomial, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, bm25_logreg, lemmatized_logreg, logreg, nb_multinomial, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, stemmed_logreg, word_pos_logreg}
  {bm25_linear_svc, lemmatized_logreg, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {bm25_linear_svc, logreg, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {bm25_linear_svc, nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg, word_pos_logreg}
  {nb_multinomial, pos_sequence_logreg, random_projection_logreg, skipgram_logreg}
  {nb_multinomial, pos_sequence_logreg, random_projection_logreg}
  {pos_sequence_logreg, random_projection_logreg}
  {pos_sequence_logreg}
```