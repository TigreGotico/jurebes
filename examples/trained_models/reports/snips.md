# snips training report

- train size: **13084**
- test size: **1400**
- intents: **7**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9856 | 0.9855 | 0.9856 | 0.811 | 1.18 | 1.37 | 10545.4 |
| feature_engineering | union_bm25_pos_logreg | 0.9852 | 0.9851 | 0.9852 | 3.473 | 1.12 | 1.29 | 1069.5 |
| linear | linear_svc | 0.9848 | 0.9848 | 0.9848 | 0.203 | 1.20 | 1.36 | 1789.3 |
| linear | bm25_logreg | 0.9846 | 0.9846 | 0.9846 | 1.410 | 0.35 | 0.43 | 724.2 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9844 | 0.9844 | 0.9844 | 10.456 | 1.36 | 1.53 | 7479.1 |
| linear | logreg | 0.9823 | 0.9822 | 0.9823 | 6.631 | 0.30 | 0.39 | 724.0 |
| reduced_dim | label_guided_logreg | 0.9818 | 0.9818 | 0.9818 | 134.566 | 3.34 | 6.70 | 60996.4 |
| reduced_dim | label_guided_linear_svc | 0.9805 | 0.9805 | 0.9805 | 134.693 | 3.39 | 6.98 | 61004.9 |
| naive_bayes | nb_multinomial | 0.9765 | 0.9763 | 0.9765 | 0.058 | 0.27 | 0.32 | 1253.5 |
| reduced_dim | autoencoder_logreg_wide | 0.9708 | 0.9708 | 0.9708 | 2512.167 | 3.34 | 6.69 | 118190.9 |
| reduced_dim | autoencoder_logreg | 0.9675 | 0.9675 | 0.9675 | 681.108 | 1.75 | 6.62 | 90591.2 |
| reduced_dim | lsa_logreg | 0.9625 | 0.9623 | 0.9625 | 1.029 | 0.56 | 0.76 | 3981.4 |
| reduced_dim | denoising_autoencoder_logreg | 0.9224 | 0.9225 | 0.9224 | 225.195 | 3.34 | 6.67 | 90589.5 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 8.1601  (n=5)

rank  baseline
1.400  linear_svc_char
2.400  union_bm25_pos_logreg
3.600  bm25_logreg
3.800  linear_svc
4.400  union_skipgram_tfidf_logreg
6.400  logreg
6.400  label_guided_logreg
7.600  label_guided_linear_svc
9.000  nb_multinomial
10.000  autoencoder_logreg_wide
11.000  autoencoder_logreg
12.000  lsa_logreg
13.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg, denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg}
```

**winning baseline:** `linear_svc_char`

## Random search (10 iter) on winner

- best CV f1_macro: **0.9858**
- best params: `{'feat__ngram_range': (3, 6), 'feat__min_df': 2, 'clf__estimator__C': 1.0}`

## Test-set evaluation

- test accuracy: **0.9843**
- test macro-F1: **0.9843**

## Artifact

- model too large to commit, size 9.00 MB (limit 5 MB) — kept locally at `snips_linear_svc_char.joblib`, excluded via `.gitignore`.