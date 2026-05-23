# clinc training report

- train size: **15000**
- test size: **4500**
- intents: **150**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9401 | 0.9399 | 0.9401 | 4.942 | 6.26 | 6.72 | 106275.7 |
| feature_engineering | union_bm25_pos_logreg | 0.9381 | 0.9382 | 0.9381 | 9.340 | 1.38 | 1.82 | 5798.8 |
| linear | linear_svc | 0.9382 | 0.9380 | 0.9382 | 2.582 | 6.18 | 6.62 | 16216.1 |
| linear | bm25_logreg | 0.9367 | 0.9369 | 0.9367 | 10.846 | 0.76 | 0.90 | 5435.6 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9351 | 0.9351 | 0.9351 | 23.469 | 38.51 | 44.14 | 80439.8 |
| reduced_dim | label_guided_linear_svc | 0.9247 | 0.9245 | 0.9247 | 119.835 | 11.70 | 23.07 | 21194.2 |
| reduced_dim | label_guided_logreg | 0.9226 | 0.9227 | 0.9226 | 85.170 | 0.70 | 3.78 | 20756.5 |
| linear | logreg | 0.9163 | 0.9161 | 0.9163 | 7.083 | 0.68 | 0.79 | 5435.4 |
| naive_bayes | nb_multinomial | 0.9163 | 0.9154 | 0.9163 | 0.088 | 0.67 | 1.04 | 10765.5 |
| reduced_dim | autoencoder_logreg_wide | 0.7452 | 0.7431 | 0.7452 | 1885.714 | 3.32 | 6.75 | 56494.4 |
| reduced_dim | autoencoder_logreg | 0.6477 | 0.6356 | 0.6477 | 721.233 | 3.34 | 6.68 | 29381.8 |
| reduced_dim | lsa_logreg | 0.6319 | 0.6158 | 0.6319 | 3.756 | 0.51 | 0.61 | 1943.0 |
| reduced_dim | denoising_autoencoder_logreg | 0.1436 | 0.0939 | 0.1436 | 57.747 | 0.62 | 5.00 | 29379.1 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 8.1601  (n=5)

rank  baseline
2.200  linear_svc_char
2.600  union_bm25_pos_logreg
3.000  linear_svc
3.400  bm25_logreg
3.800  union_skipgram_tfidf_logreg
6.000  label_guided_linear_svc
7.000  label_guided_logreg
8.400  logreg
8.600  nb_multinomial
10.000  autoencoder_logreg_wide
11.000  autoencoder_logreg
12.000  lsa_logreg
13.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg, denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg}
```

**winning baseline:** `linear_svc_char`

## Random search (10 iter) on winner

- best CV f1_macro: **0.9255**
- best params: `{'feat__ngram_range': (2, 4), 'feat__min_df': 2, 'clf__estimator__C': 2.0}`

## Test-set evaluation

- test accuracy: **0.9164**
- test macro-F1: **0.9157**

## Artifact

- model too large to commit, size 43.83 MB (limit 5 MB) — kept locally at `clinc_linear_svc_char.joblib`, excluded via `.gitignore`.