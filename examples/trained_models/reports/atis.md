# atis training report

- train size: **4972**
- test size: **884**
- intents: **17**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | bm25_logreg | 0.9632 | 0.7475 | 0.9632 | 5.180 | 0.30 | 0.43 | 124.8 |
| feature_engineering | union_bm25_pos_logreg | 0.9630 | 0.7473 | 0.9630 | 6.139 | 1.09 | 1.31 | 471.2 |
| linear | linear_svc | 0.9630 | 0.7323 | 0.9630 | 0.164 | 1.53 | 1.68 | 349.4 |
| linear | linear_svc_char | 0.9638 | 0.7310 | 0.9638 | 0.471 | 1.54 | 1.74 | 2584.2 |
| reduced_dim | label_guided_linear_svc | 0.9523 | 0.7004 | 0.9523 | 3.707 | 1.74 | 1.92 | 1868.2 |
| reduced_dim | label_guided_logreg | 0.9499 | 0.6495 | 0.9499 | 3.652 | 0.40 | 0.58 | 1844.1 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9403 | 0.5452 | 0.9403 | 5.792 | 0.91 | 1.10 | 2889.3 |
| linear | logreg | 0.9270 | 0.4350 | 0.9270 | 5.051 | 0.25 | 0.36 | 124.7 |
| reduced_dim | autoencoder_logreg_wide | 0.8966 | 0.3845 | 0.8966 | 97.904 | 0.41 | 0.58 | 11295.9 |
| reduced_dim | lsa_logreg | 0.8669 | 0.2867 | 0.8669 | 0.737 | 0.38 | 0.58 | 340.8 |
| reduced_dim | autoencoder_logreg | 0.8409 | 0.2394 | 0.8409 | 14.422 | 0.36 | 0.52 | 2572.6 |
| naive_bayes | nb_multinomial | 0.8477 | 0.2313 | 0.8477 | 0.024 | 0.22 | 0.35 | 231.8 |
| reduced_dim | denoising_autoencoder_logreg | 0.8025 | 0.1550 | 0.8025 | 3.870 | 0.37 | 0.52 | 2567.7 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 8.1601  (n=5)

rank  baseline
1.800  bm25_logreg
2.000  union_bm25_pos_logreg
3.200  linear_svc
3.600  linear_svc_char
4.800  label_guided_linear_svc
5.600  label_guided_logreg
7.000  union_skipgram_tfidf_logreg
8.200  logreg
8.800  autoencoder_logreg_wide
10.000  lsa_logreg
11.400  autoencoder_logreg
11.600  nb_multinomial
13.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, lsa_logreg, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, lsa_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc_char, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, denoising_autoencoder_logreg, nb_multinomial}
  {denoising_autoencoder_logreg, nb_multinomial}
  {denoising_autoencoder_logreg}
```

**winning baseline:** `bm25_logreg`

## Random search (10 iter) on winner

- best CV f1_macro: **0.7712**
- best params: `{'feat__bm25__k1': 0.8, 'feat__bm25__b': 0.5, 'clf__C': 4.0}`

## Test-set evaluation

- test accuracy: **0.9446**
- test macro-F1: **0.6629**

## Artifact

- saved `atis_bm25_logreg.joblib` (0.45 MB)