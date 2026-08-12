# hwu64 training report

- train size: **8954**
- test size: **1076**
- intents: **64**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_engineering | union_bm25_pos_logreg | 0.8683 | 0.8654 | 0.8683 | 4.897 | 1.10 | 1.30 | 2278.4 |
| linear | bm25_logreg | 0.8667 | 0.8639 | 0.8667 | 5.719 | 0.39 | 0.78 | 1925.9 |
| linear | linear_svc_char | 0.8663 | 0.8634 | 0.8663 | 1.446 | 2.97 | 3.22 | 36958.6 |
| linear | linear_svc | 0.8639 | 0.8616 | 0.8639 | 0.750 | 2.85 | 3.07 | 5645.9 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.8569 | 0.8547 | 0.8569 | 13.450 | 3.04 | 3.48 | 23134.8 |
| reduced_dim | label_guided_logreg | 0.8454 | 0.8435 | 0.8454 | 138.777 | 0.63 | 0.88 | 14996.1 |
| reduced_dim | label_guided_linear_svc | 0.8439 | 0.8402 | 0.8439 | 41.329 | 3.55 | 3.86 | 15135.9 |
| linear | logreg | 0.8428 | 0.8380 | 0.8428 | 5.354 | 0.33 | 0.47 | 1925.7 |
| naive_bayes | nb_multinomial | 0.8021 | 0.7627 | 0.8021 | 0.032 | 0.33 | 0.37 | 3776.9 |
| reduced_dim | autoencoder_logreg_wide | 0.7186 | 0.7046 | 0.7186 | 385.938 | 0.62 | 3.35 | 46208.9 |
| reduced_dim | autoencoder_logreg | 0.6384 | 0.6161 | 0.6384 | 123.804 | 0.41 | 0.60 | 21582.7 |
| reduced_dim | lsa_logreg | 0.6208 | 0.5944 | 0.6208 | 1.307 | 0.44 | 0.61 | 1547.6 |
| reduced_dim | denoising_autoencoder_logreg | 0.1899 | 0.1119 | 0.1899 | 50.288 | 0.47 | 0.62 | 21579.6 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 8.1601  (n=5)

rank  baseline
1.800  union_bm25_pos_logreg
2.400  linear_svc_char
2.400  bm25_logreg
3.800  linear_svc
4.600  union_skipgram_tfidf_logreg
6.600  label_guided_logreg
6.800  label_guided_linear_svc
7.600  logreg
9.000  nb_multinomial
10.000  autoencoder_logreg_wide
11.000  autoencoder_logreg
12.000  lsa_logreg
13.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_skipgram_tfidf_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg, denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg, lsa_logreg}
  {denoising_autoencoder_logreg}
```

**winning baseline:** `union_bm25_pos_logreg`

## Random search (10 iter) on winner

> [!warning]
> **no search space**
>
> `spaces.for_baseline('union_bm25_pos_logreg')` returned nothing; evaluating untuned default instead.

## Test-set evaluation

- test accuracy: **0.8690**
- test macro-F1: **0.8681**

## Artifact

- saved `hwu64_union_bm25_pos_logreg.joblib` (2.76 MB)