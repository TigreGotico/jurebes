# banking77 training report

- train size: **10003**
- test size: **3080**
- intents: **77**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.8880 | 0.8857 | 0.8880 | 2.568 | 3.91 | 4.25 | 33464.2 |
| linear | linear_svc | 0.8823 | 0.8805 | 0.8823 | 1.049 | 3.81 | 4.12 | 4021.4 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.8775 | 0.8761 | 0.8775 | 22.360 | 6.34 | 18.07 | 34132.3 |
| feature_engineering | union_bm25_pos_logreg | 0.8740 | 0.8723 | 0.8740 | 8.890 | 1.18 | 1.50 | 1699.2 |
| linear | bm25_logreg | 0.8729 | 0.8712 | 0.8729 | 6.286 | 0.38 | 0.49 | 1345.1 |
| linear | logreg | 0.8600 | 0.8551 | 0.8600 | 6.161 | 0.36 | 0.45 | 1344.9 |
| reduced_dim | label_guided_linear_svc | 0.8549 | 0.8524 | 0.8549 | 123.543 | 4.12 | 4.49 | 7080.4 |
| reduced_dim | label_guided_logreg | 0.8494 | 0.8475 | 0.8494 | 120.475 | 0.45 | 0.54 | 6864.6 |
| naive_bayes | nb_multinomial | 0.8034 | 0.7653 | 0.8034 | 0.070 | 0.34 | 0.38 | 2631.1 |
| reduced_dim | autoencoder_logreg_wide | 0.7112 | 0.6907 | 0.7112 | 638.668 | 3.33 | 6.67 | 27394.3 |
| reduced_dim | lsa_logreg | 0.6413 | 0.6129 | 0.6413 | 3.289 | 0.45 | 0.57 | 925.9 |
| reduced_dim | autoencoder_logreg | 0.5595 | 0.5191 | 0.5595 | 248.676 | 0.41 | 0.51 | 9569.0 |
| reduced_dim | denoising_autoencoder_logreg | 0.1902 | 0.1154 | 0.1902 | 62.575 | 0.40 | 0.46 | 9566.3 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 8.1601  (n=5)

rank  baseline
1.400  linear_svc_char
2.000  linear_svc
3.400  union_skipgram_tfidf_logreg
3.800  union_bm25_pos_logreg
4.400  bm25_logreg
6.600  logreg
6.800  label_guided_linear_svc
7.600  label_guided_logreg
9.000  nb_multinomial
10.000  autoencoder_logreg_wide
11.000  lsa_logreg
12.000  autoencoder_logreg
13.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_bm25_pos_logreg, union_skipgram_tfidf_logreg}
  {autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial, union_bm25_pos_logreg}
  {autoencoder_logreg, autoencoder_logreg_wide, bm25_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_linear_svc, label_guided_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, label_guided_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, denoising_autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg, denoising_autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg, denoising_autoencoder_logreg}
  {denoising_autoencoder_logreg}
```

**winning baseline:** `linear_svc_char`

## Random search (10 iter) on winner

- best CV f1_macro: **0.8931**
- best params: `{'feat__ngram_range': (2, 4), 'feat__min_df': 2, 'clf__estimator__C': 2.0}`

## Test-set evaluation

- test accuracy: **0.9062**
- test macro-F1: **0.9060**

## Artifact

- model too large to commit, size 15.44 MB (limit 5 MB) — kept locally at `banking77_linear_svc_char.joblib`, excluded via `.gitignore`.