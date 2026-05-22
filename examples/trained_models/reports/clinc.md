# clinc training report

- train size: **15000**
- test size: **4500**
- intents: **150**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9401 | 0.9399 | 0.9401 | 27.496 | 24.18 | 40.22 | 106275.7 |
| linear | linear_svc | 0.9382 | 0.9380 | 0.9382 | 17.844 | 23.77 | 43.24 | 16216.1 |
| reduced_dim | label_guided_linear_svc | 0.9247 | 0.9245 | 0.9247 | 536.103 | 32.76 | 76.90 | 21194.2 |
| reduced_dim | label_guided_logreg | 0.9226 | 0.9227 | 0.9226 | 353.835 | 3.35 | 11.13 | 20756.5 |
| linear | logreg | 0.9163 | 0.9161 | 0.9163 | 15.301 | 2.46 | 4.57 | 5435.4 |
| naive_bayes | nb_multinomial | 0.9163 | 0.9154 | 0.9163 | 0.335 | 2.51 | 4.22 | 10765.5 |
| reduced_dim | autoencoder_logreg_wide | 0.7452 | 0.7431 | 0.7452 | 2308.103 | 3.32 | 13.26 | 56494.4 |
| reduced_dim | autoencoder_logreg | 0.6477 | 0.6356 | 0.6477 | 1969.773 | 6.67 | 13.37 | 29381.8 |
| reduced_dim | lsa_logreg | 0.6306 | 0.6142 | 0.6306 | 12.285 | 1.54 | 2.32 | 1943.0 |
| reduced_dim | denoising_autoencoder_logreg | 0.1436 | 0.0939 | 0.1436 | 167.055 | 3.34 | 6.73 | 29379.1 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 6.0586  (n=5)

rank  baseline
1.400  linear_svc_char
1.600  linear_svc
3.000  label_guided_linear_svc
4.000  label_guided_logreg
5.400  logreg
5.600  nb_multinomial
7.000  autoencoder_logreg_wide
8.000  autoencoder_logreg
9.000  lsa_logreg
10.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial}
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
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