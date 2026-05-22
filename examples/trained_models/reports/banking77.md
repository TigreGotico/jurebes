# banking77 training report

- train size: **10003**
- test size: **3080**
- intents: **77**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.8880 | 0.8857 | 0.8880 | 12.539 | 13.99 | 26.61 | 33464.2 |
| linear | linear_svc | 0.8823 | 0.8805 | 0.8823 | 7.243 | 13.53 | 25.59 | 4021.4 |
| linear | logreg | 0.8600 | 0.8551 | 0.8600 | 10.486 | 1.12 | 1.50 | 1344.9 |
| reduced_dim | label_guided_linear_svc | 0.8549 | 0.8524 | 0.8549 | 435.376 | 14.86 | 28.34 | 7080.4 |
| reduced_dim | label_guided_logreg | 0.8494 | 0.8475 | 0.8494 | 362.694 | 1.38 | 1.90 | 6864.6 |
| naive_bayes | nb_multinomial | 0.8034 | 0.7653 | 0.8034 | 0.241 | 1.10 | 1.48 | 2631.1 |
| reduced_dim | autoencoder_logreg_wide | 0.7112 | 0.6907 | 0.7112 | 1022.009 | 6.67 | 13.38 | 27394.3 |
| reduced_dim | lsa_logreg | 0.6447 | 0.6166 | 0.6447 | 7.833 | 1.40 | 1.97 | 925.9 |
| reduced_dim | autoencoder_logreg | 0.5595 | 0.5191 | 0.5595 | 582.796 | 1.24 | 1.82 | 9569.0 |
| reduced_dim | denoising_autoencoder_logreg | 0.1902 | 0.1154 | 0.1902 | 143.524 | 1.24 | 1.75 | 9566.3 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 6.0586  (n=5)

rank  baseline
1.400  linear_svc_char
1.600  linear_svc
3.600  logreg
3.800  label_guided_linear_svc
4.600  label_guided_logreg
6.000  nb_multinomial
7.000  autoencoder_logreg_wide
8.000  lsa_logreg
9.000  autoencoder_logreg
10.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial}
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, lsa_logreg, nb_multinomial}
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