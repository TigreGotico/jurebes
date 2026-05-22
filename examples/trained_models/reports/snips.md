# snips training report

- train size: **13084**
- test size: **1400**
- intents: **7**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9856 | 0.9855 | 0.9856 | 6.105 | 5.88 | 7.77 | 10545.4 |
| linear | linear_svc | 0.9848 | 0.9848 | 0.9848 | 2.931 | 5.53 | 6.26 | 1789.3 |
| linear | logreg | 0.9823 | 0.9822 | 0.9823 | 21.707 | 1.41 | 1.55 | 724.0 |
| reduced_dim | label_guided_logreg | 0.9818 | 0.9818 | 0.9818 | 347.093 | 6.67 | 13.40 | 60996.4 |
| reduced_dim | label_guided_linear_svc | 0.9805 | 0.9805 | 0.9805 | 350.979 | 13.22 | 23.88 | 61004.9 |
| naive_bayes | nb_multinomial | 0.9765 | 0.9763 | 0.9765 | 0.299 | 1.40 | 1.53 | 1253.5 |
| reduced_dim | autoencoder_logreg_wide | 0.9708 | 0.9708 | 0.9708 | 2418.795 | 4.74 | 15.56 | 118190.9 |
| reduced_dim | autoencoder_logreg | 0.9675 | 0.9675 | 0.9675 | 2564.952 | 6.63 | 36.56 | 90591.2 |
| reduced_dim | lsa_logreg | 0.9622 | 0.9620 | 0.9622 | 19.475 | 2.44 | 2.75 | 3981.4 |
| reduced_dim | denoising_autoencoder_logreg | 0.9224 | 0.9225 | 0.9224 | 567.407 | 6.60 | 13.43 | 90589.5 |

## Friedman + Nemenyi

- Friedman p-value: **0.0000**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 6.0586  (n=5)

rank  baseline
1.000  linear_svc_char
2.200  linear_svc
3.600  logreg
3.600  label_guided_logreg
4.600  label_guided_linear_svc
6.000  nb_multinomial
7.000  autoencoder_logreg_wide
8.000  autoencoder_logreg
9.000  lsa_logreg
10.000  denoising_autoencoder_logreg

statistically indistinguishable groups:
  {autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, linear_svc_char, logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, linear_svc, logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, autoencoder_logreg_wide, label_guided_linear_svc, label_guided_logreg, lsa_logreg, nb_multinomial}
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