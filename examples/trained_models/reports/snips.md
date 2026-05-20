# snips training report

_generated: 2026-05-19 23:49:24_

- train size: **13084**
- test size: **1400**
- intents: **7**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9856 | 0.9855 | 0.9856 | 9.752 | 5.74 | 14.59 | 10545.4 |
| linear | linear_svc | 0.9848 | 0.9848 | 0.9848 | 6.385 | 5.77 | 20.62 | 1789.3 |
| linear | logreg | 0.9823 | 0.9822 | 0.9823 | 36.906 | 1.43 | 4.90 | 724.0 |
| naive_bayes | nb_multinomial | 0.9765 | 0.9763 | 0.9765 | 0.305 | 1.44 | 1.87 | 1253.5 |
| reduced_dim | lsa_logreg | 0.9622 | 0.9620 | 0.9622 | 43.856 | 2.64 | 15.00 | 3981.4 |
| reduced_dim | autoencoder_logreg | 0.9398 | 0.9397 | 0.9398 | 1071.525 | 39.81 | 160.88 | 39308.6 |

## Friedman + Nemenyi

- Friedman p-value: **0.0002**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 3.3722  (n=5)

rank  baseline
1.000  linear_svc_char
2.200  linear_svc
2.800  logreg
4.000  nb_multinomial
5.000  lsa_logreg
6.000  autoencoder_logreg

statistically indistinguishable groups:
  {linear_svc, linear_svc_char, logreg, nb_multinomial}
  {linear_svc, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, lsa_logreg, nb_multinomial}
  {autoencoder_logreg, lsa_logreg}
  {autoencoder_logreg}
```

**winning baseline:** `linear_svc_char`

## Random search (10 iter) on winner

> [!warning]
> **no search space**
>
> `spaces.for_baseline('linear_svc_char')` returned nothing; evaluating untuned default instead.

## Test-set evaluation

- test accuracy: **0.9850**
- test macro-F1: **0.9850**

## Artifact

- model too large to commit, size 12.06 MB (limit 5 MB) — kept locally at `snips_linear_svc_char.joblib`, excluded via `.gitignore`.