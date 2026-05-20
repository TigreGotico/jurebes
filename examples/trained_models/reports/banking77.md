# banking77 training report

- train size: **10003**
- test size: **3080**
- intents: **77**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.8880 | 0.8857 | 0.8880 | 78.208 | 29.52 | 117.84 | 33464.2 |
| linear | linear_svc | 0.8823 | 0.8805 | 0.8823 | 36.548 | 21.01 | 69.01 | 4021.4 |
| linear | logreg | 0.8600 | 0.8551 | 0.8600 | 34.425 | 1.52 | 8.04 | 1344.9 |
| naive_bayes | nb_multinomial | 0.8034 | 0.7653 | 0.8034 | 0.429 | 1.49 | 8.15 | 2631.1 |
| reduced_dim | lsa_logreg | 0.6455 | 0.6186 | 0.6455 | 45.033 | 1.89 | 12.23 | 925.9 |
| reduced_dim | autoencoder_logreg | 0.2711 | 0.2052 | 0.2711 | 572.768 | 1.88 | 29.07 | 8761.6 |

## Friedman + Nemenyi

- Friedman p-value: **0.0002**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 3.3722  (n=5)

rank  baseline
1.400  linear_svc_char
1.600  linear_svc
3.000  logreg
4.000  nb_multinomial
5.000  lsa_logreg
6.000  autoencoder_logreg

statistically indistinguishable groups:
  {linear_svc, linear_svc_char, logreg, nb_multinomial}
  {linear_svc, logreg, nb_multinomial}
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

- test accuracy: **0.9029**
- test macro-F1: **0.9027**

## Artifact

- model too large to commit, size 35.62 MB (limit 5 MB) — kept locally at `banking77_linear_svc_char.joblib`, excluded via `.gitignore`.