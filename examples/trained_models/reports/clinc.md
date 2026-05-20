# clinc training report

- train size: **15000**
- test size: **4500**
- intents: **150**

## Compare with 5-fold CV

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear_svc_char | 0.9401 | 0.9399 | 0.9401 | 247.039 | 77.10 | 377.21 | 106275.7 |
| linear | linear_svc | 0.9382 | 0.9380 | 0.9382 | 119.092 | 47.35 | 183.02 | 16216.1 |
| linear | logreg | 0.9163 | 0.9161 | 0.9163 | 29.391 | 3.55 | 14.51 | 5435.4 |
| naive_bayes | nb_multinomial | 0.9163 | 0.9154 | 0.9163 | 0.983 | 4.09 | 27.36 | 10765.5 |
| reduced_dim | lsa_logreg | 0.6291 | 0.6120 | 0.6291 | 87.979 | 2.74 | 35.96 | 1943.0 |
| reduced_dim | autoencoder_logreg | 0.2433 | 0.1867 | 0.2433 | 410.224 | 1.72 | 2.55 | 18533.9 |

## Friedman + Nemenyi

- Friedman p-value: **0.0003**
- reject H0 (all baselines equal): **True**

```
Critical Difference = 3.3722  (n=5)

rank  baseline
1.400  linear_svc_char
1.600  linear_svc
3.400  logreg
3.600  nb_multinomial
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

- test accuracy: **0.9082**
- test macro-F1: **0.9071**

## Artifact

- model too large to commit, size 113.37 MB (limit 5 MB) — kept locally at `clinc_linear_svc_char.joblib`, excluded via `.gitignore`.