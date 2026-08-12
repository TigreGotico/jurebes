# calibration analysis — canonical datasets

ECE = expected calibration error (lower is better, 0 = perfect).
Brier = mean squared error of probability vs one-hot ground truth.
`log_loss` = cross-entropy of predicted probabilities.
Friedman+Nemenyi disabled — calibration is dataset-specific.

## snips

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb | ece | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_bayes | nb_multinomial | 0.9757 | 0.9755 | 0.9757 | 0.051 | 0.28 | 0.32 | 1119.9 | 0.1337 | 0.0808 | 0.2164 |
| linear | logreg | 0.9810 | 0.9809 | 0.9810 | 7.411 | 0.33 | 0.41 | 646.8 | 0.0771 | 0.0497 | 0.1389 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9833 | 0.9832 | 0.9833 | 8.762 | 1.26 | 1.44 | 6464.4 | 0.0628 | 0.0417 | 0.1172 |
| reduced_dim | label_guided_logreg | 0.9793 | 0.9792 | 0.9793 | 41.693 | 0.87 | 3.35 | 51456.8 | 0.0183 | 0.0328 | 0.0747 |
| linear | linear_svc_char | 0.9843 | 0.9842 | 0.9843 | 0.794 | 1.35 | 1.54 | 9663.5 | 0.0164 | 0.0266 | 0.0640 |
| linear | linear_svc | 0.9842 | 0.9841 | 0.9842 | 0.205 | 1.26 | 1.41 | 1599.3 | 0.0154 | 0.0287 | 0.0664 |
| feature_engineering | union_bm25_pos_logreg | 0.9842 | 0.9841 | 0.9842 | 3.335 | 1.16 | 1.36 | 992.4 | 0.0045 | 0.0254 | 0.0550 |
| linear | bm25_logreg | 0.9840 | 0.9840 | 0.9840 | 1.693 | 0.37 | 0.48 | 647.0 | 0.0044 | 0.0255 | 0.0552 |

Best-calibrated: **`bm25_logreg`** (ECE 0.0044). Worst-calibrated: **`nb_multinomial`** (ECE 0.1337).

## banking77

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb | ece | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_bayes | nb_multinomial | 0.7939 | 0.7548 | 0.7939 | 0.051 | 0.29 | 0.31 | 2455.4 | 0.5879 | 0.6883 | 1.9559 |
| linear | logreg | 0.8535 | 0.8459 | 0.8535 | 4.289 | 0.31 | 0.36 | 1255.6 | 0.4323 | 0.4431 | 1.2113 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.8680 | 0.8656 | 0.8680 | 14.995 | 5.63 | 7.67 | 30744.8 | 0.3982 | 0.3941 | 1.0830 |
| linear | linear_svc | 0.8767 | 0.8745 | 0.8767 | 0.815 | 3.59 | 3.71 | 3759.4 | 0.1693 | 0.2203 | 0.6101 |
| linear | linear_svc_char | 0.8836 | 0.8811 | 0.8836 | 1.949 | 3.72 | 4.06 | 31593.6 | 0.1551 | 0.2048 | 0.5634 |
| reduced_dim | label_guided_logreg | 0.8393 | 0.8383 | 0.8393 | 23.757 | 0.43 | 0.53 | 6205.3 | 0.1215 | 0.2563 | 0.6894 |
| linear | bm25_logreg | 0.8703 | 0.8690 | 0.8703 | 4.501 | 0.34 | 0.40 | 1255.8 | 0.0165 | 0.1902 | 0.4904 |
| feature_engineering | union_bm25_pos_logreg | 0.8714 | 0.8704 | 0.8714 | 6.188 | 1.08 | 1.36 | 1609.9 | 0.0157 | 0.1890 | 0.4869 |

Best-calibrated: **`union_bm25_pos_logreg`** (ECE 0.0157). Worst-calibrated: **`nb_multinomial`** (ECE 0.5879).

## clinc

| group | baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms_pooled | p95_ms_pooled | size_kb | ece | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_bayes | nb_multinomial | 0.9085 | 0.9079 | 0.9085 | 0.058 | 0.54 | 0.68 | 9912.8 | 0.7540 | 0.7407 | 2.2219 |
| linear | logreg | 0.9102 | 0.9101 | 0.9102 | 4.731 | 0.55 | 0.85 | 5005.4 | 0.4500 | 0.3853 | 1.1108 |
| feature_engineering | union_skipgram_tfidf_logreg | 0.9281 | 0.9281 | 0.9281 | 21.964 | 32.20 | 38.37 | 71540.6 | 0.4004 | 0.3222 | 0.9434 |
| reduced_dim | label_guided_logreg | 0.9155 | 0.9157 | 0.9155 | 57.255 | 0.69 | 3.32 | 18437.4 | 0.1806 | 0.1806 | 0.5357 |
| linear | linear_svc | 0.9333 | 0.9333 | 0.9333 | 1.918 | 5.62 | 5.90 | 14940.7 | 0.1591 | 0.1329 | 0.4109 |
| linear | linear_svc_char | 0.9345 | 0.9344 | 0.9345 | 3.673 | 5.92 | 6.24 | 99247.4 | 0.1517 | 0.1290 | 0.3929 |
| feature_engineering | union_bm25_pos_logreg | 0.9310 | 0.9312 | 0.9310 | 8.254 | 1.30 | 1.57 | 5368.9 | 0.0307 | 0.1056 | 0.2866 |
| linear | bm25_logreg | 0.9305 | 0.9307 | 0.9305 | 6.094 | 0.58 | 0.83 | 5005.6 | 0.0305 | 0.1067 | 0.2902 |

Best-calibrated: **`bm25_logreg`** (ECE 0.0305). Worst-calibrated: **`nb_multinomial`** (ECE 0.7540).
