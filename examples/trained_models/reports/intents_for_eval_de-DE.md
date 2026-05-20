# intents-for-eval (de-DE) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8406 | 0.8431 |
| logreg_char | 0.8229 | 0.8239 |
| voting_soft | 0.8182 | 0.8222 |
| linear_svc | 0.8118 | 0.8168 |
| ovr_linear_svc | 0.8118 | 0.8168 |
| logreg | 0.7918 | 0.7958 |
| nb_multinomial | 0.6559 | 0.6292 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8663 | 0.8134 | 0.8390 | 0.8141 | 1700 |
| template | 0.6380 | 0.4394 | 0.5204 | 0.5929 | 1700 |
| sklearn_iob | 0.8731 | 0.8340 | 0.8531 | 0.8647 | 1700 |
| knn | 0.5381 | 0.4858 | 0.5106 | 0.6229 | 1700 |
| hybrid | 0.6907 | 0.8831 | 0.7751 | 0.7465 | 1700 |
| crf | 0.8993 | 0.8366 | 0.8668 | 0.8694 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.8372 |
| communication | 172 | 0.8895 |
| media | 166 | 0.7892 |
| navigation | 170 | 0.8529 |
| news | 176 | 0.7898 |
| search_qa | 168 | 0.8988 |
| smarthome | 170 | 0.8529 |
| system_control | 166 | 0.8253 |
| timers_alarms | 170 | 0.8353 |
| weather | 170 | 0.8353 |