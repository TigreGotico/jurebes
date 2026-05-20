# intents-for-eval (gl-ES) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.7106 | 0.7085 |
| voting_soft | 0.6647 | 0.6476 |
| logreg | 0.6624 | 0.6503 |
| linear_svc | 0.6512 | 0.6280 |
| ovr_linear_svc | 0.6512 | 0.6280 |
| linear_svc_char | 0.6253 | 0.5887 |
| logreg_char | 0.5982 | 0.5552 |

**winning baseline (intent):** `nb_multinomial`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8271 | 0.8129 | 0.8199 | 0.8041 | 1700 |
| template | 0.6473 | 0.4395 | 0.5235 | 0.6112 | 1700 |
| sklearn_iob | 0.8113 | 0.8378 | 0.8243 | 0.8459 | 1700 |
| knn | 0.5800 | 0.5571 | 0.5683 | 0.6835 | 1700 |
| hybrid | 0.6516 | 0.8670 | 0.7440 | 0.7218 | 1700 |
| crf | 0.8796 | 0.8532 | 0.8662 | 0.8741 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.5241 |
| communication | 172 | 0.7733 |
| media | 168 | 0.7381 |
| navigation | 178 | 0.6910 |
| news | 168 | 0.6905 |
| search_qa | 170 | 0.6235 |
| smarthome | 170 | 0.7353 |
| system_control | 166 | 0.7771 |
| timers_alarms | 174 | 0.8333 |
| weather | 168 | 0.7143 |