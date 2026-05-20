# intents-for-eval (ca-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8471 | 0.8502 |
| logreg_char | 0.8253 | 0.8263 |
| voting_soft | 0.8253 | 0.8283 |
| linear_svc | 0.8200 | 0.8243 |
| ovr_linear_svc | 0.8200 | 0.8243 |
| logreg | 0.8076 | 0.8098 |
| nb_multinomial | 0.6965 | 0.6759 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8157 | 0.8171 | 0.8164 | 0.7924 | 1700 |
| template | 0.6397 | 0.4479 | 0.5269 | 0.6082 | 1700 |
| sklearn_iob | 0.7109 | 0.7536 | 0.7316 | 0.7718 | 1700 |
| knn | 0.4875 | 0.4793 | 0.4833 | 0.6159 | 1700 |
| hybrid | 0.6374 | 0.8738 | 0.7371 | 0.6924 | 1700 |
| crf | 0.7912 | 0.7570 | 0.7737 | 0.8112 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 176 | 0.8182 |
| communication | 164 | 0.9146 |
| media | 172 | 0.7965 |
| navigation | 170 | 0.8765 |
| news | 180 | 0.7889 |
| search_qa | 170 | 0.8941 |
| smarthome | 162 | 0.9012 |
| system_control | 166 | 0.8012 |
| timers_alarms | 166 | 0.8795 |
| weather | 174 | 0.8103 |