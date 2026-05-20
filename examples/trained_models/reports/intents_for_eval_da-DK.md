# intents-for-eval (da-DK) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8576 | 0.8612 |
| logreg_char | 0.8300 | 0.8329 |
| linear_svc | 0.8135 | 0.8192 |
| ovr_linear_svc | 0.8135 | 0.8192 |
| voting_soft | 0.8124 | 0.8174 |
| logreg | 0.7794 | 0.7846 |
| nb_multinomial | 0.6571 | 0.6355 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.7805 | 0.7142 | 0.7459 | 0.7418 | 1700 |
| template | 0.6671 | 0.4970 | 0.5696 | 0.6171 | 1700 |
| sklearn_iob | 0.8118 | 0.7442 | 0.7765 | 0.8035 | 1700 |
| knn | 0.6059 | 0.5820 | 0.5937 | 0.6718 | 1700 |
| hybrid | 0.6153 | 0.8129 | 0.7004 | 0.6600 | 1700 |
| crf | 0.8447 | 0.7330 | 0.7849 | 0.8041 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 174 | 0.8276 |
| communication | 164 | 0.9146 |
| media | 168 | 0.8512 |
| navigation | 172 | 0.8663 |
| news | 174 | 0.7989 |
| search_qa | 168 | 0.8988 |
| smarthome | 168 | 0.8452 |
| system_control | 172 | 0.8372 |
| timers_alarms | 166 | 0.9096 |
| weather | 174 | 0.8333 |