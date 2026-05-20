# intents-for-eval (en-US) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8306 | 0.8339 |
| voting_soft | 0.8282 | 0.8323 |
| linear_svc | 0.8247 | 0.8292 |
| ovr_linear_svc | 0.8247 | 0.8292 |
| logreg_char | 0.8176 | 0.8198 |
| logreg | 0.8018 | 0.8022 |
| nb_multinomial | 0.6835 | 0.6665 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8447 | 0.8622 | 0.8534 | 0.8176 | 1700 |
| template | 0.6376 | 0.4668 | 0.5390 | 0.5971 | 1700 |
| sklearn_iob | 0.8237 | 0.8372 | 0.8304 | 0.8506 | 1700 |
| knn | 0.5387 | 0.5099 | 0.5239 | 0.6294 | 1700 |
| hybrid | 0.6545 | 0.9121 | 0.7621 | 0.7129 | 1700 |
| crf | 0.9038 | 0.8501 | 0.8762 | 0.8829 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.7791 |
| communication | 168 | 0.8810 |
| media | 168 | 0.7738 |
| navigation | 172 | 0.8198 |
| news | 168 | 0.7738 |
| search_qa | 170 | 0.9000 |
| smarthome | 168 | 0.8512 |
| system_control | 174 | 0.8621 |
| timers_alarms | 170 | 0.8765 |
| weather | 170 | 0.7882 |