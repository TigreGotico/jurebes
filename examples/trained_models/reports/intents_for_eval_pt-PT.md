# intents-for-eval (pt-PT) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.7112 | 0.7114 |
| logreg | 0.6606 | 0.6519 |
| voting_soft | 0.6441 | 0.6313 |
| linear_svc | 0.6324 | 0.6092 |
| ovr_linear_svc | 0.6324 | 0.6092 |
| logreg_char | 0.5929 | 0.5535 |
| linear_svc_char | 0.5906 | 0.5540 |

**winning baseline (intent):** `nb_multinomial`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.7978 | 0.8204 | 0.8090 | 0.7947 | 1700 |
| template | 0.6921 | 0.4519 | 0.5468 | 0.6294 | 1700 |
| sklearn_iob | 0.7683 | 0.7749 | 0.7716 | 0.8171 | 1700 |
| knn | 0.5261 | 0.5026 | 0.5141 | 0.6200 | 1700 |
| hybrid | 0.6572 | 0.8746 | 0.7505 | 0.7294 | 1700 |
| crf | 0.8348 | 0.7947 | 0.8143 | 0.8365 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.5465 |
| communication | 164 | 0.7622 |
| media | 170 | 0.7235 |
| navigation | 170 | 0.7706 |
| news | 172 | 0.6860 |
| search_qa | 174 | 0.6034 |
| smarthome | 168 | 0.7321 |
| system_control | 166 | 0.7289 |
| timers_alarms | 168 | 0.8690 |
| weather | 176 | 0.6989 |