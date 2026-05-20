# intents-for-eval (pt-PT) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8400 | 0.8449 |
| voting_soft | 0.8194 | 0.8221 |
| logreg_char | 0.8159 | 0.8176 |
| linear_svc | 0.8147 | 0.8187 |
| ovr_linear_svc | 0.8147 | 0.8187 |
| logreg | 0.7988 | 0.8030 |
| nb_multinomial | 0.6824 | 0.6668 |

**winning baseline (intent):** `linear_svc_char`

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

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.7674 |
| communication | 164 | 0.8963 |
| media | 170 | 0.8176 |
| navigation | 170 | 0.8882 |
| news | 172 | 0.8488 |
| search_qa | 174 | 0.9080 |
| smarthome | 168 | 0.7976 |
| system_control | 166 | 0.7711 |
| timers_alarms | 168 | 0.8750 |
| weather | 176 | 0.8295 |