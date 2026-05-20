# intents-for-eval (en-US) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6971 | 0.6960 |
| voting_soft | 0.6729 | 0.6643 |
| linear_svc | 0.6635 | 0.6529 |
| ovr_linear_svc | 0.6635 | 0.6529 |
| logreg | 0.6535 | 0.6426 |
| linear_svc_char | 0.5835 | 0.5434 |
| logreg_char | 0.5565 | 0.5063 |

**winning baseline (intent):** `nb_multinomial`

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

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.5523 |
| communication | 168 | 0.6905 |
| media | 168 | 0.7083 |
| navigation | 172 | 0.7500 |
| news | 168 | 0.6667 |
| search_qa | 170 | 0.5647 |
| smarthome | 168 | 0.7798 |
| system_control | 174 | 0.7644 |
| timers_alarms | 170 | 0.8353 |
| weather | 170 | 0.6588 |