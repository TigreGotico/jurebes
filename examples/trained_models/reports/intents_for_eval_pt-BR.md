# intents-for-eval (pt-BR) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.7129 | 0.7101 |
| logreg | 0.6571 | 0.6480 |
| voting_soft | 0.6488 | 0.6333 |
| linear_svc | 0.6365 | 0.6183 |
| ovr_linear_svc | 0.6365 | 0.6183 |
| logreg_char | 0.5853 | 0.5379 |
| linear_svc_char | 0.5812 | 0.5362 |

**winning baseline (intent):** `nb_multinomial`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8605 | 0.8291 | 0.8445 | 0.8171 | 1700 |
| template | 0.6797 | 0.4422 | 0.5358 | 0.6165 | 1700 |
| sklearn_iob | 0.7800 | 0.7900 | 0.7850 | 0.8288 | 1700 |
| knn | 0.5449 | 0.5009 | 0.5219 | 0.6406 | 1700 |
| hybrid | 0.6878 | 0.8861 | 0.7744 | 0.7500 | 1700 |
| crf | 0.8344 | 0.7925 | 0.8129 | 0.8371 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 170 | 0.5824 |
| communication | 172 | 0.7733 |
| media | 174 | 0.7126 |
| navigation | 166 | 0.7048 |
| news | 166 | 0.7169 |
| search_qa | 178 | 0.6011 |
| smarthome | 166 | 0.7530 |
| system_control | 170 | 0.7118 |
| timers_alarms | 170 | 0.8647 |
| weather | 168 | 0.7143 |