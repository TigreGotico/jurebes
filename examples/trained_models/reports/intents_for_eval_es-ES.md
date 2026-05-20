# intents-for-eval (es-ES) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6888 | 0.6809 |
| voting_soft | 0.6600 | 0.6461 |
| logreg | 0.6559 | 0.6422 |
| linear_svc | 0.6429 | 0.6260 |
| ovr_linear_svc | 0.6429 | 0.6260 |
| linear_svc_char | 0.6129 | 0.5791 |
| logreg_char | 0.6082 | 0.5722 |

**winning baseline (intent):** `nb_multinomial`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8297 | 0.7986 | 0.8139 | 0.7906 | 1700 |
| template | 0.6024 | 0.4299 | 0.5017 | 0.6035 | 1700 |
| sklearn_iob | 0.7841 | 0.8020 | 0.7929 | 0.8265 | 1700 |
| knn | 0.5690 | 0.5361 | 0.5521 | 0.6671 | 1700 |
| hybrid | 0.6234 | 0.8564 | 0.7215 | 0.6994 | 1700 |
| crf | 0.8482 | 0.8071 | 0.8272 | 0.8435 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.4477 |
| communication | 174 | 0.7874 |
| media | 174 | 0.6782 |
| navigation | 164 | 0.7073 |
| news | 166 | 0.6265 |
| search_qa | 170 | 0.6000 |
| smarthome | 172 | 0.7500 |
| system_control | 166 | 0.7831 |
| timers_alarms | 172 | 0.8256 |
| weather | 170 | 0.6824 |