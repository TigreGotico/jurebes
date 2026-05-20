# intents-for-eval (ca-ES) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6824 | 0.6793 |
| voting_soft | 0.6659 | 0.6531 |
| linear_svc | 0.6635 | 0.6489 |
| ovr_linear_svc | 0.6635 | 0.6489 |
| logreg | 0.6618 | 0.6466 |
| linear_svc_char | 0.6218 | 0.5856 |
| logreg_char | 0.6029 | 0.5645 |

**winning baseline (intent):** `nb_multinomial`

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

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 176 | 0.4830 |
| communication | 164 | 0.7866 |
| media | 172 | 0.7500 |
| navigation | 170 | 0.7059 |
| news | 180 | 0.6056 |
| search_qa | 170 | 0.6176 |
| smarthome | 162 | 0.7778 |
| system_control | 166 | 0.7229 |
| timers_alarms | 166 | 0.8313 |
| weather | 174 | 0.5690 |