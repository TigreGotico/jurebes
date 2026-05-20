# intents-for-eval (it-IT) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8382 | 0.8418 |
| voting_soft | 0.8165 | 0.8205 |
| logreg_char | 0.8153 | 0.8165 |
| linear_svc | 0.8135 | 0.8180 |
| ovr_linear_svc | 0.8135 | 0.8180 |
| logreg | 0.7929 | 0.7968 |
| nb_multinomial | 0.7006 | 0.6828 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8306 | 0.8207 | 0.8256 | 0.7935 | 1700 |
| template | 0.6881 | 0.4654 | 0.5553 | 0.6135 | 1700 |
| sklearn_iob | 0.7611 | 0.7728 | 0.7669 | 0.8053 | 1700 |
| knn | 0.5441 | 0.5269 | 0.5354 | 0.6435 | 1700 |
| hybrid | 0.6658 | 0.8676 | 0.7534 | 0.7212 | 1700 |
| crf | 0.8086 | 0.7720 | 0.7899 | 0.8153 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 168 | 0.8095 |
| communication | 172 | 0.8779 |
| media | 176 | 0.7614 |
| navigation | 168 | 0.8690 |
| news | 176 | 0.7841 |
| search_qa | 166 | 0.9096 |
| smarthome | 168 | 0.8750 |
| system_control | 172 | 0.7907 |
| timers_alarms | 170 | 0.9118 |
| weather | 164 | 0.7988 |