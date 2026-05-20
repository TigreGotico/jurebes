# intents-for-eval (it-IT) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6918 | 0.6890 |
| voting_soft | 0.6729 | 0.6624 |
| linear_svc | 0.6635 | 0.6531 |
| ovr_linear_svc | 0.6635 | 0.6531 |
| logreg | 0.6594 | 0.6477 |
| linear_svc_char | 0.6382 | 0.6099 |
| logreg_char | 0.5959 | 0.5543 |

**winning baseline (intent):** `nb_multinomial`

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

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 168 | 0.5952 |
| communication | 172 | 0.7616 |
| media | 176 | 0.6761 |
| navigation | 168 | 0.7083 |
| news | 176 | 0.5909 |
| search_qa | 166 | 0.6265 |
| smarthome | 168 | 0.7143 |
| system_control | 172 | 0.7907 |
| timers_alarms | 170 | 0.8824 |
| weather | 164 | 0.5671 |