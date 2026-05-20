# intents-for-eval (da-DK) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6929 | 0.6972 |
| voting_soft | 0.6688 | 0.6586 |
| linear_svc | 0.6671 | 0.6550 |
| ovr_linear_svc | 0.6671 | 0.6550 |
| logreg | 0.6547 | 0.6518 |
| linear_svc_char | 0.6106 | 0.5813 |
| logreg_char | 0.5924 | 0.5560 |

**winning baseline (intent):** `nb_multinomial`

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

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 174 | 0.5805 |
| communication | 164 | 0.7744 |
| media | 168 | 0.6905 |
| navigation | 172 | 0.7209 |
| news | 174 | 0.6609 |
| search_qa | 168 | 0.6071 |
| smarthome | 168 | 0.8036 |
| system_control | 172 | 0.7035 |
| timers_alarms | 166 | 0.7952 |
| weather | 174 | 0.6034 |