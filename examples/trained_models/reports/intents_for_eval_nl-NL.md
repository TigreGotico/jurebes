# intents-for-eval (nl-NL) training report

- intents: **50**
- templates: **1000**
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| nb_multinomial | 0.6941 | 0.6957 |
| voting_soft | 0.6753 | 0.6626 |
| linear_svc | 0.6688 | 0.6559 |
| ovr_linear_svc | 0.6688 | 0.6559 |
| logreg | 0.6553 | 0.6421 |
| linear_svc_char | 0.6018 | 0.5751 |
| logreg_char | 0.5876 | 0.5486 |

**winning baseline (intent):** `nb_multinomial`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8546 | 0.8266 | 0.8404 | 0.8129 | 1700 |
| template | 0.6678 | 0.4978 | 0.5704 | 0.6253 | 1700 |
| sklearn_iob | 0.8164 | 0.8059 | 0.8111 | 0.8412 | 1700 |
| knn | 0.5614 | 0.5168 | 0.5382 | 0.6335 | 1700 |
| hybrid | 0.6700 | 0.8792 | 0.7604 | 0.7194 | 1700 |
| crf | 0.8520 | 0.8145 | 0.8328 | 0.8500 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `nb_multinomial`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.6325 |
| communication | 170 | 0.7353 |
| media | 178 | 0.7191 |
| navigation | 170 | 0.7118 |
| news | 168 | 0.5714 |
| search_qa | 168 | 0.5476 |
| smarthome | 174 | 0.7816 |
| system_control | 168 | 0.7262 |
| timers_alarms | 166 | 0.8434 |
| weather | 172 | 0.6686 |