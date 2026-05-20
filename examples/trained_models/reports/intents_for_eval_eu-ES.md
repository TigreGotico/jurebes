# intents-for-eval (eu-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8388 | 0.8396 |
| logreg_char | 0.8194 | 0.8201 |
| voting_soft | 0.7776 | 0.7823 |
| linear_svc | 0.7771 | 0.7829 |
| ovr_linear_svc | 0.7771 | 0.7829 |
| logreg | 0.7535 | 0.7575 |
| nb_multinomial | 0.6553 | 0.6371 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8662 | 0.6885 | 0.7672 | 0.7494 | 1700 |
| template | 0.5625 | 0.3447 | 0.4274 | 0.5529 | 1700 |
| sklearn_iob | 0.8053 | 0.6723 | 0.7328 | 0.7641 | 1700 |
| knn | 0.4436 | 0.4417 | 0.4426 | 0.5876 | 1700 |
| hybrid | 0.6528 | 0.7345 | 0.6912 | 0.6659 | 1700 |
| crf | 0.8438 | 0.6621 | 0.7420 | 0.7747 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 174 | 0.7586 |
| communication | 168 | 0.8869 |
| media | 172 | 0.8372 |
| navigation | 170 | 0.8706 |
| news | 170 | 0.8059 |
| search_qa | 180 | 0.8889 |
| smarthome | 168 | 0.8631 |
| system_control | 162 | 0.8272 |
| timers_alarms | 168 | 0.8214 |
| weather | 168 | 0.8274 |