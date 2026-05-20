# intents-for-eval (fr-FR) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8382 | 0.8418 |
| voting_soft | 0.8182 | 0.8200 |
| linear_svc | 0.8147 | 0.8180 |
| ovr_linear_svc | 0.8147 | 0.8180 |
| logreg_char | 0.8024 | 0.8019 |
| logreg | 0.7847 | 0.7850 |
| nb_multinomial | 0.6653 | 0.6391 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8316 | 0.8295 | 0.8305 | 0.7976 | 1700 |
| template | 0.6346 | 0.4584 | 0.5323 | 0.6118 | 1700 |
| sklearn_iob | 0.6743 | 0.7095 | 0.6914 | 0.7394 | 1700 |
| knn | 0.4781 | 0.4396 | 0.4580 | 0.5965 | 1700 |
| hybrid | 0.6438 | 0.8860 | 0.7458 | 0.6947 | 1700 |
| crf | 0.7578 | 0.7266 | 0.7419 | 0.7888 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.8012 |
| communication | 180 | 0.8889 |
| media | 168 | 0.8036 |
| navigation | 170 | 0.8294 |
| news | 172 | 0.7791 |
| search_qa | 168 | 0.9107 |
| smarthome | 166 | 0.8795 |
| system_control | 170 | 0.7706 |
| timers_alarms | 168 | 0.9167 |
| weather | 172 | 0.8023 |