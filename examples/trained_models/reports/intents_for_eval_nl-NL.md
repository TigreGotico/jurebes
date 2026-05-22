# intents-for-eval (nl-NL) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8359 | 0.8393 |
| label_guided_logreg | 0.8212 | 0.8257 |
| label_guided_linear_svc | 0.8206 | 0.8229 |
| logreg_char | 0.8147 | 0.8182 |
| voting_soft | 0.8112 | 0.8157 |
| linear_svc | 0.8106 | 0.8149 |
| ovr_linear_svc | 0.8106 | 0.8149 |
| logreg | 0.7924 | 0.7981 |
| nb_multinomial | 0.6565 | 0.6282 |
| autoencoder_logreg | 0.4465 | 0.4046 |
| denoising_autoencoder_logreg | 0.2124 | 0.1316 |

**winning baseline (intent):** `linear_svc_char`

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

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.8313 |
| communication | 170 | 0.8588 |
| media | 178 | 0.8202 |
| navigation | 170 | 0.8588 |
| news | 168 | 0.7679 |
| search_qa | 168 | 0.8750 |
| smarthome | 174 | 0.8276 |
| system_control | 168 | 0.8036 |
| timers_alarms | 166 | 0.8614 |
| weather | 172 | 0.8547 |