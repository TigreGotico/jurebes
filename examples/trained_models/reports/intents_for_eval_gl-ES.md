# intents-for-eval (gl-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8441 | 0.8471 |
| voting_soft | 0.8188 | 0.8222 |
| label_guided_logreg | 0.8176 | 0.8211 |
| label_guided_linear_svc | 0.8176 | 0.8215 |
| linear_svc | 0.8165 | 0.8214 |
| ovr_linear_svc | 0.8165 | 0.8214 |
| logreg_char | 0.8141 | 0.8144 |
| logreg | 0.7906 | 0.7957 |
| nb_multinomial | 0.6724 | 0.6575 |
| autoencoder_logreg | 0.4629 | 0.4036 |
| denoising_autoencoder_logreg | 0.1518 | 0.0767 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8271 | 0.8129 | 0.8199 | 0.8041 | 1700 |
| template | 0.6473 | 0.4395 | 0.5235 | 0.6112 | 1700 |
| sklearn_iob | 0.8113 | 0.8378 | 0.8243 | 0.8459 | 1700 |
| knn | 0.5800 | 0.5571 | 0.5683 | 0.6835 | 1700 |
| hybrid | 0.6516 | 0.8670 | 0.7440 | 0.7218 | 1700 |
| crf | 0.8796 | 0.8532 | 0.8662 | 0.8741 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.8072 |
| communication | 172 | 0.8605 |
| media | 168 | 0.8036 |
| navigation | 178 | 0.8764 |
| news | 168 | 0.8095 |
| search_qa | 170 | 0.9118 |
| smarthome | 170 | 0.8647 |
| system_control | 166 | 0.8012 |
| timers_alarms | 174 | 0.8678 |
| weather | 168 | 0.8333 |