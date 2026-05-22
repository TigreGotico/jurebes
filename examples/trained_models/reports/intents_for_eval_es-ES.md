# intents-for-eval (es-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8335 | 0.8379 |
| voting_soft | 0.8312 | 0.8340 |
| linear_svc | 0.8271 | 0.8307 |
| ovr_linear_svc | 0.8271 | 0.8307 |
| label_guided_logreg | 0.8235 | 0.8266 |
| label_guided_linear_svc | 0.8171 | 0.8194 |
| logreg_char | 0.8141 | 0.8152 |
| logreg | 0.7959 | 0.7992 |
| nb_multinomial | 0.6824 | 0.6602 |
| autoencoder_logreg | 0.4541 | 0.3982 |
| denoising_autoencoder_logreg | 0.2476 | 0.1673 |

**winning baseline (intent):** `linear_svc_char`

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

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.8023 |
| communication | 174 | 0.8678 |
| media | 174 | 0.7586 |
| navigation | 164 | 0.8841 |
| news | 166 | 0.8072 |
| search_qa | 170 | 0.8882 |
| smarthome | 172 | 0.8256 |
| system_control | 166 | 0.8012 |
| timers_alarms | 172 | 0.8721 |
| weather | 170 | 0.8294 |