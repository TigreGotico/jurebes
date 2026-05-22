# intents-for-eval (pt-BR) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8453 | 0.8468 |
| voting_soft | 0.8271 | 0.8288 |
| linear_svc | 0.8224 | 0.8253 |
| ovr_linear_svc | 0.8224 | 0.8253 |
| label_guided_linear_svc | 0.8141 | 0.8160 |
| logreg_char | 0.8135 | 0.8121 |
| label_guided_logreg | 0.8082 | 0.8106 |
| logreg | 0.8059 | 0.8064 |
| nb_multinomial | 0.6906 | 0.6704 |
| autoencoder_logreg | 0.4953 | 0.4504 |
| denoising_autoencoder_logreg | 0.2600 | 0.1796 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8605 | 0.8291 | 0.8445 | 0.8171 | 1700 |
| template | 0.6797 | 0.4422 | 0.5358 | 0.6165 | 1700 |
| sklearn_iob | 0.7800 | 0.7900 | 0.7850 | 0.8288 | 1700 |
| knn | 0.5449 | 0.5009 | 0.5219 | 0.6406 | 1700 |
| hybrid | 0.6878 | 0.8861 | 0.7744 | 0.7500 | 1700 |
| crf | 0.8344 | 0.7925 | 0.8129 | 0.8371 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 170 | 0.8353 |
| communication | 172 | 0.8779 |
| media | 174 | 0.8276 |
| navigation | 166 | 0.8494 |
| news | 166 | 0.8133 |
| search_qa | 178 | 0.9213 |
| smarthome | 166 | 0.8434 |
| system_control | 170 | 0.7647 |
| timers_alarms | 170 | 0.8882 |
| weather | 168 | 0.8274 |