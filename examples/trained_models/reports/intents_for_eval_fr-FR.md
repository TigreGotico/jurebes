# intents-for-eval (fr-FR) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8394 | 0.8430 |
| union_bm25_pos_logreg | 0.8253 | 0.8272 |
| bm25_logreg | 0.8235 | 0.8256 |
| label_guided_linear_svc | 0.8218 | 0.8235 |
| voting_soft | 0.8206 | 0.8226 |
| label_guided_logreg | 0.8200 | 0.8225 |
| linear_svc | 0.8159 | 0.8190 |
| ovr_linear_svc | 0.8159 | 0.8190 |
| union_skipgram_tfidf_logreg | 0.8141 | 0.8158 |
| logreg_char | 0.8000 | 0.7994 |
| logreg | 0.7859 | 0.7862 |
| nb_multinomial | 0.6647 | 0.6386 |
| autoencoder_logreg | 0.4524 | 0.4044 |
| denoising_autoencoder_logreg | 0.1835 | 0.1063 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8316 | 0.8295 | 0.8305 | 0.7976 | 1700 |
| template | 0.6629 | 0.4499 | 0.5360 | 0.6135 | 1700 |
| sklearn_iob | 0.6669 | 0.7018 | 0.6839 | 0.7335 | 1700 |
| knn | 0.4777 | 0.4319 | 0.4536 | 0.5971 | 1700 |
| hybrid | 0.6523 | 0.8809 | 0.7495 | 0.6994 | 1700 |
| crf | 0.7576 | 0.7258 | 0.7414 | 0.7894 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.8133 |
| communication | 180 | 0.8889 |
| media | 168 | 0.8036 |
| navigation | 170 | 0.8294 |
| news | 172 | 0.7791 |
| search_qa | 168 | 0.9107 |
| smarthome | 166 | 0.8795 |
| system_control | 170 | 0.7706 |
| timers_alarms | 168 | 0.9167 |
| weather | 172 | 0.8023 |