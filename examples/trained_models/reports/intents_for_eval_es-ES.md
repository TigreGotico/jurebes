# intents-for-eval (es-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| union_bm25_pos_logreg | 0.8394 | 0.8414 |
| bm25_logreg | 0.8371 | 0.8400 |
| linear_svc_char | 0.8341 | 0.8386 |
| voting_soft | 0.8312 | 0.8344 |
| linear_svc | 0.8265 | 0.8303 |
| ovr_linear_svc | 0.8265 | 0.8303 |
| label_guided_logreg | 0.8229 | 0.8259 |
| label_guided_linear_svc | 0.8176 | 0.8202 |
| union_skipgram_tfidf_logreg | 0.8165 | 0.8186 |
| logreg_char | 0.8147 | 0.8159 |
| logreg | 0.7959 | 0.7991 |
| nb_multinomial | 0.6824 | 0.6601 |
| autoencoder_logreg | 0.4624 | 0.3995 |
| denoising_autoencoder_logreg | 0.2506 | 0.1693 |

**winning baseline (intent):** `union_bm25_pos_logreg`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8297 | 0.7986 | 0.8139 | 0.7906 | 1700 |
| template | 0.6429 | 0.4282 | 0.5140 | 0.6065 | 1700 |
| sklearn_iob | 0.7892 | 0.8046 | 0.7968 | 0.8276 | 1700 |
| knn | 0.5578 | 0.5208 | 0.5387 | 0.6559 | 1700 |
| hybrid | 0.6454 | 0.8598 | 0.7373 | 0.7100 | 1700 |
| crf | 0.8489 | 0.8020 | 0.8248 | 0.8418 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `union_bm25_pos_logreg`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.7791 |
| communication | 174 | 0.8966 |
| media | 174 | 0.7989 |
| navigation | 164 | 0.8598 |
| news | 166 | 0.7892 |
| search_qa | 170 | 0.9176 |
| smarthome | 172 | 0.8488 |
| system_control | 166 | 0.8012 |
| timers_alarms | 172 | 0.8895 |
| weather | 170 | 0.8118 |