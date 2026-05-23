# intents-for-eval (pt-PT) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8394 | 0.8440 |
| union_bm25_pos_logreg | 0.8247 | 0.8280 |
| bm25_logreg | 0.8218 | 0.8247 |
| voting_soft | 0.8212 | 0.8242 |
| logreg_char | 0.8159 | 0.8176 |
| linear_svc | 0.8153 | 0.8193 |
| ovr_linear_svc | 0.8153 | 0.8193 |
| union_skipgram_tfidf_logreg | 0.8124 | 0.8177 |
| label_guided_logreg | 0.8088 | 0.8126 |
| label_guided_linear_svc | 0.8047 | 0.8090 |
| logreg | 0.7994 | 0.8034 |
| nb_multinomial | 0.6829 | 0.6671 |
| autoencoder_logreg | 0.4506 | 0.4042 |
| denoising_autoencoder_logreg | 0.1688 | 0.0945 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.7978 | 0.8204 | 0.8090 | 0.7947 | 1700 |
| template | 0.7033 | 0.4459 | 0.5457 | 0.6282 | 1700 |
| sklearn_iob | 0.7676 | 0.7775 | 0.7725 | 0.8159 | 1700 |
| knn | 0.5364 | 0.5069 | 0.5212 | 0.6247 | 1700 |
| hybrid | 0.6623 | 0.8746 | 0.7538 | 0.7329 | 1700 |
| crf | 0.8375 | 0.7973 | 0.8169 | 0.8382 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.7558 |
| communication | 164 | 0.9024 |
| media | 170 | 0.8176 |
| navigation | 170 | 0.8882 |
| news | 172 | 0.8488 |
| search_qa | 174 | 0.9080 |
| smarthome | 168 | 0.7976 |
| system_control | 166 | 0.7711 |
| timers_alarms | 168 | 0.8750 |
| weather | 176 | 0.8295 |