# intents-for-eval (it-IT) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8388 | 0.8424 |
| union_bm25_pos_logreg | 0.8329 | 0.8362 |
| bm25_logreg | 0.8294 | 0.8335 |
| label_guided_logreg | 0.8206 | 0.8243 |
| label_guided_linear_svc | 0.8188 | 0.8234 |
| logreg_char | 0.8159 | 0.8170 |
| voting_soft | 0.8153 | 0.8194 |
| linear_svc | 0.8129 | 0.8174 |
| ovr_linear_svc | 0.8129 | 0.8174 |
| union_skipgram_tfidf_logreg | 0.8124 | 0.8173 |
| logreg | 0.7929 | 0.7967 |
| nb_multinomial | 0.7000 | 0.6823 |
| autoencoder_logreg | 0.4453 | 0.3887 |
| denoising_autoencoder_logreg | 0.2241 | 0.1384 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8306 | 0.8207 | 0.8256 | 0.7935 | 1700 |
| template | 0.6931 | 0.4552 | 0.5495 | 0.6112 | 1700 |
| sklearn_iob | 0.7546 | 0.7669 | 0.7607 | 0.8018 | 1700 |
| knn | 0.5410 | 0.5184 | 0.5294 | 0.6359 | 1700 |
| hybrid | 0.6658 | 0.8659 | 0.7528 | 0.7224 | 1700 |
| crf | 0.8071 | 0.7720 | 0.7892 | 0.8141 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 168 | 0.8155 |
| communication | 172 | 0.8837 |
| media | 176 | 0.7614 |
| navigation | 168 | 0.8690 |
| news | 176 | 0.7841 |
| search_qa | 166 | 0.9096 |
| smarthome | 168 | 0.8750 |
| system_control | 172 | 0.7907 |
| timers_alarms | 170 | 0.9059 |
| weather | 164 | 0.7988 |