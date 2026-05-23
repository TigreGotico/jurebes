# intents-for-eval (en-US) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| union_bm25_pos_logreg | 0.8318 | 0.8344 |
| linear_svc_char | 0.8294 | 0.8328 |
| bm25_logreg | 0.8294 | 0.8328 |
| voting_soft | 0.8288 | 0.8330 |
| union_skipgram_tfidf_logreg | 0.8288 | 0.8308 |
| linear_svc | 0.8247 | 0.8292 |
| ovr_linear_svc | 0.8247 | 0.8292 |
| label_guided_linear_svc | 0.8206 | 0.8232 |
| logreg_char | 0.8194 | 0.8214 |
| label_guided_logreg | 0.8194 | 0.8227 |
| logreg | 0.8018 | 0.8022 |
| nb_multinomial | 0.6835 | 0.6665 |
| autoencoder_logreg | 0.4865 | 0.4380 |
| denoising_autoencoder_logreg | 0.2706 | 0.1974 |

**winning baseline (intent):** `union_bm25_pos_logreg`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8447 | 0.8622 | 0.8534 | 0.8176 | 1700 |
| template | 0.6484 | 0.4479 | 0.5298 | 0.5959 | 1700 |
| sklearn_iob | 0.8294 | 0.8372 | 0.8333 | 0.8547 | 1700 |
| knn | 0.5309 | 0.5039 | 0.5170 | 0.6259 | 1700 |
| hybrid | 0.6707 | 0.9156 | 0.7742 | 0.7229 | 1700 |
| crf | 0.9047 | 0.8501 | 0.8766 | 0.8841 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `union_bm25_pos_logreg`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.7558 |
| communication | 168 | 0.8690 |
| media | 168 | 0.8095 |
| navigation | 172 | 0.8140 |
| news | 168 | 0.8095 |
| search_qa | 170 | 0.9176 |
| smarthome | 168 | 0.8036 |
| system_control | 174 | 0.8563 |
| timers_alarms | 170 | 0.8706 |
| weather | 170 | 0.8118 |