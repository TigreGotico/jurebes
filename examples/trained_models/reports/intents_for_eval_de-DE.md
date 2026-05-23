# intents-for-eval (de-DE) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8406 | 0.8432 |
| logreg_char | 0.8235 | 0.8243 |
| voting_soft | 0.8176 | 0.8217 |
| union_skipgram_tfidf_logreg | 0.8165 | 0.8216 |
| label_guided_logreg | 0.8159 | 0.8196 |
| bm25_logreg | 0.8129 | 0.8173 |
| linear_svc | 0.8118 | 0.8168 |
| ovr_linear_svc | 0.8118 | 0.8168 |
| union_bm25_pos_logreg | 0.8118 | 0.8161 |
| label_guided_linear_svc | 0.8059 | 0.8098 |
| logreg | 0.7906 | 0.7950 |
| nb_multinomial | 0.6559 | 0.6292 |
| autoencoder_logreg | 0.4341 | 0.3816 |
| denoising_autoencoder_logreg | 0.1506 | 0.0873 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8663 | 0.8134 | 0.8390 | 0.8141 | 1700 |
| template | 0.6529 | 0.4205 | 0.5115 | 0.5918 | 1700 |
| sklearn_iob | 0.8708 | 0.8289 | 0.8493 | 0.8618 | 1700 |
| knn | 0.5277 | 0.4755 | 0.5002 | 0.6153 | 1700 |
| hybrid | 0.7054 | 0.8831 | 0.7843 | 0.7518 | 1700 |
| crf | 0.9000 | 0.8358 | 0.8667 | 0.8694 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 172 | 0.8372 |
| communication | 172 | 0.8895 |
| media | 166 | 0.7892 |
| navigation | 170 | 0.8529 |
| news | 176 | 0.7898 |
| search_qa | 168 | 0.8988 |
| smarthome | 170 | 0.8529 |
| system_control | 166 | 0.8253 |
| timers_alarms | 170 | 0.8353 |
| weather | 170 | 0.8353 |