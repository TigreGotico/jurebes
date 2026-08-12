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
| bm25_logreg | 0.8229 | 0.8281 |
| union_bm25_pos_logreg | 0.8229 | 0.8272 |
| label_guided_logreg | 0.8212 | 0.8257 |
| label_guided_linear_svc | 0.8200 | 0.8225 |
| logreg_char | 0.8147 | 0.8182 |
| union_skipgram_tfidf_logreg | 0.8129 | 0.8187 |
| voting_soft | 0.8112 | 0.8156 |
| linear_svc | 0.8106 | 0.8149 |
| ovr_linear_svc | 0.8106 | 0.8149 |
| logreg | 0.7918 | 0.7977 |
| nb_multinomial | 0.6565 | 0.6281 |
| autoencoder_logreg | 0.4424 | 0.3959 |
| denoising_autoencoder_logreg | 0.2159 | 0.1373 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8546 | 0.8266 | 0.8404 | 0.8129 | 1700 |
| template | 0.6667 | 0.4763 | 0.5556 | 0.6212 | 1700 |
| sklearn_iob | 0.8136 | 0.8024 | 0.8080 | 0.8388 | 1700 |
| knn | 0.5563 | 0.5116 | 0.5330 | 0.6294 | 1700 |
| hybrid | 0.6748 | 0.8792 | 0.7636 | 0.7206 | 1700 |
| crf | 0.8533 | 0.8128 | 0.8325 | 0.8512 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 166 | 0.8313 |
| communication | 170 | 0.8647 |
| media | 178 | 0.8202 |
| navigation | 170 | 0.8588 |
| news | 168 | 0.7679 |
| search_qa | 168 | 0.8750 |
| smarthome | 174 | 0.8218 |
| system_control | 168 | 0.8036 |
| timers_alarms | 166 | 0.8614 |
| weather | 172 | 0.8547 |