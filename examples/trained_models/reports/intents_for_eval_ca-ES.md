# intents-for-eval (ca-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8488 | 0.8518 |
| union_bm25_pos_logreg | 0.8471 | 0.8486 |
| bm25_logreg | 0.8459 | 0.8480 |
| union_skipgram_tfidf_logreg | 0.8300 | 0.8330 |
| voting_soft | 0.8259 | 0.8290 |
| label_guided_logreg | 0.8253 | 0.8284 |
| label_guided_linear_svc | 0.8253 | 0.8279 |
| logreg_char | 0.8241 | 0.8250 |
| linear_svc | 0.8194 | 0.8237 |
| ovr_linear_svc | 0.8194 | 0.8237 |
| logreg | 0.8076 | 0.8097 |
| nb_multinomial | 0.6976 | 0.6770 |
| autoencoder_logreg | 0.4724 | 0.4157 |
| denoising_autoencoder_logreg | 0.1594 | 0.0827 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8157 | 0.8171 | 0.8164 | 0.7924 | 1700 |
| template | 0.6561 | 0.4378 | 0.5251 | 0.6071 | 1700 |
| sklearn_iob | 0.7131 | 0.7536 | 0.7328 | 0.7747 | 1700 |
| knn | 0.4830 | 0.4682 | 0.4755 | 0.6129 | 1700 |
| hybrid | 0.6450 | 0.8738 | 0.7422 | 0.6959 | 1700 |
| crf | 0.7915 | 0.7587 | 0.7748 | 0.8124 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 176 | 0.8352 |
| communication | 164 | 0.9146 |
| media | 172 | 0.7965 |
| navigation | 170 | 0.8765 |
| news | 180 | 0.7889 |
| search_qa | 170 | 0.8941 |
| smarthome | 162 | 0.9012 |
| system_control | 166 | 0.8012 |
| timers_alarms | 166 | 0.8795 |
| weather | 174 | 0.8103 |