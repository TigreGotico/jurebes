# intents-for-eval (da-DK) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8535 | 0.8569 |
| logreg_char | 0.8294 | 0.8323 |
| union_bm25_pos_logreg | 0.8194 | 0.8240 |
| bm25_logreg | 0.8171 | 0.8226 |
| voting_soft | 0.8118 | 0.8166 |
| linear_svc | 0.8112 | 0.8166 |
| ovr_linear_svc | 0.8112 | 0.8166 |
| union_skipgram_tfidf_logreg | 0.8100 | 0.8148 |
| label_guided_logreg | 0.8006 | 0.8059 |
| label_guided_linear_svc | 0.7971 | 0.8029 |
| logreg | 0.7800 | 0.7848 |
| nb_multinomial | 0.6594 | 0.6390 |
| autoencoder_logreg | 0.4288 | 0.3921 |
| denoising_autoencoder_logreg | 0.1394 | 0.0736 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.7805 | 0.7142 | 0.7459 | 0.7418 | 1700 |
| template | 0.6848 | 0.4755 | 0.5613 | 0.6147 | 1700 |
| sklearn_iob | 0.7871 | 0.7142 | 0.7489 | 0.7876 | 1700 |
| knn | 0.5962 | 0.5665 | 0.5810 | 0.6576 | 1700 |
| hybrid | 0.6201 | 0.8000 | 0.6987 | 0.6588 | 1700 |
| crf | 0.8644 | 0.7494 | 0.8028 | 0.8159 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 174 | 0.8046 |
| communication | 164 | 0.9085 |
| media | 168 | 0.8512 |
| navigation | 172 | 0.8663 |
| news | 174 | 0.7989 |
| search_qa | 168 | 0.8988 |
| smarthome | 168 | 0.8452 |
| system_control | 172 | 0.8314 |
| timers_alarms | 166 | 0.9036 |
| weather | 174 | 0.8333 |