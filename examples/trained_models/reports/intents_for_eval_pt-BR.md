# intents-for-eval (pt-BR) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8441 | 0.8455 |
| union_bm25_pos_logreg | 0.8306 | 0.8315 |
| bm25_logreg | 0.8294 | 0.8306 |
| voting_soft | 0.8265 | 0.8284 |
| union_skipgram_tfidf_logreg | 0.8229 | 0.8233 |
| linear_svc | 0.8218 | 0.8248 |
| ovr_linear_svc | 0.8218 | 0.8248 |
| label_guided_linear_svc | 0.8147 | 0.8170 |
| logreg_char | 0.8112 | 0.8099 |
| label_guided_logreg | 0.8100 | 0.8121 |
| logreg | 0.8053 | 0.8058 |
| nb_multinomial | 0.6894 | 0.6694 |
| autoencoder_logreg | 0.5171 | 0.4777 |
| denoising_autoencoder_logreg | 0.2559 | 0.1783 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8605 | 0.8291 | 0.8445 | 0.8171 | 1700 |
| template | 0.6886 | 0.4362 | 0.5341 | 0.6153 | 1700 |
| sklearn_iob | 0.7722 | 0.7840 | 0.7781 | 0.8229 | 1700 |
| knn | 0.5438 | 0.4966 | 0.5191 | 0.6382 | 1700 |
| hybrid | 0.6867 | 0.8835 | 0.7728 | 0.7488 | 1700 |
| crf | 0.8350 | 0.7917 | 0.8127 | 0.8365 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 170 | 0.8294 |
| communication | 172 | 0.8837 |
| media | 174 | 0.8218 |
| navigation | 166 | 0.8494 |
| news | 166 | 0.8133 |
| search_qa | 178 | 0.9213 |
| smarthome | 166 | 0.8434 |
| system_control | 170 | 0.7647 |
| timers_alarms | 170 | 0.8824 |
| weather | 168 | 0.8274 |