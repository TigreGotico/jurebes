# intents-for-eval (eu-ES) training report

- intents: **50**
- templates: **1000** → **2188** after slot expansion
- entities: **24**
- test utterances: **1700** in-domain (+50 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8376 | 0.8382 |
| logreg_char | 0.8212 | 0.8217 |
| union_bm25_pos_logreg | 0.7853 | 0.7891 |
| bm25_logreg | 0.7847 | 0.7914 |
| union_skipgram_tfidf_logreg | 0.7794 | 0.7846 |
| voting_soft | 0.7753 | 0.7806 |
| linear_svc | 0.7747 | 0.7814 |
| ovr_linear_svc | 0.7747 | 0.7814 |
| label_guided_logreg | 0.7747 | 0.7777 |
| label_guided_linear_svc | 0.7700 | 0.7749 |
| logreg | 0.7518 | 0.7557 |
| nb_multinomial | 0.6541 | 0.6358 |
| autoencoder_logreg | 0.4524 | 0.4062 |
| denoising_autoencoder_logreg | 0.1888 | 0.1204 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on the same templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.8662 | 0.6885 | 0.7672 | 0.7494 | 1700 |
| template | 0.5798 | 0.3370 | 0.4263 | 0.5518 | 1700 |
| sklearn_iob | 0.8075 | 0.6570 | 0.7245 | 0.7624 | 1700 |
| knn | 0.4443 | 0.4281 | 0.4361 | 0.5794 | 1700 |
| hybrid | 0.6552 | 0.7311 | 0.6911 | 0.6653 | 1700 |
| crf | 0.8696 | 0.6698 | 0.7567 | 0.7776 | 1700 |

## Per-domain intent accuracy

Using the winning baseline `linear_svc_char`.
| domain | n | accuracy |
|---|---|---|
| calendar | 174 | 0.7471 |
| communication | 168 | 0.8810 |
| media | 172 | 0.8372 |
| navigation | 170 | 0.8706 |
| news | 170 | 0.8000 |
| search_qa | 180 | 0.8889 |
| smarthome | 168 | 0.8750 |
| system_control | 162 | 0.8272 |
| timers_alarms | 168 | 0.8214 |
| weather | 168 | 0.8274 |