# massive-templates (hi-IN) training report

- intents: **60**
- templates: **13547** → **13814** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8349 | 0.7906 |
| logreg_char | 0.7922 | 0.7286 |
| linear_svc | 0.5908 | 0.5205 |
| ovr_linear_svc | 0.5908 | 0.5205 |
| voting_soft | 0.5901 | 0.5164 |
| logreg | 0.5837 | 0.4977 |
| nb_multinomial | 0.4469 | 0.2965 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1138 | 0.5427 | 0.1881 | 0.0545 | 2974 |
| template | 0.0075 | 0.0238 | 0.0114 | 0.0034 | 2974 |