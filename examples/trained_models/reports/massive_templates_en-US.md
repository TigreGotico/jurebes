# massive-templates (en-US) training report

- intents: **60**
- templates: **13547** → **13821** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8467 | 0.8305 |
| linear_svc | 0.8433 | 0.8103 |
| ovr_linear_svc | 0.8433 | 0.8103 |
| voting_soft | 0.8289 | 0.7843 |
| logreg_char | 0.8147 | 0.7526 |
| logreg | 0.8067 | 0.7537 |
| nb_multinomial | 0.6234 | 0.4739 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2422 | 0.6639 | 0.3549 | 0.1453 | 2974 |
| template | 0.0098 | 0.0311 | 0.0149 | 0.0044 | 2974 |