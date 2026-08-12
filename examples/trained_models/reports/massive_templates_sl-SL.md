# massive-templates (sl-SL) training report

- intents: **60**
- templates: **13547** → **13811** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8366 | 0.8011 |
| logreg_char | 0.8124 | 0.7549 |
| linear_svc | 0.8104 | 0.7748 |
| ovr_linear_svc | 0.8104 | 0.7748 |
| voting_soft | 0.8053 | 0.7498 |
| logreg | 0.7751 | 0.7122 |
| nb_multinomial | 0.6106 | 0.4706 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2463 | 0.5391 | 0.3381 | 0.1893 | 2974 |
| template | 0.0094 | 0.0297 | 0.0143 | 0.0013 | 2974 |