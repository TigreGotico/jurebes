# massive-templates (sv-SE) training report

- intents: **60**
- templates: **13547** → **13819** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8453 | 0.8211 |
| linear_svc | 0.8157 | 0.7796 |
| ovr_linear_svc | 0.8157 | 0.7796 |
| logreg_char | 0.8130 | 0.7617 |
| voting_soft | 0.8073 | 0.7674 |
| logreg | 0.7740 | 0.7123 |
| nb_multinomial | 0.5841 | 0.4235 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2754 | 0.6293 | 0.3831 | 0.1947 | 2974 |
| template | 0.0112 | 0.0352 | 0.0170 | 0.0044 | 2974 |