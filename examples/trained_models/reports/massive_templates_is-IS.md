# massive-templates (is-IS) training report

- intents: **60**
- templates: **13547** → **13818** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8430 | 0.8170 |
| linear_svc | 0.8198 | 0.7801 |
| ovr_linear_svc | 0.8198 | 0.7801 |
| logreg_char | 0.8070 | 0.7569 |
| voting_soft | 0.8053 | 0.7540 |
| logreg | 0.7623 | 0.6940 |
| nb_multinomial | 0.5921 | 0.4378 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2765 | 0.6122 | 0.3810 | 0.2317 | 2974 |
| template | 0.0103 | 0.0327 | 0.0157 | 0.0024 | 2974 |