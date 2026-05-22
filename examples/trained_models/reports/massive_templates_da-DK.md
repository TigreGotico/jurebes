# massive-templates (da-DK) training report

- intents: **60**
- templates: **13547** → **13820** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8453 | 0.8128 |
| linear_svc | 0.8379 | 0.8049 |
| ovr_linear_svc | 0.8379 | 0.8049 |
| voting_soft | 0.8184 | 0.7654 |
| logreg_char | 0.8100 | 0.7464 |
| logreg | 0.7787 | 0.7069 |
| nb_multinomial | 0.5864 | 0.4152 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2257 | 0.6408 | 0.3339 | 0.1621 | 2974 |
| template | 0.0130 | 0.0296 | 0.0180 | 0.0044 | 2974 |