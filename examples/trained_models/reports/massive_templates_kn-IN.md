# massive-templates (kn-IN) training report

- intents: **60**
- templates: **13547** → **13762** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8309 | 0.8083 |
| logreg_char | 0.7902 | 0.7185 |
| linear_svc | 0.5709 | 0.5170 |
| ovr_linear_svc | 0.5709 | 0.5170 |
| voting_soft | 0.5689 | 0.5039 |
| logreg | 0.5602 | 0.4839 |
| nb_multinomial | 0.4452 | 0.2866 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1485 | 0.5121 | 0.2303 | 0.0955 | 2974 |
| template | 0.0098 | 0.0315 | 0.0150 | 0.0054 | 2974 |