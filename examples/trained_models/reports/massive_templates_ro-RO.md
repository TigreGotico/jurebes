# massive-templates (ro-RO) training report

- intents: **60**
- templates: **13547** → **13843** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8460 | 0.7980 |
| linear_svc | 0.8272 | 0.7713 |
| ovr_linear_svc | 0.8272 | 0.7713 |
| voting_soft | 0.8114 | 0.7430 |
| logreg_char | 0.8090 | 0.7428 |
| logreg | 0.7804 | 0.7019 |
| nb_multinomial | 0.5905 | 0.4455 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2340 | 0.5814 | 0.3337 | 0.1705 | 2974 |
| template | 0.0092 | 0.0290 | 0.0139 | 0.0040 | 2974 |