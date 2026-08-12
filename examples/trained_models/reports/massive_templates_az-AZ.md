# massive-templates (az-AZ) training report

- intents: **60**
- templates: **13547** → **13718** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8510 | 0.8362 |
| linear_svc | 0.8208 | 0.7951 |
| ovr_linear_svc | 0.8208 | 0.7951 |
| logreg_char | 0.8201 | 0.7521 |
| voting_soft | 0.8124 | 0.7808 |
| logreg | 0.7603 | 0.6831 |
| nb_multinomial | 0.6268 | 0.5011 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2877 | 0.6272 | 0.3944 | 0.2582 | 2974 |
| template | 0.0082 | 0.0260 | 0.0124 | 0.0020 | 2974 |