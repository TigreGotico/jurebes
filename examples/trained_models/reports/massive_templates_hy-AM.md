# massive-templates (hy-AM) training report

- intents: **60**
- templates: **13547** → **13755** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8393 | 0.7809 |
| linear_svc | 0.8117 | 0.7431 |
| ovr_linear_svc | 0.8117 | 0.7431 |
| logreg_char | 0.8003 | 0.7426 |
| voting_soft | 0.7999 | 0.7352 |
| logreg | 0.7532 | 0.6798 |
| nb_multinomial | 0.5935 | 0.4546 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2500 | 0.5499 | 0.3438 | 0.1859 | 2974 |
| template | 0.0080 | 0.0252 | 0.0122 | 0.0017 | 2974 |