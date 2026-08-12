# massive-templates (fa-IR) training report

- intents: **60**
- templates: **13547** → **13817** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8480 | 0.8112 |
| linear_svc | 0.8473 | 0.8105 |
| ovr_linear_svc | 0.8473 | 0.8105 |
| voting_soft | 0.8366 | 0.7855 |
| logreg_char | 0.8161 | 0.7456 |
| logreg | 0.8114 | 0.7456 |
| nb_multinomial | 0.6298 | 0.4789 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1466 | 0.6104 | 0.2364 | 0.0716 | 2974 |
| template | 0.0223 | 0.0692 | 0.0337 | 0.0050 | 2974 |