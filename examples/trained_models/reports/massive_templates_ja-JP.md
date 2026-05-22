# massive-templates (ja-JP) training report

- intents: **60**
- templates: **13547** → **13813** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8073 | 0.7787 |
| logreg_char | 0.7367 | 0.6704 |
| linear_svc | 0.1483 | 0.1803 |
| ovr_linear_svc | 0.1483 | 0.1803 |
| voting_soft | 0.1469 | 0.1751 |
| nb_multinomial | 0.1187 | 0.0457 |
| logreg | 0.1184 | 0.0838 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.4130 | 0.0349 | 0.0644 | 0.3393 | 2974 |
| template | 0.0015 | 0.0018 | 0.0016 | 0.0003 | 2974 |