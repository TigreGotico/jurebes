# massive-templates (fi-FI) training report

- intents: **60**
- templates: **13547** → **13752** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8450 | 0.8138 |
| logreg_char | 0.8124 | 0.7485 |
| linear_svc | 0.7993 | 0.7510 |
| ovr_linear_svc | 0.7993 | 0.7510 |
| voting_soft | 0.7898 | 0.7363 |
| logreg | 0.7418 | 0.6801 |
| nb_multinomial | 0.6002 | 0.4342 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.3642 | 0.6111 | 0.4563 | 0.2858 | 2974 |
| template | 0.0033 | 0.0099 | 0.0050 | 0.0000 | 2974 |