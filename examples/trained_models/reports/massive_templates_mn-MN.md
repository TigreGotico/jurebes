# massive-templates (mn-MN) training report

- intents: **60**
- templates: **13547** → **13691** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8393 | 0.8129 |
| linear_svc | 0.8218 | 0.7834 |
| ovr_linear_svc | 0.8218 | 0.7834 |
| voting_soft | 0.8124 | 0.7664 |
| logreg_char | 0.8050 | 0.7494 |
| logreg | 0.7737 | 0.7041 |
| nb_multinomial | 0.6113 | 0.4698 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2102 | 0.5749 | 0.3078 | 0.1678 | 2974 |
| template | 0.0042 | 0.0135 | 0.0064 | 0.0010 | 2974 |