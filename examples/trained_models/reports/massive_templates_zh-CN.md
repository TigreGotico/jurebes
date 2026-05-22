# massive-templates (zh-CN) training report

- intents: **60**
- templates: **13547** → **13804** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.7586 | 0.7209 |
| logreg_char | 0.6449 | 0.5636 |
| voting_soft | 0.1106 | 0.1228 |
| linear_svc | 0.1093 | 0.1224 |
| ovr_linear_svc | 0.1093 | 0.1224 |
| nb_multinomial | 0.0915 | 0.0178 |
| logreg | 0.0831 | 0.0386 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.3276 | 0.0070 | 0.0136 | 0.3312 | 2974 |
| template | 0.0007 | 0.0007 | 0.0007 | 0.0003 | 2974 |