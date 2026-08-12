# massive-templates (am-ET) training report

- intents: **60**
- templates: **13547** → **13769** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8107 | 0.7655 |
| linear_svc | 0.7885 | 0.7384 |
| ovr_linear_svc | 0.7885 | 0.7384 |
| voting_soft | 0.7761 | 0.7297 |
| logreg_char | 0.7697 | 0.6966 |
| logreg | 0.7394 | 0.6585 |
| nb_multinomial | 0.5602 | 0.3983 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2477 | 0.5984 | 0.3504 | 0.2182 | 2974 |
| template | 0.0028 | 0.0088 | 0.0042 | 0.0003 | 2974 |