# massive-templates (sw-KE) training report

- intents: **60**
- templates: **13547** → **13869** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8349 | 0.8051 |
| linear_svc | 0.8225 | 0.7956 |
| ovr_linear_svc | 0.8225 | 0.7956 |
| voting_soft | 0.8127 | 0.7755 |
| logreg_char | 0.8090 | 0.7661 |
| logreg | 0.7912 | 0.7466 |
| nb_multinomial | 0.5962 | 0.4521 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1048 | 0.5455 | 0.1759 | 0.0733 | 2974 |
| template | 0.0086 | 0.0196 | 0.0119 | 0.0020 | 2974 |