# massive-templates (ka-GE) training report

- intents: **60**
- templates: **13547** → **13697** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.7962 | 0.7759 |
| logreg_char | 0.7603 | 0.7084 |
| linear_svc | 0.7579 | 0.7351 |
| ovr_linear_svc | 0.7579 | 0.7351 |
| voting_soft | 0.7529 | 0.7179 |
| logreg | 0.7165 | 0.6624 |
| nb_multinomial | 0.5841 | 0.4663 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2846 | 0.5796 | 0.3817 | 0.2596 | 2974 |
| template | 0.0058 | 0.0172 | 0.0086 | 0.0024 | 2974 |