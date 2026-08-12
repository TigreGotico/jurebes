# massive-templates (id-ID) training report

- intents: **60**
- templates: **13547** → **13828** after slot expansion
- entities: **54**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8484 | 0.8264 |
| linear_svc | 0.8436 | 0.8208 |
| ovr_linear_svc | 0.8436 | 0.8208 |
| voting_soft | 0.8346 | 0.7774 |
| logreg_char | 0.8204 | 0.7617 |
| logreg | 0.8198 | 0.7609 |
| nb_multinomial | 0.6382 | 0.5061 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1443 | 0.5794 | 0.2311 | 0.1002 | 2974 |
| template | 0.0123 | 0.0390 | 0.0187 | 0.0034 | 2974 |