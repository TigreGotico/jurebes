# massive-templates (ur-PK) training report

- intents: **60**
- templates: **13547** → **13808** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8336 | 0.7884 |
| linear_svc | 0.8171 | 0.7718 |
| ovr_linear_svc | 0.8171 | 0.7718 |
| voting_soft | 0.8114 | 0.7477 |
| logreg_char | 0.7946 | 0.7289 |
| logreg | 0.7861 | 0.7229 |
| nb_multinomial | 0.5894 | 0.4320 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.0917 | 0.5388 | 0.1568 | 0.0336 | 2974 |
| template | 0.0075 | 0.0239 | 0.0114 | 0.0024 | 2974 |