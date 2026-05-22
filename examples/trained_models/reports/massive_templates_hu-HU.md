# massive-templates (hu-HU) training report

- intents: **60**
- templates: **13547** → **13741** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8386 | 0.7908 |
| logreg_char | 0.8080 | 0.7457 |
| linear_svc | 0.8030 | 0.7539 |
| ovr_linear_svc | 0.8030 | 0.7539 |
| voting_soft | 0.7885 | 0.7392 |
| logreg | 0.7492 | 0.6863 |
| nb_multinomial | 0.5827 | 0.4164 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.3598 | 0.5638 | 0.4393 | 0.3295 | 2974 |
| template | 0.0076 | 0.0243 | 0.0116 | 0.0020 | 2974 |