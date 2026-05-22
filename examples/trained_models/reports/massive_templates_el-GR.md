# massive-templates (el-GR) training report

- intents: **60**
- templates: **13547** → **13834** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8366 | 0.8072 |
| linear_svc | 0.8252 | 0.7963 |
| ovr_linear_svc | 0.8252 | 0.7963 |
| voting_soft | 0.8117 | 0.7650 |
| logreg_char | 0.8063 | 0.7518 |
| logreg | 0.7824 | 0.7360 |
| nb_multinomial | 0.5834 | 0.4417 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2540 | 0.5777 | 0.3528 | 0.1994 | 2974 |
| template | 0.0088 | 0.0278 | 0.0134 | 0.0044 | 2974 |