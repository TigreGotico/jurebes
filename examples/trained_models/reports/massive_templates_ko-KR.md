# massive-templates (ko-KR) training report

- intents: **60**
- templates: **13547** → **13766** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8215 | 0.7870 |
| logreg_char | 0.7919 | 0.7341 |
| linear_svc | 0.7848 | 0.7422 |
| ovr_linear_svc | 0.7848 | 0.7422 |
| voting_soft | 0.7818 | 0.7407 |
| logreg | 0.7461 | 0.6784 |
| nb_multinomial | 0.6126 | 0.4749 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2827 | 0.6056 | 0.3855 | 0.2424 | 2974 |
| template | 0.0068 | 0.0216 | 0.0104 | 0.0007 | 2974 |