# massive-templates (bn-BD) training report

- intents: **60**
- templates: **13547** → **13783** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8248 | 0.7804 |
| logreg_char | 0.7888 | 0.7220 |
| linear_svc | 0.5750 | 0.5060 |
| ovr_linear_svc | 0.5750 | 0.5060 |
| logreg | 0.5676 | 0.4959 |
| voting_soft | 0.5625 | 0.4973 |
| nb_multinomial | 0.4398 | 0.2976 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1195 | 0.5658 | 0.1974 | 0.0444 | 2974 |
| template | 0.0040 | 0.0128 | 0.0061 | 0.0010 | 2974 |