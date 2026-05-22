# massive-templates (vi-VN) training report

- intents: **60**
- templates: **13547** → **13828** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8433 | 0.8043 |
| linear_svc | 0.8319 | 0.7940 |
| ovr_linear_svc | 0.8319 | 0.7940 |
| voting_soft | 0.8191 | 0.7698 |
| logreg_char | 0.8137 | 0.7516 |
| logreg | 0.8087 | 0.7467 |
| nb_multinomial | 0.6510 | 0.5271 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1392 | 0.5539 | 0.2224 | 0.0760 | 2974 |
| template | 0.0078 | 0.0246 | 0.0118 | 0.0037 | 2974 |