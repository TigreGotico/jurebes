# massive-templates (zh-TW) training report

- intents: **60**
- templates: **13547** → **13795** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.7441 | 0.7145 |
| logreg_char | 0.6328 | 0.5670 |
| voting_soft | 0.1701 | 0.1785 |
| linear_svc | 0.1698 | 0.1800 |
| ovr_linear_svc | 0.1698 | 0.1800 |
| nb_multinomial | 0.1379 | 0.0581 |
| logreg | 0.1375 | 0.0941 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2481 | 0.0517 | 0.0856 | 0.3517 | 2974 |
| template | 0.0003 | 0.0004 | 0.0003 | 0.0000 | 2974 |