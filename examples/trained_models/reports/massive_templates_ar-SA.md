# massive-templates (ar-SA) training report

- intents: **60**
- templates: **13547** → **13781** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.7952 | 0.7485 |
| linear_svc | 0.7781 | 0.7277 |
| ovr_linear_svc | 0.7781 | 0.7277 |
| voting_soft | 0.7734 | 0.7187 |
| logreg_char | 0.7603 | 0.6881 |
| logreg | 0.7347 | 0.6627 |
| nb_multinomial | 0.5713 | 0.4012 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2412 | 0.6057 | 0.3451 | 0.2031 | 2974 |
| template | 0.0105 | 0.0495 | 0.0173 | 0.0000 | 2974 |