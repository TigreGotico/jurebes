# massive-templates (ta-IN) training report

- intents: **60**
- templates: **13547** → **13723** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8282 | 0.7986 |
| logreg_char | 0.7949 | 0.7279 |
| linear_svc | 0.6288 | 0.5917 |
| ovr_linear_svc | 0.6288 | 0.5917 |
| voting_soft | 0.6194 | 0.5774 |
| logreg | 0.6120 | 0.5678 |
| nb_multinomial | 0.4808 | 0.3589 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1359 | 0.5445 | 0.2175 | 0.0720 | 2974 |
| template | 0.0082 | 0.0260 | 0.0124 | 0.0027 | 2974 |