# massive-templates (af-ZA) training report

- intents: **60**
- templates: **13547** → **13831** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8420 | 0.8131 |
| linear_svc | 0.8312 | 0.7992 |
| ovr_linear_svc | 0.8312 | 0.7992 |
| voting_soft | 0.8141 | 0.7722 |
| logreg_char | 0.8003 | 0.7421 |
| logreg | 0.7835 | 0.7307 |
| nb_multinomial | 0.5978 | 0.4288 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1696 | 0.6010 | 0.2645 | 0.1063 | 2974 |
| template | 0.0096 | 0.0236 | 0.0136 | 0.0027 | 2974 |