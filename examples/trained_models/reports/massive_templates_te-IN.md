# massive-templates (te-IN) training report

- intents: **60**
- templates: **13547** → **13772** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8245 | 0.7972 |
| logreg_char | 0.7902 | 0.7270 |
| linear_svc | 0.4933 | 0.4320 |
| ovr_linear_svc | 0.4933 | 0.4320 |
| logreg | 0.4916 | 0.4214 |
| voting_soft | 0.4906 | 0.4219 |
| nb_multinomial | 0.3904 | 0.2450 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1741 | 0.5141 | 0.2601 | 0.1244 | 2974 |
| template | 0.0107 | 0.0340 | 0.0163 | 0.0044 | 2974 |