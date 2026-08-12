# massive-templates (km-KH) training report

- intents: **60**
- templates: **13547** → **13728** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.7552 | 0.6998 |
| logreg_char | 0.7192 | 0.6395 |
| linear_svc | 0.5118 | 0.4715 |
| ovr_linear_svc | 0.5118 | 0.4715 |
| voting_soft | 0.5061 | 0.4651 |
| logreg | 0.5040 | 0.4472 |
| nb_multinomial | 0.3648 | 0.2501 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1607 | 0.6568 | 0.2583 | 0.0861 | 2974 |
| template | 0.0139 | 0.0336 | 0.0196 | 0.0064 | 2974 |