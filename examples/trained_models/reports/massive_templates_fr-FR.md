# massive-templates (fr-FR) training report

- intents: **60**
- templates: **13547** → **13831** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8447 | 0.8023 |
| linear_svc | 0.8245 | 0.7795 |
| ovr_linear_svc | 0.8245 | 0.7795 |
| voting_soft | 0.8120 | 0.7642 |
| logreg_char | 0.8067 | 0.7457 |
| logreg | 0.7929 | 0.7337 |
| nb_multinomial | 0.5770 | 0.4313 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1353 | 0.5246 | 0.2151 | 0.0841 | 2974 |
| template | 0.0362 | 0.0846 | 0.0507 | 0.0269 | 2974 |