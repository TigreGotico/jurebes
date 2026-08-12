# massive-templates (ml-IN) training report

- intents: **60**
- templates: **13547** → **13691** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8376 | 0.7923 |
| logreg_char | 0.7959 | 0.7362 |
| linear_svc | 0.6584 | 0.5911 |
| ovr_linear_svc | 0.6584 | 0.5911 |
| voting_soft | 0.6493 | 0.5932 |
| logreg | 0.6426 | 0.5845 |
| nb_multinomial | 0.4943 | 0.3595 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2133 | 0.5835 | 0.3124 | 0.1627 | 2974 |
| template | 0.0073 | 0.0230 | 0.0111 | 0.0030 | 2974 |