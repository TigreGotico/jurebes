# massive-templates (it-IT) training report

- intents: **60**
- templates: **13547** → **13830** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8379 | 0.8020 |
| linear_svc | 0.8346 | 0.8009 |
| ovr_linear_svc | 0.8346 | 0.8009 |
| voting_soft | 0.8302 | 0.7878 |
| logreg_char | 0.8154 | 0.7533 |
| logreg | 0.8073 | 0.7542 |
| nb_multinomial | 0.6338 | 0.5104 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1555 | 0.5469 | 0.2421 | 0.1153 | 2974 |
| template | 0.0139 | 0.0440 | 0.0211 | 0.0057 | 2974 |