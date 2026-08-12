# massive-templates (ca-ES) training report

- intents: **60**
- templates: **13547** → **13815** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8413 | 0.8016 |
| linear_svc | 0.8332 | 0.7899 |
| ovr_linear_svc | 0.8332 | 0.7899 |
| voting_soft | 0.8225 | 0.7721 |
| logreg_char | 0.8110 | 0.7666 |
| logreg | 0.7956 | 0.7412 |
| nb_multinomial | 0.6032 | 0.4602 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1921 | 0.5698 | 0.2874 | 0.1291 | 2974 |
| template | 0.0108 | 0.0341 | 0.0164 | 0.0047 | 2974 |