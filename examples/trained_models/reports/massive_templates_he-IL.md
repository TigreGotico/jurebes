# massive-templates (he-IL) training report

- intents: **60**
- templates: **13547** → **13681** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8460 | 0.8086 |
| linear_svc | 0.8241 | 0.7729 |
| ovr_linear_svc | 0.8241 | 0.7729 |
| voting_soft | 0.8184 | 0.7612 |
| logreg_char | 0.8063 | 0.7413 |
| logreg | 0.7727 | 0.7017 |
| nb_multinomial | 0.6059 | 0.4565 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2652 | 0.5979 | 0.3674 | 0.2196 | 2974 |
| template | 0.0112 | 0.0355 | 0.0170 | 0.0017 | 2974 |