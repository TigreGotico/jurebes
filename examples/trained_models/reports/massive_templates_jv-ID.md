# massive-templates (jv-ID) training report

- intents: **60**
- templates: **13547** → **13830** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8393 | 0.7966 |
| linear_svc | 0.8248 | 0.7829 |
| ovr_linear_svc | 0.8248 | 0.7829 |
| voting_soft | 0.8154 | 0.7637 |
| logreg_char | 0.8077 | 0.7401 |
| logreg | 0.7861 | 0.7126 |
| nb_multinomial | 0.6214 | 0.4842 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1541 | 0.5663 | 0.2423 | 0.1147 | 2974 |
| template | 0.0139 | 0.0436 | 0.0211 | 0.0027 | 2974 |