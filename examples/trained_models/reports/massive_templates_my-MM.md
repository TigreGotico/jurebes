# massive-templates (my-MM) training report

- intents: **60**
- templates: **13547** → **13799** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8339 | 0.7742 |
| logreg_char | 0.7996 | 0.7318 |
| logreg | 0.4758 | 0.3797 |
| voting_soft | 0.4724 | 0.3826 |
| linear_svc | 0.4670 | 0.3880 |
| ovr_linear_svc | 0.4670 | 0.3880 |
| nb_multinomial | 0.3796 | 0.2490 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.0997 | 0.6075 | 0.1713 | 0.0215 | 2974 |
| template | 0.0077 | 0.0225 | 0.0114 | 0.0017 | 2974 |