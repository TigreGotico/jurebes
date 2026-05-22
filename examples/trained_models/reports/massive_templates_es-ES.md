# massive-templates (es-ES) training report

- intents: **60**
- templates: **13547** → **13833** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8362 | 0.8086 |
| linear_svc | 0.8262 | 0.7884 |
| ovr_linear_svc | 0.8262 | 0.7884 |
| voting_soft | 0.8124 | 0.7592 |
| logreg_char | 0.7972 | 0.7367 |
| logreg | 0.7851 | 0.7212 |
| nb_multinomial | 0.5736 | 0.4179 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.0964 | 0.4661 | 0.1598 | 0.0501 | 2974 |
| template | 0.0094 | 0.0302 | 0.0143 | 0.0047 | 2974 |