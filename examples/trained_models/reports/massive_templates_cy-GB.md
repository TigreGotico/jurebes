# massive-templates (cy-GB) training report

- intents: **60**
- templates: **13547** → **13835** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8480 | 0.8097 |
| linear_svc | 0.8356 | 0.7883 |
| ovr_linear_svc | 0.8356 | 0.7883 |
| voting_soft | 0.8255 | 0.7742 |
| logreg_char | 0.8137 | 0.7588 |
| logreg | 0.7996 | 0.7469 |
| nb_multinomial | 0.6170 | 0.4725 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2029 | 0.6037 | 0.3038 | 0.1348 | 2974 |
| template | 0.0120 | 0.0268 | 0.0166 | 0.0047 | 2974 |