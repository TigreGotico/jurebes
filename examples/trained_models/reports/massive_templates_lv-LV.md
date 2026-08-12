# massive-templates (lv-LV) training report

- intents: **60**
- templates: **13547** → **13794** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8460 | 0.8382 |
| linear_svc | 0.8167 | 0.7991 |
| ovr_linear_svc | 0.8167 | 0.7991 |
| logreg_char | 0.8083 | 0.7603 |
| voting_soft | 0.8053 | 0.7858 |
| logreg | 0.7653 | 0.7073 |
| nb_multinomial | 0.6120 | 0.4733 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.3204 | 0.6089 | 0.4199 | 0.2919 | 2974 |
| template | 0.0100 | 0.0314 | 0.0152 | 0.0020 | 2974 |