# massive-templates (ru-RU) training report

- intents: **60**
- templates: **13547** → **13814** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8433 | 0.8068 |
| linear_svc | 0.8181 | 0.7850 |
| ovr_linear_svc | 0.8181 | 0.7850 |
| voting_soft | 0.8083 | 0.7623 |
| logreg_char | 0.7989 | 0.7388 |
| logreg | 0.7666 | 0.7008 |
| nb_multinomial | 0.6214 | 0.4617 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2831 | 0.5635 | 0.3769 | 0.2135 | 2974 |
| template | 0.0114 | 0.0355 | 0.0172 | 0.0030 | 2974 |