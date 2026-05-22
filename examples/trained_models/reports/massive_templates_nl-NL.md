# massive-templates (nl-NL) training report

- intents: **60**
- templates: **13547** → **13828** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8514 | 0.8243 |
| linear_svc | 0.8339 | 0.8022 |
| ovr_linear_svc | 0.8339 | 0.8022 |
| voting_soft | 0.8201 | 0.7812 |
| logreg_char | 0.8104 | 0.7573 |
| logreg | 0.7905 | 0.7338 |
| nb_multinomial | 0.5797 | 0.4036 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1633 | 0.5728 | 0.2542 | 0.0763 | 2974 |
| template | 0.0127 | 0.0283 | 0.0175 | 0.0047 | 2974 |