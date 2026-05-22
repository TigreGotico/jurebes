# massive-templates (sq-AL) training report

- intents: **60**
- templates: **13547** → **13870** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8460 | 0.8057 |
| linear_svc | 0.8231 | 0.7818 |
| ovr_linear_svc | 0.8231 | 0.7818 |
| logreg_char | 0.8181 | 0.7501 |
| voting_soft | 0.8110 | 0.7510 |
| logreg | 0.7781 | 0.7156 |
| nb_multinomial | 0.6083 | 0.4660 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1704 | 0.5371 | 0.2587 | 0.1301 | 2974 |
| template | 0.0110 | 0.0349 | 0.0167 | 0.0050 | 2974 |