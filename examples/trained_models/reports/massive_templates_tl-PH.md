# massive-templates (tl-PH) training report

- intents: **60**
- templates: **13547** → **13828** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8379 | 0.8152 |
| linear_svc | 0.8322 | 0.8066 |
| ovr_linear_svc | 0.8322 | 0.8066 |
| voting_soft | 0.8137 | 0.7549 |
| logreg_char | 0.8046 | 0.7501 |
| logreg | 0.7922 | 0.7200 |
| nb_multinomial | 0.5847 | 0.4242 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.0653 | 0.4616 | 0.1144 | 0.0514 | 2974 |
| template | 0.0145 | 0.0458 | 0.0220 | 0.0087 | 2974 |