# massive-templates (pt-PT) training report

- intents: **60**
- templates: **13547** → **13828** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8517 | 0.8258 |
| linear_svc | 0.8403 | 0.8001 |
| ovr_linear_svc | 0.8403 | 0.8001 |
| voting_soft | 0.8272 | 0.7799 |
| logreg_char | 0.8124 | 0.7573 |
| logreg | 0.7959 | 0.7265 |
| nb_multinomial | 0.6113 | 0.4772 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1618 | 0.5502 | 0.2501 | 0.1066 | 2974 |
| template | 0.0078 | 0.0342 | 0.0128 | 0.0020 | 2974 |