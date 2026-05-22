# massive-templates (tr-TR) training report

- intents: **60**
- templates: **13547** → **13755** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8359 | 0.8221 |
| linear_svc | 0.8077 | 0.7747 |
| ovr_linear_svc | 0.8077 | 0.7747 |
| logreg_char | 0.8003 | 0.7425 |
| voting_soft | 0.7996 | 0.7529 |
| logreg | 0.7586 | 0.7045 |
| nb_multinomial | 0.5965 | 0.4579 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2134 | 0.5618 | 0.3093 | 0.1829 | 2974 |
| template | 0.0055 | 0.0175 | 0.0084 | 0.0017 | 2974 |