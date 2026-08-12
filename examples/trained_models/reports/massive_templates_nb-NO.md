# massive-templates (nb-NO) training report

- intents: **60**
- templates: **13547** → **13819** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8410 | 0.8021 |
| linear_svc | 0.8211 | 0.7737 |
| ovr_linear_svc | 0.8211 | 0.7737 |
| voting_soft | 0.8087 | 0.7534 |
| logreg_char | 0.8060 | 0.7510 |
| logreg | 0.7616 | 0.6947 |
| nb_multinomial | 0.5760 | 0.4135 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2707 | 0.6281 | 0.3783 | 0.2061 | 2974 |
| template | 0.0047 | 0.0147 | 0.0072 | 0.0007 | 2974 |