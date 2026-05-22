# massive-templates (pl-PL) training report

- intents: **60**
- templates: **13547** → **13814** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8487 | 0.8240 |
| linear_svc | 0.8359 | 0.8084 |
| ovr_linear_svc | 0.8359 | 0.8084 |
| voting_soft | 0.8265 | 0.7969 |
| logreg_char | 0.8225 | 0.7566 |
| logreg | 0.7851 | 0.7260 |
| nb_multinomial | 0.6402 | 0.4976 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.2111 | 0.5249 | 0.3011 | 0.1812 | 2974 |
| template | 0.0117 | 0.0366 | 0.0178 | 0.0024 | 2974 |