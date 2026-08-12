# massive-templates (th-TH) training report

- intents: **60**
- templates: **13547** → **13807** after slot expansion
- entities: **55**
- test utterances: **2974** in-domain (+0 OOD rows excluded)

## Intent classification

Train on Padatious-style templates, evaluate top-1 intent on the test split.

| baseline | accuracy | macro_f1 |
|---|---|---|
| linear_svc_char | 0.8241 | 0.7766 |
| logreg_char | 0.7818 | 0.7161 |
| linear_svc | 0.7428 | 0.7045 |
| ovr_linear_svc | 0.7428 | 0.7045 |
| voting_soft | 0.7310 | 0.6923 |
| logreg | 0.6876 | 0.6257 |
| nb_multinomial | 0.5370 | 0.3809 |

**winning baseline (intent):** `linear_svc_char`

## Slot extraction

Train each tagger on raw templates + entity gazetteer; evaluate against the gold `expected_slots` on the test split.

> Slot extraction on the MASSIVE sweep uses the regex taggers only (`dictionary`, `template`); the ML taggers are memory-bound on this corpus and are benchmarked on intents-for-eval instead.

| tagger | slot_precision | slot_recall | slot_f1 | exact_match | n_test |
| --- | --- | --- | --- | --- | --- |
| dictionary | 0.1676 | 0.6651 | 0.2677 | 0.1147 | 2974 |
| template | 0.0118 | 0.0246 | 0.0159 | 0.0030 | 2974 |