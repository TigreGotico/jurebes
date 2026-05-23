# intents-for-eval — multi-language summary

Headline of the best intent baseline and best slot tagger per language.
Full per-language reports under `intents_for_eval_<lang>.md`.

| lang | intent_top1_baseline | intent_accuracy | intent_macro_f1 | slot_top_tagger | slot_precision | slot_f1 | slot_exact_match | wall_s |
|---|---|---|---|---|---|---|---|---|
| en-US | union_bm25_pos_logreg | 0.8318 | 0.8344 | crf | 0.9047 | 0.8766 | 0.8841 | 0.0 |
| pt-PT | linear_svc_char | 0.8394 | 0.8440 | crf | 0.8375 | 0.8169 | 0.8382 | 0.0 |
| pt-BR | linear_svc_char | 0.8441 | 0.8455 | crf | 0.8350 | 0.8127 | 0.8365 | 0.0 |
| es-ES | union_bm25_pos_logreg | 0.8394 | 0.8414 | crf | 0.8489 | 0.8248 | 0.8418 | 0.0 |
| fr-FR | linear_svc_char | 0.8394 | 0.8430 | dictionary | 0.8316 | 0.8305 | 0.7976 | 0.0 |
| de-DE | linear_svc_char | 0.8406 | 0.8432 | crf | 0.9000 | 0.8667 | 0.8694 | 0.0 |
| it-IT | linear_svc_char | 0.8388 | 0.8424 | crf | 0.8071 | 0.7892 | 0.8141 | 0.0 |
| nl-NL | linear_svc_char | 0.8359 | 0.8393 | crf | 0.8533 | 0.8325 | 0.8512 | 0.0 |
| ca-ES | linear_svc_char | 0.8488 | 0.8518 | crf | 0.7915 | 0.7748 | 0.8124 | 0.0 |
| gl-ES | linear_svc_char | 0.8447 | 0.8477 | crf | 0.8767 | 0.8621 | 0.8718 | 0.0 |
| da-DK | linear_svc_char | 0.8535 | 0.8569 | crf | 0.8644 | 0.8028 | 0.8159 | 0.0 |
| eu-ES | linear_svc_char | 0.8376 | 0.8382 | crf | 0.8696 | 0.7567 | 0.7776 | 144.8 |
