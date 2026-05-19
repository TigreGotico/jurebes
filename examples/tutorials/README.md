# 🐾 jurebes tutorials

Sixty-nine runnable examples covering every public concept in jurebes.

## Core API

1. [01_quickstart](01_quickstart.py) — minimal `IntentClassifier` + add_intent + fit + predict.
2. [02_intent_classifier_api](02_intent_classifier_api.py) — full add/remove/predict/predict_proba/predict_batch tour.
3. [03_save_load](03_save_load.py) — joblib round-trip with prediction parity check.
4. [04_predict_batch](04_predict_batch.py) — per-item vs batched predict equivalence and latency.
5. [05_calibration_modes](05_calibration_modes.py) — `calibrate="if_missing"` vs `"always"` vs `False` (with the expected ValueError).

## Featurizers

10. [10_tfidf_word](10_tfidf_word.py) — word-level TF-IDF.
11. [11_tfidf_char](11_tfidf_char.py) — char n-gram TF-IDF.
12. [12_count_word](12_count_word.py) — raw / binary counts.
13. [13_hashing_word](13_hashing_word.py) — fixed-dim hashing vectorizer.
14. [14_char_word_union](14_char_word_union.py) — word + char FeatureUnion.
15. [15_tfidf_sublinear](15_tfidf_sublinear.py) — sublinear_tf scaling.
16. [16_lsa_dimensionality_reduction](16_lsa_dimensionality_reduction.py) — Truncated-SVD bottleneck.
17. [17_nmf_topics](17_nmf_topics.py) — non-negative latent factors.
18. [18_lda_topics](18_lda_topics.py) — Latent Dirichlet Allocation topics.
19. [19_text_stats](19_text_stats.py) — handcrafted text-statistics features.
20. [20_feature_union](20_feature_union.py) — combine arbitrary featurizer builders.
21. [21_sklearn_autoencoder](21_sklearn_autoencoder.py) — fit/transform/inverse_transform/reconstruction_error.
22. [22_categorical_vectorizer](22_categorical_vectorizer.py) — dict-of-string one-hot + JSON persistence.
23. [23_custom_sklearn_pipeline](23_custom_sklearn_pipeline.py) — hand-built `sklearn.pipeline.Pipeline`.

## Baselines

30. [30_baseline_naive_bayes](30_baseline_naive_bayes.py) — multinomial / complement / bernoulli NB.
31. [31_baseline_linear](31_baseline_linear.py) — LogReg / LinearSVC / Ridge.
32. [32_baseline_kernel_svm](32_baseline_kernel_svm.py) — rbf_svc / nusvc.
33. [33_baseline_tree_ensemble](33_baseline_tree_ensemble.py) — RF / ExtraTrees / GB / HistGBM / Bagging.
34. [34_baseline_mlp_shallow](34_baseline_mlp_shallow.py) — single hidden-layer MLP.
35. [35_baseline_voting_stacking](35_baseline_voting_stacking.py) — voting_soft and stacking ensembles.
36. [36_baseline_reduced_dim](36_baseline_reduced_dim.py) — lsa/nmf/autoencoder + LogReg.
37. [37_baseline_online_learners](37_baseline_online_learners.py) — SGD losses, perceptron, passive-aggressive.
38. [38_baseline_multiclass_strategies](38_baseline_multiclass_strategies.py) — ovr vs ovo LinearSVC.
39. [39_baseline_discriminant](39_baseline_discriminant.py) — LDA and QDA.
40. [40_baselines_registry](40_baselines_registry.py) — names / resolve / groups.
41. [41_register_custom_baseline](41_register_custom_baseline.py) — register a factory under a group.

## Slot tagger

50. [50_iob_tagger_basics](50_iob_tagger_basics.py) — train and predict on a toy entity-bearing sentence.
51. [51_classifier_with_tagger](51_classifier_with_tagger.py) — IntentClassifier with SklearnIOBTagger.
52. [52_custom_token_features](52_custom_token_features.py) — custom token-feature function.
53. [53_tagger_save_load](53_tagger_save_load.py) — joblib round-trip.

## Datasets

60. [60_load_csv](60_load_csv.py) — `load_csv` on an inline tempfile.
61. [61_load_jsonl](61_load_jsonl.py) — `load_jsonl` on an inline tempfile.
62. [62_load_ovos_intents](62_load_ovos_intents.py) — alternation expansion and entity collection.
63. [63_load_huggingface](63_load_huggingface.py) — `load_hf` signature (requires `jurebes[hf]`).
64. [64_canonical_loaders](64_canonical_loaders.py) — six canonical benchmark loaders.

## Benchmark

70. [70_compare_two_baselines](70_compare_two_baselines.py) — minimal `compare()` call.
71. [71_compare_group_selector](71_compare_group_selector.py) — expand `@linear` and compare.
72. [72_multi_metric_scoring](72_multi_metric_scoring.py) — accuracy / f1_macro / log_loss.
73. [73_cross_validation_folds](73_cross_validation_folds.py) — k=3 vs k=5 stability.
74. [74_runresult_inspection](74_runresult_inspection.py) — RunResult / ComparisonResult fields.
75. [75_markdown_report](75_markdown_report.py) — `to_markdown` with sort_by + precision + group.
76. [76_json_export](76_json_export.py) — `to_json` round-trip.
77. [77_latency_analysis](77_latency_analysis.py) — pooled p50/p95/p99 percentiles.

## Search

80. [80_grid_search](80_grid_search.py) — exhaustive grid.
81. [81_halving_grid](81_halving_grid.py) — successive halving on a grid.
82. [82_random_search](82_random_search.py) — random sampling from a discrete space.
83. [83_random_with_scipy_distributions](83_random_with_scipy_distributions.py) — `loguniform` + `randint`.
84. [84_halving_random](84_halving_random.py) — successive halving on random candidates.
85. [85_bayes_search](85_bayes_search.py) — skopt backend (requires `jurebes[search-bayes]`).
86. [86_genetic_search](86_genetic_search.py) — sklearn-genetic backend (requires `jurebes[search-genetic]`).
87. [87_custom_search_space](87_custom_search_space.py) — start from `spaces.for_baseline()` and tweak.

## Statistical comparison

90. [90_paired_t_test](90_paired_t_test.py) — paired Student's t-test on per-fold scores.
91. [91_wilcoxon_signed_rank](91_wilcoxon_signed_rank.py) — non-parametric paired test.
92. [92_mcnemar_test](92_mcnemar_test.py) — McNemar on paired predictions.
93. [93_friedman_nemenyi](93_friedman_nemenyi.py) — multi-classifier Friedman + Nemenyi post-hoc.
94. [94_critical_difference_diagram](94_critical_difference_diagram.py) — Demsar-style CD diagram.

## Research workflows

100. [100_compare_reduced_dim_methods](100_compare_reduced_dim_methods.py) — LSA / NMF / LDA / autoencoder.
101. [101_find_best_nb_variant](101_find_best_nb_variant.py) — NB sweep + paired t-test.
102. [102_tune_winning_baseline](102_tune_winning_baseline.py) — compare → pick winner → random-search.
103. [103_latency_vs_accuracy_pareto](103_latency_vs_accuracy_pareto.py) — accuracy vs p95 scatter (matplotlib-gated).
104. [104_calibration_diagnostics](104_calibration_diagnostics.py) — Brier and ECE on calibrated LinearSVC.

## OVOS integration

110. [110_opm_pipeline_basic](110_opm_pipeline_basic.py) — `JurebesPipeline` + `FakeBus` + Message registration.
111. [111_opm_with_slots](111_opm_with_slots.py) — same with `enable_slots: true` and an entity.
112. [112_opm_configuration](112_opm_configuration.py) — full `mycroft.conf` JSON block.

---

Each script is standalone. Run with `python examples/tutorials/NN_name.py`. The 60–64 dataset examples and the bayes/genetic search examples need optional extras (`jurebes[hf,search-bayes,search-genetic]`); they print clear install hints when missing.
