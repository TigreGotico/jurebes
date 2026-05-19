# API reference

Module-by-module reference for every public surface.

## Pages

- [intent-classifier.md](intent-classifier.md) — `IntentClassifier`, `IntentResult`
- [featurizers.md](featurizers.md) — every featurizer in `jurebes.featurizers`
- [baselines.md](baselines.md) — the `BASELINES` registry and the 48-row baseline table
- [datasets.md](datasets.md) — `load_csv`, `load_jsonl`, `load_ovos_intents`, `load_hf`, and the canonical loaders
- [benchmark.md](benchmark.md) — `compare`, `cross_validate`, `train_test`, `RunResult`, `ComparisonResult`, `SCORING`
- [stats.md](stats.md) — `paired_t_test_cv`, `wilcoxon_signed_rank_cv`, `mcnemar_test`, `friedman_nemenyi`, `critical_difference`
- [cli.md](cli.md) — every subcommand and flag

Related pages outside the reference tree:

- [../search.md](../search.md) — search subsystem narrative and backend matrix
- [../slots.md](../slots.md) — `SklearnIOBTagger` and `token_features`
- [../opm.md](../opm.md) — OVOS pipeline plugin overview
- [../research.md](../research.md) — research-workflow cookbook

---
- Back to [docs index](../index.md)
