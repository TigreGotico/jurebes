# jurebes documentation 🐾

**J**ust-sklearn **U**tility for **R**eproducible **E**valuation of **B**aselines, **E**stimators and **S**olvers.

A classical-ML intent classification research framework.

> *Named in memory of Jurebes, the best dog.*

## Start here

- [What is jurebes?](getting-started/01-what-is-jurebes.md)
- [Installation](getting-started/02-installation.md)
- [Your first classifier](getting-started/03-first-classifier.md)
- [Core concepts](getting-started/04-core-concepts.md)
- [Troubleshooting](getting-started/05-troubleshooting.md)

## How-to guides

- [Choosing a baseline](guides/choosing-a-baseline.md)
- [Adding slots](guides/adding-slots.md)
- [Saving and loading](guides/saving-and-loading.md)
- [Confidence thresholds](guides/confidence-thresholds.md)
- [Handling class imbalance](guides/handling-class-imbalance.md)
- [Reproducibility](guides/reproducibility.md)
- [Out-of-domain detection](guides/out-of-domain-detection.md)
- [Debugging bad predictions](guides/debugging-bad-predictions.md)
- [LLM-driven data augmentation](guides/active-learning-llm-augmentation.md)

## Theory

- [Intent classification](theory/intent-classification.md)
- [Text featurization](theory/text-featurization.md)
- [Linear classifiers](theory/linear-classifiers.md)
- [Naive Bayes](theory/naive-bayes.md)
- [Kernel methods](theory/kernel-methods.md)
- [Ensemble methods](theory/ensemble-methods.md)
- [Dimensionality reduction](theory/dimensionality-reduction.md)
- [Calibration](theory/calibration.md)
- [Cross-validation](theory/cross-validation.md)
- [Multi-class strategies](theory/multi-class-strategies.md)
- [Statistical comparison](theory/statistical-comparison.md)

## Advanced

- [Custom featurizers](advanced/custom-featurizers.md)
- [Custom baselines](advanced/custom-baselines.md)
- [Custom search backends](advanced/custom-search-backends.md)
- [Custom scoring metrics](advanced/custom-scoring-metrics.md)
- [Custom datasets](advanced/custom-datasets.md)
- [Performance tuning](advanced/performance-tuning.md)
- [OVOS integration deep dive](advanced/ovos-integration-deep-dive.md)
- [Architecture](advanced/architecture.md)

## API reference

- [Reference index](reference/index.md)
- [IntentClassifier](reference/intent-classifier.md)
- [Featurizers](reference/featurizers.md)
- [Baselines registry](reference/baselines.md)
- [Datasets](reference/datasets.md)
- [Benchmark harness](reference/benchmark.md)
- [Statistical tests](reference/stats.md)
- [CLI](reference/cli.md)
- [Search subsystem](search.md)
- [Slot taggers](slots.md) — dictionary, template, sklearn IOB, hybrid, CRF (`TAGGERS` registry)
- [Slot tagger reference](reference/slots.md)
- [OVOS pipeline plugin](opm.md)

## Cookbook

- [Compare every linear baseline](cookbook/compare-all-linear.md)
- [Tune with random search](cookbook/tune-with-random-search.md)
- [Reduced-dim featurizer comparison](cookbook/reduced-dim-comparison.md)
- [OOD detection with an autoencoder](cookbook/ood-with-autoencoder.md)
- [Full research pipeline](cookbook/full-research-pipeline.md)
- [Production deployment](cookbook/production-deployment.md)
- [End-to-end pipeline-plugin testing](cookbook/e2e-testing.md)

## Other

- [Research framework deep dive](research.md)
- [Tutorial scripts library](../examples/tutorials/README.md) — runnable concept demos.
- [Migration from previous APIs](../MIGRATION.md)
