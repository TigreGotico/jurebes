# Cookbook: production deployment

Wrapping a trained jurebes model as an OVOS pipeline plugin. Assumes you have a `banking_winner.joblib` from [full-research-pipeline.md](full-research-pipeline.md).

## Approach: OVOS plugin with the bundled JurebesPipeline

The bundled `JurebesPipeline` (`jurebes.opm:JurebesPipeline`) is the path of least resistance. It trains *its own* `IntentClassifier` from samples registered via the bus — it does not load a pre-trained joblib model. Two deployment shapes follow from this:

1. **Plugin-trained.** Skills register their intent samples on the bus; the plugin trains on-startup.
2. **Pre-trained.** A separate harness trains and saves a joblib model; a thin custom adapter loads it.

### Shape 1: plugin-trained

`mycroft.conf`:

```json
{
  "intents": {
    "pipeline": [
      "ovos-jurebes-pipeline-plugin-high",
      "ovos-jurebes-pipeline-plugin-medium",
      "ovos-jurebes-pipeline-plugin-low"
    ]
  },
  "jurebes": {
    "baseline": "linear_svc",
    "enable_slots": true,
    "conf_high": 0.80,
    "conf_med":  0.60,
    "conf_low":  0.40
  }
}
```

Restart ovos-core. Skills' intent registrations are gathered on the bus; on `mycroft.ready` the plugin trains an `IntentClassifier` per configured language.

The three confidence thresholds correspond to the three OPM match levels — `match_high`, `match_medium`, `match_low`. OVOS runs higher-confidence pipelines first.

### Pinning the baseline

`"baseline"` in `mycroft.conf` accepts any registered baseline name. Pick from a benchmark of your typical skill inventory rather than the default — `linear_svc_char` is a good choice for typo-tolerance; `union_logreg` for short utterances; `nb_complement` for imbalanced inventories.

### Calibrating the confidence thresholds

The defaults (0.4 / 0.6 / 0.8) are a starting point. After a benchmark on representative data, set thresholds based on the precision/recall curve — see [../guides/confidence-thresholds.md](../guides/confidence-thresholds.md). A typical recipe:

1. Run a 5-fold CV with the chosen baseline.
2. Plot the calibration curve (predicted confidence vs empirical accuracy) — see [../theory/calibration.md](../theory/calibration.md).
3. Set `conf_high` at the lowest confidence where empirical accuracy exceeds 95 %.
4. Set `conf_med` at 80 % accuracy.
5. Set `conf_low` at 60 % accuracy.

## Shape 2: pre-trained custom adapter

When you want the *exact* trained artefact in production (rather than retraining from registered samples), subclass `JurebesPipeline` and load:

```python
# my_pipeline.py
from jurebes import IntentClassifier
from jurebes.opm import JurebesPipeline


class PretrainedJurebesPipeline(JurebesPipeline):
    def __init__(self, bus=None, config=None):
        super().__init__(bus=bus, config=config or {})
        path = self.config.get("pretrained_model")
        if path:
            lang = self.lang
            self.containers[lang] = IntentClassifier.load(path)
            self._fitted[lang] = True
```

Register an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."opm.pipeline"]
"my-pretrained-jurebes-pipeline" = "my_package.my_pipeline:PretrainedJurebesPipeline"
```

`mycroft.conf`:

```json
{
  "intents": {
    "pipeline": ["my-pretrained-jurebes-pipeline-high"]
  },
  "my_pretrained_jurebes": {
    "pretrained_model": "/opt/skills/banking_winner.joblib",
    "conf_high": 0.85
  }
}
```

This shape bypasses the bus-driven training entirely. Useful when:

- The training data is sensitive and not available at deployment time.
- The training run takes long enough that doing it on every restart is unacceptable.
- Reproducible audit trails require an exact artefact, not a retrained-on-startup classifier.

## Dependency pinning

Pin both jurebes and scikit-learn in production:

```
scikit-learn==1.4.2
jurebes==<version-from-development>
```

joblib pickles are sklearn-version-sensitive. A mismatch can produce `ModuleNotFoundError` at load time. `IntentClassifier.load(...)._loaded_from_version` exposes the jurebes version stamped at save time.

## Restart story

When intent samples change:

- **Shape 1.** Skills emit `padatious:register_intent` events on next startup; the plugin's `handle_initial_train` refits on `mycroft.ready`. No manual step.
- **Shape 2.** Retrain the joblib offline, replace the file, restart ovos-core.

## Observability

The bundled plugin logs through `ovos_utils.log.LOG`:

- `DEBUG`: match-level decisions and lazy-fit skips.
- `ERROR`: initial-train failures, prediction exceptions.

Enable debug logging by setting `"log_level": "DEBUG"` in `mycroft.conf` during deployment validation.

## Latency budgets

Voice-assistant utterance dispatch typically allows ~50 ms total. Pick a baseline whose `predict_ms_p95_pooled` (from your benchmark) stays well under that bound. Linear baselines (`logreg`, `linear_svc`) and naive Bayes are usually <1 ms; kernel and ensemble baselines can exceed the budget on large inventories.

## Memory budgets

`RunResult.model_size_bytes` is the *uncompressed* pickle size. On-disk size with `compress=3` is roughly half. For embedded deployments, prefer linear / hashing baselines (tens of kB) over tree ensembles (MBs).

---
- Back to [docs index](../index.md)
