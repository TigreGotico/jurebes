# Installation

## Core install

```bash
pip install jurebes
```

This pulls in:

- `scikit-learn>=1.4`
- `joblib`
- `ovos-utils`, `ovos-bus-client`, `ovos-config`, `ovos-plugin-manager`
- `langcodes`

Python 3.10 or newer is required.

## Optional extras

jurebes uses [PEP 621 optional dependencies](https://peps.python.org/pep-0621/) to keep the core install light. Pick extras for the features you need:

| extra | adds | enables |
| --- | --- | --- |
| `hf` | `datasets` | HuggingFace dataset loaders (`load_hf`, canonical loaders for SNIPS, CLINC150, BANKING77, HWU64, ATIS, MASSIVE) |
| `search-bayes` | `scikit-optimize>=0.10` | Bayesian hyperparameter search backend |
| `search-genetic` | `sklearn-genetic-opt>=0.10` | Genetic-algorithm hyperparameter search backend |
| `search-all` | both of the above | both bayes and genetic |
| `bench-plot` | `matplotlib` | critical-difference diagrams via `CDDiagram.to_matplotlib()` |
| `test` | `pytest`, `pytest-cov` | unit-test suite |
| `e2e` | `ovoscope`, `ovos-core`, `ovos-skill-hello-world`, `pytest`, `pytest-timeout` | OVOS end-to-end pipeline tests |

Examples:

```bash
pip install jurebes[hf]                 # canonical dataset loaders
pip install jurebes[search-bayes]       # Bayesian search
pip install jurebes[search-all]         # both optional search backends
pip install jurebes[hf,bench-plot]      # docs cookbook prerequisites
pip install jurebes[e2e]                # full pipeline-plugin testing
```

## Verifying the install

```bash
python -c "from jurebes import IntentClassifier, BASELINES; print(len(BASELINES.names()), 'baselines')"
```

A working install prints `48 baselines`.

```bash
jurebes list-baselines | head -5
```

Lists the registered baselines alphabetically.

## Editable install for development

```bash
git clone https://github.com/OpenVoiceOS/jurebes
cd jurebes
pip install -e .[test,hf,bench-plot,search-all,e2e]
pytest -q
```

## OVOS pipeline plugin discovery

jurebes registers an entry point under `opm.pipeline`:

```toml
"ovos-jurebes-pipeline-plugin" = "jurebes.opm:JurebesPipeline"
```

Once installed in the same environment as `ovos-core`, the plugin is discoverable by name. See [advanced/ovos-integration-deep-dive.md](../advanced/ovos-integration-deep-dive.md) for `mycroft.conf` wiring.

---
- Previous: [What is jurebes?](01-what-is-jurebes.md)
- Next: [Your first classifier](03-first-classifier.md)
