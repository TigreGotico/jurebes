# Cookbook: end-to-end testing the pipeline plugin

ovoscope tests the *pipeline plugin* — bus traffic in, bus traffic out — not the underlying classifier. The shape is different from skill tests.

## Prerequisites

```bash
pip install jurebes[e2e]
```

`e2e` pulls in `ovoscope`, `ovos-core`, `ovos-skill-hello-world`, `pytest`, and `pytest-timeout`.

## What gets tested

For the pipeline plugin:

- Bus events on `padatious:register_intent` populate the classifier.
- `match_high`/`match_medium`/`match_low` produce `IntentHandlerMatch` payloads.
- Exact-match short-circuits work.
- Slot extraction populates `match_data`.
- `detach_intent` / `detach_skill` clean up correctly.

For skill tests: spin up a skill and assert on the dialogue flow. Different fixture, different ovoscope helpers — see the ovoscope documentation.

## Minimal pipeline test

```python
# test/end2end/test_minimal.py
import pytest
from ovoscope import get_minicroft
from ovos_bus_client.message import Message


@pytest.fixture
def minicroft():
    mc = get_minicroft(
        skill_ids=["skill-hello-world.openvoiceos"],
        config={
            "intents": {
                "pipeline": [
                    "ovos-jurebes-pipeline-plugin-high",
                ],
            },
            "jurebes": {
                "baseline": "linear_svc",
                "enable_slots": True,
                "conf_high": 0.4,
            },
        },
    )
    yield mc
    mc.stop()


def test_hello_world_matches(minicroft):
    captured = []
    minicroft.bus.on("speak", lambda m: captured.append(m))

    minicroft.bus.emit(Message(
        "recognizer_loop:utterance",
        {"utterances": ["hello"], "lang": "en-US"},
    ))

    # Wait for the bus to settle.
    import time; time.sleep(2)
    assert any("hello" in m.data.get("utterance", "").lower() for m in captured)
```

The test:

1. Constructs a minimal ovos-core instance with the jurebes pipeline plugin and one skill.
2. Subscribes to the `speak` bus event to capture skill responses.
3. Injects an utterance via `recognizer_loop:utterance`.
4. Asserts that the skill responded.

`get_minicroft` from ovoscope handles the lifecycle: starts the bus, loads skills, registers intents, fires `mycroft.ready` to trigger jurebes' initial training.

## Asserting on captured bus traffic

For sharper assertions, capture the `IntentHandlerMatch` emission directly:

```python
def test_intent_match_data(minicroft):
    matches = []
    minicroft.bus.on("intent.service.intent.reply",
                     lambda m: matches.append(m))

    minicroft.bus.emit(Message(
        "recognizer_loop:utterance",
        {"utterances": ["hello"], "lang": "en-US"},
    ))

    import time; time.sleep(2)
    assert any(m.data.get("intent_type") == "skill-hello-world.openvoiceos:HelloWorld.intent"
               for m in matches)
```

The exact bus event names depend on the ovos-core version; consult ovoscope helpers for the most current dispatch shape.

## Fixture recording

ovoscope supports recording bus traffic to a fixture file, then replaying it. Useful for regression testing against a known-good trace. The recording flow:

```python
mc = get_minicroft(..., record="trace.jsonl")
# Drive the bus.
mc.stop()
```

Replay:

```python
mc = get_minicroft(..., replay="trace.jsonl")
```

See ovoscope's docs for the current recording API.

## Timeout safety

Pipeline tests are deadlock-prone — a stuck skill leaves the test hanging. Decorate with a timeout:

```python
import pytest

@pytest.mark.timeout(10)
def test_hello_world_matches(minicroft):
    ...
```

The `pytest-timeout` dep (pulled in by `jurebes[e2e]`) wires this up automatically.

## CI integration

The bundled `e2e` extra installs everything needed. A GitHub Actions step:

```yaml
- name: Install e2e deps
  run: pip install -e .[e2e]
- name: Run e2e tests
  run: pytest test/end2end -q --timeout=60
```

## Sanity-check the plugin is loaded

If a test fails because the plugin never fired, verify:

```python
def test_plugin_in_pipeline(minicroft):
    # The pipeline list should include jurebes.
    from ovos_config import Configuration
    pipeline = Configuration().get("intents", {}).get("pipeline", [])
    assert any("jurebes" in p for p in pipeline)
```

Common silent failures:

- The plugin entry-point is not installed (`pip install -e .` in the jurebes repo before running tests).
- The plugin name in `mycroft.conf` does not match the entry-point registration (`ovos-jurebes-pipeline-plugin-high` vs `ovos-jurebes-pipeline-plugin`).

## Related reading

- [../advanced/ovos-integration-deep-dive.md](../advanced/ovos-integration-deep-dive.md) — the bus-event surface jurebes exposes.
- [../opm.md](../opm.md) — the plugin overview.

---
- Back to [docs index](../index.md)
