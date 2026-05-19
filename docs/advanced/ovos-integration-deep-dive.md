# OVOS integration deep dive

`jurebes.opm.JurebesPipeline` is an OPM `ConfidenceMatcherPipeline` registered under the entry-point `ovos-jurebes-pipeline-plugin`. It listens on the standard OVOS message bus and exposes high/medium/low confidence buckets.

## Class hierarchy

```
ConfidenceMatcherPipeline (ovos-plugin-manager)
        ▲
        │
        │
JurebesPipeline (jurebes.opm)
        │
        │ wraps
        ▼
IntentClassifier(BASELINES.build(config["baseline"]),
                 tagger=SklearnIOBTagger() if enable_slots else None)
```

One `IntentClassifier` per configured language. Languages come from `Configuration().get("lang")` plus `Configuration().get("secondary_langs")`.

## Bus events

| event | direction | payload | action |
| --- | --- | --- | --- |
| `padatious:register_intent` | in | `{name, samples or file_name, lang, skill_id}` | append samples to the per-language classifier; mark unfitted |
| `padatious:register_entity` | in | `{name, samples or file_name, lang}` | append entity samples to the per-language tagger (when `enable_slots`) |
| `detach_intent` | in | `{intent_name}` | remove from all per-language classifiers |
| `detach_skill` | in | `{skill_id}` | remove every intent and entity whose name contains `skill_id` |
| `mycroft.ready` | in | — | call `fit()` on every per-language classifier |

After `mycroft.ready` any subsequent registration marks the affected language unfitted; the next `match_*` call triggers a lazy refit.

## Sample-file resolution

`register_intent` and `register_entity` accept either an inline `samples` list or a `file_name`. If only the file name is provided, the plugin reads the file (UTF-8) and uses each non-empty stripped line as a sample.

```python
# Skill side
self.bus.emit(Message("padatious:register_intent", {
    "name": "hello.intent",
    "samples": ["hello", "hi"],
    "lang": "en-US",
    "skill_id": "hello.skill",
}))
```

## Match levels

The three `match_*` methods all forward to `_match_level(utterances, threshold, lang, message)`:

```python
def match_high(self, utterances, lang, message):
    return self._match_level(utterances, self.conf_high, lang, message)
```

`_match_level` computes the best intent via `calc_intent(...)` and returns an `IntentHandlerMatch` if `match.confidence > threshold`.

`IntentHandlerMatch` carries:

- `match_type`: the intent name (e.g. `"hello.skill:hello.intent"`)
- `match_data`: the entity dict (e.g. `{"name": "bob"}`)
- `skill_id`: derived from the intent-to-skill mapping
- `utterance`: the original input

OVOS routes the result to the registered skill.

## Exact-match short circuit

The plugin maintains `_exact: dict[(lang, normalised_utt), intent]` — populated at register time with every sample that contains no `{entity}` placeholder. On match, exact lookups bypass the sklearn estimator entirely:

```python
exact = self._exact.get((lang, _normalize(utt)))
if exact and exact not in sess.blacklisted_intents:
    results.append(_Match(exact, 1.0, {}, utt))
```

`_normalize` lower-cases and collapses whitespace. For typo-tolerant matching, pick a char-ngram baseline (`logreg_char`, `union_logreg`, `linear_svc_char`).

## Session awareness

`SessionManager.get(message)` retrieves the current session; the plugin respects `sess.blacklisted_intents` and `sess.blacklisted_skills` — predictions matching either are filtered out before the best is chosen.

## Lazy fitting

Newly registered intents do not trigger an immediate `fit()` — that would be expensive on every skill load. Instead `_fitted[lang]` is set to `False`, and the next `_match_level` call invokes `_maybe_fit(lang)`:

```python
def _maybe_fit(self, lang):
    if not self._fitted.get(lang):
        try:
            self.containers[lang].fit()
            self._fitted[lang] = True
        except Exception as e:
            LOG.debug(f"Jurebes lazy fit skipped for {lang}: {e}")
```

Failures (e.g. only one intent registered for the language) are logged but do not block other languages.

## Configuration

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
    "conf_high": 0.8,
    "conf_med":  0.6,
    "conf_low":  0.4
  }
}
```

| key | default | meaning |
| --- | --- | --- |
| `baseline` | `"linear_svc"` | any registered name from `BASELINES.names()` |
| `enable_slots` | `true` | attach a `SklearnIOBTagger` per language |
| `conf_high` | `0.8` | threshold for `match_high` |
| `conf_med` | `0.6` | threshold for `match_medium` |
| `conf_low` | `0.4` | threshold for `match_low` |

## Max utterance length

The plugin drops utterances longer than `max_words = 50` words to avoid pathological inputs. Adjust by overriding the attribute post-construction if needed.

## Shutdown

`shutdown()` removes all bus listeners. Called by OVOS on plugin teardown; do not call manually unless replacing the plugin instance.

## Custom baseline plumbing

The plugin reads `config["baseline"]` and calls `BASELINES.build(name)`. To use a custom baseline, register it before the plugin is constructed:

```python
# In an import statement before ovos-core starts the pipeline
from my_jurebes_extensions import register_my_baselines
register_my_baselines()
```

See [custom-baselines.md](custom-baselines.md).

## Testing the plugin

End-to-end tests use `ovoscope` and the `e2e` extra. See [../cookbook/e2e-testing.md](../cookbook/e2e-testing.md).

---
- Back to [docs index](../index.md)
