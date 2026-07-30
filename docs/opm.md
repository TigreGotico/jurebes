# OVOS pipeline plugin

`jurebes.opm:JurebesPipeline` is a `ConfidenceMatcherPipeline` registered under the entry point `ovos-jurebes-pipeline-plugin`.

## Behavior

- One `IntentClassifier` runs per configured language.
- The plugin listens on the standard padatious messagebus events:
  - `padatious:register_intent`
  - `padatious:register_entity`
  - `detach_intent`
  - `detach_skill`
  - `mycroft.ready`
- An internal `{(lang, norm_utt): intent}` dict short-circuits exact normalized matches. Everything else goes through the sklearn estimator.
- Fit is lazy. Any registration after `mycroft.ready` triggers a re-fit on the next match.
- The plugin respects `SessionManager.blacklisted_intents` and `blacklisted_skills`.

## Configuration

In `mycroft.conf` (JSON):

```json
{
  "intents": {
    "pipeline": [
      "ovos-jurebes-pipeline-plugin"
    ]
  },
  "jurebes": {
    "baseline": "linear_svc",
    "enable_slots": true,
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4
  }
}
```

`baseline` accepts any name registered in `BASELINES`. `enable_slots` toggles the `SklearnIOBTagger`. The three `conf_*` thresholds map onto the `ConfidenceMatcherPipeline` high, medium, and low buckets.

## Caveats

- Slot extraction comes only from the trained `SklearnIOBTagger`. Disabling slots removes all slot extraction.
- The plugin handles exact matches in-process through a `{(lang, norm_utt): intent}` cache. For typo tolerance, pick a char-ngram baseline such as `logreg_char`, `union_logreg`, or `linear_svc_char`.

---
[Home](index.md)
