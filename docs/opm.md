# OVOS pipeline plugin

`jurebes.opm:JurebesPipeline` is a `ConfidenceMatcherPipeline` registered under the entry-point `ovos-jurebes-pipeline-plugin`.

## Behaviour

- One `IntentClassifier` per configured language.
- Listens on the standard padatious messagebus events:
  - `padatious:register_intent`
  - `padatious:register_entity`
  - `detach_intent`
  - `detach_skill`
  - `mycroft.ready`
- Exact normalised matches are short-circuited via an internal `{(lang, norm_utt): intent}` dict; everything else goes through the sklearn estimator.
- Lazy fit: any registration after `mycroft.ready` triggers a re-fit on next match.
- Session-aware: respects `SessionManager.blacklisted_intents` / `blacklisted_skills`.

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

`baseline` accepts any name registered in `BASELINES`. `enable_slots` toggles the `SklearnIOBTagger`. The three `conf_*` thresholds map onto the `ConfidenceMatcherPipeline` high/med/low buckets.

## Caveats

- Slot extraction comes only from the trained `SklearnIOBTagger`; disabling slots removes all slot extraction.
- Exact matches are handled in-process via a `{(lang, norm_utt): intent}` cache. For typo tolerance, pick a char-ngram baseline such as `logreg_char`, `union_logreg`, or `linear_svc_char`.

---
[← back to docs index](index.md)
