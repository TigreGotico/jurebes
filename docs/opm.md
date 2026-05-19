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

```yaml
intents:
  pipeline:
    - ovos-jurebes-pipeline-plugin
ovos-jurebes-pipeline-plugin:
  baseline: linear_svc      # any name from BASELINES
  enable_slots: true         # turn on SklearnIOBTagger
  conf_high: 0.8
  conf_med: 0.6
  conf_low: 0.4
```

## Caveats

- Padatious-style template + slot exact-matching (previously handled by `padacioso`) is intentionally absent in v2. Slots come only from the trained `SklearnIOBTagger`; if you disable slots, you get no slot extraction at all.

The padacioso runtime dependency is gone; exact matches are handled in-process. Fuzzy matching is no longer a configuration knob — use a baseline (e.g. `logreg_char`, `union_logreg`, `linear_svc_char`) for typo tolerance.
