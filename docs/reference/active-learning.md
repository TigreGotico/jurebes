# Active learning — `jurebes.active_learning`

Primitives for building sample-selection loops on top of `IntentClassifier`. Pure-sklearn / pure-jurebes; no network or LLM dependency. The LLM-specific glue belongs to applications and lives in [`examples/llm_augmentation/`](../../examples/llm_augmentation/).

The four primitives cover the canonical strategies in Settles (2009), *Active Learning Literature Survey*: uncertainty sampling, query-by-committee, and confusion-driven targeted augmentation.

## `uncertainty_scores(clf, utterances, labels=None) -> list[UncertaintyScore]`

Score utterances by both top-1 prediction confidence and (optionally) the confidence assigned to a requested label. `labels` is parallel to `utterances`; pass `None` for entries where no ground-truth label is known.

```python
from jurebes.active_learning import uncertainty_scores

scores = uncertainty_scores(clf, ["play africa", "wake me up"], ["play_song", "set_timer"])
for s in scores:
    print(s.utterance, s.top_pred, s.top_conf, s.requested_conf)
```

`UncertaintyScore` fields: `utterance`, `top_pred`, `top_conf`, `requested`, `requested_conf`.

## `bucket_paraphrases(clf, paraphrases_by_intent, *, hard_conf_max=0.6, dedupe_against=None) -> dict[str, ParaphraseBucket]`

Split paraphrases by intent into three buckets:

- `skip`    — predicted intent matches requested AND confidence ≥ threshold (model already knows it).
- `hard`    — predicted intent matches but confidence is below threshold (uncertainty signal — keep).
- `suspect` — predicted intent differs (LLM may have drifted off-intent — verify before keeping).

```python
from jurebes.active_learning import bucket_paraphrases

paras = {
    "play_song":  ["spin africa", "put on hey jude"],
    "set_timer":  ["wake me in five", "remind me later"],
}
buckets = bucket_paraphrases(clf, paras, hard_conf_max=0.6)
for intent, b in buckets.items():
    print(intent, len(b.skip), len(b.hard), len(b.suspect))
```

The `dedupe_against` argument (typically the current training set) is used to drop paraphrases already present.

## `confusion_pairs(comparison_result, *, top_n=5) -> list[tuple[str, str, int]]`

Extract the most-confused intent pairs from a `ComparisonResult.confusion_matrix`. Returns `[(true_label, predicted_label, count), ...]` sorted by count descending.

```python
from jurebes.active_learning import confusion_pairs
from jurebes.benchmark import compare

result = compare(["nb_multinomial"], X, y, k=3)
for true_lbl, pred_lbl, n in confusion_pairs(result, top_n=10):
    print(f"{true_lbl} confused as {pred_lbl}: {n} times")
```

Drives **hard-negative pair mining** — targeted LLM augmentation focused on the worst-confused intent pair has higher signal than uniform paraphrase generation.

## `disagreement_score(classifiers, utterance) -> float`

Query-by-committee entropy over top-1 predictions of multiple fitted classifiers. Higher = more disagreement = a better candidate for manual labeling.

```python
from jurebes.active_learning import disagreement_score

clfs = [
    IntentClassifier(BASELINES.build(name))
    for name in ("logreg", "nb_multinomial", "linear_svc")
]
for c in clfs:
    for intent, samples in seed_data.items():
        c.add_intent(intent, samples)
    c.fit()

score = disagreement_score(clfs, "play africa")  # 0.0 = unanimous
```

## End-to-end pattern

See [`guides/active-learning-llm-augmentation.md`](../guides/active-learning-llm-augmentation.md) for the conceptual three-bucket loop and [`examples/llm_augmentation/loop.py`](../../examples/llm_augmentation/loop.py) for a runnable implementation built on these primitives.

---
[← back to docs index](../index.md)
