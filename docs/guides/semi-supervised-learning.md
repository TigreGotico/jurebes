# Semi-supervised learning

Semi-supervised learning lets a small labeled seed set bootstrap a
classifier over a much larger unlabeled pool. `jurebes.semi_supervised`
ships three classical strategies that work on top of any
`IntentClassifier`:

- `self_train` — single-view self-training (Yarowsky 1995).
- `co_train` — two-view alternating self-training
  (Blum & Mitchell 1998).
- `label_propagation` — graph-based label spreading via
  scikit-learn (related to Zhou & Li 2005's tri-training family).

For a wider survey of the field see
**Triguero, García & Herrera (2015)**, *Self-labeled techniques for
semi-supervised learning: taxonomy, software and empirical study*.

## When to use which

**`self_train` — cheap baseline.** Use it when you have one good
featurization and a sizable unlabeled pool. Single-view self-training
is the cheapest way to convert "lots of un-curated utterances" into
training signal. The main failure mode is class drift: the seed
classifier picks confident but systematically wrong pseudo-labels and
the loop reinforces them.

**`co_train` — two genuinely-different feature views.** Use when you
can construct two featurizations that capture different aspects of
the input (e.g. word TF-IDF vs char n-gram, or text vs metadata).
Co-training is only useful when the views disagree on hard cases:
each view's confident picks must be informative to the other. If both
views collapse to the same decisions, co-training degenerates to
self-training with twice the compute.

**`label_propagation` — small enough to fit a dense graph.**
scikit-learn's `LabelPropagation` / `LabelSpreading` require a dense
affinity matrix, so the combined labeled + unlabeled pool must fit in
memory at O(n²). For pools above a few thousand utterances prefer
`self_train` or `co_train` instead.

## Confidence threshold tuning

The confidence threshold gates which pseudo-labels are promoted into
the labeled pool each round.

- **Fixed threshold** (e.g. `confidence_threshold=0.9`): safest,
  slowest growth.
- **Annealed schedule**: pass a `threshold_schedule` callable to
  `self_train` that starts high and decays over rounds. Useful when
  the seed classifier is well-calibrated; lets confidence "spend
  itself down" as easy examples are exhausted.

Calibration matters here. Run
`jurebes.benchmark.calibration` on a held-out set to measure ECE
before deciding on a threshold — a classifier with ECE > 0.1 emits
confidences that look high but mean little, and aggressive thresholds
will not protect you. Pair `IntentClassifier(..., calibrate="always")`
with `self_train` when your estimator is a tree, forest, kNN or
LinearSVC.

## Out-of-domain risk and mitigation

Pseudo-labels concentrate on confident-but-wrong samples, producing
systematic drift toward whichever class the seed classifier is most
biased about. Two mitigations:

1. **Use the `per_class_quota` selection strategy.** It caps how many
   pseudo-labels each class may contribute per round, which keeps the
   labeled pool's class balance close to the seed distribution.
2. **Always pass `eval_X` / `eval_y` to `self_train`.** A frozen
   held-out set with `early_stop_patience` lets the loop abort once
   macro-F1 stops improving. Without it, every additional round can
   only over-fit the pseudo-labels.

## Strategy comparison

| Strategy | Strengths | Weaknesses |
| --- | --- | --- |
| `self_train` | Cheap, simple, scales to large unlabeled pools | Class drift; relies on calibration |
| `co_train` | Catches errors one view makes | Needs two genuinely-different views; 2× compute |
| `label_propagation` | Uses unlabeled geometry directly | O(n²) memory; opaque hyper-parameters |

See `examples/semi_supervised/compare_strategies.py` for a runnable
side-by-side comparison on a shared seed set.

## References

- Yarowsky, D. (1995). *Unsupervised word sense disambiguation
  rivaling supervised methods.* ACL.
- Blum, A. & Mitchell, T. (1998). *Combining labeled and unlabeled
  data with co-training.* COLT.
- Zhou, Z.-H. & Li, M. (2005). *Tri-training: exploiting unlabeled
  data using three classifiers.* IEEE TKDE.
- Triguero, I., García, S. & Herrera, F. (2015). *Self-labeled
  techniques for semi-supervised learning: taxonomy, software and
  empirical study.* KAIS.
