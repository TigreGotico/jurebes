"""Active-learning primitives.

Pure-sklearn / pure-jurebes helpers for building sample-selection
loops on top of :class:`jurebes.IntentClassifier`. No LLM or network
dependency lives in this module — that belongs to the application.

The four primitives below cover the canonical active-learning
strategies described in Settles (2009):

- :func:`uncertainty_scores` — score utterances by top-1 and
  requested-label calibrated probabilities.
- :func:`bucket_paraphrases` — split LLM-generated paraphrases into
  ``skip`` / ``hard`` / ``suspect`` buckets per intent.
- :func:`confusion_pairs` — extract the most-confused intent pairs
  from a :class:`ComparisonResult` (hard-negative pair mining).
- :func:`disagreement_score` — query-by-committee entropy across a
  list of classifiers.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class UncertaintyScore:
    utterance: str
    top_pred: str
    top_conf: float
    requested: Optional[str]
    requested_conf: float


def uncertainty_scores(
    clf,
    utterances: Sequence[str],
    labels: Optional[Sequence[Optional[str]]] = None,
) -> List[UncertaintyScore]:
    """Score utterances by top-1 prediction confidence and (optionally)
    by the confidence assigned to a requested label.

    ``labels`` is parallel to ``utterances``; pass ``None`` for entries
    where no ground-truth label is known.
    """
    if labels is None:
        labels = [None] * len(utterances)
    out: List[UncertaintyScore] = []
    for utt, label in zip(utterances, labels):
        ranked = clf.predict_proba(utt)
        if not ranked:
            out.append(UncertaintyScore(utt, "", 0.0, label, 0.0))
            continue
        top = ranked[0]
        req_conf = 0.0
        if label is not None:
            req_conf = next(
                (r.confidence for r in ranked if r.intent == label),
                0.0,
            )
        out.append(UncertaintyScore(
            utterance=utt,
            top_pred=top.intent,
            top_conf=float(top.confidence),
            requested=label,
            requested_conf=float(req_conf),
        ))
    return out


@dataclass
class ParaphraseBucket:
    skip: List[str]
    hard: List[str]
    suspect: List[str]


def bucket_paraphrases(
    clf,
    paraphrases_by_intent: Dict[str, List[str]],
    *,
    hard_conf_max: float = 0.6,
    dedupe_against: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, ParaphraseBucket]:
    """Sort paraphrases into skip / hard / suspect buckets per intent.

    - ``skip``    : predicted intent matches the requested one AND
      requested-label confidence ≥ ``hard_conf_max``.
    - ``hard``    : predicted intent matches but confidence is below
      threshold — keep, this is where the gradient lives.
    - ``suspect`` : predicted intent differs from the requested one —
      caller should verify before keeping (LLM drift risk).

    ``dedupe_against`` is an optional mapping (typically the current
    training set) used to drop paraphrases already present.
    """
    out: Dict[str, ParaphraseBucket] = {}
    for intent, paras in paraphrases_by_intent.items():
        seen = set((dedupe_against or {}).get(intent, []))
        b = ParaphraseBucket(skip=[], hard=[], suspect=[])
        for p in paras:
            if not p or p in seen:
                continue
            seen.add(p)
            scores = uncertainty_scores(clf, [p], [intent])[0]
            if scores.top_pred == intent and scores.requested_conf >= hard_conf_max:
                b.skip.append(p)
            elif scores.top_pred == intent:
                b.hard.append(p)
            else:
                b.suspect.append(p)
        out[intent] = b
    return out


def confusion_pairs(comparison_result, *, top_n: int = 5) -> List[Tuple[str, str, int]]:
    """Extract the most-confused intent pairs from a benchmark result.

    Reads the first row's ``confusion_matrix`` + ``labels``. Returns
    ``[(true_label, predicted_label, count), ...]`` sorted by count
    descending, diagonal excluded.

    Drives hard-negative pair mining: targeted LLM augmentation focused
    on the worst-confused intent pair has higher signal than uniform
    paraphrase generation.
    """
    if not comparison_result.rows:
        return []
    row = comparison_result.rows[0]
    cm = row.confusion_matrix
    labels = row.per_class_f1.keys() if hasattr(row, "per_class_f1") else None
    labels = list(labels) if labels is not None else []
    if not labels and hasattr(row, "labels"):
        labels = list(row.labels)
    if not labels and cm:
        labels = [str(i) for i in range(len(cm))]

    pairs: List[Tuple[str, str, int]] = []
    for i, row_counts in enumerate(cm):
        for j, count in enumerate(row_counts):
            if i == j or count <= 0:
                continue
            pairs.append((labels[i], labels[j], int(count)))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_n]


def disagreement_score(classifiers, utterance: str) -> float:
    """Query-by-committee entropy over top-1 predictions of multiple classifiers.

    ``classifiers`` is a sequence of fitted :class:`IntentClassifier`
    instances. Returns Shannon entropy (natural log) of the
    distribution of top-1 predictions; higher = more disagreement = a
    better candidate for manual labeling.
    """
    if not classifiers:
        return 0.0
    votes = Counter(c.predict(utterance).intent for c in classifiers)
    total = sum(votes.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for n in votes.values():
        p = n / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


__all__ = [
    "UncertaintyScore",
    "ParaphraseBucket",
    "uncertainty_scores",
    "bucket_paraphrases",
    "confusion_pairs",
    "disagreement_score",
]
