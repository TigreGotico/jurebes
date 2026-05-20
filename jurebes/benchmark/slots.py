"""Slot-tagger benchmark harness — compare strategies on a labelled set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class SlotRunResult:
    name: str
    slot_precision: float
    slot_recall: float
    slot_f1: float
    exact_match: float
    n_test: int
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class SlotComparisonResult:
    rows: List[SlotRunResult]
    scoring: Tuple[str, ...] = ()

    def to_markdown(self, *, precision: int = 4) -> str:
        cols = ["tagger", "slot_precision", "slot_recall", "slot_f1", "exact_match", "n_test"]
        fmt = f"{{:.{precision}f}}"
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for r in self.rows:
            lines.append(
                "| "
                + " | ".join([
                    r.name,
                    fmt.format(r.slot_precision),
                    fmt.format(r.slot_recall),
                    fmt.format(r.slot_f1),
                    fmt.format(r.exact_match),
                    str(r.n_test),
                ])
                + " |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scoring": list(self.scoring),
            "rows": [r.__dict__ for r in self.rows],
        }


def _score(
    preds: Sequence[Dict[str, str]],
    gold: Sequence[Dict[str, str]],
) -> Tuple[float, float, float, float]:
    tp = fp = fn = 0
    exact = 0
    for p, g in zip(preds, gold):
        if p == g:
            exact += 1
        gset = {(k, v.lower()) for k, v in g.items()}
        pset = {(k, v.lower()) for k, v in p.items()}
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    em = exact / len(gold) if gold else 0.0
    return precision, recall, f1, em


def compare_taggers(
    tagger_names: Iterable[str],
    intent_samples: Dict[str, List[str]],
    entity_samples: Dict[str, List[str]],
    test_utterances: Sequence[Tuple[str, Dict[str, str]]],
    *,
    scoring: Tuple[str, ...] = ("slot_f1", "exact_match"),
) -> SlotComparisonResult:
    """Compare slot taggers on a labelled test set.

    Args:
        tagger_names: registry names to evaluate (see :data:`jurebes.slots.TAGGERS`).
        intent_samples: mapping of intent → list of template-bearing utterances.
        entity_samples: mapping of entity → list of gazetteer values.
        test_utterances: list of ``(utterance, gold_slot_dict)`` pairs.
        scoring: surfaced metric names (informational; all metrics are computed).
    """
    from jurebes.slots import TAGGERS

    rows: List[SlotRunResult] = []
    gold = [g for _, g in test_utterances]
    utts = [u for u, _ in test_utterances]
    for name in tagger_names:
        tg = TAGGERS.build(name)
        for ent_name, ent_vals in entity_samples.items():
            if hasattr(tg, "add_entity"):
                tg.add_entity(ent_name, ent_vals)
        for intent_name, samples in intent_samples.items():
            if hasattr(tg, "add_intent"):
                tg.add_intent(intent_name, samples)
        try:
            tg.fit(intent_samples)
        except TypeError:
            tg.fit()
        preds = [tg.predict(u) or {} for u in utts]
        p, r, f1, em = _score(preds, gold)
        rows.append(SlotRunResult(
            name=name,
            slot_precision=p,
            slot_recall=r,
            slot_f1=f1,
            exact_match=em,
            n_test=len(test_utterances),
        ))
    return SlotComparisonResult(rows=rows, scoring=tuple(scoring))
