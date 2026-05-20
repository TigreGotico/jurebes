"""Three-bucket active-learning loop driven by jurebes.

Pattern:
    1. For each intent, ask an LLM for N paraphrases.
    2. Run jurebes on every paraphrase.
    3. Bucket by (predicted == requested, confidence of requested intent):
       - SKIP    : correct & high confidence  — model already knows it.
       - HARD    : correct & low confidence   — uncertainty signal, keep.
       - SUSPECT : wrong prediction           — judge before keeping.
    4. Retrain jurebes on (original + HARD + judge-approved SUSPECT).
    5. Re-evaluate on a FROZEN held-out set; stop when macro-F1 plateaus.

The judge step is optional and pluggable: pass any callable
``judge(intent, utterance) -> bool``. The default judge calls the LLM
again with a yes/no preservation prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sklearn.metrics import f1_score

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES


@dataclass
class AugmentRound:
    n_generated: int = 0
    n_skip: int = 0
    n_hard: int = 0
    n_suspect: int = 0
    n_judged_keep: int = 0
    n_judged_drop: int = 0
    eval_macro_f1: Optional[float] = None
    added_samples: Dict[str, List[str]] = field(default_factory=dict)


def _proba_for(clf: IntentClassifier, utt: str, intent: str) -> Tuple[str, float, float]:
    """Return (top1_intent, top1_conf, conf_for_requested_intent)."""
    ranked = clf.predict_proba(utt)
    if not ranked:
        return "", 0.0, 0.0
    top = ranked[0]
    for r in ranked:
        if r.intent == intent:
            return top.intent, top.confidence, r.confidence
    return top.intent, top.confidence, 0.0


def augment_loop(
    intent_samples: Dict[str, List[str]],
    paraphrase_fn: Callable[..., List[str]],
    *,
    eval_X: List[str],
    eval_y: List[str],
    baseline: str = "logreg",
    n_per_intent: int = 20,
    n_rounds: int = 5,
    hard_conf_max: float = 0.6,
    judge_fn: Optional[Callable[[str, str], bool]] = None,
    seeds_per_intent: int = 3,
    early_stop_patience: int = 2,
    **paraphrase_kwargs,
) -> Tuple[IntentClassifier, List[AugmentRound]]:
    """Run the three-bucket augmentation loop.

    Returns the final fitted classifier and per-round diagnostics.
    """
    samples = {k: list(v) for k, v in intent_samples.items()}
    history: List[AugmentRound] = []
    best_f1 = -1.0
    plateau = 0

    for round_idx in range(n_rounds):
        clf = IntentClassifier(BASELINES.build(baseline))
        for intent, vals in samples.items():
            clf.add_intent(intent, vals)
        clf.fit()

        f1 = f1_score(eval_y, [clf.predict(x).intent for x in eval_X],
                      average="macro", zero_division=0)
        diag = AugmentRound(eval_macro_f1=f1)
        history.append(diag)

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            plateau = 0
        else:
            plateau += 1
            if plateau >= early_stop_patience:
                break

        if round_idx == n_rounds - 1:
            break

        for intent, seeds in samples.items():
            try:
                paras = paraphrase_fn(
                    intent, seeds[:seeds_per_intent],
                    n=n_per_intent, **paraphrase_kwargs,
                )
            except Exception:
                continue
            diag.n_generated += len(paras)
            for p in paras:
                if not p or p in samples[intent]:
                    continue
                top, top_conf, req_conf = _proba_for(clf, p, intent)
                if top == intent and req_conf >= hard_conf_max:
                    diag.n_skip += 1
                elif top == intent:
                    diag.n_hard += 1
                    samples[intent].append(p)
                    diag.added_samples.setdefault(intent, []).append(p)
                else:
                    diag.n_suspect += 1
                    keep = bool(judge_fn(intent, p)) if judge_fn else False
                    if keep:
                        diag.n_judged_keep += 1
                        samples[intent].append(p)
                        diag.added_samples.setdefault(intent, []).append(p)
                    else:
                        diag.n_judged_drop += 1

    return clf, history
