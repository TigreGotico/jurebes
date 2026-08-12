"""Shared helpers for canonical-benchmark training scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple

PORTFOLIO = [
    "nb_multinomial",
    "logreg",
    "linear_svc",
    "linear_svc_char",
    "bm25_logreg",
    "lsa_logreg",
    "autoencoder_logreg",
    "autoencoder_logreg_wide",
    "denoising_autoencoder_logreg",
    "label_guided_logreg",
    "label_guided_linear_svc",
    # Ablation channels — union variants that complement the lexical references.
    "union_skipgram_tfidf_logreg",
    "union_bm25_pos_logreg",
]

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
MODELS = HERE / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SIZE_LIMIT_BYTES = 5 * 1024 * 1024


def _fit_intent_classifier(baseline_name: str, X, y):
    """Fit an IntentClassifier with the given baseline on the dataset."""
    from collections import defaultdict
    from jurebes import IntentClassifier
    from jurebes.baselines import BASELINES

    clf = IntentClassifier(BASELINES.build(baseline_name))
    grouped = defaultdict(list)
    for utt, label in zip(X, y):
        grouped[label].append(utt)
    for label, samples in grouped.items():
        clf.add_intent(label, samples)
    clf.fit()
    return clf


def _predict_label(est, utterance: str) -> str:
    """Get the predicted label from either an IntentClassifier or a sklearn Pipeline."""
    if hasattr(est, "predict"):
        out = est.predict(utterance)
        if hasattr(out, "intent"):
            return out.intent
        # sklearn returns an array-like for a list, scalar for single string fitted on str input
        try:
            return out[0] if hasattr(out, "__len__") else out
        except Exception:
            return str(out)
    raise TypeError(f"don't know how to predict with {type(est)!r}")


def _warn_block(title: str, body: str) -> str:
    return f"> [!warning]\n> **{title}**\n>\n> {body}\n"


def run_pipeline(
    name: str,
    loader: Callable[[str], Tuple[List[str], List[str]]],
    *,
    cv: int = 5,
    n_iter: int = 10,
) -> str:
    """Run compare → stats → tune → test-eval → save. Return markdown report."""
    md = [f"# {name} training report\n"]

    try:
        X_train, y_train = loader("train")
        X_test, y_test = loader("test")
    except Exception as e:
        md.append(_warn_block(
            "Dataset load failed",
            f"`{type(e).__name__}: {e}` — could not fetch {name}. "
            "Verify HF dataset id and network access.",
        ))
        return "\n".join(md)

    md.append(f"- train size: **{len(X_train)}**")
    md.append(f"- test size: **{len(X_test)}**")
    md.append(f"- intents: **{len(set(y_train))}**")
    md.append("")

    from jurebes.benchmark import compare, to_markdown

    md.append(f"## Compare with {cv}-fold CV\n")
    try:
        result = compare(PORTFOLIO, X_train, y_train, k=cv)
        md.append(to_markdown(result, sort_by="macro_f1"))
        md.append("")
    except Exception as e:
        md.append(_warn_block("compare() failed", f"`{type(e).__name__}: {e}`"))
        return "\n".join(md)

    # Friedman + Nemenyi
    from jurebes.benchmark.stats import critical_difference, friedman_nemenyi

    fold_scores = {
        baseline_name: scores["f1_macro"]
        for baseline_name, scores in result.fold_scores_by_baseline.items()
    }
    md.append("## Friedman + Nemenyi\n")
    try:
        fr = friedman_nemenyi(fold_scores)
        md.append(f"- Friedman p-value: **{fr.pvalue:.4f}**")
        md.append(f"- reject H0 (all baselines equal): **{fr.reject_null}**")
        cd = critical_difference(fold_scores)
        md.append("\n```")
        md.append(cd.to_ascii())
        md.append("```\n")
        winner = min(fold_scores, key=lambda n: cd.mean_ranks[n])
    except Exception as e:
        md.append(_warn_block("stats failed", f"`{type(e).__name__}: {e}`"))
        # fall back to mean f1
        winner = max(fold_scores, key=lambda n: sum(fold_scores[n]) / len(fold_scores[n]))

    md.append(f"**winning baseline:** `{winner}`\n")

    # Random search tuning
    md.append(f"## Random search ({n_iter} iter) on winner\n")
    try:
        from jurebes.search import search, spaces

        try:
            space = spaces.for_baseline(winner)
        except Exception:
            space = None

        if space:
            tuned = search(
                winner, space, X_train, y_train,
                backend="random", n_iter=n_iter, cv=cv, scoring="f1_macro",
            )
            md.append(f"- best CV f1_macro: **{tuned.best_score:.4f}**")
            md.append(f"- best params: `{tuned.best_params}`\n")
            est = tuned.best_estimator
        else:
            md.append(_warn_block(
                "no search space",
                f"`spaces.for_baseline({winner!r})` returned nothing; "
                "evaluating untuned default instead.",
            ))
            est = _fit_intent_classifier(winner, X_train, y_train)
    except Exception as e:
        md.append(_warn_block("search failed", f"`{type(e).__name__}: {e}`"))
        est = _fit_intent_classifier(winner, X_train, y_train)

    # Test evaluation
    md.append("## Test-set evaluation\n")
    try:
        from sklearn.metrics import accuracy_score, f1_score

        preds = [_predict_label(est, x) for x in X_test]
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro")
        md.append(f"- test accuracy: **{acc:.4f}**")
        md.append(f"- test macro-F1: **{f1m:.4f}**\n")
    except Exception as e:
        md.append(_warn_block("test-set eval failed", f"`{type(e).__name__}: {e}`"))

    # Save
    md.append("## Artifact\n")
    out_path = MODELS / f"{name}_{winner}.joblib"
    try:
        if hasattr(est, "save"):
            est.save(str(out_path))
        else:
            import joblib
            joblib.dump(est, out_path)
        size = os.path.getsize(out_path)
        size_mb = size / (1024 * 1024)
        if size > SIZE_LIMIT_BYTES:
            md.append(
                f"- model too large to commit, size {size_mb:.2f} MB "
                f"(limit 5 MB) — kept locally at `{out_path.name}`, excluded "
                "via `.gitignore`."
            )
        else:
            md.append(f"- saved `{out_path.name}` ({size_mb:.2f} MB)")
    except Exception as e:
        md.append(_warn_block("save failed", f"`{type(e).__name__}: {e}`"))

    return "\n".join(md)


def write_report(name: str, content: str) -> Path:
    path = REPORTS / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path
