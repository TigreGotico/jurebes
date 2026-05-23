"""Benchmark jurebes against OpenVoiceOS/intents-for-eval.

Trains every BASELINE in the linear portfolio + the autoencoder family on
the Padatious-style templates and evaluates intent classification on the
held-out test set. Then evaluates every TAGGER against the gold slot
annotations using ``compare_taggers``.

Requires: pip install jurebes[hf,slots-crf]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
MODELS = HERE / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def _md_table(rows, headers):
    sep = "| " + " | ".join(headers) + " |"
    bar = "|" + "|".join(["---"] * len(headers)) + "|"
    out = [sep, bar]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def run(lang: str = "en-US") -> str:
    from sklearn.metrics import accuracy_score, f1_score

    from jurebes import IntentClassifier
    from jurebes.baselines import BASELINES
    from jurebes.datasets.canonical import load_intents_for_eval
    from jurebes.benchmark.slots import compare_taggers
    from jurebes.slots import TAGGERS

    md = [f"# intents-for-eval ({lang}) training report\n"]

    try:
        data = load_intents_for_eval(lang)
    except Exception as e:
        return f"# intents-for-eval ({lang}) — load failed\n\n`{type(e).__name__}: {e}`"

    intent_samples = data["intent_samples"]
    template_samples = data["template_samples"]
    entity_samples = data["entity_samples"]
    test_all = data["test"]
    test = [r for r in test_all if r.get("expected_intent")]
    ood_count = len(test_all) - len(test)

    md.append(f"- intents: **{len(intent_samples)}**")
    md.append(f"- templates: **{sum(len(v) for v in template_samples.values())}** "
              f"→ **{sum(len(v) for v in intent_samples.values())}** after slot expansion")
    md.append(f"- entities: **{len(entity_samples)}**")
    md.append(f"- test utterances: **{len(test)}** in-domain (+{ood_count} OOD rows excluded)")
    md.append("")

    # ── intent classification ────────────────────────────────────────
    md.append("## Intent classification\n")
    md.append("Train on Padatious-style templates, evaluate top-1 intent on the test split.\n")

    test_utts = [r["utterance"] for r in test]
    test_gold = [r["expected_intent"] for r in test]

    portfolio = [
        "nb_multinomial", "logreg", "linear_svc", "linear_svc_char",
        "logreg_char", "ovr_linear_svc", "voting_soft",
        "bm25_logreg",
        "autoencoder_logreg", "denoising_autoencoder_logreg",
        "label_guided_logreg", "label_guided_linear_svc",
        "union_skipgram_tfidf_logreg", "union_bm25_pos_logreg",
    ]
    rows = []
    for name in portfolio:
        try:
            clf = IntentClassifier(BASELINES.build(name))
            for intent_id, samples in intent_samples.items():
                clf.add_intent(intent_id, samples)
            clf.fit()
            preds = [clf.predict(u).intent for u in test_utts]
            acc = accuracy_score(test_gold, preds)
            f1m = f1_score(test_gold, preds, average="macro", zero_division=0)
            rows.append([name, f"{acc:.4f}", f"{f1m:.4f}"])
        except Exception as e:
            rows.append([name, f"ERR: {type(e).__name__}", "-"])

    rows.sort(key=lambda r: r[1], reverse=True)
    md.append(_md_table(rows, ["baseline", "accuracy", "macro_f1"]))
    md.append("")
    winner = rows[0][0] if rows and rows[0][1] != "-" else "logreg"
    md.append(f"**winning baseline (intent):** `{winner}`\n")

    # ── slot tagging ────────────────────────────────────────────────
    md.append("## Slot extraction\n")
    md.append("Train each tagger on the same templates + entity gazetteer; "
              "evaluate against the gold `expected_slots` on the test split.\n")

    test_slot_pairs = [
        (
            r["utterance"],
            {k: str(v) for k, v in (r.get("expected_slots") or {}).items() if v is not None},
        )
        for r in test
    ]

    tagger_names = ["dictionary", "template", "sklearn_iob", "knn", "hybrid"]
    try:
        import sklearn_crfsuite  # noqa: F401
        tagger_names.append("crf")
    except ImportError:
        md.append("> CRF tagger skipped: install `jurebes[slots-crf]` to include it.\n")

    try:
        result = compare_taggers(
            tagger_names,
            template_samples,
            entity_samples,
            test_slot_pairs,
        )
        md.append(result.to_markdown())
    except Exception as e:
        md.append(f"> [!warning]\n> **compare_taggers() failed**\n>\n> `{type(e).__name__}: {e}`")

    # ── per-domain intent breakdown ──────────────────────────────────
    md.append("\n## Per-domain intent accuracy\n")
    md.append(f"Using the winning baseline `{winner}`.")
    by_domain_true = defaultdict(list)
    by_domain_pred = defaultdict(list)
    try:
        clf = IntentClassifier(BASELINES.build(winner))
        for intent_id, samples in intent_samples.items():
            clf.add_intent(intent_id, samples)
        clf.fit()
        for r, p in zip(test, [clf.predict(u).intent for u in test_utts]):
            dom = r.get("domain") or "(unknown)"
            by_domain_true[dom].append(r["expected_intent"])
            by_domain_pred[dom].append(p)
        domain_rows = []
        for d in sorted(by_domain_true):
            acc = accuracy_score(by_domain_true[d], by_domain_pred[d])
            domain_rows.append([d, len(by_domain_true[d]), f"{acc:.4f}"])
        md.append(_md_table(domain_rows, ["domain", "n", "accuracy"]))
    except Exception as e:
        md.append(f"> per-domain breakdown failed: `{type(e).__name__}: {e}`")

    return "\n".join(md)


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en-US"
    report = run(lang)
    path = REPORTS / f"intents_for_eval_{lang}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
