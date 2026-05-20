"""Benchmark jurebes against OpenVoiceOS/massive-templates.

Same shape as ``train_intents_for_eval.py`` but for MASSIVE's 60-intent
inventory across 51 languages. Each language has ~13.5k Padatious-style
templates and ~3k test utterances.

Requires: pip install jurebes[hf,slots-crf]

Run for one language::

    python examples/trained_models/train_massive_templates.py en-US

Or sweep every language via ``train_massive_templates_all_langs.py``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)


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
    from jurebes.benchmark.slots import compare_taggers
    from jurebes.datasets.canonical import load_massive_templates

    md = [f"# massive-templates ({lang}) training report\n"]

    try:
        data = load_massive_templates(lang)
    except Exception as e:
        return f"# massive-templates ({lang}) — load failed\n\n`{type(e).__name__}: {e}`"

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

    md.append("## Intent classification\n")
    md.append("Train on Padatious-style templates, evaluate top-1 intent on the test split.\n")

    test_utts = [r["utterance"] for r in test]
    test_gold = [r["expected_intent"] for r in test]

    portfolio = [
        "nb_multinomial", "logreg", "linear_svc", "linear_svc_char",
        "logreg_char", "ovr_linear_svc", "voting_soft",
        "autoencoder_logreg", "denoising_autoencoder_logreg",
        "label_guided_logreg", "label_guided_linear_svc",
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

    md.append("## Slot extraction\n")
    md.append("Train each tagger on raw templates + entity gazetteer; "
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
            tagger_names, template_samples, entity_samples, test_slot_pairs,
        )
        md.append(result.to_markdown())
    except Exception as e:
        md.append(f"> [!warning]\n> **compare_taggers() failed**\n>\n> `{type(e).__name__}: {e}`")

    return "\n".join(md)


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en-US"
    report = run(lang)
    path = REPORTS / f"massive_templates_{lang}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport written to: {path}")


if __name__ == "__main__":
    main()
