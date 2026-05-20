"""Follow-up training on BANKING77 and SNIPS for the new AE variants.

Answers the question raised by REPORT.md: does the label-guided
supervised bottleneck recover the gap that vanilla autoencoders leave
on high-class-count intent classification?

Compares:
- autoencoder_logreg        (default, now hidden_layer_sizes='auto')
- autoencoder_logreg_wide   (256/128/256 layers)
- denoising_autoencoder_logreg
- label_guided_logreg       (supervised bottleneck — the new port)
- label_guided_linear_svc
- linear_svc_char           (current overall winner, for reference)

Writes ae_followup_<dataset>.md per dataset.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.datasets.canonical import load_banking77, load_snips


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)

PORTFOLIO = [
    "autoencoder_logreg",
    "autoencoder_logreg_wide",
    "denoising_autoencoder_logreg",
    "label_guided_logreg",
    "label_guided_linear_svc",
    "linear_svc_char",
]


def _train_and_eval(name, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    clf = IntentClassifier(BASELINES.build(name))
    grouped = defaultdict(list)
    for x, lbl in zip(X_tr, y_tr):
        grouped[lbl].append(x)
    for lbl, samples in grouped.items():
        clf.add_intent(lbl, samples)
    try:
        clf.fit()
    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {e}", "train_s": time.perf_counter() - t0}
    train_s = time.perf_counter() - t0
    preds = [clf.predict(x).intent for x in X_te]
    return {
        "name": name,
        "accuracy": accuracy_score(y_te, preds),
        "macro_f1": f1_score(y_te, preds, average="macro", zero_division=0),
        "train_s": train_s,
    }


def _write_report(dataset, rows, X_tr, X_te, n_intents):
    md = [f"# {dataset} AE follow-up\n"]
    md.append(f"- train size: **{len(X_tr)}**")
    md.append(f"- test size: **{len(X_te)}**")
    md.append(f"- intents: **{n_intents}**")
    md.append("")
    md.append("| baseline | test accuracy | test macro_f1 | train (s) |")
    md.append("| --- | ---: | ---: | ---: |")
    for r in rows:
        if "error" in r:
            md.append(f"| `{r['name']}` | ERR | {r['error']} | {r['train_s']:.1f} |")
        else:
            md.append(f"| `{r['name']}` | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['train_s']:.1f} |")
    out = REPORTS / f"ae_followup_{dataset}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(md))


def run_dataset(name, loader):
    X_tr, y_tr = loader("train")
    X_te, y_te = loader("test")
    rows = []
    for b in PORTFOLIO:
        print(f"[{name}] training {b}...")
        r = _train_and_eval(b, X_tr, y_tr, X_te, y_te)
        rows.append(r)
        if "error" in r:
            print(f"   ERR: {r['error']}")
        else:
            print(f"   acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}  train={r['train_s']:.1f}s")
    rows.sort(key=lambda r: r.get("accuracy", -1), reverse=True)
    _write_report(name, rows, X_tr, X_te, len(set(y_tr)))


def main():
    run_dataset("snips", load_snips)
    run_dataset("banking77", load_banking77)


if __name__ == "__main__":
    main()
