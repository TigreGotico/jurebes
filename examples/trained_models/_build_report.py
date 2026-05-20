"""Parse per-dataset reports and generate figures + dataframes for REPORT.md.

Reads every Markdown report under reports/ and emits:
- figures/*.png         — per-chart matplotlib outputs
- _report_data.json     — structured data extracted from the reports
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(exist_ok=True)


# ── parsing helpers ─────────────────────────────────────────────────


_TABLE_HEADER = re.compile(r"^\|\s*[^|]+\|")
_BAR_LINE = re.compile(r"^\|\s*[-:\s|]+\|\s*$")


def _parse_md_table(text: str, after_header: str) -> List[Dict[str, str]]:
    """Locate a markdown table that follows the given section header line."""
    idx = text.find(after_header)
    if idx == -1:
        return []
    sub = text[idx:]
    lines = sub.splitlines()
    rows: List[Dict[str, str]] = []
    headers: List[str] = []
    state = "search"
    for line in lines:
        if state == "search":
            if _TABLE_HEADER.match(line):
                headers = [c.strip() for c in line.strip("|").split("|")]
                state = "bar"
        elif state == "bar":
            if _BAR_LINE.match(line):
                state = "rows"
            else:
                state = "search"
        elif state == "rows":
            if not _TABLE_HEADER.match(line):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
    return rows


def _as_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _parse_canonical(name: str) -> Dict[str, Any]:
    text = (REPORTS / f"{name}.md").read_text()
    train = re.search(r"train size.*?(\d+)", text)
    test = re.search(r"test size.*?(\d+)", text)
    intents = re.search(r"intents.*?(\d+)", text)
    rows = _parse_md_table(text, "## Compare with")
    final_acc = re.search(r"test accuracy:.*?\*\*([\d.]+)\*\*", text)
    final_f1 = re.search(r"test macro-F1:.*?\*\*([\d.]+)\*\*", text)
    return {
        "name": name,
        "train_size": int(train.group(1)) if train else None,
        "test_size": int(test.group(1)) if test else None,
        "n_intents": int(intents.group(1)) if intents else None,
        "rows": rows,
        "final_test_accuracy": _as_float(final_acc.group(1)) if final_acc else float("nan"),
        "final_test_macro_f1": _as_float(final_f1.group(1)) if final_f1 else float("nan"),
    }


def _parse_ife(lang: str) -> Dict[str, Any]:
    text = (REPORTS / f"intents_for_eval_{lang}.md").read_text()
    intents = re.search(r"- intents:.*?\*\*(\d+)\*\*", text)
    templates = re.search(r"templates:.*?\*\*(\d+)\*\*.*?\*\*(\d+)\*\*", text)
    test = re.search(r"test utterances:.*?\*\*(\d+)\*\*", text)
    intent_rows = _parse_md_table(text, "## Intent classification")
    slot_rows = _parse_md_table(text, "## Slot extraction")
    return {
        "lang": lang,
        "n_intents": int(intents.group(1)) if intents else None,
        "n_templates_raw": int(templates.group(1)) if templates else None,
        "n_templates_expanded": int(templates.group(2)) if templates else None,
        "n_test": int(test.group(1)) if test else None,
        "intent_rows": intent_rows,
        "slot_rows": slot_rows,
    }


# ── extract ─────────────────────────────────────────────────────────


canonical = {n: _parse_canonical(n) for n in ("snips", "banking77", "clinc")}
LANGS = ["en-US", "pt-PT", "pt-BR", "es-ES", "fr-FR", "de-DE",
         "it-IT", "nl-NL", "ca-ES", "gl-ES", "da-DK", "eu-ES"]
ife = {lang: _parse_ife(lang) for lang in LANGS}


# ── plot 1: cross-dataset baseline accuracy ─────────────────────────


def _row_metric(rows: List[Dict[str, str]], name_col: str, metric_col: str, baseline: str) -> float:
    for r in rows:
        if r.get(name_col) == baseline:
            return _as_float(r.get(metric_col, "nan"))
    return float("nan")


PORTFOLIO = ["nb_multinomial", "logreg", "linear_svc",
             "linear_svc_char", "lsa_logreg", "autoencoder_logreg"]
DATASETS_CV = ["snips", "banking77", "clinc"]

# CV accuracy table from each canonical report (column "accuracy")
acc_matrix = np.full((len(PORTFOLIO), len(DATASETS_CV)), np.nan)
for i, b in enumerate(PORTFOLIO):
    for j, ds in enumerate(DATASETS_CV):
        acc_matrix[i, j] = _row_metric(canonical[ds]["rows"], "baseline", "accuracy", b)

fig, ax = plt.subplots(figsize=(9, 5))
xs = np.arange(len(PORTFOLIO))
width = 0.27
for j, ds in enumerate(DATASETS_CV):
    ax.bar(xs + j * width, acc_matrix[:, j], width, label=ds)
ax.set_xticks(xs + width)
ax.set_xticklabels(PORTFOLIO, rotation=30, ha="right")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Portfolio accuracy across canonical NLU benchmarks")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "01_portfolio_across_datasets.png", dpi=120)
plt.close(fig)


# ── plot 2: per-language intent accuracy ────────────────────────────


# Use the top baseline per language (whatever ranks first in each report)
def _top_intent(lang_data) -> tuple[str, float, float]:
    rows = lang_data["intent_rows"]
    if not rows:
        return ("", float("nan"), float("nan"))
    sorted_rows = sorted(rows, key=lambda r: _as_float(r.get("accuracy", "0")), reverse=True)
    top = sorted_rows[0]
    return (top.get("baseline", ""),
            _as_float(top.get("accuracy", "nan")),
            _as_float(top.get("macro_f1", "nan")))

top_by_lang = {lang: _top_intent(ife[lang]) for lang in LANGS}
top_accs = np.array([top_by_lang[l][1] for l in LANGS])

fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(len(LANGS))
ax.bar(xs, top_accs, color="steelblue")
for x, acc in zip(xs, top_accs):
    ax.text(x, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(LANGS, rotation=30, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Best baseline test accuracy")
ax.set_title("intents-for-eval — best baseline per language")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "02_ife_per_language_intent.png", dpi=120)
plt.close(fig)


# ── plot 3: slot tagger × language heatmap (exact_match) ────────────


TAGGERS = ["dictionary", "template", "sklearn_iob", "knn", "hybrid", "crf"]
heat = np.full((len(TAGGERS), len(LANGS)), np.nan)
for j, lang in enumerate(LANGS):
    rows = ife[lang]["slot_rows"]
    for r in rows:
        name = r.get("tagger", "")
        if name in TAGGERS:
            heat[TAGGERS.index(name), j] = _as_float(r.get("exact_match", "nan"))

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
ax.set_xticks(range(len(LANGS)))
ax.set_xticklabels(LANGS, rotation=30, ha="right")
ax.set_yticks(range(len(TAGGERS)))
ax.set_yticklabels(TAGGERS)
for i in range(len(TAGGERS)):
    for j in range(len(LANGS)):
        v = heat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.7 else "black", fontsize=8)
ax.set_title("Slot-tagger exact-match by language (intents-for-eval)")
fig.colorbar(im, ax=ax, label="exact-match")
fig.tight_layout()
fig.savefig(FIGS / "03_slot_tagger_by_language.png", dpi=120)
plt.close(fig)


# ── plot 4: latency vs accuracy Pareto on banking77 ─────────────────


b77_rows = canonical["banking77"]["rows"]
xs = []
ys = []
names = []
for r in b77_rows:
    name = r.get("baseline", "")
    p95 = _as_float(r.get("p95_ms_pooled", "nan"))
    acc = _as_float(r.get("accuracy", "nan"))
    if np.isfinite(p95) and np.isfinite(acc):
        xs.append(p95)
        ys.append(acc)
        names.append(name)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(xs, ys, s=80, color="darkred", zorder=3)
for x, y, n in zip(xs, ys, names):
    ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("p95 prediction latency (ms, pooled)")
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Latency vs accuracy Pareto — BANKING77")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "04_latency_vs_accuracy.png", dpi=120)
plt.close(fig)


# ── plot 5: intents-for-eval intent classification, all baselines × all langs ─


IFE_BASELINES = ["nb_multinomial", "logreg", "linear_svc",
                 "linear_svc_char", "logreg_char", "ovr_linear_svc", "voting_soft"]
ife_matrix = np.full((len(IFE_BASELINES), len(LANGS)), np.nan)
for i, b in enumerate(IFE_BASELINES):
    for j, lang in enumerate(LANGS):
        ife_matrix[i, j] = _row_metric(ife[lang]["intent_rows"], "baseline", "accuracy", b)

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(ife_matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
ax.set_xticks(range(len(LANGS)))
ax.set_xticklabels(LANGS, rotation=30, ha="right")
ax.set_yticks(range(len(IFE_BASELINES)))
ax.set_yticklabels(IFE_BASELINES)
for i in range(len(IFE_BASELINES)):
    for j in range(len(LANGS)):
        v = ife_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
ax.set_title("intents-for-eval test accuracy: baseline × language")
fig.colorbar(im, ax=ax, label="accuracy")
fig.tight_layout()
fig.savefig(FIGS / "05_ife_baseline_by_language.png", dpi=120)
plt.close(fig)


# ── persist structured data for the prose-writer ───────────────────


payload: Dict[str, Any] = {
    "canonical": canonical,
    "ife": ife,
    "top_by_lang": {l: list(v) for l, v in top_by_lang.items()},
    "portfolio": PORTFOLIO,
    "ife_baselines": IFE_BASELINES,
    "taggers": TAGGERS,
    "langs": LANGS,
    "datasets_cv": DATASETS_CV,
    "acc_matrix": acc_matrix.tolist(),
    "ife_matrix": ife_matrix.tolist(),
    "slot_heat": heat.tolist(),
    "latency_xs": xs,
    "latency_ys": ys,
    "latency_names": names,
}
(REPORTS / "_report_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"figures saved to {FIGS}")
print(f"data dump: {REPORTS / '_report_data.json'}")
