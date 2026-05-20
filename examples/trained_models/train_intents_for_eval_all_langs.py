"""Benchmark intents-for-eval across every supported language.

Calls :func:`train_intents_for_eval.run` once per language, writes a
per-language markdown report, then aggregates headline intent and slot
metrics into ``reports/intents_for_eval_summary.md``.

Requires: pip install jurebes[hf,slots-crf]
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from train_intents_for_eval import run as run_one

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)

LANGS = (
    "en-US", "pt-PT", "pt-BR", "es-ES", "fr-FR", "de-DE",
    "it-IT", "nl-NL", "ca-ES", "gl-ES", "da-DK", "eu-ES",
)


def _extract_best_intent_row(report: str):
    """Pick the first data row from the intent-classification table."""
    m = re.search(r"## Intent classification.*?\| baseline \| accuracy \| macro_f1 \|\s*\|[^\n]+\|\s*((?:\|[^\n]+\|\s*)+)", report, re.S)
    if not m:
        return None, None, None
    first_row = m.group(1).strip().splitlines()[0]
    cells = [c.strip() for c in first_row.strip("|").split("|")]
    if len(cells) < 3:
        return None, None, None
    return cells[0], cells[1], cells[2]


def _extract_best_slot_row(report: str):
    """Pick the highest exact-match row from the compare_taggers table."""
    m = re.search(r"## Slot extraction.*?\|\s*tagger\s*\|[^\n]+\|\s*\|[^\n]+\|\s*((?:\|[^\n]+\|\s*)+)", report, re.S)
    if not m:
        return None, None, None, None
    rows = []
    for line in m.group(1).strip().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4:
            rows.append(cells)
    if not rows:
        return None, None, None, None
    # row layout: tagger | precision | recall | f1 | exact_match (varies)
    def _as_float(s):
        try:
            return float(s)
        except ValueError:
            return -1.0
    rows.sort(key=lambda r: _as_float(r[-1] if len(r) >= 5 else r[-1]), reverse=True)
    best = rows[0]
    return best[0], best[1] if len(best) > 1 else "-", best[3] if len(best) > 3 else "-", best[-1]


def main():
    summary_rows = []
    timings = []
    for lang in LANGS:
        t0 = time.perf_counter()
        try:
            report = run_one(lang)
            ok = True
        except Exception as e:
            report = f"# intents-for-eval ({lang}) — failed\n\n`{type(e).__name__}: {e}`"
            ok = False
        elapsed = time.perf_counter() - t0
        timings.append((lang, elapsed))

        out = REPORTS / f"intents_for_eval_{lang}.md"
        out.write_text(report, encoding="utf-8")
        print(f"[{lang}] wrote {out.name} ({elapsed:.1f}s) ok={ok}")

        intent_name, intent_acc, intent_f1 = _extract_best_intent_row(report)
        slot_name, slot_prec, slot_f1, slot_em = _extract_best_slot_row(report)
        summary_rows.append({
            "lang": lang,
            "intent_top1_baseline": intent_name or "ERR",
            "intent_accuracy": intent_acc or "-",
            "intent_macro_f1": intent_f1 or "-",
            "slot_top_tagger": slot_name or "ERR",
            "slot_precision": slot_prec or "-",
            "slot_f1": slot_f1 or "-",
            "slot_exact_match": slot_em or "-",
            "wall_s": f"{elapsed:.1f}",
        })

    # Consolidated summary
    headers = [
        "lang", "intent_top1_baseline", "intent_accuracy", "intent_macro_f1",
        "slot_top_tagger", "slot_precision", "slot_f1", "slot_exact_match", "wall_s",
    ]
    lines = ["# intents-for-eval — multi-language summary\n"]
    lines.append("Headline of the best intent baseline and best slot tagger per language.")
    lines.append("Full per-language reports under `intents_for_eval_<lang>.md`.\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in summary_rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    summary_path = REPORTS / "intents_for_eval_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary written to: {summary_path}")


if __name__ == "__main__":
    main()
