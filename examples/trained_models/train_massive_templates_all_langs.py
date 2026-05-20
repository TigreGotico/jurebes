"""Sweep ``train_massive_templates.run`` across all 51 MASSIVE languages.

Writes one per-language report + a consolidated summary mirroring the
intents-for-eval orchestrator.

This will take a while. ~10-20 minutes per language on commodity CPU,
so plan for 8-15 hours total. Run in a terminal multiplexer.

Requires: pip install jurebes[hf,slots-crf]
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from train_massive_templates import run as run_one


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPORTS.mkdir(exist_ok=True)

# all 51 configs reported by HF list_configs
LANGS = (
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "he-IL",
    "hi-IN", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID",
    "ka-GE", "km-KH", "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY",
    "my-MM", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL",
    "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR",
    "ur-PK", "vi-VN", "zh-CN", "zh-TW",
)


def _extract_best_intent_row(report: str):
    m = re.search(r"## Intent classification.*?\| baseline \| accuracy \| macro_f1 \|\s*\|[^\n]+\|\s*((?:\|[^\n]+\|\s*)+)", report, re.S)
    if not m:
        return None, None, None
    first = m.group(1).strip().splitlines()[0]
    cells = [c.strip() for c in first.strip("|").split("|")]
    if len(cells) < 3:
        return None, None, None
    return cells[0], cells[1], cells[2]


def _extract_best_slot_row(report: str):
    m = re.search(r"## Slot extraction.*?\|\s*tagger\s*\|[^\n]+\|\s*\|[^\n]+\|\s*((?:\|[^\n]+\|\s*)+)", report, re.S)
    if not m:
        return None, None, None, None
    rows = []
    for line in m.group(1).strip().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 5:
            rows.append(cells)
    if not rows:
        return None, None, None, None

    def _as_float(s):
        try:
            return float(s)
        except ValueError:
            return -1.0

    rows.sort(key=lambda r: _as_float(r[4]), reverse=True)
    best = rows[0]
    return best[0], best[1], best[3], best[4]


def main():
    summary_rows = []
    for lang in LANGS:
        t0 = time.perf_counter()
        try:
            report = run_one(lang)
            ok = True
        except Exception as e:
            report = f"# massive-templates ({lang}) — failed\n\n`{type(e).__name__}: {e}`"
            ok = False
        dt = time.perf_counter() - t0
        out = REPORTS / f"massive_templates_{lang}.md"
        out.write_text(report, encoding="utf-8")
        print(f"[{lang}] wrote {out.name} ({dt:.1f}s) ok={ok}")

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
            "wall_s": f"{dt:.1f}",
        })

    headers = [
        "lang", "intent_top1_baseline", "intent_accuracy", "intent_macro_f1",
        "slot_top_tagger", "slot_precision", "slot_f1", "slot_exact_match", "wall_s",
    ]
    lines = ["# massive-templates — 51-language summary\n"]
    lines.append("Headline of the best intent baseline and best slot tagger per language.")
    lines.append("Full per-language reports under `massive_templates_<lang>.md`.\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in summary_rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    (REPORTS / "massive_templates_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary written to: {REPORTS / 'massive_templates_summary.md'}")


if __name__ == "__main__":
    main()
