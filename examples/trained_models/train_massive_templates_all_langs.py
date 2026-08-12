"""Sweep ``train_massive_templates`` across all 51 MASSIVE languages.

Each language is trained in its own subprocess so the OS reclaims all
memory between languages — the in-process design peaked high enough on
~13.5k-template corpora to trip the OOM killer. The sweep is resumable:
a language whose report already exists is skipped, so a kill costs at
most the one in-progress language.

Writes one per-language report + a consolidated summary.

Requires: pip install jurebes[hf]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path


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
    single = HERE / "train_massive_templates.py"
    summary_rows = []
    for lang in LANGS:
        out = REPORTS / f"massive_templates_{lang}.md"
        # Resumable: skip a language whose report already exists.
        if out.exists() and "— failed" not in out.read_text(encoding="utf-8"):
            report = out.read_text(encoding="utf-8")
            print(f"[{lang}] skip — report already present", flush=True)
        else:
            t0 = time.perf_counter()
            # One subprocess per language: the OS reclaims every byte on
            # exit, so peak memory is one language's worth, not 51.
            rc = subprocess.run(
                [sys.executable, str(single), lang], cwd=HERE,
            ).returncode
            dt = time.perf_counter() - t0
            if out.exists():
                report = out.read_text(encoding="utf-8")
                print(f"[{lang}] done ({dt:.1f}s) rc={rc}", flush=True)
            else:
                report = f"# massive-templates ({lang}) — failed\n\n`subprocess rc={rc}`"
                out.write_text(report, encoding="utf-8")
                print(f"[{lang}] subprocess died rc={rc} ({dt:.1f}s)", flush=True)

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
