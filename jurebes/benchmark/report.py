"""Report formatters — markdown + JSON."""

from __future__ import annotations

import json
from typing import Optional

from jurebes.benchmark.harness import ComparisonResult


def to_markdown(
    comparison: ComparisonResult,
    *,
    sort_by: Optional[str] = None,
    precision: int = 4,
) -> str:
    """Render a markdown table.

    Args:
        comparison: result from :func:`compare`.
        sort_by: optional column to sort rows by, descending. One of:
            ``name``, ``group``, ``accuracy``, ``macro_f1``, ``micro_f1``,
            ``train_s``, ``p50_ms``, ``p95_ms``, ``size_kb``, or any
            extra-scoring metric name.
        precision: number of decimal places for float columns.
    """
    rows = list(comparison.rows)
    if sort_by:
        rows = _sort_rows(rows, sort_by)

    extra_cols = list(comparison.scoring or ())
    # accuracy + f1_macro are surfaced in dedicated columns already; drop
    # them from extra-cols to avoid duplication unless the caller asks
    # for something else.
    extra_cols = [c for c in extra_cols if c not in {"accuracy", "f1_macro"}]

    base_cols = [
        "group", "baseline", "accuracy", "macro_f1", "micro_f1",
        "train_s", "p50_ms_pooled", "p95_ms_pooled", "size_kb",
    ] + extra_cols
    header = "| " + " | ".join(base_cols) + " |"
    sep = "| " + " | ".join("---" for _ in base_cols) + " |"
    lines = [header, sep]
    fmt = f"{{:.{precision}f}}"
    for r in rows:
        cells = [
            r.group or "",
            r.name,
            fmt.format(r.accuracy),
            fmt.format(r.macro_f1),
            fmt.format(r.micro_f1),
            f"{r.train_seconds:.3f}",
            f"{r.predict_ms_p50_pooled:.2f}",
            f"{r.predict_ms_p95_pooled:.2f}",
            f"{r.model_size_bytes / 1024:.1f}",
        ]
        for c in extra_cols:
            v = r.extra_scores.get(c, float("nan"))
            cells.append(fmt.format(v) if v == v else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sort_rows(rows, key: str):
    def keyfn(r):
        if key == "name":
            return r.name
        if key == "group":
            return r.group or ""
        if key in {"accuracy", "macro_f1", "micro_f1"}:
            return -getattr(r, key)
        if key == "train_s":
            return r.train_seconds
        if key in {"p50_ms", "p95_ms"}:
            return getattr(r, "predict_ms_p50_pooled" if key == "p50_ms" else "predict_ms_p95_pooled")
        if key == "size_kb":
            return r.model_size_bytes
        return -float(r.extra_scores.get(key, float("nan")))
    return sorted(rows, key=keyfn)


def to_json(comparison: ComparisonResult, indent: int = 2) -> str:
    return json.dumps(comparison.to_dict(), indent=indent)
