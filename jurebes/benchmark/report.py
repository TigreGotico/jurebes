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
    extra_cols_all = list(comparison.scoring or ())
    if sort_by:
        rows = _sort_rows(rows, sort_by, extra_cols_all)

    extra_cols = extra_cols_all
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


def _sort_rows(rows, key: str, extra_cols=None):
    # Valid sort keys = the rendered table columns. Mirror to_markdown().
    base_cols = {
        "group", "baseline", "accuracy", "macro_f1", "micro_f1",
        "train_s", "p50_ms_pooled", "p95_ms_pooled", "size_kb",
    }
    extras = set(extra_cols or ())
    # extras drop dedicated columns
    extras = {c for c in extras if c not in {"accuracy", "f1_macro"}}
    # union extra_scores keys actually present on any row, as a fallback
    for r in rows:
        extras.update(r.extra_scores.keys())
    # legacy aliases also accepted (and "name" → "baseline")
    aliases = {"name": "baseline", "p50_ms": "p50_ms_pooled", "p95_ms": "p95_ms_pooled"}
    valid = base_cols | extras | set(aliases.keys())
    if key not in valid:
        raise KeyError(f"sort_by={key!r} not in result columns: {sorted(valid)}")
    canon = aliases.get(key, key)

    def keyfn(r):
        if canon == "baseline":
            return r.name
        if canon == "group":
            return r.group or ""
        if canon in {"accuracy", "macro_f1", "micro_f1"}:
            return -getattr(r, canon)
        if canon == "train_s":
            return r.train_seconds
        if canon == "p50_ms_pooled":
            return r.predict_ms_p50_pooled
        if canon == "p95_ms_pooled":
            return r.predict_ms_p95_pooled
        if canon == "size_kb":
            return r.model_size_bytes
        return -float(r.extra_scores.get(canon, float("nan")))
    return sorted(rows, key=keyfn)


def to_json(comparison: ComparisonResult, indent: int = 2) -> str:
    return json.dumps(comparison.to_dict(), indent=indent)
