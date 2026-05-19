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
    with_significance: bool = False,
    significance_alpha: float = 0.05,
    significance_metric: Optional[str] = None,
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
    if with_significance:
        sig = _significance_block(
            comparison,
            metric=significance_metric,
            alpha=significance_alpha,
            precision=precision,
        )
        if sig:
            lines.append("")
            lines.append(sig)
    return "\n".join(lines)


def _significance_block(comparison: ComparisonResult, *, metric, alpha: float, precision: int) -> str:
    fsbb = comparison.fold_scores_by_baseline
    if not fsbb:
        return ""
    chosen = metric or "f1_macro"
    fold_scores = {name: scores[chosen] for name, scores in fsbb.items() if chosen in scores}
    if len(fold_scores) < 2:
        return ""
    lines: list = []
    fmt = f"{{:.{precision}f}}"
    if len(fold_scores) >= 3:
        from jurebes.benchmark.stats import critical_difference, friedman_nemenyi
        fr = friedman_nemenyi(fold_scores, alpha=alpha)
        cd = critical_difference(fold_scores, alpha=alpha)
        lines.append(f"### Critical Difference (metric={chosen}, alpha={alpha})")
        lines.append("")
        lines.append(f"Friedman statistic={fmt.format(fr.statistic)}, p={fmt.format(fr.pvalue)}, reject_null={fr.reject_null}")
        lines.append("")
        lines.append("| baseline | mean_rank |")
        lines.append("| --- | --- |")
        for n, r in sorted(cd.mean_ranks.items(), key=lambda kv: kv[1]):
            lines.append(f"| {n} | {fmt.format(r)} |")
        lines.append("")
        lines.append(f"CD threshold = {fmt.format(cd.cd_threshold)}")
        lines.append("Statistically indistinguishable groups:")
        for grp in cd.groups:
            lines.append("- {" + ", ".join(sorted(grp)) + "}")
    else:
        from jurebes.benchmark.stats import paired_t_test_cv, wilcoxon_signed_rank_cv
        names = list(fold_scores.keys())
        a, b = fold_scores[names[0]], fold_scores[names[1]]
        rt = paired_t_test_cv(a, b, alpha=alpha)
        rw = wilcoxon_signed_rank_cv(a, b, alpha=alpha)
        lines.append(f"### Pairwise significance ({names[0]} vs {names[1]}, metric={chosen}, alpha={alpha})")
        lines.append("")
        lines.append("| test | statistic | pvalue | reject_null |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(f"| paired_t | {fmt.format(rt.statistic)} | {fmt.format(rt.pvalue)} | {rt.reject_null} |")
        lines.append(f"| wilcoxon | {fmt.format(rw.statistic)} | {fmt.format(rw.pvalue)} | {rw.reject_null} |")
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
