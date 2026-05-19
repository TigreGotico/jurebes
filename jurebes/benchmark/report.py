"""Report formatters — markdown + JSON."""

from __future__ import annotations

import json

from jurebes.benchmark.harness import ComparisonResult


def to_markdown(comparison: ComparisonResult) -> str:
    header = "| baseline | accuracy | macro_f1 | micro_f1 | train_s | p50_ms | p95_ms | size_kb |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in comparison.rows:
        lines.append(
            f"| {r.name} | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.micro_f1:.4f} "
            f"| {r.train_seconds:.3f} | {r.predict_ms_p50:.2f} | {r.predict_ms_p95:.2f} "
            f"| {r.model_size_bytes / 1024:.1f} |"
        )
    return "\n".join(lines)


def to_json(comparison: ComparisonResult, indent: int = 2) -> str:
    return json.dumps(comparison.to_dict(), indent=indent)
