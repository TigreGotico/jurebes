"""Benchmark harness — train_test, cross_validate, compare."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from jurebes.baselines import BASELINES
from jurebes.benchmark.metrics import RunResult, pooled_percentiles
from jurebes.benchmark.scoring import get as get_scorer

ClassifierFactory = Union[str, Callable[[], "Pipeline"]]  # noqa: F821


def _resolve(factory: ClassifierFactory):
    if isinstance(factory, str):
        name = factory
        return name, lambda: BASELINES.build(name)
    name = getattr(factory, "__name__", "custom")
    return name, factory


def _measure(estimator, X_test, y_test, labels, scoring_names: Tuple[str, ...]):
    pred_times: List[float] = []
    preds = []
    for x in X_test:
        t = time.perf_counter()
        p = estimator.predict([x])[0]
        pred_times.append((time.perf_counter() - t) * 1000.0)
        preds.append(p)
    # probabilities (optional)
    try:
        probs = estimator.predict_proba(list(X_test))
        classes = list(estimator.classes_)
    except (AttributeError, NotImplementedError):
        probs = None
        classes = labels

    macro = f1_score(y_test, preds, average="macro", zero_division=0)
    micro = f1_score(y_test, preds, average="micro", zero_division=0)
    per_class = f1_score(y_test, preds, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels).tolist()
    buf = io.BytesIO()
    joblib.dump(estimator, buf)

    # within-fold percentiles (legacy)
    p50 = float(np.percentile(pred_times, 50)) if pred_times else 0.0
    p95 = float(np.percentile(pred_times, 95)) if pred_times else 0.0
    p99 = float(np.percentile(pred_times, 99)) if pred_times else 0.0

    extra_scores = {}
    for name in scoring_names:
        scorer = get_scorer(name)
        try:
            extra_scores[name] = scorer(y_test, preds, probs, classes)
        except Exception:
            extra_scores[name] = float("nan")

    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_class_f1": dict(zip(labels, [float(v) for v in per_class])),
        "predict_ms_p50": p50,
        "predict_ms_p95": p95,
        "predict_ms_p99": p99,
        "predict_ms_samples": pred_times,
        "model_size_bytes": buf.tell(),
        "confusion_matrix": cm,
        "labels": list(labels),
        "extra_scores": extra_scores,
    }


def train_test(
    classifier_factory: ClassifierFactory,
    X: List[str],
    y: List[str],
    *,
    test_size: float = 0.2,
    seed: int = 0,
    scoring: Sequence[str] = ("accuracy", "f1_macro"),
) -> RunResult:
    name, factory = _resolve(classifier_factory)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    est = factory()
    t = time.perf_counter()
    est.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t
    labels = sorted(set(y))
    m = _measure(est, X_test, y_test, labels, tuple(scoring))
    pooled = pooled_percentiles(m["predict_ms_samples"])
    return RunResult(
        name=name,
        accuracy=float(m["extra_scores"].get("accuracy", 0.0)),
        macro_f1=m["macro_f1"],
        micro_f1=m["micro_f1"],
        per_class_f1=m["per_class_f1"],
        train_seconds=train_seconds,
        predict_ms_p50=m["predict_ms_p50"],
        predict_ms_p95=m["predict_ms_p95"],
        predict_ms_p99=m["predict_ms_p99"],
        predict_ms_p50_pooled=pooled["p50"],
        predict_ms_p95_pooled=pooled["p95"],
        predict_ms_p99_pooled=pooled["p99"],
        predict_ms_mean=pooled["mean"],
        model_size_bytes=m["model_size_bytes"],
        confusion_matrix=m["confusion_matrix"],
        labels=m["labels"],
        extra_scores=m["extra_scores"],
        group=_baseline_group(name),
    )


def _baseline_group(name: str) -> str:
    try:
        groups = sorted(BASELINES.in_group(name))
        return ",".join(groups) if groups else ""
    except Exception:
        return ""


def cross_validate(
    classifier_factory: ClassifierFactory,
    X: List[str],
    y: List[str],
    *,
    k: int = 5,
    seed: int = 0,
    scoring: Sequence[str] = ("accuracy", "f1_macro"),
) -> RunResult:
    name, factory = _resolve(classifier_factory)
    scoring = tuple(scoring)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    X_arr = np.array(X)
    y_arr = np.array(y)
    labels = sorted(set(y))

    macros, micros, trains, sizes = [], [], [], []
    p50s, p95s, p99s = [], [], []
    pooled_samples: List[float] = []
    per_class_acc = {lab: [] for lab in labels}
    cm_total = np.zeros((len(labels), len(labels)), dtype=int)
    extra: dict = {n: [] for n in scoring}

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        est = factory()
        t = time.perf_counter()
        est.fit(X_arr[train_idx].tolist(), y_arr[train_idx].tolist())
        trains.append(time.perf_counter() - t)
        m = _measure(est, X_arr[test_idx].tolist(), y_arr[test_idx].tolist(), labels, scoring)
        macros.append(m["macro_f1"])
        micros.append(m["micro_f1"])
        sizes.append(m["model_size_bytes"])
        p50s.append(m["predict_ms_p50"])
        p95s.append(m["predict_ms_p95"])
        p99s.append(m["predict_ms_p99"])
        pooled_samples.extend(m["predict_ms_samples"])
        for lab, v in m["per_class_f1"].items():
            per_class_acc[lab].append(v)
        cm_total += np.array(m["confusion_matrix"])
        for n, v in m["extra_scores"].items():
            extra[n].append(v)

    pooled = pooled_percentiles(pooled_samples)
    extra_mean = {n: float(np.mean(vs)) if vs and not all(np.isnan(vs)) else float("nan") for n, vs in extra.items()}
    fold_scores: dict = {n: [float(v) for v in vs] for n, vs in extra.items()}
    fold_scores["f1_macro"] = [float(v) for v in macros]
    fold_scores["f1_micro"] = [float(v) for v in micros]
    return RunResult(
        name=name,
        accuracy=float(extra_mean.get("accuracy", 0.0)),
        macro_f1=float(np.mean(macros)),
        micro_f1=float(np.mean(micros)),
        per_class_f1={k_: float(np.mean(v)) for k_, v in per_class_acc.items()},
        train_seconds=float(np.mean(trains)),
        predict_ms_p50=float(np.mean(p50s)),
        predict_ms_p95=float(np.mean(p95s)),
        predict_ms_p99=float(np.mean(p99s)),
        predict_ms_p50_pooled=pooled["p50"],
        predict_ms_p95_pooled=pooled["p95"],
        predict_ms_p99_pooled=pooled["p99"],
        predict_ms_mean=pooled["mean"],
        model_size_bytes=int(np.mean(sizes)),
        confusion_matrix=cm_total.tolist(),
        labels=labels,
        extra_scores=extra_mean,
        group=_baseline_group(name),
        fold_scores=fold_scores,
    )


@dataclass
class ComparisonResult:
    rows: List[RunResult] = field(default_factory=list)
    scoring: Tuple[str, ...] = ("accuracy", "f1_macro")

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "scoring": list(self.scoring),
            "fold_scores_by_baseline": self.fold_scores_by_baseline,
        }

    @property
    def fold_scores_by_baseline(self) -> dict:
        """Map baseline name → {scoring_metric → per-fold scores}."""
        return {r.name: dict(r.fold_scores) for r in self.rows}


def compare(
    baselines: List[ClassifierFactory],
    X: List[str],
    y: List[str],
    *,
    k: int = 5,
    seed: int = 0,
    scoring: Sequence[str] = ("accuracy", "f1_macro"),
) -> ComparisonResult:
    if isinstance(scoring, str):
        scoring = (scoring,)
    rows = [
        cross_validate(b, X, y, k=k, seed=seed, scoring=scoring)
        for b in baselines
    ]
    return ComparisonResult(rows=rows, scoring=tuple(scoring))
