"""Benchmark harness — train_test, cross_validate, compare."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, List, Union

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from jurebes.baselines import BASELINES
from jurebes.benchmark.metrics import RunResult

ClassifierFactory = Union[str, Callable]


def _resolve(factory: ClassifierFactory):
    if isinstance(factory, str):
        name = factory
        return name, lambda: BASELINES.build(name)
    name = getattr(factory, "__name__", "custom")
    return name, factory


def _percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _measure(estimator, X_test, y_test, labels):
    pred_times: List[float] = []
    preds = []
    for x in X_test:
        t = time.perf_counter()
        p = estimator.predict([x])[0]
        pred_times.append((time.perf_counter() - t) * 1000.0)
        preds.append(p)
    acc = accuracy_score(y_test, preds)
    macro = f1_score(y_test, preds, average="macro", zero_division=0)
    micro = f1_score(y_test, preds, average="micro", zero_division=0)
    per_class = f1_score(y_test, preds, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels).tolist()
    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_class_f1": dict(zip(labels, [float(v) for v in per_class])),
        "predict_ms_p50": _percentile(pred_times, 50),
        "predict_ms_p95": _percentile(pred_times, 95),
        "predict_ms_p99": _percentile(pred_times, 99),
        "model_size_bytes": buf.tell(),
        "confusion_matrix": cm,
        "labels": list(labels),
    }


def train_test(
    classifier_factory: ClassifierFactory,
    X: List[str],
    y: List[str],
    *,
    test_size: float = 0.2,
    seed: int = 0,
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
    m = _measure(est, X_test, y_test, labels)
    return RunResult(name=name, train_seconds=train_seconds, **m)


def cross_validate(
    classifier_factory: ClassifierFactory,
    X: List[str],
    y: List[str],
    *,
    k: int = 5,
    seed: int = 0,
) -> RunResult:
    name, factory = _resolve(classifier_factory)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    X_arr = np.array(X)
    y_arr = np.array(y)
    labels = sorted(set(y))
    accs, macros, micros, trains, sizes = [], [], [], [], []
    p50s, p95s, p99s = [], [], []
    per_class_acc = {lab: [] for lab in labels}
    cm_total = np.zeros((len(labels), len(labels)), dtype=int)
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        est = factory()
        t = time.perf_counter()
        est.fit(X_arr[train_idx].tolist(), y_arr[train_idx].tolist())
        trains.append(time.perf_counter() - t)
        m = _measure(est, X_arr[test_idx].tolist(), y_arr[test_idx].tolist(), labels)
        accs.append(m["accuracy"])
        macros.append(m["macro_f1"])
        micros.append(m["micro_f1"])
        sizes.append(m["model_size_bytes"])
        p50s.append(m["predict_ms_p50"])
        p95s.append(m["predict_ms_p95"])
        p99s.append(m["predict_ms_p99"])
        for lab, v in m["per_class_f1"].items():
            per_class_acc[lab].append(v)
        cm_total += np.array(m["confusion_matrix"])
    return RunResult(
        name=name,
        accuracy=float(np.mean(accs)),
        macro_f1=float(np.mean(macros)),
        micro_f1=float(np.mean(micros)),
        per_class_f1={k_: float(np.mean(v)) for k_, v in per_class_acc.items()},
        train_seconds=float(np.mean(trains)),
        predict_ms_p50=float(np.mean(p50s)),
        predict_ms_p95=float(np.mean(p95s)),
        predict_ms_p99=float(np.mean(p99s)),
        model_size_bytes=int(np.mean(sizes)),
        confusion_matrix=cm_total.tolist(),
        labels=labels,
    )


@dataclass
class ComparisonResult:
    rows: List[RunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rows": [r.to_dict() for r in self.rows]}


def compare(
    baselines: List[ClassifierFactory],
    X: List[str],
    y: List[str],
    *,
    k: int = 5,
    seed: int = 0,
) -> ComparisonResult:
    rows = [cross_validate(b, X, y, k=k, seed=seed) for b in baselines]
    return ComparisonResult(rows=rows)
