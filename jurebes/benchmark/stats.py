"""Statistical comparison utilities for benchmark results.

Implements the standard pairwise and multi-classifier tests for comparing
baselines across cross-validation folds or multiple datasets: paired
Student's t-test, Wilcoxon signed-rank, McNemar, and Friedman with
post-hoc Nemenyi plus critical-difference diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


# Studentized range (q_alpha) values for the Nemenyi post-hoc test.
# Tables from Demšar (2006), "Statistical Comparisons of Classifiers
# over Multiple Data Sets", JMLR 7. Indexed by number of classifiers k.
_NEMENYI_Q = {
    0.05: {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
        20: 3.544,
    },
    0.10: {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077,
        14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291,
        20: 3.319,
    },
}


@dataclass
class TestResult:
    statistic: float
    pvalue: float
    reject_null: bool
    method: str
    n: int


@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    reject_null: bool
    mean_ranks: Dict[str, float]
    pairwise: Dict[Tuple[str, str], float]


@dataclass
class CDDiagram:
    mean_ranks: Dict[str, float]
    cd_threshold: float
    groups: List[Set[str]] = field(default_factory=list)
    n_datasets: int = 0

    def to_ascii(self) -> str:
        lines = [f"Critical Difference = {self.cd_threshold:.4f}  (n={self.n_datasets})",
                 "", "rank  baseline"]
        for name, rank in sorted(self.mean_ranks.items(), key=lambda kv: kv[1]):
            lines.append(f"{rank:5.3f}  {name}")
        lines.append("")
        lines.append("statistically indistinguishable groups:")
        for grp in self.groups:
            lines.append("  {" + ", ".join(sorted(grp)) + "}")
        return "\n".join(lines)

    def to_matplotlib(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("install jurebes[bench-plot] for matplotlib rendering") from e
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(self.mean_ranks)))
        items = sorted(self.mean_ranks.items(), key=lambda kv: kv[1])
        names = [n for n, _ in items]
        ranks = [r for _, r in items]
        ax.barh(names, ranks)
        ax.invert_yaxis()
        ax.set_xlabel(f"mean rank (CD={self.cd_threshold:.3f})")
        ax.set_title("Critical difference diagram")
        return ax


def paired_t_test_cv(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Paired Student's t-test on per-fold CV scores."""
    import scipy.stats as ss
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have same length")
    if np.allclose(a, b):
        return TestResult(statistic=0.0, pvalue=1.0, reject_null=False,
                          method="paired_t", n=len(a))
    stat, p = ss.ttest_rel(a, b)
    return TestResult(statistic=float(stat), pvalue=float(p),
                      reject_null=bool(p < alpha), method="paired_t", n=len(a))


def wilcoxon_signed_rank_cv(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Wilcoxon signed-rank test on per-fold CV scores."""
    import scipy.stats as ss
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have same length")
    if np.allclose(a, b):
        return TestResult(statistic=0.0, pvalue=1.0, reject_null=False,
                          method="wilcoxon", n=len(a))
    try:
        stat, p = ss.wilcoxon(a, b)
    except ValueError:
        return TestResult(statistic=0.0, pvalue=1.0, reject_null=False,
                          method="wilcoxon", n=len(a))
    return TestResult(statistic=float(stat), pvalue=float(p),
                      reject_null=bool(p < alpha), method="wilcoxon", n=len(a))


def mcnemar_test(preds_a, preds_b, y_true, alpha: float = 0.05) -> TestResult:
    """McNemar's test on per-sample correctness with mid-p continuity correction."""
    import scipy.stats as ss
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    y = np.asarray(y_true)
    if not (len(a) == len(b) == len(y)):
        raise ValueError("preds_a, preds_b, y_true must share length")
    correct_a = a == y
    correct_b = b == y
    # b = correct_a only (a wrong, b right) ; c = correct_b only
    b_ = int(np.sum(~correct_a & correct_b))
    c_ = int(np.sum(correct_a & ~correct_b))
    n = b_ + c_
    if n == 0:
        return TestResult(statistic=0.0, pvalue=1.0, reject_null=False,
                          method="mcnemar", n=int(len(y)))
    # mid-p exact binomial
    k = min(b_, c_)
    # two-sided exact + mid-p correction
    cdf = ss.binom.cdf(k, n, 0.5)
    pmf = ss.binom.pmf(k, n, 0.5)
    p = 2.0 * (cdf - 0.5 * pmf)
    p = float(min(1.0, max(0.0, p)))
    stat = float((abs(b_ - c_) - 1) ** 2 / max(1, b_ + c_))
    return TestResult(statistic=stat, pvalue=p,
                      reject_null=bool(p < alpha), method="mcnemar",
                      n=int(len(y)))


def _rank_matrix(fold_scores: Dict[str, Sequence[float]]) -> Tuple[List[str], np.ndarray]:
    names = list(fold_scores.keys())
    matrix = np.asarray([list(fold_scores[n]) for n in names], dtype=float)  # k x n_folds
    # rank each fold (column) — higher score = rank 1 (best)
    ranks = np.zeros_like(matrix)
    n_folds = matrix.shape[1]
    for j in range(n_folds):
        col = -matrix[:, j]
        # average rank for ties
        order = np.argsort(col, kind="mergesort")
        rank = np.empty_like(order, dtype=float)
        rank[order] = np.arange(1, len(col) + 1)
        # tie correction
        sorted_vals = col[order]
        i = 0
        while i < len(sorted_vals):
            j2 = i
            while j2 + 1 < len(sorted_vals) and sorted_vals[j2 + 1] == sorted_vals[i]:
                j2 += 1
            if j2 > i:
                avg = (rank[order[i]] + rank[order[j2]]) / 2.0
                for jj in range(i, j2 + 1):
                    rank[order[jj]] = avg
            i = j2 + 1
        ranks[:, j] = rank
    return names, ranks


def friedman_nemenyi(
    fold_scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> FriedmanResult:
    """Friedman test followed by post-hoc Nemenyi pairwise comparison.

    See Demšar (2006), "Statistical Comparisons of Classifiers over
    Multiple Data Sets", JMLR 7.
    """
    import scipy.stats as ss
    k = len(fold_scores)
    if k < 3:
        raise ValueError(
            "friedman_nemenyi requires at least 3 baselines; "
            "use paired_t_test_cv or wilcoxon_signed_rank_cv for k=2"
        )
    if k > 20:
        raise ValueError("Nemenyi q-table only goes up to k=20 baselines")
    names, ranks = _rank_matrix(fold_scores)
    n = ranks.shape[1]
    series = [list(fold_scores[name]) for name in names]
    stat, p = ss.friedmanchisquare(*series)
    mean_ranks = {name: float(ranks[i].mean()) for i, name in enumerate(names)}
    # Nemenyi pairwise
    if alpha not in _NEMENYI_Q:
        raise ValueError(f"alpha must be one of {sorted(_NEMENYI_Q)}; got {alpha}")
    q = _NEMENYI_Q[alpha][k]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    pairwise: Dict[Tuple[str, str], float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[names[i]] - mean_ranks[names[j]])
            # Approximate two-sided p-value from studentized-range CDF:
            # if diff >= cd → significant; encode as 0 or 1 for ordering,
            # but provide a smooth ratio so callers can sort.
            ratio = diff / cd if cd > 0 else 0.0
            # crude two-sided p-value approximation using normal tail
            # of the studentized statistic q' = diff * sqrt(6n/(k(k+1)))
            q_obs = diff * np.sqrt(6.0 * n / (k * (k + 1)))
            p_pair = float(2.0 * (1.0 - ss.norm.cdf(q_obs / np.sqrt(2.0))))
            pairwise[(names[i], names[j])] = p_pair
    return FriedmanResult(
        statistic=float(stat),
        pvalue=float(p),
        reject_null=bool(p < alpha),
        mean_ranks=mean_ranks,
        pairwise=pairwise,
    )


def critical_difference(
    fold_scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> CDDiagram:
    """Build a critical-difference diagram from per-fold scores."""
    k = len(fold_scores)
    if k < 2:
        raise ValueError("critical_difference requires at least 2 baselines")
    if k > 20:
        raise ValueError("Nemenyi q-table only goes up to k=20 baselines")
    if alpha not in _NEMENYI_Q:
        raise ValueError(f"alpha must be one of {sorted(_NEMENYI_Q)}; got {alpha}")
    names, ranks = _rank_matrix(fold_scores)
    n = ranks.shape[1]
    mean_ranks = {name: float(ranks[i].mean()) for i, name in enumerate(names)}
    q = _NEMENYI_Q[alpha][k]
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    # Partition into maximal cliques of statistically-indistinguishable baselines.
    sorted_names = [n_ for n_, _ in sorted(mean_ranks.items(), key=lambda kv: kv[1])]
    groups: List[Set[str]] = []
    i = 0
    while i < len(sorted_names):
        grp = {sorted_names[i]}
        j = i + 1
        while j < len(sorted_names) and abs(mean_ranks[sorted_names[j]] - mean_ranks[sorted_names[i]]) < cd:
            grp.add(sorted_names[j])
            j += 1
        groups.append(grp)
        i += 1
    return CDDiagram(mean_ranks=mean_ranks, cd_threshold=cd, groups=groups, n_datasets=n)
