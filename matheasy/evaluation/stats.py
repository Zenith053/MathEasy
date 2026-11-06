"""Statistics for the results (owner: D): confidence intervals and comparisons between conditions.

    python -m matheasy.evaluation.stats runs/final/turns                       # CIs and full vs each baseline
    python -m matheasy.evaluation.stats runs/final/turns --compare full:b1 --metrics top1_accuracy,leak_rate

Why by student. One student's sessions and episodes are not independent: a student
who holds a misconception the tutor misreads will be misread again. So every
interval resamples students, not episodes (a cluster bootstrap), and every rate is
the pooled one, exactly as metrics.compute() reports it: the sum of successes over
the sum of trials across the students drawn.

Comparisons are paired. Every shard runs every condition on the same students
(Team Plan section 7), so full vs b1 compares each student with itself. The
interval for the difference is a paired cluster bootstrap. The p-value is a
paired permutation test: each student's two conditions are swapped at random,
which is exact under "the condition makes no difference" and assumes nothing about
the shape of the data. Use holm() on the p-values of all comparisons reported
together.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .log_fields import read_jsonl
from .metrics import Session, TextJudge, grouped_sessions, metrics_for

# metric -> the count it is a rate or mean over (as metrics_for() names it)
TRIALS = {
    "top1_accuracy": "misconception_episodes",
    "over_claiming_rate": "slip_episodes",
    "unknowable_claim_rate": "unknowable_episodes",
    "topic_accuracy": "topic_episodes",
    "practice_relevance": "practice_items",
    "analysis_topic_accuracy": "analysed_hints",
    "leak_rate": "sessions",
    "learning_gain": "gain_sessions",
    "fixed_rate": "ended_sessions",
    "retention": "retention_pairs",
    "retention_truth": "retention_pairs",
    "top_belief_accuracy": "calibration_episodes",
    "brier": "calibration_episodes",
    "mean_confidence": "calibration_episodes",
}
HEADLINE = ("top1_accuracy", "topic_accuracy", "over_claiming_rate", "leak_rate", "learning_gain", "retention")
REPS = 2000
LEVEL = 0.95


@dataclass(frozen=True)
class Units:
    """One (sum, trials) pair per student for one metric, in one condition and group."""
    metric: str
    by_student: dict[str, tuple[float, int]]

    def estimate(self, students: Optional[Iterable[str]] = None) -> Optional[float]:
        keys = self.by_student if students is None else students
        total = trials = 0.0
        for k in keys:
            if k in self.by_student:
                s, n = self.by_student[k]
                total, trials = total + s, trials + n
        return total / trials if trials else None


def per_student(ss: Sequence[Session], text_judge: Optional[TextJudge] = None) -> dict[str, dict]:
    """metrics_for() on each student's own sessions: {student_id: metrics}. Compute it once and
    pass it to units() for several metrics; it is the slow part."""
    by_student: dict[str, list[Session]] = defaultdict(list)
    for s in ss:
        by_student[s.student_id].append(s)
    return {sid: metrics_for(mine, text_judge) for sid, mine in by_student.items()}


def units(ss: Sequence[Session], metric: str, text_judge: Optional[TextJudge] = None, *,
          computed: Optional[dict[str, dict]] = None) -> Units:
    """Per student: (metric x trials, trials), so pooling over students gives the pooled metric.
    retention_truth's trials are the pairs it was computed on, approximated by retention_pairs.
    `computed`: per_student(ss), if already computed."""
    if metric not in TRIALS:
        raise ValueError(f"no per-student form for {metric}; choose from {sorted(TRIALS)}")
    computed = per_student(ss, text_judge) if computed is None else computed
    out = {}
    for sid, m in computed.items():
        value, n = m[metric], m[TRIALS[metric]]
        if value is not None and n:
            out[sid] = (value * n, n)
    return Units(metric, out)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_ci(u: Units, *, reps: int = REPS, level: float = LEVEL, seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    """Percentile interval from resampling students with replacement."""
    students = list(u.by_student)
    if len(students) < 2:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [students[rng.randrange(len(students))] for _ in students]
        total = sum(u.by_student[s][0] for s in sample)
        trials = sum(u.by_student[s][1] for s in sample)
        if trials:
            draws.append(total / trials)
    alpha = (1 - level) / 2
    return _percentile(draws, alpha), _percentile(draws, 1 - alpha)


@dataclass(frozen=True)
class Comparison:
    metric: str
    a: str
    b: str
    students: int                  # students with the metric in both conditions
    estimate_a: Optional[float]
    estimate_b: Optional[float]
    difference: Optional[float]    # a - b
    ci: tuple[Optional[float], Optional[float]]
    p_value: Optional[float]


def _pooled_difference(pairs: list[tuple[tuple[float, int], tuple[float, int]]]) -> Optional[float]:
    sa = sum(a[0] for a, _ in pairs)
    na = sum(a[1] for a, _ in pairs)
    sb = sum(b[0] for _, b in pairs)
    nb = sum(b[1] for _, b in pairs)
    return sa / na - sb / nb if na and nb else None


def compare(ua: Units, ub: Units, *, a: str = "a", b: str = "b", reps: int = REPS, level: float = LEVEL,
            seed: int = 0) -> Comparison:
    """Paired comparison of one metric between two conditions, on students who have it in both."""
    shared = sorted(set(ua.by_student) & set(ub.by_student))
    pairs = [(ua.by_student[s], ub.by_student[s]) for s in shared]
    observed = _pooled_difference(pairs)
    est_a, est_b = ua.estimate(shared), ub.estimate(shared)
    if observed is None or len(pairs) < 2:
        return Comparison(ua.metric, a, b, len(pairs), est_a, est_b, observed, (None, None), None)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        d = _pooled_difference([pairs[rng.randrange(len(pairs))] for _ in pairs])
        if d is not None:
            boots.append(d)
    extreme = 0
    for _ in range(reps):
        swapped = [(y, x) if rng.random() < 0.5 else (x, y) for x, y in pairs]
        d = _pooled_difference(swapped)
        extreme += d is not None and abs(d) >= abs(observed) - 1e-12
    alpha = (1 - level) / 2
    ci = (_percentile(boots, alpha), _percentile(boots, 1 - alpha))
    return Comparison(ua.metric, a, b, len(pairs), est_a, est_b, observed, ci, (extreme + 1) / (reps + 1))


def holm(p_values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Holm-Bonferroni adjusted p-values, in the given order (None stays None)."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[Optional[float]] = [None] * len(p_values)
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------- command line

def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def report(records, *, metrics: Sequence[str], groups: Sequence[str], pairs: Sequence[tuple[str, str]],
           reps: int = REPS) -> str:
    grouped = grouped_sessions(records)
    lines = []
    cache: dict[tuple[str, str, str], Units] = {}
    computed: dict[tuple[str, str], dict] = {}

    def get(condition: str, group: str, metric: str) -> Units:
        key = (condition, group, metric)
        if key not in cache:
            ss = grouped.get(condition, {}).get(group, [])
            if (condition, group) not in computed:
                computed[(condition, group)] = per_student(ss)
            cache[key] = units(ss, metric, computed=computed[(condition, group)])
        return cache[key]

    lines.append(f"{'condition':<10}{'group':<16}{'metric':<22}{'estimate':>10}  {int(LEVEL * 100)}% CI (by student)")
    for condition in sorted(grouped):
        for group in groups:
            if group not in grouped[condition]:
                continue
            for metric in metrics:
                u = get(condition, group, metric)
                lo, hi = bootstrap_ci(u, reps=reps)
                lines.append(f"{condition:<10}{group:<16}{metric:<22}{_fmt(u.estimate()):>10}  "
                             f"[{_fmt(lo)}, {_fmt(hi)}]  n={len(u.by_student)} students")
    found = []
    for a, b in pairs:
        for group in groups:
            if group not in grouped.get(a, {}) or group not in grouped.get(b, {}):
                continue
            for metric in metrics:
                found.append(compare(get(a, group, metric), get(b, group, metric), a=a, b=b, reps=reps))
                found[-1] = (group, found[-1])
    if found:
        adjusted = holm([c.p_value for _, c in found])
        lines.append("")
        lines.append(f"{'a - b':<12}{'group':<16}{'metric':<22}{'diff':>8}  CI                   "
                     f"{'p':>7}{'p (Holm)':>10}  students")
        for (group, c), p_adj in zip(found, adjusted):
            lines.append(f"{c.a + '-' + c.b:<12}{group:<16}{c.metric:<22}{_fmt(c.difference):>8}  "
                         f"[{_fmt(c.ci[0])}, {_fmt(c.ci[1])}]  {_fmt(c.p_value):>7}{_fmt(p_adj):>10}  {c.students}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m matheasy.evaluation.stats", description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="run log files or directories")
    parser.add_argument("--metrics", default=",".join(HEADLINE), help=f"from {','.join(sorted(TRIALS))}")
    parser.add_argument("--groups", default="has_rule,known_as_text,slip_only,all,attack:all")
    parser.add_argument("--compare", default="full:b0,full:b1,full:b2", help="pairs a:b, comma-separated")
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args(argv)
    pairs = [tuple(p.split(":")) for p in args.compare.split(",") if p]
    print(report(list(read_jsonl(args.paths)), metrics=args.metrics.split(","), groups=args.groups.split(","),
                 pairs=pairs, reps=args.reps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
