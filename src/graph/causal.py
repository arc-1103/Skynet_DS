# src/graph/causal.py
from collections import defaultdict
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.telemetry import NormalizedRun

Kind = Literal["code", "config", "data", "unknown"]


class CauseCandidate(BaseModel):
    kind: Kind
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


def apply_history_prior(candidates: list[CauseCandidate], history: list[CauseCandidate]) -> list[CauseCandidate]:
    """Adjust candidate confidences using this model's past verdicts
    (Workflow.md "Cross-Workflow Notes": 'config changes to learning_rate
    have historically been the cause 70% of the time for this model
    family'). `history` is newest-first, one top verdict per past diagnosis.

    Must return candidates with confidence still in [0, 1]; ordering is
    done by the caller."""
    # TODO(user): decide how much past verdicts should move today's scores.
    return candidates


class CausalGraph:
    """Holds candidate causes for a single diagnosis and ranks them.

    This is the causal-candidate graph (data structure + scoring), distinct
    from any agent-orchestration/execution graph in agents/ - BluePrint.md
    section 1 flags these as easy to conflate under the shared `graph/` name.

    Heuristic scoring only, no ML (BluePrint.md Build Order Phase 0), plus
    an optional prior from this model's diagnosis history.
    """

    def __init__(self) -> None:
        self._candidates: list[CauseCandidate] = []

    def add_candidate(self, candidate: CauseCandidate) -> None:
        self._candidates.append(candidate)

    def rank(self, history: list[CauseCandidate] | None = None) -> list[CauseCandidate]:
        candidates = apply_history_prior(self._candidates, history) if history else self._candidates
        return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _variables(runs: list[NormalizedRun]) -> dict[tuple[Kind, str], list[Any]]:
    """Every run-level variable that could explain a metric spread."""
    keys = dict.fromkeys(k for r in runs for k in r.config)  # ordered union
    variables: dict[tuple[Kind, str], list[Any]] = {
        ("config", f"`{k}`"): [r.config.get(k) for r in runs] for k in keys
    }
    variables[("data", "data version")] = [r.data_version for r in runs]
    variables[("code", "code version")] = [r.code_ref for r in runs]
    return variables


def rank_correlates(runs: list[NormalizedRun], metric: str) -> list[CauseCandidate]:
    """N-way comparison mode (Workflow.md Workflow 3): instead of one
    baseline vs. one current run, rank which varying variable best explains
    the spread of `metric` across all runs.

    Score = eta^2 (share of metric variance explained by grouping runs on
    that variable) x a support factor: a variable with a distinct value in
    every run explains all variance by construction, so it is capped at
    half confidence - with only 2 runs every changed variable ties there,
    which is the honest answer.

    ponytail: eta^2 ignores ordering of numeric values (lr 1e-4 < 1e-3);
    swap in rank correlation for numeric keys if trends matter more than
    groups."""
    scored = [(r, r.metrics[metric]) for r in runs if metric in r.metrics]
    if len(scored) < 2:
        return []
    runs, scores = [r for r, _ in scored], [s for _, s in scored]
    n, mean = len(scores), fmean(scores)
    ss_total = sum((s - mean) ** 2 for s in scores)
    if ss_total == 0:
        return []

    candidates = []
    for (kind, label), values in _variables(runs).items():
        groups: dict[str, list[float]] = defaultdict(list)
        for value, score in zip(values, scores):
            groups[str(value)].append(score)
        k = len(groups)
        if k < 2:
            continue
        ss_between = sum(len(g) * (fmean(g) - mean) ** 2 for g in groups.values())
        support = 0.5 + 0.5 * (n - k) / (n - 1)
        by_mean = sorted(groups.items(), key=lambda kv: fmean(kv[1]), reverse=True)
        detail = ", ".join(f"{v} -> mean {fmean(g):.3f} ({len(g)} run(s))" for v, g in by_mean)
        candidates.append(
            CauseCandidate(
                kind=kind,
                description=f"{label} separates {metric} across {n} runs: {detail}",
                confidence=round(min(1.0, ss_between / ss_total * support), 3),
                evidence_refs=[r.run_id for r in runs],
            )
        )
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)
