# src/graph/causal.py
from typing import Literal
from pydantic import BaseModel, Field


class CauseCandidate(BaseModel):
    kind: Literal["code", "config", "data", "unknown"]
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


class CausalGraph:
    """Holds candidate causes for a single diagnosis and ranks them.

    This is the causal-candidate graph (data structure + scoring), distinct
    from any agent-orchestration/execution graph in agents/ - BluePrint.md
    section 1 flags these as easy to conflate under the shared `graph/` name.

    Phase 0: heuristic scoring only, no ML (BluePrint.md Build Order Phase 0).
    """

    def __init__(self) -> None:
        self._candidates: list[CauseCandidate] = []

    def add_candidate(self, candidate: CauseCandidate) -> None:
        self._candidates.append(candidate)

    def rank(self) -> list[CauseCandidate]:
        return sorted(self._candidates, key=lambda c: c.confidence, reverse=True)
