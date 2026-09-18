# src/schemas/drift.py
"""Shared contracts for data-drift detection (Workflow.md Workflow 2).

Every detector in drift/methods.py returns a MethodResult, so the caller
never has to know whether the number came from a hypothesis test, a
distance, or a classifier's ROC AUC - only whether it crossed its own
threshold."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class DriftMethod(str, Enum):
    KS = "ks"
    PSI = "psi"
    PAGE_HINKLEY = "page_hinkley"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MODEL_BASED = "model_based"
    MMD = "mmd"


class ColumnKind(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class MethodResult(BaseModel):
    """One detector's verdict on one comparison."""

    method: DriftMethod
    statistic: float = Field(
        ..., description="The method's score - scale and direction differ per method"
    )
    threshold: float = Field(..., description="Cutoff the verdict was decided against")
    drifted: bool = Field(..., description="Whether this method alone flags drift")
    p_value: Optional[float] = Field(
        default=None, description="Only set by the methods that produce one (KS, MMD)"
    )
    detail: Optional[str] = Field(
        default=None, description="How to read the statistic, or anything notable"
    )


class ColumnDriftReport(BaseModel):
    """All applicable detectors run against a single column."""

    column: str
    kind: ColumnKind
    n_reference: int
    n_current: int
    results: list[MethodResult] = Field(default_factory=list)
    skipped: dict[str, str] = Field(
        default_factory=dict, description="method -> why it did not apply to this column"
    )
    drifted: bool = Field(..., description="Verdict after combining the applicable methods")
    vote: str = Field(..., description="How results were combined: 'majority' or 'any'")

    @property
    def n_drifted_methods(self) -> int:
        return sum(r.drifted for r in self.results)


class DatasetDriftReport(BaseModel):
    """Dataset-level roll-up. share_drifted is the 'Share of Drifted
    Components' metric: the fraction of columns their own detectors flagged.

    The four roll-up fields are @computed_field, not plain @property, so the
    verdict survives model_dump()/model_dump_json() - worker/jobs.py persists
    these reports to disk and ml_debugger_tools hands them to the LLM, and
    both were getting a report with no verdict in it."""

    columns: list[ColumnDriftReport] = Field(default_factory=list)
    share_threshold: float = Field(
        ..., description="Dataset counts as drifted above this share of drifted columns"
    )
    label: Optional[str] = Field(
        default=None, description="What was compared, e.g. a data file path"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_drifted(self) -> int:
        return sum(c.drifted for c in self.columns)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def share_drifted(self) -> float:
        return self.n_drifted / self.n_columns if self.columns else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_drifted(self) -> bool:
        return bool(self.columns) and self.share_drifted > self.share_threshold
