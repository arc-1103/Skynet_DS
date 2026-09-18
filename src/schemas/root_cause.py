# src/schemas/root_cause.py
"""Contracts for the SHAP root-cause engine (drift/root_cause.py).

Where schemas/drift.py answers 'did the data change?', these answer 'which
feature changed?' - the attribution a surrogate model assigns to each
feature when asked to tell reference data from production data."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ShiftDirection(str, Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    # Same centre, different shape - a variance or category-mix change.
    RESHAPED = "reshaped"


class FeatureAttribution(BaseModel):
    """One feature's share of the blame for reference/production separability."""

    feature: str
    mean_abs_shap: float = Field(
        ..., description="Mean |SHAP value| - how much this feature moves the surrogate's output"
    )
    share: float = Field(
        ..., ge=0.0, le=1.0, description="This feature's fraction of total attribution"
    )
    direction: ShiftDirection
    reference_mean: Optional[float] = Field(
        default=None, description="Feature mean in reference data (numeric features only)"
    )
    current_mean: Optional[float] = Field(
        default=None, description="Feature mean in production data (numeric features only)"
    )


class DriftRootCause(BaseModel):
    """Ranked feature attributions behind a reference -> production shift.

    `separable` is the gate that decides whether any of this means anything:
    a surrogate that cannot beat chance has found no drift, and its SHAP
    ranking is then just noise ordered by magnitude. Callers must check it
    before reading `attributions`.

    That gate has two halves, because either alone is misleading. The AUC
    threshold is an effect-size floor - a surrogate scoring 0.51 has found
    nothing worth acting on however many rows back it. `auc_p_value` is the
    significance half: on a small test set the AUC of pure noise swings
    wildly (with ~15 rows per class its standard error is around 0.1), so a
    0.56 means nothing there while meaning a great deal across thousands of
    rows."""

    surrogate_auc: float = Field(
        ..., description="Held-out ROC AUC of the reference-vs-production classifier"
    )
    auc_threshold: float = Field(..., description="AUC above which the samples count as separable")
    auc_p_value: float = Field(
        ...,
        description="Probability of an AUC this high by chance, given the test-set size "
        "(Mann-Whitney U, which is equivalent to ROC AUC)",
    )
    alpha: float = Field(..., description="Significance level auc_p_value is judged against")
    separable: bool = Field(
        ...,
        description="True only when the AUC clears its threshold AND is significant for the "
        "test-set size; False means there is no detectable drift to attribute",
    )
    n_reference: int
    n_current: int
    attributions: list[FeatureAttribution] = Field(
        default_factory=list, description="Ranked by mean_abs_shap, descending"
    )
    label: Optional[str] = Field(default=None, description="What was compared, e.g. a file path")

    @property
    def top_features(self) -> list[str]:
        return [a.feature for a in self.attributions]

    @property
    def separability_strength(self) -> float:
        """AUC rescaled to 0-1, where 0.5 (chance) maps to 0 and 1.0 maps to 1.

        Used to discount attribution shares when the surrogate is only
        marginally better than guessing."""
        return min(1.0, max(0.0, (self.surrogate_auc - 0.5) * 2.0))
