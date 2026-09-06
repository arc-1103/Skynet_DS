# src/schemas/telemetry.py
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class NormalizedRun(BaseModel):
    """Backend-agnostic run record."""

    # Disallow any attributes not explicitly defined below (Maximum criteria)
    model_config = ConfigDict(extra="forbid")

    # Required: Minimum criteria (no default values provided)
    run_id: str = Field(..., description="Unique identifier for this run/execution")
    model_id: str = Field(..., description="Model or pipeline identifier this run belongs to")
    timestamp: datetime = Field(..., description="When the run completed")

    # Optional: Can be omitted, default factories and None handle them
    metrics: dict[str, float] = Field(
        default_factory=dict, 
        description="Evaluation metrics, e.g. {'accuracy': 0.91}"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, 
        description="Hyperparameters/config used for this run"
    )
    code_ref: Optional[str] = Field(
        default=None, 
        description="Commit SHA the run was executed against, if known"
    )
    data_version: Optional[str] = Field(
        default=None, 
        description="Identifier/hash of the training or eval data version"
    )