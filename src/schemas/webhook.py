# webhook.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RepositoryRef(BaseModel):
    full_name: str = Field(...,description="e.g. 'owner/repo'")
    clone_url: str = Field(...,description="Git clone URL")
    default_branch: str = Field(default="main")


class WebhookUser(BaseModel):
    login: str = Field(...,description="GitHub handle")


class WebhookPR(BaseModel):
    number: int
    title: str
    body: Optional[str] = ""
    state: str
    head: Dict[str, Any] = Field(...,description="Head commit metadata (sha, ref)")
    base: Dict[str, Any] = Field(...,description="Base commit metadata")


class GitHubWebhookEvent(BaseModel):
    """Parses raw HTTP POST body from /api/webhooks."""
    action: str = Field(..., description="'opened', 'synchronize', 'reopened', 'closed'")
    number: Optional[int] = None
    pull_request: Optional[WebhookPR] = None
    repository: RepositoryRef
    sender: WebhookUser


class DeploymentRef(BaseModel):
    sha: str = Field(..., description="Commit SHA that was deployed")
    ref: str = Field(..., description="Git ref (branch/tag) that was deployed")
    environment: str = Field(default="production", description="Deployment target environment")


class DeploymentStatusRef(BaseModel):
    state: str = Field(..., description="'pending', 'success', 'failure', 'error', etc.")


class GitHubDeploymentStatusEvent(BaseModel):
    """Parses a GitHub 'deployment_status' webhook payload - the trigger for
    Workflow 4 (Deployment Regression Triage) per Workflow.md."""

    action: Optional[str] = None
    deployment: DeploymentRef
    deployment_status: DeploymentStatusRef
    repository: RepositoryRef
    sender: WebhookUser