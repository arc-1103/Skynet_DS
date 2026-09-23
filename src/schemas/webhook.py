# webhook.py
from typing import Optional
from pydantic import BaseModel, Field

# A commit SHA, full or abbreviated. Anything interpolated into a path or a
# GitHub API URL is held to this: webhook payloads and run metadata can
# claim any "sha" they like, including "../../../../x".
SHA_PATTERN = r"^[0-9a-fA-F]{4,64}$"


class RepositoryRef(BaseModel):
    full_name: str = Field(...,description="e.g. 'owner/repo'")
    clone_url: str = Field(...,description="Git clone URL")
    default_branch: str = Field(default="main")


class WebhookUser(BaseModel):
    login: str = Field(...,description="GitHub handle")


class CommitRef(BaseModel):
    # Constrained because worker/jobs.py builds a cache filename out of this:
    # a payload claiming sha="../../../../x" would otherwise write outside
    # settings.pr_cache_dir, and signature verification is skipped entirely
    # when GITHUB_WEBHOOK_SECRET is unset (src/core/config.py).
    sha: str = Field(..., pattern=SHA_PATTERN, description="Commit SHA")
    ref: str = Field(..., description="Git ref (branch name)")


class WebhookPR(BaseModel):
    number: int
    title: str
    body: Optional[str] = ""
    state: str
    head: CommitRef = Field(..., description="Head commit metadata (sha, ref)")
    base: CommitRef = Field(..., description="Base commit metadata")


class GitHubPullRequestEvent(BaseModel):
    """Parses a GitHub 'pull_request' webhook payload."""
    action: str = Field(..., description="'opened', 'synchronize', 'reopened', 'closed'")
    number: int
    pull_request: WebhookPR
    repository: RepositoryRef
    sender: WebhookUser


class DeploymentRef(BaseModel):
    # Same constraint as CommitRef.sha: this reaches a GitHub compare URL
    # (retrieval/evidence.py), so a payload can't smuggle path segments in.
    sha: str = Field(..., pattern=SHA_PATTERN, description="Commit SHA that was deployed")
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