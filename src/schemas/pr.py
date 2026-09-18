# src/schemas/pr.py
from pydantic import BaseModel, Field
from typing import List, Optional


class ChangedFile(BaseModel):
    filename: str = Field(..., description="Relative file path in the repository")
    status: str = Field(..., description="File change status: added, modified, deleted, renamed")
    additions: int = Field(default=0, description="Number of lines added")
    deletions: int = Field(default=0, description="Number of lines deleted")
    patch: Optional[str] = Field(None, description="Raw unified git diff patch hunk")


class PRPayload(BaseModel):
    pr_id: int = Field(..., description="Pull Request number")
    repo_full_name: str = Field(..., description="Repository slug, e.g. 'owner/repo'")
    commit_sha: str = Field(..., description="Head commit SHA for inline diff commenting")
    base_sha: Optional[str] = Field(
        None, description="Base commit SHA the PR targets - the 'before' side of any comparison"
    )
    base_ref: Optional[str] = Field(None, description="Base branch name, e.g. 'main'")
    author: str = Field(..., description="GitHub handle of the PR author")
    title: str = Field(..., description="Title of the Pull Request")
    body: Optional[str] = Field("", description="Description/body provided in the PR")
    changed_files: List[ChangedFile] = Field(default_factory=list)