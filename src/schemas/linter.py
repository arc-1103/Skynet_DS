# linter.py
from typing import List, Optional
from pydantic import BaseModel, Field


class LinterIssue(BaseModel):
    code: str = Field(..., description="Rule code, e.g. 'E501' or 'F401'")
    message: str = Field(..., description="Explanation from linter")
    file_path: str
    line: int
    column: int
    fix_suggested: Optional[str] = None


class LinterReport(BaseModel):
    tool_name: str = Field(default="Ruff")
    total_issues: int = 0
    issues: List[LinterIssue] = Field(default_factory=list)