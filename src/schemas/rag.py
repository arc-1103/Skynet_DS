
from pydantic import BaseModel, Field
from typing import Dict, Any, List


class CodeChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique hash ID of the code chunk")
    file_path: str = Field(..., description="Source file path relative to repo root")
    symbol_name: str = Field(..., description="Target class, function, or module extracted via Tree-sitter")
    content: str = Field(..., description="Raw code lines of the chunk")
    start_line: int = Field(..., description="Start line in source file")
    end_line: int = Field(..., description="End line in source file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Imports, callers, language context")


class RetrievalResult(BaseModel):
    query_symbol: str = Field(..., description="Modified function or class symbol queried")
    chunks: List[CodeChunk] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)