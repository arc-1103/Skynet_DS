# src/indexing/code_index.py
from typing import Dict, List, Optional

from src.indexing.ast_parser import ASTParser
from src.indexing.chunker import PythonDSChunker
from src.schemas.rag import CodeChunk, RetrievalResult


class CodeIndex:
    """Builds a searchable index over code so retrieval/ and agents/ can
    query it instead of re-parsing raw files each time (Architecture.md
    section 2, indexing/).

    Phase 0: in-memory store, heuristic substring scoring for search() - no
    vector DB/embeddings yet, matching graph/causal.py's "heuristic scoring,
    no ML yet" approach for this build phase. `schemas/rag.py`'s CodeChunk/
    RetrievalResult are the shared contracts this produces, so a later swap
    to a real vector store only touches this module.
    """

    def __init__(self, language: str = "python") -> None:
        if language != "python":
            # PythonDSChunker is Python-only by design (docstring, chunker.py) -
            # only the raw ASTParser (used by symbols_touching_lines/find_symbol)
            # is language-generic.
            raise ValueError(f"CodeIndex only supports Python chunking, got: {language!r}")
        self._chunker = PythonDSChunker()
        self._ast_parser = ASTParser(language)
        self._chunks_by_id: Dict[str, CodeChunk] = {}
        self._chunk_ids_by_file: Dict[str, List[str]] = {}

    def index_file(
        self, file_path: str, source_code: str, metadata: Optional[dict] = None
    ) -> List[CodeChunk]:
        """(Re)indexes a single file, replacing any chunks previously
        indexed for that path."""
        chunks = self._chunker.chunk_file(file_path, source_code, metadata)

        self.remove_file(file_path)
        self._chunk_ids_by_file[file_path] = [c.chunk_id for c in chunks]
        for chunk in chunks:
            self._chunks_by_id[chunk.chunk_id] = chunk

        return chunks

    def index_files(self, sources: Dict[str, str]) -> int:
        """Indexes {file_path: source_code} in bulk. Returns the total
        number of chunks indexed."""
        total = 0
        for file_path, source_code in sources.items():
            total += len(self.index_file(file_path, source_code))
        return total

    def remove_file(self, file_path: str) -> None:
        for chunk_id in self._chunk_ids_by_file.pop(file_path, []):
            self._chunks_by_id.pop(chunk_id, None)

    def get_chunks_for_file(self, file_path: str) -> List[CodeChunk]:
        return [
            self._chunks_by_id[cid]
            for cid in self._chunk_ids_by_file.get(file_path, [])
        ]

    def find_symbol(self, symbol_name: str) -> List[CodeChunk]:
        """Exact match on the trailing symbol name, e.g. "baz" also matches
        a chunk indexed as "Bar.baz"."""
        return [
            chunk
            for chunk in self._chunks_by_id.values()
            if chunk.symbol_name == symbol_name
            or chunk.symbol_name.endswith(f".{symbol_name}")
        ]

    def symbols_touching_lines(
        self, source_code: str, modified_lines: List[int]
    ) -> List[str]:
        """Resolves diff line numbers to enclosing function/class symbols,
        without re-indexing the file. This is the piece retrieval/'s
        get_code_diff stub will call once github/ supplies real diffs
        (BluePrint.md Phase 1) - it lets a code-change evidence lookup go
        straight to affected symbols instead of the raw diff text."""
        return self._ast_parser.extract_modified_symbols(source_code, modified_lines)

    def search(self, query: str, limit: int = 5) -> RetrievalResult:
        """Heuristic substring search over indexed chunks - symbol-name
        matches score higher than content-only matches. Placeholder for a
        real embedding/vector search once the "code" leg of PRD.md's
        evidence sources is prioritized."""
        needle = query.lower()
        scored: List[tuple[CodeChunk, float]] = []

        for chunk in self._chunks_by_id.values():
            score = 0.0
            if needle in chunk.symbol_name.lower():
                score += 1.0
            if needle in chunk.content.lower():
                score += 0.25
            if score > 0.0:
                scored.append((chunk, min(score, 1.0)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:limit]

        return RetrievalResult(
            query_symbol=query,
            chunks=[chunk for chunk, _ in top],
            scores=[score for _, score in top],
        )
