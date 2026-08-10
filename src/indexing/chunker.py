# src/indexing/chunker.py
import hashlib
from typing import List, Dict, Any
from tree_sitter import Node

from src.schemas.rag import CodeChunk
from src.indexing.ast_parser import ASTParser


class SemanticChunker:
    """
    Uses the AST to intelligently slice files into structural chunks 
    (classes and functions) for the Vector DB.
    """
    def __init__(self, language: str = "python"):
        self.ast_parser = ASTParser(language)

    def _generate_chunk_id(self, file_path: str, symbol_name: str) -> str:
        """Creates a deterministic hash for the chunk."""
        unique_string = f"{file_path}::{symbol_name}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    def _extract_nodes(self, node: Node, target_types: List[str]) -> List[Node]:
        """Recursively finds all nodes of specific types (e.g., function_definition)."""
        extracted = []
        if node.type in target_types:
            extracted.append(node)
            
        for child in node.children:
            extracted.extend(self._extract_nodes(child, target_types))
            
        return extracted

    def chunk_file(self, file_path: str, source_code: str, metadata: Dict[str, Any] = None) -> List[CodeChunk]:
        """
        Parses a file and returns a list of strict CodeChunk objects 
        ready for the Vector DB.
        """
        tree = self.ast_parser.parse_code(source_code)
        target_types = ["function_definition", "class_definition"]
        
        # Find all structural blocks in the file
        structural_nodes = self._extract_nodes(tree.root_node, target_types)
        
        chunks = []
        source_lines = source_code.splitlines()

        for node in structural_nodes:
            # Extract the name of the function/class
            name_node = node.child_by_field_name("name")
            symbol_name = name_node.text.decode("utf8") if name_node else "anonymous"

            # Tree-sitter rows are 0-indexed
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            # Extract exact text using line indices
            chunk_content = "\n".join(source_lines[start_line:end_line + 1])

            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, symbol_name),
                file_path=file_path,
                symbol_name=symbol_name,
                content=chunk_content,
                start_line=start_line + 1,  # Convert to 1-indexed for human readability
                end_line=end_line + 1,
                metadata=metadata or {}
            )
            chunks.append(chunk)

        return chunks