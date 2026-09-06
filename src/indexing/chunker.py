# src/indexing/chunker.py
import hashlib
from typing import List, Dict, Any, Optional
from tree_sitter import Node

from src.schemas.rag import CodeChunk
from src.indexing.ast_parser import ASTParser


class PythonDSChunker:
    """
    AST chunker optimized for Data Science & ML Python files:
    - Splits large classes into methods while preserving class headers/signatures.
    - Captures top-level script logic, pipelines, and train loops.
    - Includes decorators, types, and signature context for vectors.
    """

    MAX_CLASS_CHARS_STANDALONE = 1500  # If a class is longer, chunk its methods individually

    def __init__(self):
        self.ast_parser = ASTParser("python")

    def _generate_chunk_id(self, file_path: str, symbol_name: str, start_line: int) -> str:
        unique_string = f"{file_path}::{symbol_name}::{start_line}"
        return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:16]

    def _get_node_name(self, node: Node) -> str:
        """Extracts identifier name, handling decorated nodes."""
        target = node
        if node.type == "decorated_definition":
            target = next((c for c in reversed(node.children) if c.type in {"function_definition", "class_definition"}), node)
        name_node = target.child_by_field_name("name")
        return name_node.text.decode("utf-8") if name_node and name_node.text else "anonymous"

    def _slice_source(self, source_lines: List[str], start_line: int, end_line: int) -> str:
        return "\n".join(source_lines[start_line:end_line + 1])

    def _append_class_member_chunk(
        self,
        chunks: List[CodeChunk],
        file_path: str,
        source_lines: List[str],
        class_header: str,
        class_symbol: str,
        member_node: Node,
        member_name: str,
        chunk_type: str,
        meta: Dict[str, Any],
    ) -> None:
        m_start = member_node.start_point[0]
        m_end = member_node.end_point[0]
        m_content = self._slice_source(source_lines, m_start, m_end)
        enriched_content = f"# Class Context: {class_header}\n{m_content}"
        qualified_symbol = f"{class_symbol}.{member_name}"

        chunks.append(
            CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, qualified_symbol, m_start + 1),
                file_path=file_path,
                symbol_name=qualified_symbol,
                content=enriched_content,
                start_line=m_start + 1,
                end_line=m_end + 1,
                metadata={**meta, "chunk_type": chunk_type, "parent_class": class_symbol},
            )
        )

    def chunk_file(
        self, 
        file_path: str, 
        source_code: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[CodeChunk]:
        tree = self.ast_parser.parse_code(source_code)
        source_lines = source_code.splitlines()
        chunks: List[CodeChunk] = []
        meta = metadata or {}

        # Track top-level statements for DS scripts (train loops, data loading, imports)
        top_level_buffer: List[Node] = []

        def flush_top_level_buffer():
            if not top_level_buffer:
                return
            start_line = top_level_buffer[0].start_point[0]
            end_line = top_level_buffer[-1].end_point[0]
            content = self._slice_source(source_lines, start_line, end_line).strip()
            
            if content:
                chunks.append(
                    CodeChunk(
                        chunk_id=self._generate_chunk_id(file_path, "module_script_block", start_line + 1),
                        file_path=file_path,
                        symbol_name="module_script_block",
                        content=content,
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                        metadata={**meta, "chunk_type": "top_level_script"}
                    )
                )
            top_level_buffer.clear()

        for child in tree.root_node.children:
            # Handle comments / standalone imports aggregation
            if child.type not in {"class_definition", "function_definition", "decorated_definition"}:
                top_level_buffer.append(child)
                continue

            # If we hit a structural block, flush any pending script code above it
            flush_top_level_buffer()

            node_to_inspect = child
            if child.type == "decorated_definition":
                inner = next((c for c in reversed(child.children) if c.type in {"function_definition", "class_definition"}), None)
                if inner:
                    node_to_inspect = inner

            symbol_name = self._get_node_name(child)
            start_line = child.start_point[0]
            end_line = child.end_point[0]
            content = self._slice_source(source_lines, start_line, end_line)

            # Case 1: Standard function/closure
            if node_to_inspect.type == "function_definition":
                chunks.append(
                    CodeChunk(
                        chunk_id=self._generate_chunk_id(file_path, symbol_name, start_line + 1),
                        file_path=file_path,
                        symbol_name=symbol_name,
                        content=content,
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                        metadata={**meta, "chunk_type": "function"}
                    )
                )

            # Case 2: Class definition (e.g., torch.nn.Module, Dataset, Sklearn Transformer)
            elif node_to_inspect.type == "class_definition":
                if len(content) <= self.MAX_CLASS_CHARS_STANDALONE:
                    chunks.append(
                        CodeChunk(
                            chunk_id=self._generate_chunk_id(file_path, symbol_name, start_line + 1),
                            file_path=file_path,
                            symbol_name=symbol_name,
                            content=content,
                            start_line=start_line + 1,
                            end_line=end_line + 1,
                            metadata={**meta, "chunk_type": "class"}
                        )
                    )
                else:
                    # Break large classes into methods, injecting class header for context
                    header_line = node_to_inspect.start_point[0]
                    class_header = source_lines[header_line].strip()
                    
                    body = node_to_inspect.child_by_field_name("body")
                    if body:
                        for class_child in body.children:
                            method_target = class_child
                            if class_child.type == "decorated_definition":
                                method_target = next(
                                    (c for c in reversed(class_child.children) if c.type in {"function_definition", "class_definition"}),
                                    class_child
                                )

                            # Prepend class context so the embedding model knows which model/layer this is
                            if method_target.type == "function_definition":
                                self._append_class_member_chunk(
                                    chunks, file_path, source_lines, class_header, symbol_name,
                                    class_child, self._get_node_name(class_child), "class_method", meta,
                                )
                            elif method_target.type == "class_definition":
                                # Nested class (e.g. a Config/Meta inner class) - without this
                                # branch it falls through unindexed: it's not a function, and
                                # it's too deep to be picked up by the top-level script scan.
                                self._append_class_member_chunk(
                                    chunks, file_path, source_lines, class_header, symbol_name,
                                    class_child, self._get_node_name(class_child), "nested_class", meta,
                                )

        # Flush any trailing script logic (e.g., model.fit(), evaluation metrics at bottom)
        flush_top_level_buffer()

        return chunks