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

    def _make_chunk(
        self,
        file_path: str,
        symbol_name: str,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        meta: Dict[str, Any],
        **extra_metadata: Any,
    ) -> CodeChunk:
        """Every chunk in this module is built the same way; only the symbol
        name, the slice and the chunk_type differ. Line numbers are taken
        0-based (as tree-sitter reports them) and stored 1-based."""
        return CodeChunk(
            chunk_id=self._generate_chunk_id(file_path, symbol_name, start_line + 1),
            file_path=file_path,
            symbol_name=symbol_name,
            content=content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            metadata={**meta, "chunk_type": chunk_type, **extra_metadata},
        )

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

        chunks.append(
            self._make_chunk(
                file_path,
                f"{class_symbol}.{member_name}",
                f"# Class Context: {class_header}\n{m_content}",
                m_start,
                m_end,
                chunk_type,
                meta,
                parent_class=class_symbol,
            )
        )

    def _append_nested_function_chunks(
        self,
        chunks: List[CodeChunk],
        file_path: str,
        source_lines: List[str],
        parent_symbol: str,
        func_node: Node,
        meta: Dict[str, Any],
    ) -> None:
        """Emit a chunk per function defined inside func_node, at any depth.

        Without this, a closure (the `train_step` inside `train` that this
        chunker is meant to be good at) is absorbed into its parent's single
        chunk and can't be resolved as its own symbol. Recursive, because the
        nesting can be: the outer walk in chunk_file only sees module-level
        nodes plus one level of class body."""
        body = func_node.child_by_field_name("body")
        if not body:
            return

        for stmt in body.children:
            target = stmt
            if stmt.type == "decorated_definition":
                target = next(
                    (c for c in reversed(stmt.children) if c.type in {"function_definition", "class_definition"}),
                    stmt,
                )
            if target.type != "function_definition":
                continue

            qualified_symbol = f"{parent_symbol}.{self._get_node_name(stmt)}"
            start_line = stmt.start_point[0]
            end_line = stmt.end_point[0]
            chunks.append(
                self._make_chunk(
                    file_path,
                    qualified_symbol,
                    self._slice_source(source_lines, start_line, end_line),
                    start_line,
                    end_line,
                    "nested_function",
                    meta,
                    parent_symbol=parent_symbol,
                )
            )
            self._append_nested_function_chunks(
                chunks, file_path, source_lines, qualified_symbol, target, meta
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
                    self._make_chunk(
                        file_path, "module_script_block", content,
                        start_line, end_line, "top_level_script", meta,
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
                    self._make_chunk(
                        file_path, symbol_name, content,
                        start_line, end_line, "function", meta,
                    )
                )
                self._append_nested_function_chunks(
                    chunks, file_path, source_lines, symbol_name, node_to_inspect, meta
                )

            # Case 2: Class definition (e.g., torch.nn.Module, Dataset, Sklearn Transformer)
            elif node_to_inspect.type == "class_definition":
                if len(content) <= self.MAX_CLASS_CHARS_STANDALONE:
                    chunks.append(
                        self._make_chunk(
                            file_path, symbol_name, content,
                            start_line, end_line, "class", meta,
                        )
                    )
                else:
                    # Break large classes into methods, injecting class header for context.
                    # The header runs from `child` (the decorated_definition, when
                    # there is one) through the `class X:` line, so a decorated class
                    # keeps its @dataclass/@register_model - the whole-class chunk
                    # above already includes it, and it's often the only thing that
                    # identifies the class. Joined into one line to stay a comment.
                    class_header = " ".join(
                        source_lines[line].strip()
                        for line in range(child.start_point[0], node_to_inspect.start_point[0] + 1)
                    )

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
                                method_name = self._get_node_name(class_child)
                                self._append_class_member_chunk(
                                    chunks, file_path, source_lines, class_header, symbol_name,
                                    class_child, method_name, "class_method", meta,
                                )
                                # a closure inside a method is the same case as one
                                # inside a plain function
                                self._append_nested_function_chunks(
                                    chunks, file_path, source_lines,
                                    f"{symbol_name}.{method_name}", method_target, meta,
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