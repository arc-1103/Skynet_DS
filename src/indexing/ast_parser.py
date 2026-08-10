# src/indexing/ast_parser.py
from typing import List, Set, Optional
from tree_sitter_language_pack import get_parser
from tree_sitter import Node


class ASTParser:
    def __init__(self, language: str = "python"):
        """Initializes the Tree-sitter parser for the specified language."""
        try:
            self.parser = get_parser(language)
        except Exception as e:
            raise ValueError(f"Failed to load parser for {language}: {str(e)}")

    def parse_code(self, code: str):
        """Parses raw source code into an Abstract Syntax Tree."""
        return self.parser.parse(bytes(code, "utf8"))

    def get_enclosing_symbol(self, node: Node, line_number: int) -> Optional[str]:
        """
        Recursively finds the function or class definition that encloses a given line.
        Tree-sitter uses 0-indexed rows, so we subtract 1 from the line_number.
        """
        target_row = line_number - 1
        
        # Check if the current node spans the target line
        if not (node.start_point[0] <= target_row <= node.end_point[0]):
            return None

        # We are looking for function or class definitions
        symbol_name = None
        if node.type in ["function_definition", "class_definition"]:
            # The 'name' field in Tree-sitter usually holds the identifier
            name_node = node.child_by_field_name("name")
            if name_node:
                symbol_name = name_node.text.decode("utf8")

        # Recursively check children to find the deepest enclosing scope
        for child in node.children:
            child_symbol = self.get_enclosing_symbol(child, line_number)
            if child_symbol:
                # Append child scope to parent scope (e.g., ClassName.function_name)
                if symbol_name:
                    return f"{symbol_name}.{child_symbol}"
                return child_symbol

        return symbol_name

    def extract_modified_symbols(self, code: str, modified_lines: List[int]) -> List[str]:
        """
        Takes the full file code and a list of modified lines from the diff,
        returning a deduplicated list of affected functions/classes.
        """
        tree = self.parse_code(code)
        root_node = tree.root_node
        
        affected_symbols: Set[str] = set()
        
        for line in modified_lines:
            symbol = self.get_enclosing_symbol(root_node, line)
            if symbol:
                affected_symbols.add(symbol)
                
        return list(affected_symbols)