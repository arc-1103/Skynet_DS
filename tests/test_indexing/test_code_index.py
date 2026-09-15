from src.indexing.code_index import CodeIndex

SOURCE = """
def compute_score(x):
    return x * 2


class Model:
    def predict(self, x):
        return compute_score(x)

    def fit(self, data):
        pass
"""

OTHER_SOURCE = """
def helper():
    return 1
"""

# PythonDSChunker only splits a class into per-method chunks once its content
# exceeds MAX_CLASS_CHARS_STANDALONE (1500 chars) - pad with an oversized
# docstring so tests can exercise that per-method/qualified-name path
# without needing dozens of throwaway methods.
_FILLER = "x" * 1600

BIG_SOURCE = f'''
class Model:
    def predict(self, x):
        """{_FILLER}"""
        return x

    def fit(self, data):
        pass
'''

COLLISION_SOURCE = f'''
class A:
    def predict(self, x):
        """{_FILLER}"""
        return x + 1

class B:
    def predict(self, x):
        """{_FILLER}"""
        return x * 2
'''


def test_index_file_returns_chunks_for_top_level_function_and_small_class():
    index = CodeIndex()

    chunks = index.index_file("model.py", SOURCE)

    symbol_names = {c.symbol_name for c in chunks}
    assert symbol_names == {"compute_score", "Model"}


def test_index_file_splits_large_class_into_qualified_method_chunks():
    index = CodeIndex()

    chunks = index.index_file("model.py", BIG_SOURCE)

    symbol_names = {c.symbol_name for c in chunks}
    assert symbol_names == {"Model.predict", "Model.fit"}


def test_get_chunks_for_file_returns_indexed_chunks():
    index = CodeIndex()
    index.index_file("model.py", SOURCE)

    chunks = index.get_chunks_for_file("model.py")

    assert {c.symbol_name for c in chunks} == {"compute_score", "Model"}
    assert index.get_chunks_for_file("missing.py") == []


def test_index_file_reindex_replaces_previous_chunks():
    index = CodeIndex()
    index.index_file("model.py", SOURCE)

    index.index_file("model.py", OTHER_SOURCE)

    chunks = index.get_chunks_for_file("model.py")
    assert {c.symbol_name for c in chunks} == {"helper"}


def test_index_files_bulk_indexes_and_returns_total_chunk_count():
    index = CodeIndex()

    total = index.index_files({"model.py": SOURCE, "util.py": OTHER_SOURCE})

    assert total == 3
    assert index.get_chunks_for_file("util.py")[0].symbol_name == "helper"


def test_remove_file_drops_its_chunks():
    index = CodeIndex()
    index.index_file("model.py", SOURCE)

    index.remove_file("model.py")

    assert index.get_chunks_for_file("model.py") == []


def test_find_symbol_matches_qualified_names_from_split_classes():
    index = CodeIndex()
    index.index_file("model.py", BIG_SOURCE)

    predict_matches = index.find_symbol("predict")
    assert {c.symbol_name for c in predict_matches} == {"Model.predict"}

    assert index.find_symbol("nonexistent") == []


def test_index_file_does_not_collide_same_named_methods_on_different_large_classes():
    """Regression test: chunk_id must stay unique even when two different
    (split) classes in the same file both have a same-named method, since
    a hash collision would silently clobber one chunk with the other."""
    index = CodeIndex()

    chunks = index.index_file("model.py", COLLISION_SOURCE)

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    predict_chunks = {c.symbol_name: c for c in chunks if c.symbol_name.endswith(".predict")}
    assert set(predict_chunks) == {"A.predict", "B.predict"}
    assert "x + 1" in predict_chunks["A.predict"].content
    assert "x * 2" in predict_chunks["B.predict"].content


def test_symbols_touching_lines_resolves_enclosing_functions():
    index = CodeIndex()

    symbols = index.symbols_touching_lines(SOURCE, [3])

    assert symbols == ["compute_score"]


def test_search_ranks_symbol_name_matches_above_content_only_matches():
    index = CodeIndex()
    index.index_file("model.py", BIG_SOURCE)

    result = index.search("predict")

    assert result.query_symbol == "predict"
    assert result.chunks[0].symbol_name == "Model.predict"
    assert result.scores[0] == 1.0


def test_search_respects_limit_and_returns_empty_for_no_match():
    index = CodeIndex()
    index.index_files({"model.py": SOURCE, "util.py": OTHER_SOURCE})

    limited = index.search("def", limit=1)
    assert len(limited.chunks) <= 1

    empty = index.search("nope_not_present")
    assert empty.chunks == []
    assert empty.scores == []
