from src.indexing.chunker import PythonDSChunker

_FILLER = "x" * 1600


def test_chunk_file_splits_top_level_function_and_leaves_small_class_whole():
    source = """
def compute_score(x):
    return x * 2


class Model:
    def predict(self, x):
        return compute_score(x)

    def fit(self, data):
        pass
"""
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("model.py", source)

    by_name = {c.symbol_name: c for c in chunks}
    assert set(by_name) == {"compute_score", "Model"}
    assert by_name["Model"].metadata["chunk_type"] == "class"
    assert "def predict" in by_name["Model"].content
    assert "def fit" in by_name["Model"].content


def test_chunk_file_splits_large_class_into_qualified_method_chunks():
    source = f'''
class Model:
    def predict(self, x):
        """{_FILLER}"""
        return x

    def fit(self, data):
        pass
'''
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("model.py", source)

    by_name = {c.symbol_name: c for c in chunks}
    assert set(by_name) == {"Model.predict", "Model.fit"}
    assert by_name["Model.predict"].metadata["chunk_type"] == "class_method"
    assert by_name["Model.predict"].metadata["parent_class"] == "Model"
    assert by_name["Model.predict"].content.startswith("# Class Context: class Model:")


def test_chunk_file_captures_nested_class_inside_a_split_class():
    """Regression test: a class nested inside a large (split) class used to
    be silently dropped - it isn't a function (so the method branch skipped
    it) and it's nested two levels deep (so the top-level script scan never
    saw it either)."""
    source = f'''
class Model:
    def predict(self, x):
        """{_FILLER}"""
        return x

    class Config:
        lr = 0.01

    @dataclass
    class Meta:
        name: str
'''
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("model.py", source)

    by_name = {c.symbol_name: c for c in chunks}
    assert "Model.Config" in by_name
    assert "Model.Meta" in by_name
    assert by_name["Model.Config"].metadata["chunk_type"] == "nested_class"
    assert "lr = 0.01" in by_name["Model.Config"].content
    assert "name: str" in by_name["Model.Meta"].content


def test_chunk_file_includes_decorator_in_function_chunk():
    source = """
@torch.no_grad()
def evaluate(model, x):
    return model(x)
"""
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("eval.py", source)

    assert len(chunks) == 1
    assert chunks[0].symbol_name == "evaluate"
    assert chunks[0].content.startswith("@torch.no_grad()")


def test_chunk_file_groups_top_level_statements_into_script_block_chunks():
    source = """
import torch

model = build_model()
model.fit(x, y)
"""
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("train.py", source)

    assert len(chunks) == 1
    assert chunks[0].symbol_name == "module_script_block"
    assert chunks[0].metadata["chunk_type"] == "top_level_script"
    assert "import torch" in chunks[0].content
    assert "model.fit(x, y)" in chunks[0].content


def test_chunk_file_splits_script_blocks_around_structural_definitions():
    source = """
import torch


def train():
    pass


model = train()
"""
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("train.py", source)

    script_chunks = [c for c in chunks if c.symbol_name == "module_script_block"]
    assert len(script_chunks) == 2
    assert "import torch" in script_chunks[0].content
    assert "model = train()" in script_chunks[1].content


def test_chunk_id_is_stable_and_unique_per_chunk():
    source = f'''
class A:
    def predict(self, x):
        """{_FILLER}"""
        return x + 1


class B:
    def predict(self, x):
        """{_FILLER}"""
        return x * 2
'''
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("model.py", source)

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # deterministic: re-chunking the same source yields the same ids
    again = chunker.chunk_file("model.py", source)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_chunk_file_merges_caller_metadata_with_chunk_type():
    source = """
def compute_score(x):
    return x * 2
"""
    chunker = PythonDSChunker()

    chunks = chunker.chunk_file("model.py", source, metadata={"repo": "demo"})

    assert chunks[0].metadata["repo"] == "demo"
    assert chunks[0].metadata["chunk_type"] == "function"
