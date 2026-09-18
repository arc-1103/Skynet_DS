"""The single tabular reader that replaced four divergent copies."""

import io

import pandas as pd
import pytest

from src.drift.tables import read_table


@pytest.fixture
def frame():
    return pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})


def test_reads_csv_by_suffix(tmp_path, frame):
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    pd.testing.assert_frame_equal(read_table(str(path)), frame)


def test_reads_parquet_by_suffix(tmp_path, frame):
    path = tmp_path / "t.parquet"
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(read_table(str(path)), frame)


def test_suffix_match_is_case_insensitive(tmp_path, frame):
    path = tmp_path / "t.CSV"
    frame.to_csv(path, index=False)
    assert len(read_table(str(path))) == 2


def test_unknown_suffix_raises_instead_of_being_read_as_csv(tmp_path, frame):
    """The two drift CLIs used to feed anything that wasn't .parquet to
    read_csv, which turns a .json or .xlsx into a garbage DataFrame the drift
    detectors will then happily report drift on."""
    path = tmp_path / "t.json"
    frame.to_json(path)
    with pytest.raises(ValueError, match="unsupported file type"):
        read_table(str(path))


def test_reads_a_buffer_when_given_an_explicit_suffix(frame):
    """pr_drift reads blobs out of `git show`; they never touch the filesystem."""
    buffer = io.BytesIO(frame.to_csv(index=False).encode())
    pd.testing.assert_frame_equal(read_table(buffer, suffix=".csv"), frame)


def test_a_buffer_without_a_suffix_is_an_error():
    with pytest.raises(ValueError, match="suffix is required"):
        read_table(io.BytesIO(b"x\n1\n"))
