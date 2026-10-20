import os
import sys
from pathlib import Path

# Before any src import (see tests/conftest.py): keep test runs out of the real Langfuse project.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

# repo root - lets `from src...` and `from demo...` resolve the same way
# tests/conftest.py does for the core suite, without merging the two
# suites into one pytest rootdir config.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
