"""Root test configuration for d:/RMQ.

Ensures the repository root is on ``sys.path`` so tests can import
top-level packages such as ``index`` and ``schemas`` when running
``pytest`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from config import settings as settings_module

# Insert the repository root at the front of sys.path so that
# ``import index`` and ``import schemas`` resolve correctly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic environment variables for settings tests."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HF_TOKEN", "hf_test_fake_token")
    settings_module._settings = None
    yield
    settings_module._settings = None
