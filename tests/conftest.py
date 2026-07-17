"""Root test configuration for d:/RMQ.

Ensures the repository root is on ``sys.path`` so tests can import
top-level packages such as ``index`` and ``schemas`` when running
``pytest`` from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the repository root at the front of sys.path so that
# ``import index`` and ``import schemas`` resolve correctly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
