"""Central logging infrastructure.

This module is the ONLY place in the project that configures logging
handlers/format. Every other module must only call
``logging.getLogger(__name__)`` and must never call
``logging.basicConfig()`` or attach handlers itself (engineering standard
#2, Logging Standards).

Entry points (``app/streamlit_app.py``, ``evaluation/run_baselines.py``,
each ``index/build_*.py``, etc.) call ``setup_logging()`` exactly once at
startup, before doing anything else.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Infrastructure defaults, not user-facing config: kept as module constants
# rather than settings fields until a concrete requirement (e.g. per-deploy
# rotation policy) justifies exposing them through config.settings.
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT = 5

# Sentinel attribute used to mark handlers this function installed, so
# repeated calls (e.g. Streamlit's rerun-on-interaction model) don't stack
# duplicate handlers onto the root logger.
_HANDLER_MARKER = "_medrag_managed_handler"


def setup_logging(log_filename: str = "app.log") -> None:
    """Configure the root logger's handlers and format.

    Idempotent: safe to call multiple times (e.g. from a Streamlit script
    that reruns on every interaction) without duplicating log output.
    Reads level and output directory from :func:`config.settings.get_settings`.

    Args:
        log_filename: Name of the rotating log file created under the
            configured ``log_dir``. Callers that want per-component log
            files (e.g. ``build_corpus.log``) may override this.

    Raises:
        ValueError: if the configured ``log_level`` is not a recognized
            logging level name (propagated from ``Settings.log_level_int``).
        OSError: if ``log_dir`` cannot be created (e.g. permissions issue).
    """
    settings = get_settings()
    level = settings.log_level_int()

    log_dir: Path = settings.log_dir
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Re-raised after writing to stderr directly, since the logging
        # system itself isn't configured yet at this point.
        sys.stderr.write(f"Failed to create log directory {log_dir!r}\n")
        raise

    # We configure the ROOT logger (not __name__) intentionally: this is the
    # single point of control so every module in the project inherits the
    # same handlers/format via propagation, without needing its own setup.
    root_logger = logging.getLogger()

    # Remove any handlers this function previously installed, so re-running
    # setup_logging() (e.g. Streamlit reruns) doesn't duplicate output.
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _HANDLER_MARKER, True)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        filename=log_dir / log_filename,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARKER, True)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, log_dir=%s, file=%s",
        logging.getLevelName(level),
        log_dir,
        log_filename,
    )
