"""Logging for the package: library code logs, the CLI prints results.

Every module logs through :func:`get_logger`, so tests capture messages with ``caplog`` and
a census run can attach one redacting filter (milestone 5).
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "surfcensus"


class _CliHandler(logging.Handler):
    """Writes ``level: message`` lines to whatever ``sys.stderr`` is when a record arrives.

    The stream is looked up per record, not kept: a test runner or a caller that replaces
    ``sys.stderr`` after the handler was installed still gets the lines, and a stream that
    has been closed since is never written to.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_cli_logging(verbose: bool = False) -> None:
    """One stderr handler on the package logger: warnings and above, or everything from INFO
    with ``verbose``. Calling it again replaces the handler, it never adds a second one."""
    logger = get_logger()
    for handler in [h for h in logger.handlers if isinstance(h, _CliHandler)]:
        logger.removeHandler(handler)
    handler = _CliHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
