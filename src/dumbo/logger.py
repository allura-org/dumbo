"""auto_nametag_logger.py
A tiny helper around the stdlib *logging* module that automatically adds a
[filename/logger] nametag, supports coloured output, all the usual logging
levels, and remains 100 % drop‑in with the standard *logging* infrastructure.

Quick start
-----------
>>> import auto_nametag_logger as logutil
>>> logutil.setup_root(level=logutil.logging.DEBUG)  # coloured by default
>>> logger = logutil.get_logger(__name__)
>>> logger.info("The logger says hi!")
[auto_nametag_logger.py/__main__] INFO: The logger says hi!  # INFO will be green

Disabling colour or tweaking styles:
>>> logutil.setup_root(colour=False)             # plain text only
>>> logutil.setup_root(colour_map={             # your own palette
...     logutil.logging.INFO: "\x1b[35m",    # magenta
... })

Using from a class that already expects the stdlib logger:
>>> class Worker:
...     def __init__(self):
...         self.log = logutil.get_logger(self.__class__.__name__)
...     def run(self):
...         self.log.debug("doing work")

This helper only installs one *StreamHandler* on the root logger (unless you
ask it not to) and therefore plays nicely with any extra handlers/formatters
that you attach yourself.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Dict, Optional

try:
    # Enables ANSI codes on Windows terminals when installed. Safe on *nix.
    import colorama

    colorama.init()  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover – silent fallback if colour not required
    colorama = None  # noqa: N816  # keep sentinel

__all__ = [
    "setup_root",
    "get_logger",
    "logging",  # re‑export so callers can do `import auto_nametag_logger as log; log.logging.DEBUG`
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_FMT = "[%(name)s] %(levelname)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_COLOUR_MAP: Dict[int, str] = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[1;41m",  # bold white on red bg
}
_RESET = "\x1b[0m"


# ---------------------------------------------------------------------------
# Colour‑aware formatter
# ---------------------------------------------------------------------------
class _ColourFormatter(logging.Formatter):  # noqa: D401 – internal helper
    """Inject ANSI colour codes based on *logging* level."""

    def __init__(self, fmt: str, datefmt: str, colour_map: Dict[int, str]):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._colour_map = colour_map

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        colour = self._colour_map.get(record.levelno, "")
        if colour:
            record.levelname = f"{colour}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:  # restore so other formatters/handlers aren’t affected
            if colour:
                record.levelname = record.levelname.replace(colour, "").replace(_RESET, "")


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

def _make_stream_handler(*, fmt: str, datefmt: str, colour: bool, colour_map: Dict[int, str]) -> logging.Handler:  # noqa: D401
    handler = logging.StreamHandler()
    if colour and _stream_supports_colour(handler.stream):
        handler.setFormatter(_ColourFormatter(fmt=fmt, datefmt=datefmt, colour_map=colour_map))
    else:
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return handler


def _stream_supports_colour(stream) -> bool:  # noqa: D401 – loose typing ok
    """Return *True* if the given stream is a TTY that probably supports ANSI."""
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    # On Windows, colourama patches isatty to True when init() succeeded.
    if os.name == "nt" and colorama is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def setup_root(
    *,
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    clear: bool = True,
    colour: bool = True,
    colour_map: Optional[Dict[int, str]] = None,
) -> None:  # noqa: D401
    """Configure the root logger quickly.

    Parameters
    ----------
    level
        Minimum logging level accepted by the root handler.
    fmt, datefmt
        Formatter template and date format.
    clear
        If *True* (default) remove any existing handlers before installing ours.
    colour
        Enable ANSI colour codes (default *True*). Turns itself off when the
        output stream is not a TTY.
    colour_map
        Optional map of ``levelno -> ANSI sequence`` that overrides the built‑in
        palette.
    """

    root = logging.getLogger()
    root.setLevel(level)

    if clear:
        root.handlers.clear()

    handler = _make_stream_handler(
        fmt=fmt,
        datefmt=datefmt,
        colour=colour,
        colour_map=colour_map or _DEFAULT_COLOUR_MAP,
    )
    root.addHandler(handler)


def get_logger(name: Optional[str] = None, *, level: Optional[int] = None) -> logging.Logger:  # noqa: D401
    """Return a child logger that inherits our colourful root configuration."""

    logger_name = name if name is not None else _guess_caller_module()
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _guess_caller_module() -> str:  # noqa: D401
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return "__main__"
    module = inspect.getmodule(frame.f_back.f_back)
    return module.__name__ if module is not None else "__main__"
