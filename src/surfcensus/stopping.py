"""How a command stops at SIGTERM and Ctrl-C (decision D35, review of the quick start).

Three rules, each from a fault that a review reproduced:

1. A signal handler takes no lock. :class:`StopFlag` is a stop request that a handler can
   make: a plain attribute, and a wait that polls it. ``threading.Event.set`` takes the
   lock of the event, and a second signal that arrives while the first handler, or
   ``Event.wait`` in the main thread, holds that lock blocks on it for good.
2. Once a command has taken its SIGTERM, or has served the app to its end, SIGTERM (and
   after the closing line SIGINT) is ignored by the process (``SIG_IGN``), not handled by a
   Python function. CPython puts ``SIG_DFL`` back for every Python-level handler while the
   interpreter finalizes, so a Python handler cannot protect the last moments of the
   process; an ignored signal stays ignored.
3. While a command prepares (inventory, metrics, render), SIGTERM ends it as Ctrl-C does,
   by ``KeyboardInterrupt``, which the batches, the leases and the locks are written for
   (``batches.run_batches``), and not by an exception they do not expect. Once that
   ``KeyboardInterrupt`` is raised, SIGTERM and SIGINT are ignored, so that nothing
   interrupts the unwinding a second time.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

POLL_SECONDS = 0.05
"""How often :meth:`StopFlag.wait` looks at the flag."""

TERMINATED = 128 + int(signal.SIGTERM)
"""The exit code of a command that a SIGTERM ended while it prepared (143)."""
INTERRUPTED = 128 + int(signal.SIGINT)
"""The exit code of a command that Ctrl-C ended (130)."""


@runtime_checkable
class StopRequest(Protocol):
    """What a call that serves until it is told to stop reads: a :class:`StopFlag`, or a
    ``threading.Event`` of a caller, which has the same two methods."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class StopFlag:
    """A stop request that a signal handler can make. ``set`` takes no lock, and ``wait``
    polls with short sleeps, which a signal interrupts and its handler runs in."""

    __slots__ = ("_set",)

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float | None = None) -> bool:
        """Until the flag is set or ``timeout`` seconds have passed; whether it is set."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._set:
            if deadline is None:
                time.sleep(POLL_SECONDS)
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(POLL_SECONDS, left))
        return self._set


def _ignore(*signums: int) -> None:
    for signum in signums:
        signal.signal(signum, signal.SIG_IGN)


class CommandSignals:
    """SIGTERM and Ctrl-C of one command, from its start to the end of the process.

    Before ``serving`` is set (the preparation, and the whole of a phase command): the first
    SIGTERM ignores SIGTERM and SIGINT from then on and raises ``KeyboardInterrupt``, as
    Ctrl-C does; when Ctrl-C is already ending the command, it raises nothing. Once
    ``serving`` is set: the first SIGTERM ignores SIGTERM and sets :attr:`stop`, which ends
    the serving in order. Ctrl-C keeps Python's meaning in both: it raises
    ``KeyboardInterrupt``, so that a second Ctrl-C is a person's way out of a stop that
    waits. :meth:`served` ignores both once the serving has ended and its closing line is
    out.
    """

    def __init__(self) -> None:
        self.stop = StopFlag()
        self.serving = False
        self.terminated = False
        """A SIGTERM was taken."""
        self.interrupted = False
        """Ctrl-C was taken (by this object's handler)."""
        self.ended = False
        """The command is ending at a signal, or served to its end: nothing is put back."""

    def on_sigterm(self, signum: int, frame: object) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # first: every later one is a no-op
        self.terminated = True
        if self.serving:
            self.stop.set()
            return
        if self.interrupted:
            return  # Ctrl-C is ending the command already
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    def on_sigint(self, signum: int, frame: object) -> None:
        self.interrupted = True
        raise KeyboardInterrupt

    def exit_code(self) -> int:
        """The code of a command that a ``KeyboardInterrupt`` ends: 143 when a SIGTERM
        began the end of the preparation, 130 otherwise."""
        return TERMINATED if self.terminated and not self.interrupted else INTERRUPTED

    def served(self) -> None:
        """The app was served to its end (the server and the renderer have stopped, and
        the closing line follows): from here on neither SIGTERM nor SIGINT ends the
        process."""
        self.ended = True
        if threading.current_thread() is threading.main_thread():
            _ignore(signal.SIGTERM, signal.SIGINT)

    def ending(self) -> None:
        """The command ends at a signal: a late one must not kill it while it finalizes."""
        self.ended = True
        if threading.current_thread() is threading.main_thread():
            _ignore(signal.SIGTERM, signal.SIGINT)


@contextmanager
def command_signals(
    said: Callable[[CommandSignals], None] | None = None,
) -> Iterator[CommandSignals]:
    """Handle SIGTERM and Ctrl-C with a :class:`CommandSignals` for the length of the block.

    A ``KeyboardInterrupt`` that ends the block (Ctrl-C, or a SIGTERM before ``serving``)
    is turned into ``SystemExit`` with :meth:`CommandSignals.exit_code`, after ``said`` has
    been called with the signals (to print a line) and both signals have been set to
    ``SIG_IGN``. When no signal was taken and nothing was served (a command that was
    refused, ``--no-serve``, a caller in the same process such as a test), the handlers
    that were there are put back. Outside the main thread no handler can be set: the block
    runs as it is.
    """
    signals = CommandSignals()
    if threading.current_thread() is not threading.main_thread():
        yield signals
        return
    earlier_term: Any = signal.signal(signal.SIGTERM, signals.on_sigterm)
    earlier_int: Any = signal.getsignal(signal.SIGINT)
    if earlier_int is signal.default_int_handler:
        # SIG_IGN (a job of a shell without job control) stays as it is, and so does a
        # handler of a caller's own
        signal.signal(signal.SIGINT, signals.on_sigint)
    try:
        yield signals
    except KeyboardInterrupt:
        signals.ending()
        if said is not None:
            said(signals)
        raise SystemExit(signals.exit_code()) from None
    finally:
        if not (signals.ended or signals.terminated or signals.interrupted):
            # None: a handler that was not set from Python, which cannot be put back
            if earlier_term is not None:
                signal.signal(signal.SIGTERM, earlier_term)
            if earlier_int is signal.default_int_handler:
                signal.signal(signal.SIGINT, earlier_int)
