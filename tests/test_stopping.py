"""The stop request and the signal handlers of the commands (``surfcensus.stopping``;
review of the quick start, F1 to F3)."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from surfcensus.stopping import (
    INTERRUPTED,
    TERMINATED,
    CommandSignals,
    StopFlag,
    StopRequest,
    command_signals,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")


@pytest.fixture(autouse=True)
def handlers_restored() -> Iterator[None]:
    """What a test leaves of SIGTERM and SIGINT is put back; SIGINT is Python's own."""
    earlier_term = signal.getsignal(signal.SIGTERM)
    earlier_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGTERM, earlier_term)
    signal.signal(signal.SIGINT, earlier_int)


def _to_self(signum: int) -> None:
    os.kill(os.getpid(), signum)
    time.sleep(0.01)  # the handler runs in this thread, at the next instruction


def test_a_stop_flag_is_set_waited_for_and_takes_no_lock() -> None:
    flag = StopFlag()
    assert isinstance(flag, StopRequest) and isinstance(threading.Event(), StopRequest)
    assert not flag.is_set()
    started = time.monotonic()
    assert flag.wait(0.12) is False
    assert 0.1 <= time.monotonic() - started < 2
    setter = threading.Timer(0.05, flag.set)
    setter.start()
    started = time.monotonic()
    assert flag.wait(30) is True and time.monotonic() - started < 5
    assert flag.wait() is True and flag.wait(0) is True
    # set() calls nothing of the threading module, so a handler that calls it waits for
    # no lock, whatever the main thread holds
    called: list[str] = []

    def profile(frame: Any, event: str, arg: object) -> None:
        if event == "call" and frame.f_code.co_filename == threading.__file__:
            called.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        StopFlag().set()
    finally:
        sys.setprofile(None)
    assert called == []


def test_a_stop_flag_is_set_by_a_handler_while_its_wait_runs() -> None:
    """A burst of signals while the main thread waits on the flag: each handler sets it and
    the wait returns. With a ``threading.Event`` this is where a second handler could wait
    for the lock that the wait or the first handler holds."""
    flag = StopFlag()
    earlier = signal.signal(signal.SIGUSR1, lambda *_: flag.set())
    try:
        for _ in range(5):
            threading.Timer(0.02, os.kill, (os.getpid(), signal.SIGUSR1)).start()
        assert flag.wait(30) is True
        time.sleep(0.1)  # the rest of the burst arrives after the wait
    finally:
        signal.signal(signal.SIGUSR1, earlier)


def test_before_serving_sigterm_raises_keyboard_interrupt_and_ignores_both_after() -> None:
    signals = CommandSignals()
    signal.signal(signal.SIGTERM, signals.on_sigterm)
    with pytest.raises(KeyboardInterrupt):
        _to_self(signal.SIGTERM)
    assert signals.terminated and not signals.interrupted
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    _to_self(signal.SIGTERM)  # ignored
    _to_self(signal.SIGINT)
    assert signals.exit_code() == TERMINATED == 143


def test_while_serving_sigterm_sets_the_stop_and_ignores_itself_after() -> None:
    signals = CommandSignals()
    signals.serving = True
    signal.signal(signal.SIGTERM, signals.on_sigterm)
    _to_self(signal.SIGTERM)
    assert signals.stop.is_set() and signals.terminated
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    # Ctrl-C keeps its meaning while the app stops
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    # a handler called again from within itself (a signal between two of its statements)
    # finds nothing to wait for
    signals.on_sigterm(signal.SIGTERM, None)
    signals.served()
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_ctrl_c_first_keeps_its_code_and_a_later_sigterm_raises_nothing() -> None:
    seen: list[str] = []
    with pytest.raises(SystemExit) as caught, command_signals(lambda s: seen.append("said")):
        try:
            _to_self(signal.SIGINT)
        finally:
            _to_self(signal.SIGTERM)  # raises nothing: Ctrl-C is ending the block
    assert caught.value.code == INTERRUPTED == 130 and seen == ["said"]
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_a_block_that_takes_no_signal_puts_the_handlers_back() -> None:
    def own(*_: object) -> None:
        return None

    signal.signal(signal.SIGTERM, own)
    with command_signals() as signals:
        assert signal.getsignal(signal.SIGTERM) == signals.on_sigterm
        assert signal.getsignal(signal.SIGINT) == signals.on_sigint
    assert signal.getsignal(signal.SIGTERM) is own
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    # an exception of the block, too
    with pytest.raises(ValueError), command_signals():
        raise ValueError
    assert signal.getsignal(signal.SIGTERM) is own
    # SIGINT that is ignored (a job of a shell without job control) stays ignored
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with command_signals():
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_outside_the_main_thread_no_handler_is_set() -> None:
    seen: list[object] = []

    def block() -> None:
        with command_signals() as signals:
            seen.append(signals)

    earlier = signal.getsignal(signal.SIGTERM)
    thread = threading.Thread(target=block)
    thread.start()
    thread.join()
    assert isinstance(seen[0], CommandSignals) and signal.getsignal(signal.SIGTERM) is earlier
