from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass

WATCHDOG_SECONDS = 10.0

SocketHandler = Callable[[socket.socket, threading.Event], None]


@dataclass(frozen=True, slots=True)
class DaemonCall[ResultT]:
    """Avoid joining a call while asserting watchdog behavior."""

    _entered: threading.Event
    _done: threading.Event
    _thread: threading.Thread
    _results: list[ResultT]
    _errors: list[BaseException]

    @classmethod
    def start(cls, target: Callable[[], ResultT]) -> DaemonCall[ResultT]:
        entered = threading.Event()
        done = threading.Event()
        results: list[ResultT] = []
        errors: list[BaseException] = []

        def run() -> None:
            entered.set()
            try:
                results.append(target())
            except BaseException as error:  # noqa: BLE001 -- re-raised by result
                errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        call = cls(
            _entered=entered,
            _done=done,
            _thread=thread,
            _results=results,
            _errors=errors,
        )
        thread.start()
        return call

    def wait_until_entered(self) -> None:
        if not self._entered.wait(timeout=WATCHDOG_SECONDS):
            raise TimeoutError("daemon call did not enter")

    def result(self) -> ResultT:
        if not self._done.wait(timeout=WATCHDOG_SECONDS):
            raise TimeoutError("daemon call exceeded the external watchdog")
        if self._errors:
            raise self._errors[0]
        return self._results[0]


class LocalSocketServer:
    def __init__(self, handler: SocketHandler) -> None:
        self._handler = handler
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._entered = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self._entered.set()
            conn.settimeout(WATCHDOG_SECONDS)
            try:
                self._handler(conn, self._stop)
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    conn.close()
            return

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_until_entered(self) -> None:
        if not self._entered.wait(timeout=WATCHDOG_SECONDS):
            raise TimeoutError("socket server did not accept the request")

    def __enter__(self) -> LocalSocketServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=WATCHDOG_SECONDS)
        if self._thread.is_alive():
            raise TimeoutError("socket server did not stop")
