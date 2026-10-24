"""Ordered, bounded progress logging outside the numerical computation."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple, TextIO


class _Record(NamedTuple):
    message: str
    timestamp: str
    log_path: str | None
    timing_path: str | None


def _emit(record: _Record, handles: dict[str, TextIO]) -> None:
    line = f"[trajectory] {record.message}"
    print(line, flush=True)
    for path, text in (
        (record.log_path, line + "\n"),
        (record.timing_path, f"{record.timestamp}\t{record.message}\n"),
    ):
        if not path:
            continue
        try:
            if path not in handles:
                handles[path] = Path(path).open("a", encoding="utf-8")
            handles[path].write(text)
            handles[path].flush()
        except OSError:
            # Preserve the existing best-effort file logging behavior.
            pass


def _close_handles(handles: dict[str, TextIO]) -> None:
    for handle in handles.values():
        try:
            handle.close()
        except OSError:
            pass


class _Writer:
    def __init__(self):
        self.queue: Queue[_Record | None] = Queue(maxsize=128)
        self.error: BaseException | None = None
        self.thread = Thread(target=self._run, name="wrf-progress-writer", daemon=True)
        self.thread.start()

    def _run(self):
        handles: dict[str, TextIO] = {}
        try:
            while (record := self.queue.get()) is not None:
                try:
                    _emit(record, handles)
                except BaseException as error:
                    # Drain even after a failed stdout write so close cannot deadlock.
                    if self.error is None:
                        self.error = error
        finally:
            _close_handles(handles)

    def append(self, record: _Record):
        if self.error is not None:
            raise self.error
        self.queue.put(record)

    def close(self, *, raise_errors: bool = True):
        self.queue.put(None)
        self.thread.join()
        if raise_errors and self.error is not None:
            raise self.error


_active_writer: ContextVar[_Writer | None] = ContextVar("wrf_progress_writer", default=None)


@contextmanager
def progress_logging():
    """Flush all queued messages on normal or exceptional trajectory exit."""
    writer = _Writer()
    token = _active_writer.set(writer)
    try:
        try:
            yield
        except BaseException:
            writer.close(raise_errors=False)
            raise
        else:
            writer.close()
    finally:
        _active_writer.reset(token)


def write_progress(message: str, log_path: str | None, timing_path: str | None) -> None:
    record = _Record(message, datetime.now().astimezone().isoformat(), log_path, timing_path)
    writer = _active_writer.get()
    if writer is not None:
        writer.append(record)
    else:
        handles: dict[str, TextIO] = {}
        try:
            _emit(record, handles)
        finally:
            _close_handles(handles)
