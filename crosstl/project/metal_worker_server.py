"""A long-lived Metal worker process that executes dispatch requests in order.

Starting the worker once per dispatch costs a process launch, a device
lookup, a library load and a pipeline compilation each time. The worker's
``--serve`` mode keeps the device, queue, loaded libraries and compiled
pipelines, and reads one JSON request per line. Each request still gets fresh
zeroed allocations and its own command buffer, so requests stay independent.

Each reply is one line: ``+`` followed by the response object, or ``-``
followed by ``{"failure": message, "error": details?}``. A reply is returned
as the :class:`subprocess.CompletedProcess` a one-shot worker would have
produced, so callers handle both modes alike.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time

from .runtime_verification import RuntimeAdapterSetupError

# The largest write or read per readiness event.
_CHUNK = 1 << 20


class MetalWorkerServer:
    """One ``--serve`` worker process owned by the process that started it."""

    def __init__(self, worker):
        self.worker = str(worker)
        self.command = [self.worker, "--serve"]
        self.owner = os.getpid()
        self._stderr = tempfile.TemporaryFile()
        self._buffer = bytearray()
        self._closed = False
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                start_new_session=True,
            )
        except BaseException:
            self._stderr.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    @property
    def usable(self):
        """Whether this process started the worker and the worker still runs."""
        return (
            not self._closed
            and self.owner == os.getpid()
            and self.process.poll() is None
        )

    def exchange(self, request_text, timeout_seconds):
        """Send one request and return its reply as a completed process."""
        data = memoryview(request_text.encode("utf-8") + b"\n")
        deadline = time.monotonic() + timeout_seconds
        stdin = self.process.stdin.fileno()
        stdout = self.process.stdout.fileno()
        written = 0
        with selectors.DefaultSelector() as selector:
            selector.register(stdin, selectors.EVENT_WRITE)
            selector.register(stdout, selectors.EVENT_READ)
            while True:
                newline = self._buffer.find(b"\n")
                if newline >= 0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._expire(timeout_seconds)
                for key, _ in selector.select(remaining):
                    if key.fd == stdin:
                        try:
                            written += os.write(stdin, data[written : written + _CHUNK])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            written = len(data)
                        if written >= len(data):
                            selector.unregister(stdin)
                        continue
                    try:
                        chunk = os.read(stdout, _CHUNK)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        return self._exited()
                    self._buffer += chunk
        line = bytes(self._buffer[:newline])
        del self._buffer[: newline + 1]
        return self._reply(line)

    def _reply(self, line):
        status, body = line[:1], line[1:].decode("utf-8")
        if status == b"+":
            return subprocess.CompletedProcess(self.command, 0, body, "")
        if status != b"-":
            self.terminate()
            raise RuntimeAdapterSetupError(
                "Metal worker returned a malformed reply.",
                details={"target": "metal", "reasonKind": "worker-protocol-invalid"},
            )
        failure = json.loads(body)
        stdout = (
            json.dumps({"error": failure["error"]}, sort_keys=True)
            if "error" in failure
            else ""
        )
        return subprocess.CompletedProcess(
            self.command, 1, stdout, f"{failure['failure']}\n"
        )

    def _stderr_text(self):
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace")

    def _exited(self):
        returncode = self.process.wait()
        stdout = bytes(self._buffer).decode("utf-8", errors="replace")
        stderr = self._stderr_text()
        self.terminate()
        return subprocess.CompletedProcess(
            self.command, returncode or 1, stdout, stderr
        )

    def _expire(self, timeout_seconds):
        self._kill()
        stdout = bytes(self._buffer).decode("utf-8", errors="replace")
        stderr = self._stderr_text()
        self.terminate()
        raise RuntimeAdapterSetupError(
            "Metal native operation exceeded its deadline.",
            details={
                "target": "metal",
                "reasonKind": "operation-timeout",
                "timeoutSeconds": timeout_seconds,
                "command": list(self.command),
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def _kill(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def terminate(self):
        """Stop the worker; a worker started by another process is left alone."""
        if self._closed:
            return
        self._closed = True
        if self.owner == os.getpid():
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._kill()
                self.process.wait()
        self._stderr.close()
