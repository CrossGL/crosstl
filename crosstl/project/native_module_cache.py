"""Session reuse of native runtime modules that already passed validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import weakref
from pathlib import Path
from typing import Any, Sequence

ARTIFACT_PLACEHOLDER = "{artifact}"
SCRATCH_PLACEHOLDER = "{scratch}"


def _normalized(value: Any, artifact: str, scratch: str) -> Any:
    if value is None:
        return None
    text = str(value)
    if text == artifact:
        return ARTIFACT_PLACEHOLDER
    if (
        text == scratch
        or text.startswith(scratch + "/")
        or text.startswith(scratch + "\\")
    ):
        return SCRATCH_PLACEHOLDER + text[len(scratch) :].replace("\\", "/")
    return text


class ValidatedModuleCache:
    """Reuse a validated module for later executions of byte-identical artifacts.

    A key covers the artifact bytes and suffix and every validation command
    (arguments, standard input and produced module path), with the
    per-execution artifact and scratch paths replaced by placeholders. A hit
    therefore means the exact command sequence already succeeded on identical
    input in this session; only fully successful sequences are stored, so a
    failure is always re-run and reported. Stored modules live in a private
    directory that is removed with the cache.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._directory: Path | None = None
        self._modules: dict[str, str | None] = {}
        self.hits = 0
        self.misses = 0

    def key(self, artifact_path: Path, scratch: Path, commands: Sequence[Any]) -> str:
        artifact = str(artifact_path)
        root = str(scratch)
        payload = {
            "artifactSha256": (
                hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
            ),
            "artifactSuffix": Path(artifact_path).suffix,
            "commands": [
                {
                    "action": spec.action,
                    "command": [
                        _normalized(item, artifact, root) for item in spec.command
                    ],
                    "inputText": spec.input_text,
                    "modulePath": _normalized(spec.module_path, artifact, root),
                }
                for spec in commands
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def reuse(self, key: str, artifact_path: Path, module_path: Path) -> bool:
        """Place the stored module at ``module_path``; ``False`` on a miss."""

        with self._lock:
            if key not in self._modules:
                self.misses += 1
                return False
            stored = self._modules[key]
            self.hits += 1
        if stored is None:
            # The key's module path was the artifact itself: validation only.
            return True
        destination = Path(module_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stored, destination)
        return True

    def store(self, key: str, artifact_path: Path, module_path: Path) -> None:
        if Path(module_path) == Path(artifact_path):
            with self._lock:
                self._modules[key] = None
            return
        with self._lock:
            if self._directory is None:
                directory = Path(tempfile.mkdtemp(prefix="crosstl-native-modules-"))
                weakref.finalize(self, shutil.rmtree, directory, True)
                self._directory = directory
            stored = self._directory / f"{key}{Path(module_path).suffix}"
        shutil.copyfile(module_path, stored)
        with self._lock:
            self._modules[key] = str(stored)
