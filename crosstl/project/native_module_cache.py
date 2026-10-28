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
    """Reuse validated modules for later executions of byte-identical artifacts.

    A key covers the artifact bytes and suffix and every validation command
    (arguments, standard input and produced module path), with the
    per-execution artifact and scratch paths replaced by placeholders. A hit
    therefore means the exact command sequence already succeeded on identical
    input in this session; only fully successful sequences are stored, so a
    failure is always re-run and reported. Every module the sequence produced,
    intermediate ones included, is stored in a private directory that is
    removed with the cache, and restored at the same per-execution paths.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._directory: Path | None = None
        self._modules: dict[str, list[str]] = {}
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

    def reuse(
        self, key: str, artifact_path: Path, module_paths: Sequence[Path]
    ) -> bool:
        """Restore every stored module at ``module_paths``; ``False`` on a miss.

        ``module_paths`` lists the path each validation command produces, in
        command order, so intermediate modules (for example Metal ``.air``
        before its ``.metallib``) reappear exactly where a fresh validation
        would leave them.
        """

        with self._lock:
            if key not in self._modules:
                self.misses += 1
                return False
            stored = self._modules[key]
            self.hits += 1
        destinations = [
            Path(path) for path in module_paths if Path(path) != Path(artifact_path)
        ]
        if len(destinations) != len(stored):
            raise ValueError("Validated module reuse expects the stored module count")
        for source, destination in zip(stored, destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return True

    def store(
        self, key: str, artifact_path: Path, module_paths: Sequence[Path]
    ) -> None:
        """Keep a copy of every module the successful validation produced.

        Paths equal to the artifact are validation-only steps and store nothing.
        """

        produced = [
            Path(path) for path in module_paths if Path(path) != Path(artifact_path)
        ]
        with self._lock:
            if produced and self._directory is None:
                directory = Path(tempfile.mkdtemp(prefix="crosstl-native-modules-"))
                weakref.finalize(self, shutil.rmtree, directory, True)
                self._directory = directory
            root = self._directory
        stored = []
        for index, path in enumerate(produced):
            destination = root / f"{key}-{index}{path.suffix}"
            shutil.copyfile(path, destination)
            stored.append(str(destination))
        with self._lock:
            self._modules[key] = stored
