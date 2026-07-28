"""Deterministic cache storage for bounded native deferred compilation."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .native_deferred_compilation import (
    NATIVE_DEFERRED_COMPILATION_REQUEST_KIND,
    validate_native_deferred_compilation_request,
)

NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_KIND = (
    "crosstl-native-deferred-compilation-cache-entry"
)
NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION = 1

_ERROR_PREFIX = "project.native-deferred-compilation-cache"
_ENTRY_FILENAME = "entry.json"
_ENTRY_FIELDS = frozenset(
    (
        "kind",
        "schemaVersion",
        "success",
        "cacheKey",
        "requestKind",
        "requestHash",
        "toolchain",
        "target",
        "expectedInterfaceIdentity",
        "output",
    )
)
_TOOLCHAIN_FIELDS = frozenset(("name", "version", "executableHash"))
_TARGET_FIELDS = frozenset(("outputFormat",))
_OUTPUT_FIELDS = frozenset(("relativePath", "format", "sizeBytes", "hash"))
_SHA256_FIELDS = frozenset(("algorithm", "value"))
_LOWER_HEX = frozenset("0123456789abcdef")


class NativeDeferredCompilationCacheError(ValueError):
    """A deferred native compilation cache operation is not trustworthy."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "$",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = (
            code if code.startswith(f"{_ERROR_PREFIX}.") else f"{_ERROR_PREFIX}.{code}"
        )
        self.message = message
        self.path = path
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(f"{path}: {message} ({self.code})")

    def to_json(self) -> dict[str, Any]:
        """Return the stable diagnostic representation."""

        payload: dict[str, Any] = {
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


def lookup_native_deferred_compilation_cache(
    cache_root: str | os.PathLike[str],
    request: Mapping[str, Any],
    toolchain_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a verified successful cache entry, or ``None`` for a clean miss.

    The returned mapping contains ``entry`` metadata, immutable ``outputBytes``,
    and the verified absolute ``entryPath`` and ``outputPath``. Existing data
    at the deterministic entry path is either a complete valid hit or a
    structured corruption error; incomplete entries are never treated as
    misses.
    """

    normalized_request = _validated_request(request)
    normalized_toolchain = _validated_toolchain_identity(toolchain_identity)
    request_identity = _request_identity(normalized_request)
    cache_key = _cache_key(
        request_identity["requestHash"],
        normalized_toolchain,
    )
    root = _cache_root(cache_root, create=False)
    if root is None:
        return None

    entry_directory = _entry_directory(root, cache_key)
    _validate_path_chain(root, entry_directory, final_must_be_directory=True)
    if not entry_directory.exists():
        return None
    if not entry_directory.is_dir():
        raise NativeDeferredCompilationCacheError(
            "entry-path-invalid",
            "The deterministic cache entry path is not a directory.",
            path="$.cacheRoot",
            details={"entryPath": str(entry_directory)},
        )

    entry_path = entry_directory / _ENTRY_FILENAME
    entry = _read_entry(entry_path, root=root)
    output = _validated_entry(
        entry,
        cache_key=cache_key,
        request_identity=request_identity,
        toolchain_identity=normalized_toolchain,
    )
    output_path = _entry_output_path(
        root,
        entry_directory,
        output["relativePath"],
    )
    output_bytes = _read_output(output_path)
    _verify_output(output, output_bytes, output_path=output_path)
    return {
        "entry": copy.deepcopy(entry),
        "entryPath": str(entry_path),
        "outputPath": str(output_path),
        "outputBytes": output_bytes,
    }


def publish_native_deferred_compilation_cache(
    cache_root: str | os.PathLike[str],
    request: Mapping[str, Any],
    toolchain_identity: Mapping[str, Any],
    output_bytes: bytes,
) -> dict[str, Any]:
    """Atomically publish one successful deferred compilation result.

    Empty and non-byte outputs are rejected. A verified identical entry is
    reused. A different output for the same complete request and toolchain
    identity is reported as a deterministic-output conflict.
    """

    normalized_request = _validated_request(request)
    normalized_toolchain = _validated_toolchain_identity(toolchain_identity)
    request_identity = _request_identity(normalized_request)
    content = _validated_output_bytes(output_bytes)
    cache_key = _cache_key(
        request_identity["requestHash"],
        normalized_toolchain,
    )
    existing = lookup_native_deferred_compilation_cache(
        cache_root,
        normalized_request,
        normalized_toolchain,
    )
    if existing is not None:
        _ensure_repeated_output_matches(existing, content)
        return existing

    root = _cache_root(cache_root, create=True)
    assert root is not None
    entry_directory = _entry_directory(root, cache_key)
    parent = entry_directory.parent
    _create_cache_directories(root, parent)

    output_hash = _sha256(content)
    output_relative_path = _output_relative_path(output_hash["value"])
    entry = {
        "kind": NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_KIND,
        "schemaVersion": NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION,
        "success": True,
        "cacheKey": _sha256_from_digest(cache_key),
        "requestKind": NATIVE_DEFERRED_COMPILATION_REQUEST_KIND,
        "requestHash": copy.deepcopy(request_identity["requestHash"]),
        "toolchain": copy.deepcopy(normalized_toolchain),
        "target": {"outputFormat": request_identity["outputFormat"]},
        "expectedInterfaceIdentity": copy.deepcopy(
            request_identity["interfaceIdentity"]
        ),
        "output": {
            "relativePath": output_relative_path,
            "format": request_identity["outputFormat"],
            "sizeBytes": len(content),
            "hash": output_hash,
        },
    }
    entry_text = _json_text(entry).encode("utf-8")

    temporary_directory: Path | None = None
    try:
        temporary_directory = Path(
            tempfile.mkdtemp(
                dir=parent,
                prefix=f".{cache_key}.",
                suffix=".tmp",
            )
        )
        temporary_output = temporary_directory / PurePosixPath(output_relative_path)
        temporary_output.parent.mkdir(parents=True, exist_ok=False)
        _write_new_file(temporary_output, content)
        _write_new_file(temporary_directory / _ENTRY_FILENAME, entry_text)
        _rename_new_entry(temporary_directory, entry_directory)
        temporary_directory = None
    except NativeDeferredCompilationCacheError:
        raise
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise NativeDeferredCompilationCacheError(
                "publish-failed",
                "The deferred compilation cache entry could not be published.",
                path="$.cacheRoot",
                details={
                    "entryPath": str(entry_directory),
                    "error": str(exc),
                },
            ) from exc
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    published = lookup_native_deferred_compilation_cache(
        root,
        normalized_request,
        normalized_toolchain,
    )
    if published is None:
        raise NativeDeferredCompilationCacheError(
            "publish-incomplete",
            "The deferred compilation cache entry was not visible after publication.",
            path="$.cacheRoot",
            details={"entryPath": str(entry_directory)},
        )
    _ensure_repeated_output_matches(published, content)
    return published


def _validated_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise NativeDeferredCompilationCacheError(
            "request-invalid",
            "Deferred compilation request must be an object.",
            path="$.request",
        )
    try:
        validated = validate_native_deferred_compilation_request(request)
    except (TypeError, ValueError) as exc:
        raise NativeDeferredCompilationCacheError(
            "request-invalid",
            "Deferred compilation request failed contract validation.",
            path="$.request",
            details={"error": str(exc)},
        ) from exc
    normalized = validated if isinstance(validated, Mapping) else request
    try:
        return copy.deepcopy(dict(normalized))
    except (TypeError, ValueError) as exc:
        raise NativeDeferredCompilationCacheError(
            "request-invalid",
            "Deferred compilation request could not be normalized.",
            path="$.request",
            details={"error": str(exc)},
        ) from exc


def _request_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    request_hash = _validated_sha256(
        request.get("requestHash"),
        path="$.request.requestHash",
    )
    target = request.get("target")
    if not isinstance(target, Mapping):
        raise NativeDeferredCompilationCacheError(
            "request-target-invalid",
            "Deferred compilation request target must be an object.",
            path="$.request.target",
        )
    output_format = target.get("outputFormat")
    if not isinstance(output_format, str) or not output_format.strip():
        raise NativeDeferredCompilationCacheError(
            "request-output-format-invalid",
            "Deferred compilation request output format must be nonempty.",
            path="$.request.target.outputFormat",
        )
    interface_identity = _request_interface_identity(request)
    _canonical_json(
        interface_identity,
        path="$.request.expectedInterfaceIdentity",
        code="request-interface-identity-invalid",
    )
    return {
        "requestHash": request_hash,
        "outputFormat": output_format,
        "interfaceIdentity": copy.deepcopy(interface_identity),
    }


def _request_interface_identity(request: Mapping[str, Any]) -> Any:
    identity = request.get("expectedLoaderDescriptor")
    if not isinstance(identity, Mapping):
        raise NativeDeferredCompilationCacheError(
            "request-interface-identity-invalid",
            "Deferred compilation request must expose its expected loader descriptor identity.",
            path="$.request.expectedLoaderDescriptor",
        )
    return identity


def _validated_toolchain_identity(
    identity: Mapping[str, Any],
    *,
    path: str = "$.toolchainIdentity",
) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or frozenset(identity) != _TOOLCHAIN_FIELDS:
        raise NativeDeferredCompilationCacheError(
            "toolchain-identity-invalid",
            "Toolchain identity must contain only name, version, and executableHash.",
            path=path,
        )
    name = identity.get("name")
    version = identity.get("version")
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or any(ord(character) < 32 for character in name)
    ):
        raise NativeDeferredCompilationCacheError(
            "toolchain-name-invalid",
            "Toolchain name must be a nonempty trimmed string without control characters.",
            path=f"{path}.name",
        )
    if (
        not isinstance(version, str)
        or not version
        or version != version.strip()
        or any(ord(character) < 32 for character in version)
    ):
        raise NativeDeferredCompilationCacheError(
            "toolchain-version-invalid",
            "Toolchain version must be a nonempty trimmed string without control characters.",
            path=f"{path}.version",
        )
    executable_hash = _validated_sha256(
        identity.get("executableHash"),
        path=f"{path}.executableHash",
    )
    return {
        "name": name,
        "version": version,
        "executableHash": executable_hash,
    }


def _cache_key(
    request_hash: Mapping[str, str],
    toolchain_identity: Mapping[str, Any],
) -> str:
    payload = {
        "requestHash": copy.deepcopy(dict(request_hash)),
        "toolchainIdentity": copy.deepcopy(dict(toolchain_identity)),
    }
    encoded = _canonical_json(payload, path="$", code="cache-key-invalid").encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _cache_root(
    value: str | os.PathLike[str],
    *,
    create: bool,
) -> Path | None:
    try:
        raw_value = os.fspath(value)
    except TypeError as exc:
        raise NativeDeferredCompilationCacheError(
            "cache-root-invalid",
            "Cache root must be a filesystem path.",
            path="$.cacheRoot",
        ) from exc
    if not isinstance(raw_value, str) or not raw_value or "\x00" in raw_value:
        raise NativeDeferredCompilationCacheError(
            "cache-root-invalid",
            "Cache root must be a nonempty filesystem path.",
            path="$.cacheRoot",
        )
    raw_path = Path(raw_value).expanduser()
    try:
        if raw_path.is_symlink():
            raise NativeDeferredCompilationCacheError(
                "cache-root-symlink",
                "Cache root must not be a symbolic link.",
                path="$.cacheRoot",
                details={"cacheRoot": str(raw_path)},
            )
        if not raw_path.exists():
            if not create:
                return None
            raw_path.mkdir(parents=True, exist_ok=True)
        if raw_path.is_symlink():
            raise NativeDeferredCompilationCacheError(
                "cache-root-symlink",
                "Cache root must not be a symbolic link.",
                path="$.cacheRoot",
                details={"cacheRoot": str(raw_path)},
            )
        if not raw_path.is_dir():
            raise NativeDeferredCompilationCacheError(
                "cache-root-invalid",
                "Cache root must be a directory.",
                path="$.cacheRoot",
                details={"cacheRoot": str(raw_path)},
            )
        return raw_path.resolve(strict=True)
    except NativeDeferredCompilationCacheError:
        raise
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationCacheError(
            "cache-root-invalid",
            "Cache root could not be prepared.",
            path="$.cacheRoot",
            details={"cacheRoot": str(raw_path), "error": str(exc)},
        ) from exc


def _entry_directory(root: Path, cache_key: str) -> Path:
    return (
        root
        / "entries"
        / f"v{NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION}"
        / cache_key[:2]
        / cache_key[2:4]
        / cache_key
    )


def _create_cache_directories(root: Path, directory: Path) -> None:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise NativeDeferredCompilationCacheError(
            "path-escape",
            "Cache entry directory resolves outside the cache root.",
            path="$.cacheRoot",
            details={"entryPath": str(directory)},
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(exist_ok=True)
        except FileExistsError:
            pass
        if current.is_symlink():
            raise NativeDeferredCompilationCacheError(
                "path-symlink",
                "Cache entry path must not contain symbolic links.",
                path="$.cacheRoot",
                details={"path": str(current)},
            )
        if not current.is_dir():
            raise NativeDeferredCompilationCacheError(
                "path-invalid",
                "Cache entry path contains a non-directory component.",
                path="$.cacheRoot",
                details={"path": str(current)},
            )
        _require_within_root(root, current, path="$.cacheRoot")


def _validate_path_chain(
    root: Path,
    path_value: Path,
    *,
    final_must_be_directory: bool,
) -> None:
    try:
        relative = path_value.relative_to(root)
    except ValueError as exc:
        raise NativeDeferredCompilationCacheError(
            "path-escape",
            "Cache entry path is outside the cache root.",
            path="$.cacheRoot",
            details={"path": str(path_value)},
        ) from exc
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise NativeDeferredCompilationCacheError(
                "path-symlink",
                "Cache entry path must not contain symbolic links.",
                path="$.cacheRoot",
                details={"path": str(current)},
            )
        if not current.exists():
            return
        is_final = index == len(relative.parts) - 1
        if (not is_final or final_must_be_directory) and not current.is_dir():
            raise NativeDeferredCompilationCacheError(
                "path-invalid",
                "Cache entry path contains a non-directory component.",
                path="$.cacheRoot",
                details={"path": str(current)},
            )
        _require_within_root(root, current, path="$.cacheRoot")


def _read_entry(path_value: Path, *, root: Path) -> dict[str, Any]:
    _validate_path_chain(root, path_value, final_must_be_directory=False)
    if not path_value.exists():
        raise NativeDeferredCompilationCacheError(
            "entry-incomplete",
            "Cache entry manifest is missing.",
            path="$.cacheEntry",
            details={"entryPath": str(path_value)},
        )
    if not path_value.is_file():
        raise NativeDeferredCompilationCacheError(
            "entry-path-invalid",
            "Cache entry manifest is not a regular file.",
            path="$.cacheEntry",
            details={"entryPath": str(path_value)},
        )
    try:
        content = path_value.read_bytes()
    except OSError as exc:
        raise NativeDeferredCompilationCacheError(
            "entry-read-failed",
            "Cache entry manifest could not be read.",
            path="$.cacheEntry",
            details={"entryPath": str(path_value), "error": str(exc)},
        ) from exc
    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise NativeDeferredCompilationCacheError(
            "entry-json-invalid",
            "Cache entry manifest must contain strict UTF-8 JSON.",
            path="$.cacheEntry",
            details={"entryPath": str(path_value), "error": str(exc)},
        ) from exc
    if not isinstance(decoded, dict):
        raise NativeDeferredCompilationCacheError(
            "entry-schema-invalid",
            "Cache entry manifest must be an object.",
            path="$.cacheEntry",
        )
    return decoded


def _validated_entry(
    entry: Mapping[str, Any],
    *,
    cache_key: str,
    request_identity: Mapping[str, Any],
    toolchain_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if frozenset(entry) != _ENTRY_FIELDS:
        raise NativeDeferredCompilationCacheError(
            "entry-schema-invalid",
            "Cache entry manifest contains missing or unknown fields.",
            path="$.cacheEntry",
        )
    if entry.get("kind") != NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_KIND:
        raise NativeDeferredCompilationCacheError(
            "entry-kind-invalid",
            "Cache entry kind is not supported.",
            path="$.cacheEntry.kind",
        )
    if (
        isinstance(entry.get("schemaVersion"), bool)
        or entry.get("schemaVersion") != NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION
    ):
        raise NativeDeferredCompilationCacheError(
            "entry-version-invalid",
            "Cache entry schema version is not supported.",
            path="$.cacheEntry.schemaVersion",
        )
    if entry.get("success") is not True:
        raise NativeDeferredCompilationCacheError(
            "entry-not-successful",
            "Only successful cache entries are usable.",
            path="$.cacheEntry.success",
        )
    if (
        _validated_sha256(entry.get("cacheKey"), path="$.cacheEntry.cacheKey")["value"]
        != cache_key
    ):
        raise NativeDeferredCompilationCacheError(
            "entry-cache-key-mismatch",
            "Cache entry key does not match its deterministic path.",
            path="$.cacheEntry.cacheKey",
        )
    if entry.get("requestKind") != NATIVE_DEFERRED_COMPILATION_REQUEST_KIND:
        raise NativeDeferredCompilationCacheError(
            "entry-request-kind-invalid",
            "Cache entry request kind is not supported.",
            path="$.cacheEntry.requestKind",
        )
    entry_request_hash = _validated_sha256(
        entry.get("requestHash"),
        path="$.cacheEntry.requestHash",
    )
    if entry_request_hash != request_identity["requestHash"]:
        raise NativeDeferredCompilationCacheError(
            "entry-request-mismatch",
            "Cache entry request identity does not match the lookup request.",
            path="$.cacheEntry.requestHash",
        )
    entry_toolchain = _validated_toolchain_identity(
        entry.get("toolchain"),
        path="$.cacheEntry.toolchain",
    )
    if entry_toolchain != toolchain_identity:
        raise NativeDeferredCompilationCacheError(
            "entry-toolchain-mismatch",
            "Cache entry toolchain identity does not match the lookup toolchain.",
            path="$.cacheEntry.toolchain",
        )
    target = entry.get("target")
    if not isinstance(target, Mapping) or frozenset(target) != _TARGET_FIELDS:
        raise NativeDeferredCompilationCacheError(
            "entry-target-invalid",
            "Cache entry target must contain only outputFormat.",
            path="$.cacheEntry.target",
        )
    if target.get("outputFormat") != request_identity["outputFormat"]:
        raise NativeDeferredCompilationCacheError(
            "entry-output-format-mismatch",
            "Cache entry output format does not match the request.",
            path="$.cacheEntry.target.outputFormat",
        )
    if _canonical_json(
        entry.get("expectedInterfaceIdentity"),
        path="$.cacheEntry.expectedInterfaceIdentity",
        code="entry-interface-identity-invalid",
    ) != _canonical_json(
        request_identity["interfaceIdentity"],
        path="$.request.expectedInterfaceIdentity",
        code="request-interface-identity-invalid",
    ):
        raise NativeDeferredCompilationCacheError(
            "entry-interface-mismatch",
            "Cache entry interface identity does not match the request.",
            path="$.cacheEntry.expectedInterfaceIdentity",
        )

    output = entry.get("output")
    if not isinstance(output, Mapping) or frozenset(output) != _OUTPUT_FIELDS:
        raise NativeDeferredCompilationCacheError(
            "entry-output-invalid",
            "Cache entry output contains missing or unknown fields.",
            path="$.cacheEntry.output",
        )
    output_hash = _validated_sha256(
        output.get("hash"),
        path="$.cacheEntry.output.hash",
    )
    size_bytes = output.get("sizeBytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise NativeDeferredCompilationCacheError(
            "entry-output-size-invalid",
            "Cache entry output size must be a positive integer.",
            path="$.cacheEntry.output.sizeBytes",
        )
    if output.get("format") != request_identity["outputFormat"]:
        raise NativeDeferredCompilationCacheError(
            "entry-output-format-mismatch",
            "Cache entry output format does not match the request.",
            path="$.cacheEntry.output.format",
        )
    expected_relative_path = _output_relative_path(output_hash["value"])
    if output.get("relativePath") != expected_relative_path:
        raise NativeDeferredCompilationCacheError(
            "entry-output-path-invalid",
            "Cache entry output path must be derived from its content hash.",
            path="$.cacheEntry.output.relativePath",
            details={"expected": expected_relative_path},
        )
    return copy.deepcopy(dict(output))


def _entry_output_path(
    root: Path,
    entry_directory: Path,
    relative_path: str,
) -> Path:
    parsed = PurePosixPath(relative_path)
    if (
        parsed.is_absolute()
        or relative_path != parsed.as_posix()
        or "\\" in relative_path
        or any(part in ("", ".", "..") for part in parsed.parts)
    ):
        raise NativeDeferredCompilationCacheError(
            "entry-output-path-invalid",
            "Cache entry output path must be a portable relative path.",
            path="$.cacheEntry.output.relativePath",
        )
    output_path = entry_directory.joinpath(*parsed.parts)
    _validate_path_chain(root, output_path, final_must_be_directory=False)
    if not output_path.exists():
        raise NativeDeferredCompilationCacheError(
            "entry-incomplete",
            "Cache entry output is missing.",
            path="$.cacheEntry.output.relativePath",
            details={"outputPath": str(output_path)},
        )
    if not output_path.is_file():
        raise NativeDeferredCompilationCacheError(
            "entry-output-path-invalid",
            "Cache entry output is not a regular file.",
            path="$.cacheEntry.output.relativePath",
            details={"outputPath": str(output_path)},
        )
    _require_within_root(root, output_path, path="$.cacheEntry.output.relativePath")
    return output_path


def _read_output(path_value: Path) -> bytes:
    try:
        return path_value.read_bytes()
    except OSError as exc:
        raise NativeDeferredCompilationCacheError(
            "entry-output-read-failed",
            "Cache entry output could not be read.",
            path="$.cacheEntry.output.relativePath",
            details={"outputPath": str(path_value), "error": str(exc)},
        ) from exc


def _verify_output(
    output: Mapping[str, Any],
    content: bytes,
    *,
    output_path: Path,
) -> None:
    actual_hash = hashlib.sha256(content).hexdigest()
    expected_hash = output["hash"]["value"]
    actual_size = len(content)
    expected_size = output["sizeBytes"]
    if actual_size != expected_size or actual_hash != expected_hash:
        raise NativeDeferredCompilationCacheError(
            "entry-output-identity-mismatch",
            "Cache entry output no longer matches its recorded identity.",
            path="$.cacheEntry.output",
            details={
                "outputPath": str(output_path),
                "expected": {
                    "sizeBytes": expected_size,
                    "hash": copy.deepcopy(output["hash"]),
                },
                "observed": {
                    "sizeBytes": actual_size,
                    "hash": _sha256_from_digest(actual_hash),
                },
            },
        )


def _validated_output_bytes(value: Any) -> bytes:
    if not isinstance(value, bytes):
        raise NativeDeferredCompilationCacheError(
            "output-invalid",
            "Successful deferred compilation output must be bytes.",
            path="$.outputBytes",
        )
    if not value:
        raise NativeDeferredCompilationCacheError(
            "output-empty",
            "Successful deferred compilation output must not be empty.",
            path="$.outputBytes",
        )
    return value


def _ensure_repeated_output_matches(
    cache_hit: Mapping[str, Any],
    output_bytes: bytes,
) -> None:
    if cache_hit.get("outputBytes") != output_bytes:
        expected = cache_hit["entry"]["output"]
        raise NativeDeferredCompilationCacheError(
            "output-conflict",
            "The same request and toolchain identity produced different output.",
            path="$.outputBytes",
            details={
                "cached": {
                    "sizeBytes": expected["sizeBytes"],
                    "hash": copy.deepcopy(expected["hash"]),
                },
                "published": {
                    "sizeBytes": len(output_bytes),
                    "hash": _sha256(output_bytes),
                },
            },
        )


def _rename_new_entry(source: Path, destination: Path) -> None:
    try:
        source.rename(destination)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        raise NativeDeferredCompilationCacheError(
            "publish-failed",
            "The deferred compilation cache entry could not be published atomically.",
            path="$.cacheRoot",
            details={
                "entryPath": str(destination),
                "error": str(exc),
            },
        ) from exc


def _write_new_file(path_value: Path, content: bytes) -> None:
    try:
        with path_value.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise NativeDeferredCompilationCacheError(
            "publish-write-failed",
            "Deferred compilation cache content could not be written.",
            path="$.cacheRoot",
            details={"path": str(path_value), "error": str(exc)},
        ) from exc


def _require_within_root(root: Path, path_value: Path, *, path: str) -> None:
    try:
        resolved = path_value.resolve(strict=True)
        resolved.relative_to(root)
    except ValueError as exc:
        raise NativeDeferredCompilationCacheError(
            "path-escape",
            "Cache content resolves outside the cache root.",
            path=path,
            details={"resolvedPath": str(path_value)},
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationCacheError(
            "path-invalid",
            "Cache content path could not be resolved.",
            path=path,
            details={"path": str(path_value), "error": str(exc)},
        ) from exc


def _validated_sha256(value: Any, *, path: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or frozenset(value) != _SHA256_FIELDS
        or value.get("algorithm") != "sha256"
        or not isinstance(value.get("value"), str)
        or len(value["value"]) != 64
        or any(character not in _LOWER_HEX for character in value["value"])
    ):
        raise NativeDeferredCompilationCacheError(
            "hash-invalid",
            "Content identity must be an exact lowercase SHA-256 hash object.",
            path=path,
        )
    return {"algorithm": "sha256", "value": value["value"]}


def _sha256(content: bytes) -> dict[str, str]:
    return _sha256_from_digest(hashlib.sha256(content).hexdigest())


def _sha256_from_digest(digest: str) -> dict[str, str]:
    return {"algorithm": "sha256", "value": digest}


def _output_relative_path(output_hash: str) -> str:
    return f"outputs/{output_hash}.bin"


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _canonical_json(value: Any, *, path: str, code: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NativeDeferredCompilationCacheError(
            code,
            "Identity data must be deterministic JSON.",
            path=path,
            details={"error": str(exc)},
        ) from exc


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")
