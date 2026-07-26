"""Verified package inputs for bounded native deferred compilation."""

from __future__ import annotations

import copy
import hashlib
import os
import posixpath
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from .native_deferred_compilation import (
    validate_native_deferred_compilation_request,
)

_ERROR_PREFIX = "project.native-deferred-compilation-package"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INCLUDE_DIRECTIVE_RE = re.compile(
    r"^[ \t]*#[ \t]*include(?P<suffix>[A-Za-z_][A-Za-z0-9_]*)?" r"\b(?P<operand>.*)$",
)
_QUOTED_INCLUDE_RE = re.compile(r'^"(?P<path>[^"\r\n]+)"[ \t]*$')
_ANGLE_INCLUDE_RE = re.compile(r"^<(?P<path>[^>\r\n]+)>[ \t]*$")
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_PATH_STEMS = frozenset(
    (
        "AUX",
        "CON",
        "NUL",
        "PRN",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    )
)


class NativeDeferredCompilationPackageError(ValueError):
    """A deferred compilation package cannot be verified or materialized."""

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


@dataclass(frozen=True)
class _VerifiedPackageFile:
    package_path: PurePosixPath
    content: bytes
    hash: Mapping[str, str]
    size_bytes: int
    request_path: str
    format: str | None = None


@dataclass(frozen=True)
class _IncludeDirective:
    kind: str
    value: str
    line: int


def materialize_native_deferred_compilation_inputs(
    request: Mapping[str, Any],
    package_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify and isolate all package inputs for one compiler invocation."""

    normalized = _validated_request(request)
    package_directory = _regular_package_directory(package_root)
    output_directory = _output_directory(
        output_root,
        package_directory=package_directory,
    )

    source_record = _required_mapping(normalized.get("source"), path="$.source")
    source = _verified_identity_file(
        package_directory,
        package_path_value=source_record.get("path"),
        hash_value=source_record.get("hash"),
        size_value=source_record.get("sizeBytes"),
        path="$.source",
        label="Deferred compilation source",
        format_value=source_record.get("format"),
    )
    includes = _verified_include_files(package_directory, normalized.get("includes"))
    descriptor_record = _required_mapping(
        normalized.get("expectedLoaderDescriptor"),
        path="$.expectedLoaderDescriptor",
    )
    descriptor = _verified_identity_file(
        package_directory,
        package_path_value=descriptor_record.get("path"),
        hash_value=descriptor_record.get("hash"),
        size_value=descriptor_record.get("sizeBytes"),
        path="$.expectedLoaderDescriptor",
        label="Deferred compilation interface descriptor",
    )

    _reject_package_path_collisions(source, includes, descriptor)
    angle_roots = _verify_include_closure(source, includes)
    _invalidate_stale_output(output_directory)
    return _publish_materialization(
        normalized,
        source=source,
        includes=includes,
        descriptor=descriptor,
        angle_roots=angle_roots,
        output_root=output_directory,
    )


def _regular_package_directory(value: str | os.PathLike[str]) -> Path:
    package_root = _filesystem_path(
        value,
        path="$.packageRoot",
        label="Deferred compilation package root",
    )
    try:
        metadata = package_root.lstat()
    except FileNotFoundError as exc:
        raise NativeDeferredCompilationPackageError(
            "package-root-missing",
            "Deferred compilation package root does not exist.",
            path="$.packageRoot",
            details={"packageRoot": str(package_root)},
        ) from exc
    except OSError as exc:
        raise NativeDeferredCompilationPackageError(
            "package-root-unavailable",
            "Deferred compilation package root could not be inspected.",
            path="$.packageRoot",
            details={"packageRoot": str(package_root), "error": str(exc)},
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise NativeDeferredCompilationPackageError(
            "package-root-symlink",
            "Deferred compilation package root must not be a symbolic link.",
            path="$.packageRoot",
            details={"packageRoot": str(package_root)},
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise NativeDeferredCompilationPackageError(
            "package-root-not-directory",
            "Deferred compilation package root must be a directory.",
            path="$.packageRoot",
            details={"packageRoot": str(package_root)},
        )
    try:
        return package_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationPackageError(
            "package-root-unavailable",
            "Deferred compilation package root could not be resolved.",
            path="$.packageRoot",
            details={"packageRoot": str(package_root), "error": str(exc)},
        ) from exc


def _output_directory(
    value: str | os.PathLike[str],
    *,
    package_directory: Path,
) -> Path:
    output_root = _filesystem_path(
        value,
        path="$.outputRoot",
        label="Deferred compilation output root",
    )
    if output_root.name in ("", ".", ".."):
        raise NativeDeferredCompilationPackageError(
            "output-root-invalid",
            "Deferred compilation output root must name a distinct directory.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root)},
        )
    if output_root.exists() or output_root.is_symlink():
        try:
            metadata = output_root.lstat()
        except OSError as exc:
            raise NativeDeferredCompilationPackageError(
                "output-root-unavailable",
                "Deferred compilation output root could not be inspected.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root), "error": str(exc)},
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "output-root-symlink",
                "Deferred compilation output root must not be a symbolic link.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root)},
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "output-root-collision",
                "Deferred compilation output root collides with a non-directory.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root)},
            )

    parent = output_root.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationPackageError(
            "output-root-unavailable",
            "Deferred compilation output parent could not be prepared.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root), "error": str(exc)},
        ) from exc
    if not resolved_parent.is_dir():
        raise NativeDeferredCompilationPackageError(
            "output-root-unavailable",
            "Deferred compilation output parent must be a directory.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root)},
        )

    try:
        resolved_output = (resolved_parent / output_root.name).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationPackageError(
            "output-root-unavailable",
            "Deferred compilation output root could not be resolved.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root), "error": str(exc)},
        ) from exc
    if _paths_overlap(resolved_output, package_directory):
        raise NativeDeferredCompilationPackageError(
            "output-package-overlap",
            "Deferred compilation output must not overlap the source package.",
            path="$.outputRoot",
            details={
                "outputRoot": str(resolved_output),
                "packageRoot": str(package_directory),
            },
        )
    return resolved_output


def _verified_include_files(
    package_root: Path,
    value: Any,
) -> list[_VerifiedPackageFile]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise NativeDeferredCompilationPackageError(
            "request-shape-invalid",
            "Deferred compilation include closure must be an array.",
            path="$.includes",
        )
    includes: list[_VerifiedPackageFile] = []
    for index, item in enumerate(value):
        record_path = f"$.includes[{index}]"
        record = _required_mapping(item, path=record_path)
        includes.append(
            _verified_identity_file(
                package_root,
                package_path_value=record.get("path"),
                hash_value=record.get("hash"),
                size_value=record.get("sizeBytes"),
                path=record_path,
                label="Deferred compilation include",
                format_value=record.get("format"),
            )
        )
    return includes


def _verified_identity_file(
    package_root: Path,
    *,
    package_path_value: Any,
    hash_value: Any,
    size_value: Any,
    path: str,
    label: str,
    format_value: Any = None,
) -> _VerifiedPackageFile:
    package_path_field, hash_field, size_field = ("path", "hash", "sizeBytes")
    package_path = _portable_package_path(
        package_path_value,
        path=f"{path}.{package_path_field}",
    )
    expected_hash = _sha256_identity(
        hash_value,
        path=f"{path}.{hash_field}",
    )
    if (
        not isinstance(size_value, int)
        or isinstance(size_value, bool)
        or size_value < 0
    ):
        raise NativeDeferredCompilationPackageError(
            "identity-invalid",
            f"{label} size must be a non-negative integer.",
            path=f"{path}.{size_field}",
        )

    content = _read_regular_package_file(
        package_root,
        package_path,
        path=f"{path}.{package_path_field}",
        label=label,
    )
    actual_size = len(content)
    if actual_size != size_value:
        raise NativeDeferredCompilationPackageError(
            "size-mismatch",
            f"{label} size does not match the deferred request.",
            path=f"{path}.{size_field}",
            details={
                "packagePath": package_path.as_posix(),
                "expectedSizeBytes": size_value,
                "actualSizeBytes": actual_size,
            },
        )
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash["value"]:
        raise NativeDeferredCompilationPackageError(
            "hash-mismatch",
            f"{label} hash does not match the deferred request.",
            path=f"{path}.{hash_field}.value",
            details={
                "packagePath": package_path.as_posix(),
                "expectedSHA256": expected_hash["value"],
                "actualSHA256": actual_hash,
            },
        )
    return _VerifiedPackageFile(
        package_path=package_path,
        content=content,
        hash=expected_hash,
        size_bytes=actual_size,
        request_path=f"{path}.{package_path_field}",
        format=format_value if isinstance(format_value, str) else None,
    )


def _read_regular_package_file(
    package_root: Path,
    package_path: PurePosixPath,
    *,
    path: str,
    label: str,
) -> bytes:
    candidate = package_root
    final_metadata: os.stat_result | None = None
    for index, component in enumerate(package_path.parts):
        candidate = candidate / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError as exc:
            raise NativeDeferredCompilationPackageError(
                "package-file-missing",
                f"{label} does not exist in the package.",
                path=path,
                details={"packagePath": package_path.as_posix()},
            ) from exc
        except OSError as exc:
            raise NativeDeferredCompilationPackageError(
                "package-file-unavailable",
                f"{label} could not be inspected.",
                path=path,
                details={
                    "packagePath": package_path.as_posix(),
                    "error": str(exc),
                },
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "package-path-symlink",
                f"{label} must not traverse a symbolic link.",
                path=path,
                details={
                    "packagePath": package_path.as_posix(),
                    "component": "/".join(package_path.parts[: index + 1]),
                },
            )
        is_final = index == len(package_path.parts) - 1
        if is_final:
            final_metadata = metadata
        elif not stat.S_ISDIR(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "package-path-not-directory",
                f"{label} package path traverses a non-directory.",
                path=path,
                details={
                    "packagePath": package_path.as_posix(),
                    "component": "/".join(package_path.parts[: index + 1]),
                },
            )

    assert final_metadata is not None
    if not stat.S_ISREG(final_metadata.st_mode):
        raise NativeDeferredCompilationPackageError(
            "package-file-not-regular",
            f"{label} must be a regular file.",
            path=path,
            details={"packagePath": package_path.as_posix()},
        )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise NativeDeferredCompilationPackageError(
            "package-path-escape",
            f"{label} resolves outside the package root.",
            path=path,
            details={"packagePath": package_path.as_posix()},
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationPackageError(
            "package-file-unavailable",
            f"{label} could not be resolved.",
            path=path,
            details={"packagePath": package_path.as_posix(), "error": str(exc)},
        ) from exc

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "package-file-not-regular",
                f"{label} must remain a regular file while it is read.",
                path=path,
                details={"packagePath": package_path.as_posix()},
            )
        if (
            getattr(final_metadata, "st_dev", None),
            getattr(final_metadata, "st_ino", None),
        ) != (
            getattr(opened_metadata, "st_dev", None),
            getattr(opened_metadata, "st_ino", None),
        ):
            raise NativeDeferredCompilationPackageError(
                "package-file-changed",
                f"{label} changed while it was being verified.",
                path=path,
                details={"packagePath": package_path.as_posix()},
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            return stream.read()
    except NativeDeferredCompilationPackageError:
        raise
    except OSError as exc:
        raise NativeDeferredCompilationPackageError(
            "package-file-read-failed",
            f"{label} could not be read.",
            path=path,
            details={"packagePath": package_path.as_posix(), "error": str(exc)},
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_package_path_collisions(
    source: _VerifiedPackageFile,
    includes: Sequence[_VerifiedPackageFile],
    descriptor: _VerifiedPackageFile,
) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for item, label in (
        (source, "source"),
        *((include, "include") for include in includes),
        (descriptor, "interface descriptor"),
    ):
        value = item.package_path.as_posix()
        key = value.casefold()
        previous = seen.get(key)
        if previous is not None:
            previous_value, previous_label = previous
            code = (
                "package-path-duplicate"
                if previous_value == value
                else "package-path-case-collision"
            )
            raise NativeDeferredCompilationPackageError(
                code,
                "Deferred compilation package inputs must have distinct portable paths.",
                path=item.request_path,
                details={
                    "firstPath": previous_value,
                    "firstKind": previous_label,
                    "secondPath": value,
                    "secondKind": label,
                },
            )
        seen[key] = (value, label)


def _verify_include_closure(
    source: _VerifiedPackageFile,
    includes: Sequence[_VerifiedPackageFile],
) -> set[PurePosixPath]:
    declared = {item.package_path: item for item in includes}
    directives: dict[PurePosixPath, list[_IncludeDirective]] = {}
    for item in (source, *includes):
        directives[item.package_path] = _include_directives(item)

    edges: dict[PurePosixPath, set[PurePosixPath]] = {}
    angle_roots: set[PurePosixPath] = set()
    for item in (source, *includes):
        resolved: set[PurePosixPath] = set()
        for directive in directives[item.package_path]:
            target, angle_root = _resolve_include_directive(
                item,
                directive,
                declared=declared,
            )
            resolved.add(target)
            if angle_root is not None:
                angle_roots.add(angle_root)
        edges[item.package_path] = resolved

    reachable: set[PurePosixPath] = set()
    pending = [source.package_path]
    while pending:
        current = pending.pop()
        for included_path in edges.get(current, ()):
            if included_path in reachable:
                continue
            reachable.add(included_path)
            pending.append(included_path)
    unreachable = sorted(
        (path for path in declared if path not in reachable),
        key=lambda item: item.as_posix(),
    )
    if unreachable:
        first = declared[unreachable[0]]
        raise NativeDeferredCompilationPackageError(
            "include-unreachable",
            "Declared include is unreachable from the deferred source include closure.",
            path=first.request_path,
            details={
                "packagePath": first.package_path.as_posix(),
                "unreachableIncludes": [item.as_posix() for item in unreachable],
            },
        )
    return angle_roots


def _include_directives(item: _VerifiedPackageFile) -> list[_IncludeDirective]:
    try:
        source = item.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NativeDeferredCompilationPackageError(
            "source-encoding-invalid",
            "Deferred compilation source and includes must be valid UTF-8.",
            path=item.request_path,
            details={
                "packagePath": item.package_path.as_posix(),
                "byteOffset": exc.start,
            },
        ) from exc

    without_comments = _strip_comments(source)
    directives: list[_IncludeDirective] = []
    for line_number, logical_line in _logical_lines(without_comments):
        match = _INCLUDE_DIRECTIVE_RE.match(logical_line)
        if match is None:
            continue
        operand = match.group("operand").strip()
        suffix = match.group("suffix")
        if suffix is not None:
            raise NativeDeferredCompilationPackageError(
                "include-directive-unsupported",
                "Deferred compilation packages support only literal include directives.",
                path=item.request_path,
                details={
                    "packagePath": item.package_path.as_posix(),
                    "line": line_number,
                    "directive": f"include{suffix}",
                },
            )
        quoted = _QUOTED_INCLUDE_RE.fullmatch(operand)
        if quoted is not None:
            directives.append(
                _IncludeDirective("quoted", quoted.group("path"), line_number)
            )
            continue
        angle = _ANGLE_INCLUDE_RE.fullmatch(operand)
        if angle is not None:
            directives.append(
                _IncludeDirective("angle", angle.group("path"), line_number)
            )
            continue
        raise NativeDeferredCompilationPackageError(
            "include-dynamic",
            "Deferred compilation packages require literal include directives.",
            path=item.request_path,
            details={
                "packagePath": item.package_path.as_posix(),
                "line": line_number,
                "operand": operand,
            },
        )
    return directives


def _resolve_include_directive(
    including: _VerifiedPackageFile,
    directive: _IncludeDirective,
    *,
    declared: Mapping[PurePosixPath, _VerifiedPackageFile],
) -> tuple[PurePosixPath, PurePosixPath | None]:
    include_value = _include_operand_path(
        directive.value,
        kind=directive.kind,
        including=including,
        line=directive.line,
    )
    if directive.kind == "quoted":
        candidate = _normalized_join(including.package_path.parent, include_value)
        if candidate is None:
            raise NativeDeferredCompilationPackageError(
                "include-path-escape",
                "Quoted include resolves outside the package root.",
                path=including.request_path,
                details={
                    "packagePath": including.package_path.as_posix(),
                    "line": directive.line,
                    "include": directive.value,
                },
            )
        if candidate in declared:
            return candidate, None
        _raise_undeclared_include(including, directive, candidate=candidate)

    exact = PurePosixPath(posixpath.normpath(include_value.as_posix()))
    if exact in declared:
        return exact, PurePosixPath(".")
    matches = [
        path
        for path in declared
        if len(path.parts) >= len(exact.parts)
        and path.parts[-len(exact.parts) :] == exact.parts
    ]
    if len(matches) == 1:
        match = matches[0]
        prefix_parts = match.parts[: -len(exact.parts)]
        return match, (
            PurePosixPath(*prefix_parts) if prefix_parts else PurePosixPath(".")
        )
    if len(matches) > 1:
        raise NativeDeferredCompilationPackageError(
            "include-ambiguous",
            "Angle include matches multiple files in the declared include closure.",
            path=including.request_path,
            details={
                "packagePath": including.package_path.as_posix(),
                "line": directive.line,
                "include": directive.value,
                "matches": sorted(path.as_posix() for path in matches),
            },
        )
    _raise_undeclared_include(including, directive, candidate=exact)


def _raise_undeclared_include(
    including: _VerifiedPackageFile,
    directive: _IncludeDirective,
    *,
    candidate: PurePosixPath,
) -> None:
    raise NativeDeferredCompilationPackageError(
        "include-undeclared",
        "Include directive does not resolve within the declared include closure.",
        path=including.request_path,
        details={
            "packagePath": including.package_path.as_posix(),
            "line": directive.line,
            "include": directive.value,
            "resolvedPackagePath": candidate.as_posix(),
        },
    )


def _include_operand_path(
    value: str,
    *,
    kind: str,
    including: _VerifiedPackageFile,
    line: int,
) -> PurePosixPath:
    components = value.split("/")
    nonportable_components = [
        component
        for component in components
        if (
            (component not in (".", "..") and component.endswith((" ", ".")))
            or component.partition(".")[0].upper() in _WINDOWS_RESERVED_PATH_STEMS
            or any(
                ord(character) < 32 or character in _WINDOWS_INVALID_PATH_CHARACTERS
                for character in component
            )
        )
    ]
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or any(component == "" for component in components)
        or (
            kind == "angle"
            and any(component in (".", "..") for component in components)
        )
        or nonportable_components
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
    ):
        raise NativeDeferredCompilationPackageError(
            "include-path-invalid",
            "Literal include path must be relative and portable.",
            path=including.request_path,
            details={
                "packagePath": including.package_path.as_posix(),
                "line": line,
                "include": value,
                "nonportableComponents": nonportable_components,
            },
        )
    return PurePosixPath(value)


def _normalized_join(
    parent: PurePosixPath,
    child: PurePosixPath,
) -> PurePosixPath | None:
    value = posixpath.normpath((parent / child).as_posix())
    if value in ("", ".") or value == ".." or value.startswith("../"):
        return None
    return PurePosixPath(value)


def _strip_comments(source: str) -> str:
    output: list[str] = []
    index = 0
    state = "normal"
    quote = ""
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line-comment":
            if character in "\r\n":
                output.append(character)
                state = "normal"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block-comment":
            if character == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "normal"
            else:
                output.append(character if character in "\r\n" else " ")
                index += 1
            continue
        if state == "string":
            output.append(character)
            if character == "\\" and following:
                output.append(following)
                index += 2
                continue
            if character == quote:
                state = "normal"
            index += 1
            continue
        if character == "/" and following == "/":
            output.extend((" ", " "))
            index += 2
            state = "line-comment"
            continue
        if character == "/" and following == "*":
            output.extend((" ", " "))
            index += 2
            state = "block-comment"
            continue
        if character in ('"', "'"):
            quote = character
            state = "string"
        output.append(character)
        index += 1
    return "".join(output)


def _logical_lines(source: str) -> list[tuple[int, str]]:
    physical_lines = source.splitlines(keepends=True)
    logical: list[tuple[int, str]] = []
    fragments: list[str] = []
    start_line = 1
    for line_number, line in enumerate(physical_lines, start=1):
        if not fragments:
            start_line = line_number
        stripped_newline = line.rstrip("\r\n")
        if re.search(r"\\[ \t]*$", stripped_newline):
            fragments.append(re.sub(r"\\[ \t]*$", " ", stripped_newline))
            continue
        fragments.append(stripped_newline)
        logical.append((start_line, "".join(fragments)))
        fragments = []
    if fragments:
        logical.append((start_line, "".join(fragments)))
    return logical


def _publish_materialization(
    normalized: Mapping[str, Any],
    *,
    source: _VerifiedPackageFile,
    includes: Sequence[_VerifiedPackageFile],
    descriptor: _VerifiedPackageFile,
    angle_roots: set[PurePosixPath],
    output_root: Path,
) -> dict[str, Any]:
    parent = output_root.parent
    temporary_root: Path | None = None
    try:
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.",
                suffix=".tmp",
                dir=parent,
            )
        )
        source_root = temporary_root / "source"
        interface_root = temporary_root / "interface"
        _write_verified_file(source_root, source)
        for include in includes:
            _write_verified_file(source_root, include)
        _write_verified_file(interface_root, descriptor)

        published_source_root = output_root / "source"
        published_interface_root = output_root / "interface"
        source_result = _materialized_file_result(
            source,
            path=published_source_root.joinpath(*source.package_path.parts),
        )
        include_results = [
            {
                **_materialized_file_result(
                    item,
                    path=published_source_root.joinpath(*item.package_path.parts),
                ),
                "format": item.format,
            }
            for item in sorted(includes, key=lambda item: item.package_path.as_posix())
        ]
        descriptor_result = {
            "packagePath": descriptor.package_path.as_posix(),
            "path": str(
                published_interface_root.joinpath(*descriptor.package_path.parts)
            ),
            "hash": copy.deepcopy(dict(descriptor.hash)),
            "sizeBytes": descriptor.size_bytes,
        }
        include_directories = sorted(
            {
                str(
                    published_source_root
                    if root == PurePosixPath(".")
                    else published_source_root.joinpath(*root.parts)
                )
                for root in angle_roots
            }
        )

        if output_root.exists() or output_root.is_symlink():
            raise NativeDeferredCompilationPackageError(
                "output-publication-collision",
                "Deferred compilation output appeared during publication.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root)},
            )
        temporary_root.rename(output_root)
        temporary_root = None
        return {
            "requestHash": copy.deepcopy(normalized.get("requestHash")),
            "target": copy.deepcopy(normalized.get("target")),
            "variant": copy.deepcopy(normalized.get("variant")),
            "outputRoot": str(output_root),
            "sourceRoot": str(published_source_root),
            "source": {
                **source_result,
                "format": normalized["source"]["format"],
            },
            "includes": include_results,
            "includeDirectories": include_directories,
            "interface": descriptor_result,
        }
    except NativeDeferredCompilationPackageError:
        raise
    except OSError as exc:
        raise NativeDeferredCompilationPackageError(
            "output-write-failed",
            "Deferred compilation inputs could not be materialized.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root), "error": str(exc)},
        ) from exc
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)


def _write_verified_file(root: Path, item: _VerifiedPackageFile) -> None:
    destination = root.joinpath(*item.package_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(item.content)
        stream.flush()
        os.fsync(stream.fileno())


def _materialized_file_result(
    item: _VerifiedPackageFile,
    *,
    path: Path,
) -> dict[str, Any]:
    return {
        "packagePath": item.package_path.as_posix(),
        "path": str(path),
        "hash": copy.deepcopy(dict(item.hash)),
        "sizeBytes": item.size_bytes,
    }


def _invalidate_stale_output(output_root: Path) -> None:
    if not output_root.exists() and not output_root.is_symlink():
        return
    try:
        metadata = output_root.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "output-root-symlink",
                "Deferred compilation output root must not be a symbolic link.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root)},
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise NativeDeferredCompilationPackageError(
                "output-root-collision",
                "Deferred compilation output root collides with a non-directory.",
                path="$.outputRoot",
                details={"outputRoot": str(output_root)},
            )
        shutil.rmtree(output_root)
    except NativeDeferredCompilationPackageError:
        raise
    except OSError as exc:
        raise NativeDeferredCompilationPackageError(
            "output-invalidation-failed",
            "Stale deferred compilation outputs could not be invalidated.",
            path="$.outputRoot",
            details={"outputRoot": str(output_root), "error": str(exc)},
        ) from exc


def _portable_package_path(value: Any, *, path: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise NativeDeferredCompilationPackageError(
            "package-path-invalid",
            "Package path must be a non-empty portable relative path.",
            path=path,
            details={"packagePath": value},
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    components = posix_path.parts
    nonportable_components = [
        component
        for component in components
        if (
            component.endswith((" ", "."))
            or component.partition(".")[0].upper() in _WINDOWS_RESERVED_PATH_STEMS
            or any(
                ord(character) < 32 or character in _WINDOWS_INVALID_PATH_CHARACTERS
                for character in component
            )
        )
    ]
    if (
        any(component in ("", ".", "..") for component in components)
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or nonportable_components
    ):
        raise NativeDeferredCompilationPackageError(
            "package-path-invalid",
            "Package path must be normalized, relative, traversal-free, and portable.",
            path=path,
            details={
                "packagePath": value,
                "nonportableComponents": nonportable_components,
            },
        )
    return posix_path


def _sha256_identity(value: Any, *, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "value"}:
        raise NativeDeferredCompilationPackageError(
            "identity-invalid",
            "Content identity must contain exactly algorithm and value.",
            path=path,
        )
    algorithm = value.get("algorithm")
    digest = value.get("value")
    if (
        algorithm != "sha256"
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise NativeDeferredCompilationPackageError(
            "identity-invalid",
            "Content identity must be a lowercase SHA-256 digest.",
            path=path,
        )
    return {"algorithm": "sha256", "value": digest}


def _filesystem_path(
    value: str | os.PathLike[str],
    *,
    path: str,
    label: str,
) -> Path:
    try:
        raw_value = os.fspath(value)
        if not raw_value:
            raise ValueError("path is empty")
        result = Path(value)
    except (TypeError, ValueError) as exc:
        raise NativeDeferredCompilationPackageError(
            "filesystem-path-invalid",
            f"{label} must be a filesystem path.",
            path=path,
        ) from exc
    if not os.fspath(result):
        raise NativeDeferredCompilationPackageError(
            "filesystem-path-invalid",
            f"{label} must not be empty.",
            path=path,
        )
    return result


def _required_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationPackageError(
            "request-shape-invalid",
            "Validated deferred compilation request has an invalid mapping.",
            path=path,
        )
    return value


def _validated_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise NativeDeferredCompilationPackageError(
            "request-invalid",
            "Deferred compilation request must be an object.",
            path="$.request",
        )
    try:
        normalized = validate_native_deferred_compilation_request(request)
    except (TypeError, ValueError) as exc:
        details: dict[str, Any] = {"error": str(exc)}
        request_code = getattr(exc, "code", None)
        if isinstance(request_code, str):
            details["requestCode"] = request_code
        request_path = getattr(exc, "path", "$.request")
        raise NativeDeferredCompilationPackageError(
            "request-invalid",
            "Deferred compilation request failed contract validation.",
            path=request_path if isinstance(request_path, str) else "$.request",
            details=details,
        ) from exc
    if not isinstance(normalized, Mapping):
        raise NativeDeferredCompilationPackageError(
            "request-invalid",
            "Deferred compilation request validator returned an invalid result.",
            path="$.request",
        )
    return copy.deepcopy(dict(normalized))


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False
