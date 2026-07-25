"""Generate target-specific C++ adapters for the native loader execution ABI."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

NATIVE_LOADER_TARGET_ADAPTER_KIND = "crosstl-native-loader-target-adapter"
NATIVE_LOADER_TARGET_ADAPTER_VERSION = 1

_ERROR_PREFIX = "project.native-loader-target-adapter"
_SUPPORTED_TARGETS = ("directx", "opengl")


class NativeLoaderTargetAdapterError(ValueError):
    """A native loader target adapter cannot be generated."""

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


def native_loader_target_adapter_targets() -> tuple[str, ...]:
    """Return canonical targets with generated native loader adapters."""

    return _SUPPORTED_TARGETS


def generate_native_loader_target_adapter(target: str) -> str:
    """Render a deterministic C++17 adapter header for ``target``.

    The generated header consumes the shared types declared by a native loader
    execution header. Callers include one or more unit execution headers before
    including the target adapter header.
    """

    normalized = _normalize_target(target)
    generator = _target_generators()[normalized]
    return generator()


def _normalize_target(target: Any) -> str:
    if not isinstance(target, str) or not target or target != target.strip():
        raise NativeLoaderTargetAdapterError(
            "target-invalid",
            "Native loader target must be a non-empty trimmed string.",
            path="$.target",
        )
    normalized = target.lower()
    if normalized not in _SUPPORTED_TARGETS:
        raise NativeLoaderTargetAdapterError(
            "target-unsupported",
            f"Native loader target adapter is not available for {target!r}.",
            path="$.target",
            details={"target": target, "supportedTargets": list(_SUPPORTED_TARGETS)},
        )
    return normalized


def _target_generators() -> dict[str, Callable[[], str]]:
    from .native_directx_adapter import (  # pylint: disable=import-outside-toplevel
        generate_directx_native_loader_adapter,
    )
    from .native_opengl_adapter import (  # pylint: disable=import-outside-toplevel
        generate_opengl_native_loader_adapter,
    )

    return {
        "directx": generate_directx_native_loader_adapter,
        "opengl": generate_opengl_native_loader_adapter,
    }
