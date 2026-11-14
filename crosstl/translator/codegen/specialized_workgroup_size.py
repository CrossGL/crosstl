"""Compute workgroup dimensions that the pipeline sets through specialization constants.

GLSL compiled to SPIR-V (GLSL 4.60 section 4.4.1.4, GL_KHR_vulkan_glsl) writes ``layout(local_size_x_id = N) in;``:
the x dimension of the workgroup becomes the specialization constant with SpecId ``N``, whose default is
``local_size_x``, or 1 when that is absent. CrossGL spells it the same way in a compute stage layout, and the parser
keeps ``local_size_{x,y,z}_id`` in the stage's execution metadata next to ``local_size_{x,y,z}``.

SPIR-V builds the ``WorkgroupSize`` built-in from those specialization constants, which overrides the ``LocalSize``
execution mode (SPIR-V 1.6 section 3.21, BuiltIn WorkgroupSize). GLSL writes the layout back behind the SPIR-V guard,
and a driver that compiles the source keeps the defaults, as for ``layout(constant_id)``. HLSL for Direct3D fixes
``numthreads`` with literals and has no specialization constants, so it refuses a specialized size through
:func:`require_fixed_workgroup_size`.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = [
    "SPECIALIZED_WORKGROUP_SIZE_KEYS",
    "SpecializedWorkgroupSizeError",
    "has_specialized_workgroup_size",
    "require_fixed_workgroup_size",
    "workgroup_size_specialization_ids",
]

SPECIALIZED_WORKGROUP_SIZE_KEYS = (
    "local_size_x_id",
    "local_size_y_id",
    "local_size_z_id",
)

# SpecId is a 32-bit literal, but GLSL writes the id as a signed int literal, so ids up to 2^31 - 1 are portable.
_MAX_SPECIALIZATION_ID = 2**31 - 1
_INTEGER = re.compile(r"[+]?[0-9]+")


class SpecializedWorkgroupSizeError(ValueError):
    """Raised when a target cannot express, or the source misstates, a specialized workgroup size."""

    project_diagnostic_code = (
        "project.translate.workgroup-size-specialization-unsupported"
    )
    missing_capabilities = ("execution.workgroup-size-specialization",)
    check_kind = "execution-specialization"

    def __init__(self, message: str, *, target: str, specialization_ids, reason: str):
        super().__init__(message)
        self.target = target
        self.specialization_ids = tuple(specialization_ids)
        self.reason = reason


def _raw_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _specialization_id(key: str, value: Any, target: str) -> int:
    raw = _raw_value(value)
    text = str(raw).strip()
    if isinstance(raw, bool) or not _INTEGER.fullmatch(text):
        raise SpecializedWorkgroupSizeError(
            f"{key} must be a non-negative integer literal, not '{text}'",
            target=target,
            specialization_ids=(),
            reason="non-literal-specialization-id",
        )
    spec_id = int(text)
    if spec_id > _MAX_SPECIALIZATION_ID:
        raise SpecializedWorkgroupSizeError(
            f"{key} = {spec_id} exceeds the largest specialization constant id "
            f"{_MAX_SPECIALIZATION_ID}",
            target=target,
            specialization_ids=(spec_id,),
            reason="specialization-id-out-of-range",
        )
    return spec_id


def workgroup_size_specialization_ids(
    execution_config: Mapping[str, Any] | None, target: str = "CrossGL"
) -> tuple[int | None, int | None, int | None]:
    """The SpecId of each workgroup dimension the pipeline sets, or ``None`` for a fixed dimension.

    Two dimensions sharing one id would be one specialization constant setting both, which GLSL rejects; the
    helper refuses it so no target silently ties them together.
    """
    config = execution_config or {}
    ids = tuple(
        _specialization_id(key, config[key], target) if key in config else None
        for key in SPECIALIZED_WORKGROUP_SIZE_KEYS
    )
    given = [spec_id for spec_id in ids if spec_id is not None]
    if len(given) != len(set(given)):
        raise SpecializedWorkgroupSizeError(
            "Two workgroup dimensions name the same specialization constant id: "
            + ", ".join(
                f"{key} = {spec_id}"
                for key, spec_id in zip(SPECIALIZED_WORKGROUP_SIZE_KEYS, ids)
                if spec_id is not None
            ),
            target=target,
            specialization_ids=given,
            reason="duplicate-specialization-id",
        )
    return ids


def has_specialized_workgroup_size(execution_config: Mapping[str, Any] | None) -> bool:
    """Whether the pipeline sets any workgroup dimension of the stage."""
    config = execution_config or {}
    return any(key in config for key in SPECIALIZED_WORKGROUP_SIZE_KEYS)


def require_fixed_workgroup_size(
    execution_config: Mapping[str, Any] | None, target: str, reason: str
) -> None:
    """Refuse a specialized workgroup size on a target that only has a size fixed in the shader."""
    if not has_specialized_workgroup_size(execution_config):
        return
    ids = workgroup_size_specialization_ids(execution_config, target)
    named = ", ".join(
        f"{key} = {spec_id}"
        for key, spec_id in zip(SPECIALIZED_WORKGROUP_SIZE_KEYS, ids)
        if spec_id is not None
    )
    raise SpecializedWorkgroupSizeError(
        f"{target} cannot express a workgroup size the pipeline specializes ({named}): {reason}",
        target=target,
        specialization_ids=[spec_id for spec_id in ids if spec_id is not None],
        reason="target-has-no-specialized-workgroup-size",
    )
