"""Host-interface reflection helpers for packaged runtime artifacts."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REFLECTION_DIAGNOSTIC_PREFIX = "project.runtime-package-inspection"
REFLECTION_TOOL_UNAVAILABLE = (
    f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-reflection-tool-unavailable"
)
REFLECTION_UNSUPPORTED_FORMAT = (
    f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-reflection-unsupported-format"
)
REFLECTION_AMBIGUOUS_BINDING = (
    f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-reflection-ambiguous-binding"
)
REFLECTION_INCOMPLETE_OUTPUT = (
    f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-reflection-incomplete-output"
)
REFLECTION_EMPTY = f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-empty"
REFLECTION_PARSE_FAILED = (
    f"{REFLECTION_DIAGNOSTIC_PREFIX}.host-interface-reflection-parse-failed"
)
REFLECTION_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ReflectionDiagnostic:
    """Structured reflection diagnostic carried alongside stable code strings."""

    code: str
    message: str
    severity: str = "warning"
    category: str = "reflection"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def host_interface_record(
    *,
    status: str,
    source: str,
    parser: str | None,
    artifact_format: str | None = None,
    entry_points: Sequence[Mapping[str, Any]] = (),
    resources: Sequence[Mapping[str, Any]] = (),
    constants: Sequence[Mapping[str, Any]] = (),
    diagnostics: Sequence[str | Mapping[str, Any] | ReflectionDiagnostic] = (),
) -> dict[str, Any]:
    diagnostic_records = [
        _diagnostic_record(diagnostic).to_json() for diagnostic in diagnostics
    ]
    return {
        "status": status,
        "source": source,
        "parser": parser,
        "artifactFormat": artifact_format,
        "entryPointCount": len(entry_points),
        "resourceCount": len(resources),
        "constantCount": len(constants),
        "entryPoints": [dict(entry_point) for entry_point in entry_points],
        "resources": [dict(resource) for resource in resources],
        "constants": [dict(constant) for constant in constants],
        "diagnostics": [record["code"] for record in diagnostic_records],
        "diagnosticRecords": diagnostic_records,
    }


def empty_host_interface_record(
    status: str,
    *,
    parser: str | None = None,
    source: str = "package-artifact",
    artifact_format: str | None = None,
    diagnostics: Sequence[str | Mapping[str, Any] | ReflectionDiagnostic] = (),
) -> dict[str, Any]:
    return host_interface_record(
        status=status,
        source=source,
        parser=parser,
        artifact_format=artifact_format,
        diagnostics=diagnostics,
    )


def reflect_target_host_interface(
    artifact_path: Path,
    *,
    target: str,
    artifact_format: str | None = None,
    stage: str | None = None,
    tool_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    """Reflect supported target artifacts into the common hostInterface record."""

    normalized_target = target.strip().lower()
    resolver = tool_resolver or shutil.which
    suffix = artifact_path.suffix.lower()
    try:
        if normalized_target in {"directx", "hlsl", "dxil"}:
            if suffix in {".dxil", ".cso"}:
                return empty_host_interface_record(
                    "unsupported",
                    parser="directx-reflection",
                    source="compiled-artifact",
                    artifact_format=artifact_format or "DXIL bytecode",
                    diagnostics=(
                        ReflectionDiagnostic(
                            REFLECTION_UNSUPPORTED_FORMAT,
                            "DirectX bytecode reflection is not available in this "
                            "bounded reflector; provide HLSL source or external "
                            "reflection metadata.",
                            details={"target": normalized_target, "suffix": suffix},
                        ),
                    ),
                )
            return _reflect_hlsl_source(
                artifact_path,
                artifact_format=artifact_format or "HLSL source",
                stage=stage,
            )
        if normalized_target in {"opengl", "glsl", "webgl"}:
            return _reflect_glsl_source(
                artifact_path,
                parser=(
                    "webgl-reflection"
                    if normalized_target == "webgl"
                    else "opengl-reflection"
                ),
                artifact_format=artifact_format or "GLSL source",
                stage=stage,
            )
        if normalized_target in {"vulkan", "spirv", "spv"}:
            return _reflect_spirv_artifact(
                artifact_path,
                artifact_format=artifact_format,
                tool_resolver=resolver,
            )
    except OSError as exc:
        return empty_host_interface_record(
            "failed",
            parser=f"{normalized_target}-reflection",
            source="compiled-artifact",
            artifact_format=artifact_format,
            diagnostics=(
                ReflectionDiagnostic(
                    REFLECTION_PARSE_FAILED,
                    f"Runtime artifact reflection failed: {exc}",
                    severity="error",
                    details={"target": normalized_target, "path": str(artifact_path)},
                ),
            ),
        )
    return None


def _diagnostic_record(
    diagnostic: str | Mapping[str, Any] | ReflectionDiagnostic,
) -> ReflectionDiagnostic:
    if isinstance(diagnostic, ReflectionDiagnostic):
        return diagnostic
    if isinstance(diagnostic, Mapping):
        code = diagnostic.get("code")
        return ReflectionDiagnostic(
            str(code) if code else REFLECTION_INCOMPLETE_OUTPUT,
            str(diagnostic.get("message") or code or "Reflection diagnostic."),
            severity=str(diagnostic.get("severity") or "warning"),
            category=str(diagnostic.get("category") or "reflection"),
            details=(
                dict(diagnostic.get("details"))
                if isinstance(diagnostic.get("details"), Mapping)
                else {}
            ),
        )
    return ReflectionDiagnostic(str(diagnostic), str(diagnostic))


def _status_for_reflection(
    entry_points: Sequence[Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
    constants: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[ReflectionDiagnostic],
) -> str:
    codes = {diagnostic.code for diagnostic in diagnostics}
    if REFLECTION_PARSE_FAILED in codes:
        return "failed"
    if REFLECTION_UNSUPPORTED_FORMAT in codes:
        return "unsupported"
    if REFLECTION_TOOL_UNAVAILABLE in codes:
        return "unavailable"
    if REFLECTION_AMBIGUOUS_BINDING in codes:
        return "ambiguous"
    if REFLECTION_INCOMPLETE_OUTPUT in codes:
        return "incomplete"
    if not entry_points and not resources and not constants:
        return "unavailable"
    return "ready"


def _finalize_reflection(
    *,
    parser: str,
    artifact_format: str,
    entry_points: Sequence[Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
    constants: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[ReflectionDiagnostic],
) -> dict[str, Any]:
    current_diagnostics = list(diagnostics)
    if not entry_points and not resources and not constants:
        current_diagnostics.append(
            ReflectionDiagnostic(
                REFLECTION_EMPTY,
                "No entry points, resources, or constants were reflected.",
                severity="warning",
            )
        )
    current_diagnostics.extend(_binding_diagnostics(resources))
    return host_interface_record(
        status=_status_for_reflection(
            entry_points, resources, constants, current_diagnostics
        ),
        source="compiled-artifact",
        parser=parser,
        artifact_format=artifact_format,
        entry_points=entry_points,
        resources=resources,
        constants=constants,
        diagnostics=current_diagnostics,
    )


def _binding_diagnostics(
    resources: Sequence[Mapping[str, Any]],
) -> list[ReflectionDiagnostic]:
    diagnostics = []
    seen: dict[tuple[Any, Any], tuple[str, tuple[str, ...]]] = {}
    for resource in resources:
        resource_name = str(resource.get("name") or "<unnamed>")
        binding = resource.get("binding")
        set_number = resource.get("set")
        if binding is None:
            diagnostics.append(
                ReflectionDiagnostic(
                    REFLECTION_INCOMPLETE_OUTPUT,
                    f"Reflected resource {resource_name} is missing a binding.",
                    details={"resource": resource_name},
                )
            )
            continue
        coordinate = (0 if set_number is None else set_number, binding)
        owners = _resource_entry_points(resource)
        if coordinate in seen:
            conflicting_resource, conflicting_owners = seen[coordinate]
            if (
                owners
                and conflicting_owners
                and set(owners).isdisjoint(conflicting_owners)
            ):
                continue
            diagnostics.append(
                ReflectionDiagnostic(
                    REFLECTION_AMBIGUOUS_BINDING,
                    "Multiple reflected resources use the same binding coordinate.",
                    details={
                        "resource": resource_name,
                        "conflictingResource": conflicting_resource,
                        "set": coordinate[0],
                        "binding": coordinate[1],
                    },
                )
            )
        else:
            seen[coordinate] = (resource_name, owners)
    return diagnostics


def _resource_entry_points(resource: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = resource.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    names: list[str] = []
    for key in ("entryPoint", "entry_point"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    for key in ("entryPoints", "entry_points"):
        value = metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            names.extend(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
    return tuple(dict.fromkeys(names))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _int_value(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _parse_register(register_text: str | None) -> tuple[int | None, int | None]:
    if not register_text:
        return None, None
    match = re.search(
        r"\b[btus]\s*(?P<binding>\d+)(?:\s*,\s*space\s*(?P<set>\d+))?",
        register_text,
        re.IGNORECASE,
    )
    if match is None:
        return None, None
    set_number = _int_value(match.group("set"))
    return 0 if set_number is None else set_number, _int_value(match.group("binding"))


HLSL_FUNCTION_RE = re.compile(
    r"(?P<attributes>(?:\s*\[[^\]]+\]\s*)*)"
    r"(?:[A-Za-z_][\w:<>,\s*&]*\s+)+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)"
    r"(?:\s*:\s*[A-Za-z_]\w*)?\s*\{",
    re.MULTILINE,
)
HLSL_NUMTHREADS_RE = re.compile(
    r"\[\s*numthreads\s*\(\s*([^,\]]+)\s*,\s*([^,\]]+)\s*,\s*([^,\]]+)\s*\)\s*\]",
    re.IGNORECASE,
)
HLSL_SHADER_ATTR_RE = re.compile(
    r"\[\s*shader\s*\(\s*\"([^\"]+)\"\s*\)\s*\]", re.IGNORECASE
)
HLSL_CBUFFER_RE = re.compile(
    r"\b(?P<kind>c?buffer|tbuffer)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*:\s*register\s*\((?P<register>[^)]*)\))?\s*\{",
    re.IGNORECASE,
)
HLSL_RESOURCE_RE = re.compile(
    r"\b(?P<type>(?:RW)?(?:StructuredBuffer|ByteAddressBuffer|Buffer|Texture\w+|"
    r"RWTexture\w+|SamplerState|SamplerComparisonState|ConstantBuffer)"
    r"(?:\s*<[^;>{}]+>)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:\:\s*register\s*\((?P<register>[^)]*)\))?\s*;",
    re.IGNORECASE,
)
HLSL_CONSTANT_RE = re.compile(
    r"\b(?:static\s+)?const\s+"
    r"(?P<type>bool|int|uint|float|double|half)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);",
    re.IGNORECASE,
)

HLSL_ENTRY_STAGE_BY_NAME = {
    "VSMain": "vertex",
    "PSMain": "fragment",
    "CSMain": "compute",
    "GSMain": "geometry",
    "HSMain": "tessellation_control",
    "DSMain": "tessellation_evaluation",
    "ASMain": "task",
    "MSMain": "mesh",
}
HLSL_ENTRY_STAGE_BY_ATTR = {
    "vertex": "vertex",
    "pixel": "fragment",
    "fragment": "fragment",
    "compute": "compute",
    "geometry": "geometry",
    "hull": "tessellation_control",
    "domain": "tessellation_evaluation",
    "amplification": "task",
    "mesh": "mesh",
    "raygeneration": "ray_generation",
    "intersection": "ray_intersection",
    "anyhit": "ray_any_hit",
    "closesthit": "ray_closest_hit",
    "miss": "ray_miss",
    "callable": "ray_callable",
}


def _reflect_hlsl_source(
    artifact_path: Path, *, artifact_format: str, stage: str | None
) -> dict[str, Any]:
    source = _strip_comments(_read_text(artifact_path))
    entry_points = []
    for match in HLSL_FUNCTION_RE.finditer(source):
        name = match.group("name")
        attributes = match.group("attributes") or ""
        reflected_stage = _hlsl_entry_stage(name, attributes, stage)
        if reflected_stage is None:
            continue
        execution_config = {}
        thread_match = HLSL_NUMTHREADS_RE.search(attributes)
        if thread_match:
            execution_config["numthreads"] = [
                _literal_value(thread_match.group(index)) for index in (1, 2, 3)
            ]
        entry_points.append(
            {
                "name": name,
                "stage": reflected_stage,
                "executionConfig": execution_config,
            }
        )

    resources = []
    for match in HLSL_CBUFFER_RE.finditer(source):
        set_number, binding = _parse_register(match.group("register"))
        resources.append(
            {
                "name": match.group("name"),
                "kind": "constant-buffer",
                "type": match.group("name"),
                "set": set_number,
                "binding": binding,
                "access": "read",
            }
        )
    for match in HLSL_RESOURCE_RE.finditer(source):
        type_name = " ".join(match.group("type").split())
        set_number, binding = _parse_register(match.group("register"))
        resources.append(
            {
                "name": match.group("name"),
                "kind": _hlsl_resource_kind(type_name),
                "type": type_name,
                "set": set_number,
                "binding": binding,
                "access": _hlsl_resource_access(type_name),
            }
        )

    constants = [
        {
            "name": match.group("name"),
            "kind": "scalar-constant",
            "dtype": match.group("type"),
            "value": _literal_value(match.group("value")),
            "required": False,
            "source": "hlsl.const",
        }
        for match in HLSL_CONSTANT_RE.finditer(source)
    ]
    return _finalize_reflection(
        parser="directx-reflection",
        artifact_format=artifact_format,
        entry_points=entry_points,
        resources=resources,
        constants=constants,
        diagnostics=[],
    )


def _hlsl_entry_stage(name: str, attributes: str, stage: str | None) -> str | None:
    shader_match = HLSL_SHADER_ATTR_RE.search(attributes)
    if shader_match:
        return HLSL_ENTRY_STAGE_BY_ATTR.get(shader_match.group(1).lower())
    if name in HLSL_ENTRY_STAGE_BY_NAME:
        return HLSL_ENTRY_STAGE_BY_NAME[name]
    if name == "main" and stage:
        return _stage_name(stage)
    return None


def _hlsl_resource_kind(type_name: str) -> str:
    lowered = type_name.lower()
    if "sampler" in lowered:
        return "sampler"
    if "texture" in lowered:
        return "storage-texture" if lowered.startswith("rw") else "texture"
    if "constantbuffer" in lowered:
        return "constant-buffer"
    return "buffer"


def _hlsl_resource_access(type_name: str) -> str | None:
    lowered = type_name.lower()
    if lowered.startswith("rw"):
        return "read_write"
    if "buffer" in lowered or "texture" in lowered:
        return "read"
    return None


GLSL_LAYOUT_RE = re.compile(r"layout\s*\((?P<layout>[^)]*)\)\s*", re.IGNORECASE)
GLSL_LOCAL_SIZE_RE = re.compile(
    r"layout\s*\((?P<layout>[^)]*local_size_[^)]*)\)\s*in\s*;",
    re.IGNORECASE,
)
GLSL_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(", re.IGNORECASE)
GLSL_BLOCK_RESOURCE_RE = re.compile(
    r"(?:layout\s*\((?P<layout>[^)]*)\)\s*)?"
    r"(?P<qualifiers>(?:(?:readonly|writeonly|coherent|restrict|volatile)\s+)*)"
    r"(?P<storage>uniform|buffer)\s+"
    r"(?P<block>[A-Za-z_]\w*)\s*\{[^}]*\}\s*"
    r"(?P<name>[A-Za-z_]\w*)?\s*;",
    re.IGNORECASE | re.DOTALL,
)
GLSL_RESOURCE_RE = re.compile(
    r"(?:layout\s*\((?P<layout>[^)]*)\)\s*)?"
    r"(?P<qualifiers>(?:(?:readonly|writeonly|coherent|restrict|volatile)\s+)*)"
    r"(?P<storage>uniform|buffer)\s+"
    r"(?P<type>[A-Za-z_]\w*(?:\s*<[^;]+>)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*;",
    re.IGNORECASE,
)
GLSL_SPEC_CONSTANT_RE = re.compile(
    r"layout\s*\((?P<layout>[^)]*constant_id[^)]*)\)\s*"
    r"const\s+(?P<type>[A-Za-z_]\w*)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);",
    re.IGNORECASE,
)
GLSL_SCALAR_CONSTANT_RE = re.compile(
    r"(?<!layout\s)\bconst\s+(?P<type>bool|int|uint|float|double)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);",
    re.IGNORECASE,
)


def _reflect_glsl_source(
    artifact_path: Path,
    *,
    parser: str,
    artifact_format: str,
    stage: str | None,
) -> dict[str, Any]:
    raw_source = _read_text(artifact_path)
    reflected_stage = _stage_name(stage) or _glsl_stage_from_source(
        raw_source, artifact_path
    )
    source = _strip_comments(raw_source)
    execution_config = {}
    local_size_match = GLSL_LOCAL_SIZE_RE.search(source)
    if local_size_match:
        layout = _parse_layout(local_size_match.group("layout"))
        execution_config.update(
            {
                "local_size_x": layout.get("local_size_x", 1),
                "local_size_y": layout.get("local_size_y", 1),
                "local_size_z": layout.get("local_size_z", 1),
            }
        )
    entry_points = []
    if GLSL_MAIN_RE.search(source):
        entry_points.append(
            {
                "name": "main",
                "stage": reflected_stage,
                "executionConfig": execution_config,
            }
        )

    resources = []
    occupied_spans = []
    for match in GLSL_BLOCK_RESOURCE_RE.finditer(source):
        occupied_spans.append(match.span())
        layout = _parse_layout(match.group("layout"))
        storage = match.group("storage").lower()
        qualifiers = match.group("qualifiers") or ""
        name = match.group("name") or match.group("block")
        resources.append(
            {
                "name": name,
                "kind": "constant-buffer" if storage == "uniform" else "buffer",
                "type": match.group("block"),
                "set": _layout_int(layout, ("set", "descriptor_set"), default=0),
                "binding": _layout_int(layout, ("binding",)),
                "access": _glsl_access(storage, qualifiers),
            }
        )
    for match in GLSL_RESOURCE_RE.finditer(source):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        layout = _parse_layout(match.group("layout"))
        storage = match.group("storage").lower()
        type_name = " ".join(match.group("type").split())
        resources.append(
            {
                "name": match.group("name"),
                "kind": _glsl_resource_kind(storage, type_name),
                "type": type_name,
                "set": _layout_int(layout, ("set", "descriptor_set"), default=0),
                "binding": _layout_int(layout, ("binding",)),
                "access": _glsl_access(storage, match.group("qualifiers") or ""),
            }
        )

    constants = []
    spec_spans = []
    for match in GLSL_SPEC_CONSTANT_RE.finditer(source):
        spec_spans.append(match.span())
        layout = _parse_layout(match.group("layout"))
        constants.append(
            {
                "name": match.group("name"),
                "kind": "specialization-constant",
                "id": _layout_int(layout, ("constant_id",)),
                "dtype": match.group("type"),
                "value": _literal_value(match.group("value")),
                "required": False,
                "source": "glsl.layout.constant_id",
            }
        )
    for match in GLSL_SCALAR_CONSTANT_RE.finditer(source):
        if any(start <= match.start() < end for start, end in spec_spans):
            continue
        constants.append(
            {
                "name": match.group("name"),
                "kind": "scalar-constant",
                "dtype": match.group("type"),
                "value": _literal_value(match.group("value")),
                "required": False,
                "source": "glsl.const",
            }
        )

    return _finalize_reflection(
        parser=parser,
        artifact_format=artifact_format,
        entry_points=entry_points,
        resources=resources,
        constants=constants,
        diagnostics=[],
    )


def _glsl_stage_from_source(source: str, artifact_path: Path) -> str | None:
    suffix = artifact_path.suffix.lower()
    suffix_stage = {
        ".vert": "vertex",
        ".frag": "fragment",
        ".comp": "compute",
        ".geom": "geometry",
        ".tesc": "tessellation_control",
        ".tese": "tessellation_evaluation",
    }.get(suffix)
    if suffix_stage:
        return suffix_stage
    lowered = source.lower()
    if "vertex shader" in lowered:
        return "vertex"
    if "fragment shader" in lowered or "pixel shader" in lowered:
        return "fragment"
    if "compute shader" in lowered:
        return "compute"
    if "local_size_" in lowered or "gl_globalinvocationid" in lowered:
        return "compute"
    if "gl_position" in lowered:
        return "vertex"
    if "gl_fragcoord" in lowered or "gl_fragcolor" in lowered:
        return "fragment"
    return None


def _parse_layout(layout: str | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not layout:
        return values
    for part in layout.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = _literal_value(value)
    return values


def _layout_int(
    layout: Mapping[str, Any], names: Sequence[str], *, default: int | None = None
) -> int | None:
    for name in names:
        value = layout.get(name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            parsed = _int_value(value)
            if parsed is not None:
                return parsed
    return default


def _glsl_resource_kind(storage: str, type_name: str) -> str:
    lowered = type_name.lower()
    if storage == "buffer":
        return "buffer"
    if "sampler" in lowered:
        return "sampler" if lowered == "sampler" else "texture"
    if "image" in lowered:
        return "storage-texture"
    if "texture" in lowered:
        return "texture"
    return "uniform"


def _glsl_access(storage: str, qualifiers: str) -> str | None:
    lowered = qualifiers.lower()
    if "writeonly" in lowered:
        return "write"
    if "readonly" in lowered:
        return "read"
    if storage == "buffer":
        return "read_write"
    if storage == "uniform":
        return "read"
    return None


SPIRV_STAGE_BY_MODEL = {
    "Vertex": "vertex",
    "Fragment": "fragment",
    "GLCompute": "compute",
    "Geometry": "geometry",
    "TessellationControl": "tessellation_control",
    "TessellationEvaluation": "tessellation_evaluation",
    "MeshEXT": "mesh",
    "TaskEXT": "task",
    "RayGenerationKHR": "ray_generation",
    "IntersectionKHR": "ray_intersection",
    "AnyHitKHR": "ray_any_hit",
    "ClosestHitKHR": "ray_closest_hit",
    "MissKHR": "ray_miss",
    "CallableKHR": "ray_callable",
}


def _reflect_spirv_artifact(
    artifact_path: Path,
    *,
    artifact_format: str | None,
    tool_resolver: Callable[[str], str | None],
) -> dict[str, Any]:
    suffix = artifact_path.suffix.lower()
    if suffix == ".spvasm":
        source = _read_text(artifact_path)
        reflected_format = artifact_format or "SPIR-V assembly"
    elif suffix in {".spv", ".spirv"}:
        tool = tool_resolver("spirv-dis")
        reflected_format = artifact_format or "SPIR-V binary"
        if tool is None:
            return empty_host_interface_record(
                "unavailable",
                parser="spirv-reflection",
                source="compiled-artifact",
                artifact_format=reflected_format,
                diagnostics=(
                    ReflectionDiagnostic(
                        REFLECTION_TOOL_UNAVAILABLE,
                        "SPIR-V binary reflection requires spirv-dis.",
                        details={"tool": "spirv-dis", "path": str(artifact_path)},
                    ),
                ),
            )
        source = _spirv_disassemble(tool, artifact_path)
    else:
        return empty_host_interface_record(
            "unsupported",
            parser="spirv-reflection",
            source="compiled-artifact",
            artifact_format=artifact_format,
            diagnostics=(
                ReflectionDiagnostic(
                    REFLECTION_UNSUPPORTED_FORMAT,
                    "SPIR-V reflection supports .spvasm, .spv, and .spirv artifacts.",
                    details={"suffix": suffix, "path": str(artifact_path)},
                ),
            ),
        )
    return _reflect_spirv_assembly(source, artifact_format=reflected_format)


def _spirv_disassemble(tool: str, artifact_path: Path) -> str:
    result = subprocess.run(
        [tool, str(artifact_path), "-o", "-"],
        capture_output=True,
        text=True,
        check=False,
        timeout=REFLECTION_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or "spirv-dis failed")
    return result.stdout


def _reflect_spirv_assembly(source: str, *, artifact_format: str) -> dict[str, Any]:
    names: dict[str, str] = {}
    decorations: dict[str, dict[str, Any]] = {}
    pointer_types: dict[str, tuple[str, str | None]] = {}
    variables: dict[str, dict[str, Any]] = {}
    constants_by_id: dict[str, dict[str, Any]] = {}
    entry_points = []
    execution_modes: dict[str, dict[str, Any]] = {}

    for line in source.splitlines():
        tokens = _spirv_tokens(line)
        if not tokens:
            continue
        opcode = _spirv_opcode(tokens)
        if opcode is None:
            continue
        if opcode == "OpName" and len(tokens) >= 3:
            names[tokens[1]] = tokens[2]
        elif opcode == "OpDecorate" and len(tokens) >= 3:
            target = tokens[1]
            value = _literal_value(tokens[3]) if len(tokens) >= 4 else True
            decorations.setdefault(target, {})[tokens[2]] = value
        elif opcode == "OpEntryPoint" and len(tokens) >= 4:
            model = tokens[1]
            function_id = tokens[2]
            entry_points.append(
                {
                    "name": tokens[3],
                    "stage": SPIRV_STAGE_BY_MODEL.get(model, model),
                    "executionConfig": {},
                    "metadata": {
                        "executionModel": model,
                        "function": function_id,
                    },
                }
            )
        elif opcode == "OpExecutionMode" and len(tokens) >= 3:
            function_id = tokens[1]
            if len(tokens) >= 6 and tokens[2] == "LocalSize":
                execution_modes.setdefault(function_id, {})["local_size"] = [
                    _literal_value(tokens[3]),
                    _literal_value(tokens[4]),
                    _literal_value(tokens[5]),
                ]
        elif opcode == "OpTypePointer" and len(tokens) >= 5 and tokens[1] == "=":
            pointer_types[tokens[0]] = (tokens[3], tokens[4])
        elif opcode == "OpVariable" and len(tokens) >= 4 and tokens[1] == "=":
            variable_type = tokens[3]
            storage_class = tokens[4] if len(tokens) >= 5 else None
            variables[tokens[0]] = {
                "typeId": variable_type,
                "storageClass": storage_class,
            }
        elif opcode in {"OpSpecConstant", "OpConstant"} and len(tokens) >= 5:
            constants_by_id[tokens[0]] = {
                "kind": (
                    "specialization-constant"
                    if opcode == "OpSpecConstant"
                    else "scalar-constant"
                ),
                "typeId": tokens[3],
                "value": _literal_value(tokens[4]),
            }

    for entry_point in entry_points:
        function_id = entry_point.get("metadata", {}).get("function")
        mode = execution_modes.get(function_id)
        if mode and "local_size" in mode:
            entry_point["executionConfig"] = {"local_size": list(mode["local_size"])}

    resources = []
    for variable_id, variable in variables.items():
        decorate = decorations.get(variable_id, {})
        if "BuiltIn" in decorate or (
            "Binding" not in decorate and "DescriptorSet" not in decorate
        ):
            continue
        storage_class = variable.get("storageClass")
        pointee = pointer_types.get(variable.get("typeId"), (storage_class, None))[1]
        resources.append(
            {
                "name": names.get(variable_id, variable_id.lstrip("%")),
                "kind": _spirv_resource_kind(str(storage_class), pointee, decorations),
                "type": names.get(str(pointee), str(pointee) if pointee else None),
                "set": decorate.get("DescriptorSet", 0),
                "binding": decorate.get("Binding"),
                "access": _spirv_resource_access(str(storage_class), decorate),
                "metadata": {
                    "id": variable_id,
                    "storageClass": storage_class,
                    "typeId": variable.get("typeId"),
                },
            }
        )
    resources = _annotate_spirv_resource_entry_points(resources, entry_points)

    constants = []
    for constant_id, constant in constants_by_id.items():
        decorate = decorations.get(constant_id, {})
        if constant["kind"] == "scalar-constant" and constant_id not in names:
            continue
        payload = {
            "name": names.get(constant_id, constant_id.lstrip("%")),
            "kind": constant["kind"],
            "dtype": _spirv_type_name(str(constant.get("typeId")), names),
            "value": constant.get("value"),
            "required": False,
            "source": (
                "spirv.spec-constant"
                if constant["kind"] == "specialization-constant"
                else "spirv.constant"
            ),
            "metadata": {"id": constant_id, "typeId": constant.get("typeId")},
        }
        if "SpecId" in decorate:
            payload["id"] = decorate["SpecId"]
        constants.append(payload)

    return _finalize_reflection(
        parser="spirv-reflection",
        artifact_format=artifact_format,
        entry_points=entry_points,
        resources=resources,
        constants=constants,
        diagnostics=[],
    )


def _annotate_spirv_resource_entry_points(
    resources: Sequence[Mapping[str, Any]],
    entry_points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotated = [dict(resource) for resource in resources]
    entry_names = [
        str(entry_point.get("name"))
        for entry_point in entry_points
        if isinstance(entry_point.get("name"), str)
    ]
    if not entry_names:
        return annotated

    for resource in annotated:
        owner = _spirv_resource_entry_point_by_name(resource, entry_names)
        if owner is not None:
            metadata = dict(resource.get("metadata") or {})
            metadata.setdefault("entryPoint", owner)
            resource["metadata"] = metadata

    if len(entry_names) <= 1 or len(annotated) % len(entry_names):
        return annotated

    group_size = len(annotated) // len(entry_names)
    if group_size == 0:
        return annotated
    groups = [
        annotated[index : index + group_size]
        for index in range(0, len(annotated), group_size)
    ]
    first_signature = [
        _spirv_resource_layout_signature(resource) for resource in groups[0]
    ]
    if not all(
        [_spirv_resource_layout_signature(resource) for resource in group]
        == first_signature
        for group in groups[1:]
    ):
        return annotated

    for entry_name, group in zip(entry_names, groups):
        for resource in group:
            metadata = dict(resource.get("metadata") or {})
            metadata.setdefault("entryPoint", entry_name)
            resource["metadata"] = metadata
    return annotated


def _spirv_resource_entry_point_by_name(
    resource: Mapping[str, Any], entry_names: Sequence[str]
) -> str | None:
    metadata = resource.get("metadata")
    metadata_id = metadata.get("id") if isinstance(metadata, Mapping) else ""
    candidates = (
        str(resource.get("type") or ""),
        str(resource.get("name") or ""),
        str(metadata_id or ""),
    )
    for entry_name in entry_names:
        prefixes = (f"{entry_name}_", f"{entry_name}.", f"{entry_name}$")
        if any(candidate.startswith(prefixes) for candidate in candidates):
            return entry_name
    return None


def _spirv_resource_layout_signature(resource: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        resource.get("set", 0),
        resource.get("binding"),
        resource.get("kind"),
        resource.get("name"),
    )


def _spirv_tokens(line: str) -> list[str]:
    stripped = line.split(";", 1)[0].strip()
    if not stripped:
        return []
    try:
        return shlex.split(stripped)
    except ValueError:
        return stripped.split()


def _spirv_opcode(tokens: Sequence[str]) -> str | None:
    if not tokens:
        return None
    if len(tokens) > 2 and tokens[1] == "=":
        return tokens[2]
    return tokens[0]


def _spirv_resource_kind(
    storage_class: str,
    pointee: str | None,
    decorations: Mapping[str, Mapping[str, Any]],
) -> str:
    decorate = decorations.get(str(pointee), {}) if pointee is not None else {}
    if storage_class == "UniformConstant":
        return "texture"
    if storage_class == "Uniform":
        return "buffer" if "BufferBlock" in decorate else "constant-buffer"
    if storage_class == "StorageBuffer":
        return "buffer"
    return "resource"


def _spirv_resource_access(
    storage_class: str, decorations: Mapping[str, Any]
) -> str | None:
    if "NonReadable" in decorations:
        return "write"
    if "NonWritable" in decorations:
        return "read"
    if storage_class == "Uniform":
        return "read"
    if storage_class == "StorageBuffer":
        return "read_write"
    return None


def _spirv_type_name(type_id: str, names: Mapping[str, str]) -> str | None:
    lowered = type_id.lower()
    if "uint" in lowered:
        return "uint"
    if "int" in lowered:
        return "int"
    if "float" in lowered:
        return "float"
    if "bool" in lowered:
        return "bool"
    return names.get(type_id, type_id.lstrip("%") if type_id else None)


def _literal_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered.endswith(("u", "f", "h")):
        text = text[:-1]
    try:
        return int(text, 0)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value.strip()


def _stage_name(stage: str | None) -> str | None:
    if stage is None:
        return None
    label = str(stage).strip().lower().replace("-", "_")
    if not label:
        return None
    return {
        "vert": "vertex",
        "vs": "vertex",
        "frag": "fragment",
        "fs": "fragment",
        "pixel": "fragment",
        "ps": "fragment",
        "comp": "compute",
        "cs": "compute",
        "geom": "geometry",
        "gs": "geometry",
        "tesc": "tessellation_control",
        "tese": "tessellation_evaluation",
    }.get(label, label)


__all__ = [
    "REFLECTION_AMBIGUOUS_BINDING",
    "REFLECTION_EMPTY",
    "REFLECTION_INCOMPLETE_OUTPUT",
    "REFLECTION_PARSE_FAILED",
    "REFLECTION_TOOL_UNAVAILABLE",
    "REFLECTION_UNSUPPORTED_FORMAT",
    "ReflectionDiagnostic",
    "empty_host_interface_record",
    "host_interface_record",
    "reflect_target_host_interface",
]
