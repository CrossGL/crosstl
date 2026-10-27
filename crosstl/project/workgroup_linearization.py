"""Linearize compute workgroups whose shape a target cannot declare.

Direct3D 12 bounds ``numthreads`` per dimension at x <= 1024, y <= 1024 and z <= 64, with 1024 threads in total
(``D3D12_CS_THREAD_GROUP_MAX_X``/``_Y``/``_Z`` and ``D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP``). OpenGL 4.3
guarantees the same ``GL_MAX_COMPUTE_WORK_GROUP_SIZE`` of (1024, 1024, 64) and 1024
``GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS``. Metal bounds only the total (``maxTotalThreadsPerThreadgroup``), so a
Metal kernel may run in threadgroups such as (1, 1, 512) that neither target can declare.

A workgroup (X, Y, Z) whose total fits the target but whose shape does not is emitted as (X * Y * Z, 1, 1). The
flattened local index ``i = x + X * (y + Y * z)`` is the same in both shapes, so ``gl_LocalInvocationIndex``,
``gl_WorkGroupID``, ``gl_NumWorkGroups`` and the subgroup built-ins keep their values. The entry rebuilds the
built-ins whose value depends on the shape from them:

- ``gl_LocalInvocationID = (i % X, (i / X) % Y, i / (X * Y))``;
- ``gl_GlobalInvocationID = gl_WorkGroupID * (X, Y, Z) + gl_LocalInvocationID``;
- ``gl_WorkGroupSize = (X, Y, Z)``;
- ``threads_per_grid = gl_NumWorkGroups * (X, Y, Z)``.

The host keeps dispatching the same workgroup count, so the dispatch contract still names the source shape; the
execution metadata records the declared ``targetWorkgroupSize`` next to it. An entry whose program names one of
the rebuilt built-ins anywhere other than an entry parameter, or whose parameter type is not an integer scalar or
vector, is left unchanged, and the target generator rejects the shape as before.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from crosstl.translator.ast import IdentifierNode
from crosstl.translator.semantics.builtin_values import (
    canonicalize_builtin_values,
    glsl_builtin_name,
)

__all__ = [
    "TARGET_WORKGROUP_LIMITS",
    "WorkgroupLinearization",
    "linearize_compute_workgroups",
    "linearized_workgroup_size",
    "record_workgroup_linearization",
]

# Per-dimension caps and the total-invocation cap each target guarantees.
TARGET_WORKGROUP_LIMITS: Mapping[str, tuple[tuple[int, int, int], int]] = {
    "directx": ((1024, 1024, 64), 1024),
    "opengl": ((1024, 1024, 64), 1024),
}

_LOCAL_ID = "gl_LocalInvocationID"
_GLOBAL_ID = "gl_GlobalInvocationID"
_WORKGROUP_SIZE = "gl_WorkGroupSize"
_THREADS_PER_GRID = "threads_per_grid"
_LOCAL_INDEX = "gl_LocalInvocationIndex"
_WORKGROUP_ID = "gl_WorkGroupID"
_NUM_WORKGROUPS = "gl_NumWorkGroups"

# Built-ins whose value depends on the workgroup shape, in the order the entry rebuilds them.
_REBUILT_SEMANTICS = (_LOCAL_ID, _GLOBAL_ID, _WORKGROUP_SIZE, _THREADS_PER_GRID)
_SWIZZLES = {1: ".x", 2: ".xy", 3: ""}
_VECTOR_PREFIXES = {"uint": "uvec", "int": "ivec"}
_INTEGER_NAMED_TYPE = re.compile(
    r"(?:u?int(?:16|32)?|u?short|u(?:16|32)?vec[23]|i(?:16|32)?vec[23])"
)


@dataclass(frozen=True)
class WorkgroupLinearization:
    """One compute entry emitted with a linearized workgroup."""

    entry_point: str
    source_workgroup_size: tuple[int, int, int]
    target_workgroup_size: tuple[int, int, int]
    rebuilt_parameters: tuple[tuple[str, str], ...]


def linearize_compute_workgroups(
    ast: Any, *, target: str, entry_point: str | None = None
) -> tuple[WorkgroupLinearization, ...]:
    """Rewrite compute entries of ``ast`` whose workgroup shape ``target`` cannot declare.

    Returns one record per rewritten entry, ordered by entry name. Entries the rewrite cannot prove exact are left
    untouched.
    """
    limits = TARGET_WORKGROUP_LIMITS.get(target)
    if limits is None:
        return ()
    stages = [
        stage
        for name, stage in _stage_items(ast)
        if str(getattr(name, "value", name)).strip().lower() == "compute"
        and getattr(stage, "entry_point", None) is not None
        and (entry_point is None or stage.entry_point.name == entry_point)
    ]
    if not stages or _program_names_rebuilt_builtin(ast):
        return ()
    records = []
    for stage in stages:
        size = _stage_workgroup_size(stage)
        if size is None or not _exceeds_shape_only(size, limits):
            continue
        record = _linearize_entry(stage, size, target)
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda record: record.entry_point))


def linearized_workgroup_size(
    target: str, workgroup_size: Sequence[int]
) -> tuple[int, int, int] | None:
    """The size ``target`` declares for a source workgroup it can run only linearized, else ``None``."""
    limits = TARGET_WORKGROUP_LIMITS.get(target)
    if limits is None or len(workgroup_size) != 3:
        return None
    size = tuple(int(value) for value in workgroup_size)
    if any(value < 1 for value in size) or not _exceeds_shape_only(size, limits):
        return None
    return (math.prod(size), 1, 1)


def record_workgroup_linearization(
    execution: Mapping[str, Any], records: Sequence[WorkgroupLinearization]
) -> dict[str, Any]:
    """Return ``execution`` with ``targetWorkgroupSize`` beside each linearized ``workgroupSize``.

    Per-entry records are matched by materialized entry name; a top-level size is recorded when every
    linearized entry shares it.
    """
    updated = dict(execution)
    if not records:
        return updated
    by_entry = {record.entry_point: record for record in records}
    entries = execution.get("entryPoints")
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        rewritten = []
        for entry in entries:
            record = (
                by_entry.get(str(entry.get("materializedEntryPoint", "")))
                if isinstance(entry, Mapping)
                else None
            )
            if record is not None and list(entry.get("workgroupSize", ())) == list(
                record.source_workgroup_size
            ):
                entry = {
                    **entry,
                    "targetWorkgroupSize": list(record.target_workgroup_size),
                }
            rewritten.append(entry)
        updated["entryPoints"] = rewritten
    size = execution.get("workgroupSize")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        matching = {
            record.target_workgroup_size
            for record in records
            if list(record.source_workgroup_size) == list(size)
        }
        if len(matching) == 1:
            updated["targetWorkgroupSize"] = list(next(iter(matching)))
    return updated


def _stage_items(ast: Any) -> Iterable[tuple[Any, Any]]:
    stages = getattr(ast, "stages", None)
    return list(stages.items()) if hasattr(stages, "items") else []


def _stage_workgroup_size(stage: Any) -> tuple[int, int, int] | None:
    config = dict(getattr(stage, "execution_config", None) or {})
    values: Sequence[Any] | None = None
    for key in ("local_size", "workgroup_size", "numthreads"):
        candidate = config.get(key)
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 3:
            values = candidate[:3]
            break
    if values is None:
        if not all(
            key in config for key in ("local_size_x", "local_size_y", "local_size_z")
        ):
            return None
        values = [config[f"local_size_{axis}"] for axis in "xyz"]
    resolved = []
    for value in values:
        text = str(getattr(value, "value", value)).strip()
        if not re.fullmatch(r"[1-9][0-9]*", text):
            return None
        resolved.append(int(text))
    return tuple(resolved)


def _exceeds_shape_only(
    size: tuple[int, int, int], limits: tuple[tuple[int, int, int], int]
) -> bool:
    caps, total = limits
    exceeds = any(value > cap for value, cap in zip(size, caps))
    return exceeds and math.prod(size) <= min(total, caps[0])


def _program_names_rebuilt_builtin(ast: Any) -> bool:
    names = set(_REBUILT_SEMANTICS)
    return any(
        isinstance(node, IdentifierNode) and getattr(node, "name", None) in names
        for node in ast.walk()
    )


def _parameter_semantic(parameter: Any) -> str | None:
    """The GLSL name of the built-in ``parameter`` receives, in any CrossGL spelling."""
    attributes = list(getattr(parameter, "attributes", None) or [])
    semantic = getattr(parameter, "semantic", None)
    if isinstance(semantic, str) and semantic:
        attributes.insert(0, SimpleNamespace(name=semantic, arguments=[]))
    for attribute in attributes:
        if getattr(attribute, "name", None) == _THREADS_PER_GRID:
            return _THREADS_PER_GRID
        name = glsl_builtin_name(attribute)
        if name is not None:
            return name
    return None


def _type_spelling(type_node: Any) -> tuple[str, int] | None:
    """CrossGL spelling and component count of an integer scalar or vector type."""
    kind = type(type_node).__name__
    if kind == "PrimitiveType":
        name = str(getattr(type_node, "name", ""))
        return (name, 1) if _INTEGER_NAMED_TYPE.fullmatch(name) else None
    if kind == "VectorType":
        element = getattr(type_node, "element_type", None)
        size = getattr(type_node, "size", None)
        prefix = _VECTOR_PREFIXES.get(str(getattr(element, "name", "")))
        if (
            type(element).__name__ != "PrimitiveType"
            or prefix is None
            or size not in (2, 3)
        ):
            return None
        return f"{prefix}{size}", size
    if kind == "NamedType" and not getattr(type_node, "generic_args", None):
        name = str(getattr(type_node, "name", ""))
        if not _INTEGER_NAMED_TYPE.fullmatch(name):
            return None
        match = re.search(r"vec([23])$", name)
        return name, int(match.group(1)) if match else 1
    return None


def _unique_name(base: str, taken: set[str]) -> str:
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base}_{suffix}", suffix + 1
    taken.add(name)
    return name


def _local_components(index: str, size: tuple[int, int, int]) -> str:
    x, y, z = size
    components = [
        "0u" if x == 1 else f"{index} % {x}u",
        (
            "0u"
            if y == 1
            else (f"{index} % {y}u" if x == 1 else f"({index} / {x}u) % {y}u")
        ),
        "0u" if z == 1 else (index if x * y == 1 else f"{index} / {x * y}u"),
    ]
    return f"uvec3({', '.join(components)})"


def _linearize_entry(
    stage: Any, size: tuple[int, int, int], target: str
) -> WorkgroupLinearization | None:
    function = stage.entry_point
    parameters = list(getattr(function, "parameters", None) or [])
    body = getattr(function, "body", None)
    if body is None or not isinstance(getattr(body, "statements", None), list):
        return None
    rebuilt: list[tuple[Any, str, tuple[str, int]]] = []
    existing: dict[str, Any] = {}
    for parameter in parameters:
        semantic = _parameter_semantic(parameter)
        if semantic in _REBUILT_SEMANTICS:
            spelling = _type_spelling(getattr(parameter, "param_type", None))
            if spelling is None:
                return None
            rebuilt.append((parameter, semantic, spelling))
        elif semantic in (_LOCAL_INDEX, _WORKGROUP_ID, _NUM_WORKGROUPS):
            existing.setdefault(semantic, parameter)

    taken = {str(getattr(parameter, "name", "")) for parameter in parameters}
    taken.update(
        str(getattr(node, "name", ""))
        for node in function.walk()
        if hasattr(node, "name") and isinstance(getattr(node, "name", None), str)
    )
    semantics = {semantic for _parameter, semantic, _spelling in rebuilt}
    needed = {_LOCAL_INDEX} if semantics & {_LOCAL_ID, _GLOBAL_ID} else set()
    if _GLOBAL_ID in semantics:
        needed.add(_WORKGROUP_ID)
    if _THREADS_PER_GRID in semantics:
        needed.add(_NUM_WORKGROUPS)
    inputs: dict[str, str] = {}
    added: list[str] = []
    for semantic, base, spelling in (
        (_LOCAL_INDEX, "crosstl_linear_index", "uint"),
        (_WORKGROUP_ID, "crosstl_linear_group", "uvec3"),
        (_NUM_WORKGROUPS, "crosstl_linear_groups", "uvec3"),
    ):
        if semantic not in needed:
            continue
        parameter = existing.get(semantic)
        if parameter is not None and _type_spelling(parameter.param_type) == (
            spelling,
            1 if spelling == "uint" else 3,
        ):
            inputs[semantic] = str(parameter.name)
            continue
        inputs[semantic] = _unique_name(base, taken)
        added.append(f"{spelling} {inputs[semantic]} @{semantic}")

    shape = f"uvec3({size[0]}u, {size[1]}u, {size[2]}u)"
    statements: list[str] = []
    local = None
    if needed & {_LOCAL_INDEX}:
        local = _unique_name("crosstl_linear_local", taken)
        statements.append(
            f"uvec3 {local} = {_local_components(inputs[_LOCAL_INDEX], size)};"
        )
    values = {
        _LOCAL_ID: local,
        _GLOBAL_ID: f"{inputs.get(_WORKGROUP_ID)} * {shape} + {local}",
        _WORKGROUP_SIZE: shape,
        _THREADS_PER_GRID: f"{inputs.get(_NUM_WORKGROUPS)} * {shape}",
    }
    for parameter, semantic, (spelling, count) in rebuilt:
        value = values[semantic]
        if spelling == "uvec3":
            statements.append(f"uvec3 {parameter.name} = {value};")
            continue
        temporary = _unique_name(f"crosstl_linear_{parameter.name}", taken)
        statements.append(f"uvec3 {temporary} = {value};")
        statements.append(
            f"{spelling} {parameter.name} = {spelling}({temporary}{_SWIZZLES[count]});"
        )

    snippet_entry = _parse_entry(added, statements, target)
    if snippet_entry is None:
        return None
    rebuilt_ids = {id(parameter) for parameter, _semantic, _spelling in rebuilt}
    new_parameters = [
        parameter for parameter in parameters if id(parameter) not in rebuilt_ids
    ]
    new_parameters.extend(snippet_entry.parameters)
    new_statements = list(snippet_entry.body.statements)
    for node in (*snippet_entry.parameters, *new_statements):
        if hasattr(node, "parent"):
            node.parent = function if node in snippet_entry.parameters else body
    function.parameters = new_parameters
    body.statements[:0] = new_statements
    _set_stage_workgroup_size(stage, (math.prod(size), 1, 1))
    return WorkgroupLinearization(
        entry_point=str(function.name),
        source_workgroup_size=size,
        target_workgroup_size=(math.prod(size), 1, 1),
        rebuilt_parameters=tuple(
            (str(parameter.name), semantic)
            for parameter, semantic, _spelling in rebuilt
        ),
    )


def _parse_entry(
    parameters: Sequence[str], statements: Sequence[str], target: str
) -> Any | None:
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    body = "\n".join(f"            {statement}" for statement in statements)
    source = (
        "shader crosstl_workgroup_linearization {\n"
        "    compute {\n"
        f"        void main({', '.join(parameters)}) {{\n"
        f"{body}\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    program = SOURCE_REGISTRY.get("cgl").parse(
        source, source_options={"strict_function_bodies": True}
    )
    # Spell the added built-in parameters the way the program's other parameters are spelled for this target.
    canonicalize_builtin_values(program, target)
    for _name, stage in _stage_items(program):
        entry = getattr(stage, "entry_point", None)
        if entry is not None and entry.name == "main":
            return entry
    return None


def _set_stage_workgroup_size(stage: Any, size: tuple[int, int, int]) -> None:
    x, y, z = (str(value) for value in size)
    config = dict(getattr(stage, "execution_config", None) or {})
    for key in ("local_size", "workgroup_size"):
        if isinstance(config.get(key), (list, tuple)):
            config[key] = (x, y, z)
    config.update(
        {
            "local_size_x": x,
            "local_size_y": y,
            "local_size_z": z,
            "numthreads": (x, y, z),
        }
    )
    stage.execution_config = config
