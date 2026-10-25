"""Compute entries whose workgroup size the host chooses when it dispatches them.

A Metal kernel has no compile-time threadgroup size: the host passes ``threadsPerThreadgroup`` to
``dispatchThreads`` or ``dispatchThreadgroups`` of ``MTLComputeCommandEncoder``, and MSL's
``max_total_threads_per_threadgroup`` attribute only bounds it. Vulkan, OpenGL and
Direct3D fix the size in the shader, as ``LocalSize``, ``local_size_x`` or ``numthreads``. Emitting a size of one
for such a kernel is only correct when nothing the kernel computes depends on the size. A kernel that reads a
threadgroup-relative built-in, declares threadgroup memory, waits at a barrier or calls a SIMD-group function sees
one-thread groups instead, and computes something else without any diagnostic.

The importer marks these kernels ``@ host_workgroup_size``, or ``@ host_workgroup_size(x, y, z)`` when its
``threadgroup_size`` source option names the size the host dispatches with. :func:`resolve_host_workgroup_sizes`
runs when CrossGL is parsed. A kernel that does not depend on the size drops the mark and keeps running as one
invocation per workgroup, which is exact for any dispatch. A kernel that depends on it takes the host's size when
the mark names one; the host then dispatches whole workgroups of that size, as ``dispatchThreadgroups`` does. On
any other kernel the mark records the first construct that depends on the size, and a code generator that needs
the size raises :class:`HostDefinedWorkgroupSizeError` through :func:`require_static_workgroup_size`, unless an
explicit ``@ workgroup_size`` or a project workgroup-size rule gives it. The Metal generator needs no size, so a
Metal round trip keeps the host's choice.
"""

from __future__ import annotations

from typing import Any, Iterable

from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name

__all__ = [
    "HOST_WORKGROUP_SIZE_ATTRIBUTE",
    "HOST_WORKGROUP_SIZE_KEY",
    "HostDefinedWorkgroupSizeError",
    "function_entry_execution_config",
    "has_explicit_workgroup_size",
    "host_workgroup_size_mark",
    "host_workgroup_size_attribute_text",
    "normalize_threadgroup_size_option",
    "require_static_workgroup_size",
    "resolve_host_workgroup_sizes",
    "workgroup_size_dependence",
]

HOST_WORKGROUP_SIZE_ATTRIBUTE = "host_workgroup_size"
HOST_WORKGROUP_SIZE_KEY = "host_workgroup_size"

# CrossGL built-ins whose value depends on the workgroup or subgroup partition of the dispatch.
_SIZE_DEPENDENT_BUILTINS = frozenset(
    name.lower()
    for name in (
        "gl_LocalInvocationID",
        "gl_LocalInvocationIndex",
        "gl_WorkGroupID",
        "gl_WorkGroupSize",
        "gl_NumWorkGroups",
        "gl_SubgroupSize",
        "gl_SubgroupInvocationID",
        "gl_SubgroupID",
        "gl_NumSubgroups",
        "gl_SubgroupEqMask",
        "gl_SubgroupGeMask",
        "gl_SubgroupGtMask",
        "gl_SubgroupLeMask",
        "gl_SubgroupLtMask",
        "gl_QuadGroupID",
        "gl_QuadGroupInvocationID",
        "SV_GroupThreadID",
        "SV_GroupID",
        "SV_GroupIndex",
    )
)

# Execution barriers synchronize the threads of one workgroup or subgroup.
_SIZE_DEPENDENT_CALLS = frozenset(
    name.lower()
    for name in (
        "barrier",
        "workgroupBarrier",
        "workgroupExecutionBarrier",
        "storageBarrier",
        "textureBarrier",
        "subgroupBarrier",
        "subgroupExecutionBarrier",
        "GroupMemoryBarrierWithGroupSync",
        "DeviceMemoryBarrierWithGroupSync",
        "AllMemoryBarrierWithGroupSync",
        "threadgroup_barrier",
        "simdgroup_barrier",
    )
)

# Subgroup, quad and cooperative-matrix operations act on a partition of the workgroup.
_SIZE_DEPENDENT_CALL_PREFIXES = (
    "wave",
    "quad",
    "subgroup",
    "simd",
    "cooperative",
    "coopmat",
)

_WORKGROUP_MEMORY_QUALIFIERS = frozenset(
    {"threadgroup", "shared", "groupshared", "workgroup", "threadgroup_imageblock"}
)

_SIZE_DEPENDENT_TYPE_PREFIXES = ("simdgroup_", "cooperative", "coopmat")

_EXPLICIT_LIST_KEYS = ("local_size", "workgroup_size", "numthreads")
_EXPLICIT_SCALAR_KEYS = ("local_size_x", "local_size_y", "local_size_z")


class HostDefinedWorkgroupSizeError(ValueError):
    """Raised when a target needs a compute size that only the source's host chooses."""

    project_diagnostic_code = "project.translate.workgroup-size-host-defined"
    missing_capabilities = ("execution.host-defined-workgroup-size",)
    check_kind = "execution-specialization"

    def __init__(self, entry: str, reason: str):
        super().__init__(
            f"Entry '{entry}' {reason}, so its result depends on the threadgroup size, which the host "
            "chooses at dispatch time in the source language. The target fixes the size in the shader: give it "
            "with source_options={'threadgroup_size': (x, y, z)} or a project workgroup-size rule."
        )
        self.entry = entry
        self.reason = reason


def normalize_threadgroup_size_option(value: Any) -> tuple[int, int, int] | None:
    """The ``threadgroup_size`` source option as three positive dimensions, or ``None`` when it is absent."""
    if value is None:
        return None
    dimensions = (value,) if isinstance(value, int) else value
    try:
        dimensions = tuple(dimensions)
    except TypeError:
        raise ValueError(
            "threadgroup_size must be a positive integer or a sequence of one to three positive integers"
        ) from None
    if not 1 <= len(dimensions) <= 3 or any(
        type(item) is not int or item <= 0 for item in dimensions
    ):
        raise ValueError(
            "threadgroup_size must be a positive integer or a sequence of one to three positive integers, "
            f"got {value!r}"
        )
    return dimensions + (1,) * (3 - len(dimensions))


def has_explicit_workgroup_size(config: dict | None) -> bool:
    """Whether execution metadata names a workgroup size of its own."""
    config = config or {}
    for key in _EXPLICIT_LIST_KEYS:
        value = config.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return True
    return any(config.get(key) is not None for key in _EXPLICIT_SCALAR_KEYS)


def require_static_workgroup_size(config: dict | None) -> None:
    """Refuse execution metadata of a size-dependent kernel whose size the host chooses."""
    record = (config or {}).get(HOST_WORKGROUP_SIZE_KEY)
    if not record or has_explicit_workgroup_size(config):
        return
    record = record if isinstance(record, dict) else {}
    raise HostDefinedWorkgroupSizeError(
        record.get("entry", "<compute>"),
        record.get("reason", "has a threadgroup size that the host chooses"),
    )


def _name(node: Any) -> str | None:
    name = getattr(node, "name", node)
    return name if isinstance(name, str) else None


def _children(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, (list, tuple)):
        return node
    values = getattr(node, "__dict__", None)
    if not values:
        return ()
    return [
        value
        for key, value in values.items()
        if key not in ("parent", "source_location", "annotations")
        and not key.startswith("_")
    ]


def _type_names(node: Any) -> Iterable[str]:
    for attribute in ("var_type", "param_type", "vtype", "type"):
        name = _name(getattr(node, attribute, None))
        if name:
            yield name


def _node_reason(
    node: Any, functions: dict, workgroup_variables: dict, builtin_parameters: dict
) -> str | None:
    """The size-dependent construct one AST node is, if any."""
    kind = type(node).__name__
    if kind in ("WaveOpNode", "CooperativeMatrixOpNode"):
        return f"calls {getattr(node, 'operation', kind)}"
    if kind == "SyncNode":
        return f"calls {getattr(node, 'sync_type', 'a barrier')}"
    if kind == "BuiltinVariableNode":
        name = str(getattr(node, "builtin_name", ""))
        return f"reads {name}" if name.lower() in _SIZE_DEPENDENT_BUILTINS else None
    if kind == "CooperativeMatrixType":
        return "uses a cooperative matrix"
    if kind == "FunctionCallNode":
        name = _name(getattr(node, "function", None)) or _name(node)
        lowered = (name or "").lower()
        if name in functions:
            return None
        if lowered in _SIZE_DEPENDENT_CALLS or lowered.startswith(
            _SIZE_DEPENDENT_CALL_PREFIXES
        ):
            return f"calls {name}"
        return None
    if kind == "IdentifierNode":
        name = _name(node) or ""
        if name.lower() in _SIZE_DEPENDENT_BUILTINS:
            return f"reads {name}"
        if name in builtin_parameters:
            return f"reads {builtin_parameters[name]}"
        if name in workgroup_variables:
            return f"uses the {workgroup_variables[name]} variable {name}"
        return None
    if kind == "ParameterNode":
        # A built-in parameter counts once the body reads it.
        for attribute in getattr(node, "attributes", None) or []:
            name = _name(attribute) or ""
            if name.lower() in _SIZE_DEPENDENT_BUILTINS and _name(node):
                builtin_parameters.setdefault(_name(node), name)
    qualifiers = {
        str(item).lower() for item in getattr(node, "qualifiers", None) or []
    } & _WORKGROUP_MEMORY_QUALIFIERS
    if qualifiers and kind in ("VariableNode", "ParameterNode", "ArrayNode"):
        return f"declares the {sorted(qualifiers)[0]} variable {_name(node)}"
    for name in _type_names(node):
        if name.lower().startswith(_SIZE_DEPENDENT_TYPE_PREFIXES):
            return f"uses the {name} type"
    return None


def _builtin_axis(node: Any) -> tuple[str, str] | None:
    """``(built-in, axis)`` for a ``gl_*.x``-style component read, or ``None``."""
    if type(node).__name__ != "MemberAccessNode":
        return None
    member = str(getattr(node, "member", ""))
    base = getattr(node, "object", None)
    if member not in ("x", "y", "z") or type(base).__name__ not in (
        "IdentifierNode",
        "BuiltinVariableNode",
    ):
        return None
    name = getattr(base, "builtin_name", None) or _name(base) or ""
    return name.lower(), member


def _product_axis(node: Any, first: str, second: str) -> str | None:
    """The axis of ``first.c * second.c`` in either operand order, or ``None``."""
    if type(node).__name__ != "BinaryOpNode" or getattr(node, "op", None) != "*":
        return None
    left = _builtin_axis(getattr(node, "left", None))
    right = _builtin_axis(getattr(node, "right", None))
    if not left or not right or left[1] != right[1]:
        return None
    return left[1] if {left[0], right[0]} == {first, second} else None


def _partition_invariant(node: Any) -> bool:
    """Whether an expression of partition built-ins has the same value for every workgroup size.

    ``gl_WorkGroupID.c * gl_WorkGroupSize.c + gl_LocalInvocationID.c`` is ``gl_GlobalInvocationID.c``, and
    ``gl_NumWorkGroups.c * gl_WorkGroupSize.c`` is the number of invocations along ``c``, which the dispatch fixes.
    """
    if _product_axis(node, "gl_numworkgroups", "gl_workgroupsize"):
        return True
    if type(node).__name__ != "BinaryOpNode" or getattr(node, "op", None) != "+":
        return False
    operands = (getattr(node, "left", None), getattr(node, "right", None))
    for local, product in (operands, operands[::-1]):
        local_axis = _builtin_axis(local)
        if local_axis and local_axis[0] == "gl_localinvocationid":
            axis = _product_axis(product, "gl_workgroupid", "gl_workgroupsize")
            if axis == local_axis[1]:
                return True
    return False


def _first_reason(root: Any, functions: dict, workgroup_variables: dict) -> str | None:
    """Depth-first, in source order, through the entry and every function it calls by name."""
    seen = set()
    builtin_parameters: dict = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if _partition_invariant(node):
            continue
        reason = _node_reason(node, functions, workgroup_variables, builtin_parameters)
        if reason:
            return reason
        children = list(_children(node))
        if type(node).__name__ == "FunctionCallNode":
            callee = _name(getattr(node, "function", None)) or _name(node)
            children.extend(functions.get(callee, ()))
        stack.extend(reversed(children))
    return None


def _entry_dependence(
    shader: Any, entry: Any, local_functions=(), local_variables=()
) -> str | None:
    functions: dict = {}
    for function in list(getattr(shader, "functions", None) or []) + list(
        local_functions or []
    ):
        name = _name(function)
        if name:
            functions.setdefault(name, []).append(function)
    workgroup_variables = {}
    for variable in list(getattr(shader, "global_variables", None) or []) + list(
        local_variables or []
    ):
        qualifiers = {
            str(item).lower() for item in getattr(variable, "qualifiers", None) or []
        } & _WORKGROUP_MEMORY_QUALIFIERS
        if qualifiers and _name(variable):
            workgroup_variables[_name(variable)] = sorted(qualifiers)[0]
    return _first_reason(entry, functions, workgroup_variables)


def workgroup_size_dependence(shader: Any, stage: Any) -> str | None:
    """Why a compute stage's results depend on its workgroup size, or ``None`` when they do not."""
    return _entry_dependence(
        shader,
        getattr(stage, "entry_point", None),
        getattr(stage, "local_functions", None),
        getattr(stage, "local_variables", None),
    )


def host_workgroup_size_attribute_text(size: tuple[int, int, int] | None) -> str:
    """The CrossGL mark of a kernel whose host chooses its size, naming that size when it is known."""
    if size is None:
        return HOST_WORKGROUP_SIZE_ATTRIBUTE
    return f"{HOST_WORKGROUP_SIZE_ATTRIBUTE}({size[0]}, {size[1]}, {size[2]})"


def host_workgroup_size_mark(attributes: Iterable[Any], render) -> dict | None:
    """The execution-metadata mark of ``@ host_workgroup_size`` attributes, or ``None`` when there is none.

    ``render`` turns one attribute argument into its source text. The mark holds the host's size when the last
    attribute names three dimensions; an attribute with any other number of arguments is malformed.
    """
    marks = [
        attribute
        for attribute in attributes
        if str(getattr(attribute, "name", "")).lower() == HOST_WORKGROUP_SIZE_ATTRIBUTE
    ]
    if not marks:
        return None
    arguments = list(getattr(marks[-1], "arguments", None) or [])
    if not arguments:
        return {"size": None}
    if len(arguments) != 3:
        raise ValueError(
            f"@ {HOST_WORKGROUP_SIZE_ATTRIBUTE} takes no arguments or three dimensions, got {len(arguments)}"
        )
    return {"size": tuple(render(argument) for argument in arguments)}


# Stages that run in workgroups: a Metal kernel, mesh function or object function takes its threadgroup size from
# the host's dispatch, while the targets fix it in the shader.
_WORKGROUP_STAGES = frozenset({"compute", "mesh", "task"})


def _is_workgroup_stage(stage: Any) -> bool:
    kind = getattr(stage, "stage", None)
    return str(getattr(kind, "value", kind)).lower() in _WORKGROUP_STAGES


def _is_compute_function(function: Any) -> bool:
    """Whether a top-level function is a compute entry, as ``@ compute fn`` or a ``compute`` qualifier declares it."""
    qualifiers = list(getattr(function, "qualifiers", None) or [])
    names = qualifiers[:1] + [
        getattr(attribute, "name", attribute)
        for attribute in getattr(function, "attributes", None) or []
    ]
    for name in names:
        stage = normalize_stage_name(name)
        if stage in STAGE_QUALIFIER_NAMES:
            return stage == "compute"
    return False


def _resolve_mark(entry: Any, mark: Any, config: dict, dependence) -> None:
    """Record the host's size, or the reason a target cannot pick one, on a size-dependent entry's config."""
    if has_explicit_workgroup_size(config):
        return
    reason = dependence()
    if reason is None:
        return
    size = mark.get("size") if isinstance(mark, dict) else None
    if size is not None:
        config["numthreads"] = tuple(size)
        return
    config[HOST_WORKGROUP_SIZE_KEY] = {
        "entry": _name(entry) or "main",
        "reason": reason,
    }


def function_entry_execution_config(function: Any) -> dict:
    """The execution metadata of a top-level ``@ compute fn`` entry, which the CrossGL parser records."""
    return dict(getattr(function, "execution_config", None) or {})


def _resolve_function_entries(shader: Any, render) -> None:
    for function in getattr(shader, "functions", None) or []:
        if not _is_compute_function(function):
            continue
        attributes = list(getattr(function, "attributes", None) or [])
        config: dict = {}
        for attribute in attributes:
            name = str(getattr(attribute, "name", "")).lower()
            arguments = list(getattr(attribute, "arguments", None) or [])
            if name in _EXPLICIT_LIST_KEYS and len(arguments) == 3:
                config["numthreads"] = tuple(render(item) for item in arguments)
                break
        mark = host_workgroup_size_mark(attributes, render)
        if mark is not None:
            function.attributes = [
                attribute
                for attribute in attributes
                if str(getattr(attribute, "name", "")).lower()
                != HOST_WORKGROUP_SIZE_ATTRIBUTE
            ]
            _resolve_mark(
                function,
                mark,
                config,
                lambda: _entry_dependence(shader, function),
            )
        function.execution_config = config


def resolve_host_workgroup_sizes(shader: Any, render=str) -> Any:
    """Apply the host's size to compute entries that depend on it; drop the mark from the others.

    Stage entries carry the mark in their execution metadata. Top-level ``@ compute fn`` entries carry it as an
    attribute, which ``render`` reads; their resolved metadata is recorded as the function's ``execution_config``,
    with any ``@ workgroup_size`` the function declares, for :func:`function_entry_execution_config`.
    """
    stages = getattr(shader, "stages", None) or {}
    for stage in list(stages.values()):
        config = getattr(stage, "execution_config", None)
        if not config or not config.get(HOST_WORKGROUP_SIZE_KEY):
            continue
        mark = config.pop(HOST_WORKGROUP_SIZE_KEY)
        if not _is_workgroup_stage(stage):
            continue
        _resolve_mark(
            getattr(stage, "entry_point", None),
            mark,
            config,
            lambda stage=stage: workgroup_size_dependence(shader, stage),
        )
    _resolve_function_entries(shader, render)
    return shader
