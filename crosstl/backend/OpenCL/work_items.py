"""OpenCL C work-item functions (OpenCL C 3.0, 6.15.1) whose dimension is not a literal ``0``, ``1`` or ``2``.

The importer maps ``get_global_id(1)`` to ``gl_GlobalInvocationID.y``, ``get_local_size(0)`` to
``gl_WorkGroupSize.x`` and so on (``OpenCLToCrossGLConverter.format_opencl_builtin_call``). This module imports the
other work-item queries under the same dispatch model, which every CrossGL target has: a three-dimensional grid of
uniform work-groups with a zero global offset.

- A dimension outside 0 to 2 makes ``get_global_id``, ``get_local_id``, ``get_group_id`` and ``get_global_offset``
  return 0 and the size queries return 1 (6.15.1). A constant dimension folds to that value. A dimension known only
  at run time becomes ``(u32(d) < 3u ? gl_GlobalInvocationID[u32(d)] : 0u)``, which reads ``d`` twice, so a
  dimension with a side effect is refused.
- ``get_enqueued_local_size(d)`` is ``get_local_size(d)``: CrossGL work-groups are uniform.
- ``get_global_offset(d)`` is 0 and ``get_global_linear_id()`` is ``gid.x + gid.y * gsize.x + gid.z * gsize.x *
  gsize.y``: the importer already reads ``get_global_id`` as ``gl_GlobalInvocationID``, which assumes a zero global
  offset.
- ``get_work_dim()`` is chosen by the host at enqueue time and no target exposes it, so it is refused with
  :class:`OpenCLWorkItemUnsupportedError`.

A dimension written as a suffixed or cast literal 0, 1 or 2 (``1u``, ``(uint)1``) keeps the importer's current
handling; issue #4819 tracks it. A program that defines a function with one of these names keeps its own function.
"""

from __future__ import annotations

import re

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic


class OpenCLWorkItemUnsupportedError(TranslationDiagnostic):
    """An OpenCL C work-item function whose value no CrossGL target exposes."""

    code = CODE_PREFIX + "opencl-work-item-unsupported"


# Each query with a dimension: the CrossGL vector it reads per component, and its value for a dimension of 3 or more.
_INDEXED_QUERIES = {
    "get_global_id": ("gl_GlobalInvocationID", "0u"),
    "get_local_id": ("gl_LocalInvocationID", "0u"),
    "get_group_id": ("gl_WorkGroupID", "0u"),
    "get_local_size": ("gl_WorkGroupSize", "1u"),
    "get_enqueued_local_size": ("gl_WorkGroupSize", "1u"),
    "get_num_groups": ("gl_NumWorkGroups", "1u"),
    "get_global_size": (None, "1u"),
}
_GLOBAL_SIZE = "(gl_NumWorkGroups.{c} * gl_WorkGroupSize.{c})"
_GLOBAL_LINEAR_ID = (
    "(gl_GlobalInvocationID.x"
    f" + gl_GlobalInvocationID.y * {_GLOBAL_SIZE.format(c='x')}"
    f" + gl_GlobalInvocationID.z * {_GLOBAL_SIZE.format(c='x')} * {_GLOBAL_SIZE.format(c='y')})"
)
_INTEGER_LITERAL = re.compile(r"(?P<digits>0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*")
_INTEGER_CASTS = frozenset(
    {
        "uint",
        "unsigned",
        "unsigned int",
        "int",
        "size_t",
        "ulong",
        "long",
        "unsigned long",
        "ushort",
        "short",
        "uchar",
        "char",
    }
)


def format_work_item_query(converter, name, args):
    """The CrossGL text of the work-item call ``name(args)``, or ``None`` to leave the call to the importer."""
    if name in getattr(converter, "user_function_names", ()):
        return None
    if name == "get_work_dim" and not args:
        raise OpenCLWorkItemUnsupportedError(
            "OpenCL C get_work_dim() returns the number of dimensions the host passes to clEnqueueNDRangeKernel, which no "
            "CrossGL target exposes to a kernel"
        )
    if name == "get_global_linear_id" and not args:
        return _GLOBAL_LINEAR_ID
    if name == "get_global_offset" and len(args) == 1:
        _require_single_evaluation(converter, name, args[0], reads=0)
        return "0u"
    if name not in _INDEXED_QUERIES or len(args) != 1:
        return None
    vector, beyond = _INDEXED_QUERIES[name]
    dimension = args[0]
    component = converter.opencl_dimension_component(dimension)
    if component is not None:
        return _component(name, vector, component)
    value = _constant_dimension(dimension)
    if value is not None:
        if 0 <= value <= 2:
            # A suffixed or cast literal 0, 1 or 2 (#4819).
            return f"{name}({converter.visit(dimension)})"
        return beyond
    _require_single_evaluation(converter, name, dimension, reads=2)
    index = _as_u32(converter.visit(dimension))
    if vector is None:
        read = f"(gl_NumWorkGroups[{index}] * gl_WorkGroupSize[{index}])"
    else:
        read = f"{vector}[{index}]"
    return f"({index} < 3u ? {read} : {beyond})"


def _component(name, vector, component):
    if vector is None:
        return _GLOBAL_SIZE.format(c=component)
    return f"{vector}.{component}"


def _require_single_evaluation(converter, name, dimension, reads):
    if converter.opencl_expression_is_side_effect_free(dimension):
        return
    if reads == 0:
        detail = "the import does not evaluate the dimension"
    else:
        detail = "the import reads the dimension twice"
    raise OpenCLWorkItemUnsupportedError(
        f"OpenCL C {name}() with a dimension that has a side effect: {detail}; assign the dimension to a variable first"
    )


def _constant_dimension(expression):
    """The value of an integer literal, a cast of one or a negated one; ``None`` for anything else."""
    negate = False
    while True:
        if type(expression).__name__ == "CastNode":
            if (
                str(getattr(expression, "target_type", "")).strip()
                not in _INTEGER_CASTS
            ):
                return None
            expression = getattr(expression, "expression", None)
            continue
        if (
            type(expression).__name__ == "UnaryOpNode"
            and getattr(expression, "op", None) == "-"
        ):
            negate = not negate
            expression = getattr(expression, "operand", None)
            continue
        break
    if isinstance(expression, bool):
        return None
    if isinstance(expression, int):
        value = expression
    elif isinstance(expression, str):
        match = _INTEGER_LITERAL.fullmatch(expression.strip())
        if match is None:
            return None
        digits = match["digits"]
        if digits[:2] in ("0x", "0X"):
            value = int(digits, 16)
        elif len(digits) > 1 and digits[0] == "0":
            if not all(ch in "01234567" for ch in digits):
                return None
            value = int(digits, 8)
        else:
            value = int(digits)
    else:
        return None
    return -value if negate else value


def _as_u32(rendered):
    """``rendered`` converted to ``u32`` once: OpenCL C declares every dimension parameter ``uint``."""
    if rendered.startswith("u32(") and _closes_at_end(rendered, len("u32")):
        return rendered
    return f"u32({rendered})"


def _closes_at_end(text, open_index):
    depth = 0
    for position in range(open_index, len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                return position == len(text) - 1
    return False
