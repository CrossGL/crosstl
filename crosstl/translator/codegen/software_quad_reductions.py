"""Quad-group reductions for software subgroups on DirectX and OpenGL.

A quad is four consecutive invocations of the workgroup's linear index, the
grouping of Metal's ``quadgroup_index_in_threadgroup`` and
``thread_index_in_quadgroup``. Logical subgroup widths are multiples of four,
so a quad never straddles two subgroups. When the workgroup's invocation count
is not a multiple of four, its last quad holds fewer invocations; the missing
ones do not exist and contribute nothing.

Every invocation of a quad combines the same scratch values in the same order,
``((q0 + q1) + q2) + q3``, so all of them return bit-identical results and
DirectX and OpenGL agree. Floating-point sums are ``precise`` so that the
compilers cannot reassociate them.
"""

from __future__ import annotations

QUAD_SIZE = 4

# Quad-group builtins, as linear-index arithmetic on the target's
# workgroup-local invocation index.
QUAD_BUILTIN_OPERATORS = {
    "gl_QuadGroupID": "/",
    "gl_QuadGroupInvocationID": "%",
}

# The quad reductions the software lowerings provide.
SOFTWARE_QUAD_REDUCTIONS = frozenset({"QuadActiveSum"})


def quad_builtin_expression(semantic, invocation_index):
    """``invocation_index`` split into the quad ID or the lane in its quad.

    Returns ``None`` when ``semantic`` is not a quad-group builtin.
    """
    operator = QUAD_BUILTIN_OPERATORS.get(semantic)
    if operator is None:
        return None
    return f"({invocation_index} {operator} {QUAD_SIZE}u)"


def _quad_sum_statements(
    value_type,
    scratch,
    invocation,
    invocation_count,
    indent,
    precise_keyword,
    unroll="",
):
    qualifier = precise_keyword if value_type in {"float", "float16_t"} else ""
    partial = invocation_count % QUAD_SIZE != 0
    lines = [
        f"uint quadBase = {invocation} - ({invocation} % {QUAD_SIZE}u);",
        f"{qualifier}{value_type} result = {scratch}[quadBase];",
    ]
    if unroll:
        lines.append(unroll)
    lines.append(
        f"for (uint quadOffset = 1u; quadOffset < {QUAD_SIZE}u; ++quadOffset) {{"
    )
    if partial:
        lines.append(
            f"    if (quadBase + quadOffset >= {invocation_count}u) {{ break; }}"
        )
    lines.append(f"    result = result + {scratch}[quadBase + quadOffset];")
    lines.append("}")
    return "".join(f"{indent}{line}\n" for line in lines)


def hlsl_quad_sum_body(value_type, scratch, invocation_count):
    """The body of a DirectX quad-sum helper after its scratch store.

    The caller's prologue declares ``invocation`` and stores ``value`` at
    ``scratch[invocation]`` behind a group barrier; its epilogue synchronizes
    again before returning ``result``.
    """
    return _quad_sum_statements(
        value_type,
        scratch,
        "invocation",
        invocation_count,
        "    ",
        "precise ",
        unroll="[unroll]",
    )


def glsl_quad_sum_helper(value_type, helper, scratch, invocation_count):
    """A complete OpenGL quad-sum helper over ``scratch``."""
    return (
        f"{value_type} {helper}({value_type} value) {{\n"
        "    uint invocation = gl_LocalInvocationIndex;\n"
        f"    {scratch}[invocation] = value;\n"
        "    memoryBarrierShared();\n"
        "    barrier();\n"
        + _quad_sum_statements(
            value_type,
            scratch,
            "invocation",
            invocation_count,
            "    ",
            "precise ",
        )
        + "    memoryBarrierShared();\n"
        "    barrier();\n"
        "    return result;\n"
        "}\n\n"
    )
