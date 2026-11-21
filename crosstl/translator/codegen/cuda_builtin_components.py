"""Subscripts of the CrossGL compute built-ins in CUDA and HIP output.

CrossGL, like GLSL, HLSL and Metal, reads a component of a ``uvec3`` compute built-in with a subscript, and the
index may be a runtime value: ``gl_GlobalInvocationID[d]``. CUDA and HIP spell each component of a built-in as its
own expression (``blockIdx.x * blockDim.x + threadIdx.x``), and their ``uint3`` and ``dim3`` have no subscript
operator. A constant subscript names its component directly. A runtime subscript calls one helper that takes the
index and the three components, so the index expression is evaluated exactly once, as in the source; the components
are built-in reads, which have no side effects.

GLSL leaves a vector index outside 0..2 undefined (GLSL 4.60 section 5.5), as HLSL and Metal do. The helper
returns the ``z`` component for such an index. A constant index outside 0..2 does not compile in the source
languages and is refused.
"""

from __future__ import annotations

RUNTIME_COMPONENT_HELPER = "cgl_builtin_component"

_RUNTIME_COMPONENT_TEXT = (
    "__device__ inline unsigned int {name}(unsigned int index, unsigned int x, unsigned int y, unsigned int z)\n"
    "{{\n"
    "    return index == 0u ? x : (index == 1u ? y : z);\n"
    "}}"
)


def compute_builtin_subscript(
    generator, name, role, index_node, component_expression, refused, target
):
    """The CUDA or HIP expression that reads ``name[index_node]``, a component of the compute built-in ``role``.

    ``component_expression(role, component)`` spells one component, or returns ``None`` when the built-in has no
    such component; ``refused`` is the generator's typed lowering diagnostic.
    """
    index = generator.literal_int_value(index_node)
    if index is not None:
        if index not in (0, 1, 2):
            raise refused(
                [
                    f"cannot subscript compute built-in '{name}' with index {index} in {target}: "
                    "its components are 0, 1 and 2"
                ]
            )
        component = "xyz"[index]
        expression = component_expression(role, component)
        if expression is None:
            raise refused(
                [
                    f"compute built-in '{name}' has no '.{component}' component in {target}"
                ]
            )
        return expression
    components = [component_expression(role, component) for component in "xyz"]
    if None in components:
        raise refused(
            [
                f"compute built-in '{name}' is a scalar in {target}; it has no components to subscript"
            ]
        )
    generator.subscript_index_depth += 1
    try:
        index_text = generator.visit(index_node)
    finally:
        generator.subscript_index_depth -= 1
    helpers = generator.helper_functions
    if RUNTIME_COMPONENT_HELPER not in helpers:
        helpers[RUNTIME_COMPONENT_HELPER] = _RUNTIME_COMPONENT_TEXT.format(
            name=RUNTIME_COMPONENT_HELPER
        )
    return f"{RUNTIME_COMPONENT_HELPER}({index_text}, {', '.join(components)})"
