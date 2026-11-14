"""Subscripts of vector values that have no variable, such as the constant ``gl_WorkGroupSize``.

SPIR-V declares the ``WorkgroupSize`` built-in as a constant composite rather than as an ``Input`` variable (SPIR-V
1.6 section 3.21, BuiltIn WorkgroupSize), so the generator holds a value, not a pointer, for it. ``OpAccessChain``
needs a pointer base (section 3.49.8), and the generator wrote ``gl_WorkGroupSize[0]`` as an access chain on the
constant, which spirv-val rejects. ``gl_WorkGroupSize.x`` already extracted the component.

A subscript of such a named value now extracts the component: ``OpCompositeExtract`` for a constant index and
``OpVectorExtractDynamic`` (section 3.49.11) otherwise. A vector held in a variable keeps its access chain.
"""

from __future__ import annotations

from ..ast import IdentifierNode, VariableNode

__all__ = ["vector_value_subscript"]


def _stored_value(generator, name):
    """The local, global or constant built-in ``name`` refers to, found without emitting instructions.

    ``variable_pointer_from_expression`` also resolves bare uniform-block members, and it does so by emitting an
    ``OpAccessChain``. Those members always have storage, so the caller would emit the access chain a second time.
    """
    found = generator.local_variables.get(name) or generator.resolve_global_variable(
        name
    )
    if found is not None:
        return found
    info = generator.compute_builtin_info(name)
    if info is not None and info[2] == "Constant":
        return generator.ensure_compute_builtin(name)
    return None


def vector_value_subscript(generator, array_expr, index, index_expr) -> object | None:
    """The component ``array_expr[index]`` when ``array_expr`` names a vector value without storage, else ``None``."""
    if not isinstance(array_expr, (IdentifierNode, VariableNode)):
        # Resolving a compound base emits its access chains and index code, which the caller then emits again.
        return None
    base = _stored_value(generator, array_expr.name)
    if base is None or base.type.storage_class:
        return None
    vector_type = generator.registered_value_type(base)
    vector_info = (
        generator.vector_component_type_and_count(vector_type.type.base_type)
        if vector_type is not None
        else None
    )
    if vector_info is None:
        return None
    from .SPIRV_codegen import (
        SpirvId,  # imported here: the host module imports this one
    )

    component_type = generator.register_primitive_type(vector_info[0])
    literal = generator.constant_scalar_literal_value(index_expr)
    if isinstance(literal, int) and not isinstance(literal, bool):
        if 0 <= literal < vector_info[1]:
            return generator.composite_extract(base, component_type, literal)
        generator.refuse_lowering(
            f"Component {literal} of a {vector_info[1]}-component vector is out of range"
        )
        return generator.register_constant(0, component_type)
    result = generator.get_id()
    generator.emit(
        f"%{result} = OpVectorExtractDynamic %{component_type.id} %{base.id} %{index.id}"
    )
    generator.value_types[result] = component_type
    return SpirvId(result, component_type.type)
