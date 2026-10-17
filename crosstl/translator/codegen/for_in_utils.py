"""Typed fixed-array iteration for targets without native range references."""

from ..ast import (
    ArrayAccessNode,
    ForInNode,
    IdentifierNode,
    MemberAccessNode,
    ReferenceType,
    VariableNode,
)
from .array_utils import format_c_style_array_declaration


def validate_range_reference_argument(
    generator, argument, parameter_type, qualifiers=()
):
    """Do not turn source reference identity into target copy-in/copy-out."""
    bindings = getattr(generator, "current_for_in_reference_bindings", {})
    if not bindings:
        return
    type_name = generator.type_name_string(parameter_type) or ""
    if not (
        type_name.rstrip().endswith(("&", "*"))
        # Metal const references are currently imported as explicit `in`.
        or set(qualifiers).intersection({"in", "out", "inout"})
    ):
        return
    for child in generator.walk_ast(argument):
        if isinstance(child, (IdentifierNode, VariableNode)):
            reject = bindings.get(child.name)
            if reject is not None:
                reject("reference-helper-transport-unsupported")


def generate_typed_array_for_in(generator, node, indent, *, target):
    """Keep the original array live and capture each subarray selector once.

    Callers own the surrounding lexical state. Reference bindings are expression
    aliases, not copies followed by writeback: break, continue and return must
    retain every write, and reads must observe other aliases of the same element.
    """
    glsl = target == "opengl"
    expression_type = (
        generator.glsl_source_expression_type
        if glsl
        else generator.hlsl_source_expression_type
    )
    fail = (
        generator.raise_glsl_for_in_iterable_error
        if glsl
        else generator.raise_hlsl_for_in_iterable_error
    )
    shape = (
        generator.glsl_for_in_array_shape if glsl else generator.hlsl_for_in_array_shape
    )
    pattern = node.pattern
    iterable_type = generator.type_name_string(expression_type(node.iterable))
    extent, element_type = shape(node, pattern, iterable_type)

    def reject(reason):
        fail(
            node,
            pattern,
            iterable_type,
            binding_type=generator.type_name_string(node.binding_type),
            reason=reason,
        )

    used_names = set(generator.local_variable_types)
    used_names.update(generator.global_variable_types)
    used_names.update(generator.current_identifier_aliases.values())

    def fresh(suffix):
        name = (
            generator.glsl_synthetic_local_identifier(f"{pattern}_crossgl_{suffix}")
            if glsl
            else generator.hlsl_unique_local_identifier(
                f"{pattern}_crossgl_{suffix}", used_names
            )
        )
        used_names.add(name)
        return name

    prefix = "    " * indent
    code = f"{prefix}{{\n"
    roots = set()

    def capture(expression):
        nonlocal code
        if isinstance(expression, IdentifierNode):
            roots.add(expression.name)
            return expression
        if isinstance(expression, MemberAccessNode):
            owner_type = generator.type_name_string(
                expression_type(expression.object_expr)
            )
            if owner_type is None or owner_type.rstrip().endswith("*"):
                reject("unsupported-container-identity")
            return MemberAccessNode(capture(expression.object_expr), expression.member)
        if isinstance(expression, ArrayAccessNode):
            owner_type = generator.type_name_string(expression_type(expression.array))
            if owner_type is None or owner_type.rstrip().endswith("*"):
                reject("unsupported-container-identity")
            array = capture(expression.array)
            index_type = expression_type(expression.index)
            if not generator.is_scalar_integer_type(index_type):
                reject("unresolved-selector-type")
            index = fresh("selector")
            code += generator.generate_statement(
                VariableNode(index, index_type, expression.index), indent + 1
            )
            return ArrayAccessNode(array, IdentifierNode(index))
        reject("unsupported-container-identity")

    container = capture(node.iterable)
    container_text = generator.generate_expression(container)
    reference = isinstance(node.binding_type, ReferenceType)
    raw_value_type = (
        node.binding_type.referenced_type if reference else node.binding_type
    )
    value_type = generator.type_name_string(raw_value_type)
    if value_type in {None, "auto"}:
        value_type = element_type
    if glsl and generator.type_contains_opaque_resource(element_type):
        reject("unsupported-element-type")
    if not glsl and (
        generator.is_resource_parameter_type(element_type)
        or generator.is_hlsl_global_resource_type(element_type)
    ):
        reject("unsupported-element-type")
    mapped_type = generator.map_type(value_type)
    readonly = (
        "const" in node.binding_qualifiers or "constant" in node.binding_qualifiers
    )
    readonly |= reference and not node.binding_type.is_mutable
    # A const reference to a converted scalar binds a temporary, not the source.
    if reference and mapped_type != generator.map_type(element_type):
        if not readonly:
            reject("reference-element-type-mismatch")
        reference = False
    if reference:
        for child in generator.walk_ast(node.body):
            name = (
                child.name
                if isinstance(child, VariableNode)
                else child.pattern if isinstance(child, ForInNode) else None
            )
            if name in roots or name == pattern:
                reject("reference-container-shadowed")
    if glsl and id(node) in generator.current_glsl_scalar_array_view_plans:
        reject("binding-address-view-unsupported")

    # All iterable expressions above are resolved before the binding shadows
    # an outer local, resource alias or compile-time constant of the same name.
    for attribute in (
        "current_identifier_aliases",
        "current_resource_aliases",
        "current_hlsl_resource_pointer_aliases",
        "current_glsl_workgroup_pointer_aliases",
        "current_glsl_storage_pointer_aliases",
        "current_stage_parameter_aliases",
        "current_stage_inputs",
        "current_stage_outputs",
        "current_glsl_scalar_array_views",
        "current_compile_time_int_constants",
        "current_compile_time_int_vector_constants",
        "current_compile_time_int_vector_array_constants",
        "current_hlsl_visible_int_constants",
        "current_hlsl_texture_offset_constants",
    ):
        aliases = getattr(generator, attribute, None)
        if aliases is not None:
            aliases.pop(pattern, None)

    index = fresh("index")
    initializer = f"{container_text}[{index}]"
    generator.local_variable_types[index] = "int"
    generator.local_variable_source_types[index] = "int"
    generator.local_variable_types[pattern] = value_type
    generator.local_variable_source_types[pattern] = value_type
    code += f"{prefix}    for (int {index} = 0; {index} < {extent}; ++{index}) {{\n"
    if reference:
        generator.current_identifier_aliases[pattern] = initializer
    else:
        name = fresh("value")
        generator.current_identifier_aliases[pattern] = name
        if glsl and mapped_type != generator.map_type(element_type):
            initializer = f"{mapped_type}({initializer})"
        declaration = format_c_style_array_declaration(mapped_type, name)
        code += f"{prefix}        {'const ' if readonly else ''}{declaration} = {initializer};\n"
    previous_bindings = getattr(generator, "current_for_in_reference_bindings", {})
    generator.current_for_in_reference_bindings = dict(previous_bindings)
    if reference:
        generator.current_for_in_reference_bindings[pattern] = reject
    else:
        generator.current_for_in_reference_bindings.pop(pattern, None)
    try:
        code += generator.generate_scoped_statement_body(node.body, indent + 2)
    finally:
        generator.current_for_in_reference_bindings = previous_bindings
    code += f"{prefix}    }}\n{prefix}}}\n"
    return code
