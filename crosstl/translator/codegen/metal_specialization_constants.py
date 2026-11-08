"""Emit CrossGL specialization constants as Metal function constants.

A CrossGL ``const T name @constant_id(N) = value;`` is a specialization constant: the host may replace ``value``
when it builds the pipeline (SPIR-V ``Decoration SpecId N``, GLSL ``layout(constant_id = N)``). Metal's equivalent
is a function constant, set through ``MTLFunctionConstantValues`` when the function is specialized. MSL function
constants take no initializer, so the default is expressed with ``is_function_constant_defined`` (Metal Shading
Language Specification, section 5.8.1, "Specifying a Constant Value"):

.. code-block:: c++

    constant int __crossgl_function_constant_scale [[function_constant(0)]];
    constant int scale = is_function_constant_defined(__crossgl_function_constant_scale)
        ? __crossgl_function_constant_scale : 3;

The program keeps reading ``scale``, so constant expressions built from it (``scale * 2 + 1``) follow the
specialized value, as ``OpSpecConstantOp`` does in SPIR-V. The id is preserved, so a host sets the value by index
exactly as it sets the SPIR-V SpecId.

The emitter used to print only ``constant int scale = 3;`` under a "cannot preserve" comment, which silently froze
the default. A type that MSL cannot use for a function constant is refused rather than lowered.

A function constant is not a constant expression to the Metal front end, so it cannot size an array ("array size
is not a constant expression"), and freezing such an array at the default would disagree with every specialized
value. A specialization constant that sizes an array, directly or through a constant expression, is therefore
specialized when the Metal source is compiled instead: it reads a preprocessor macro named after its id, which
defaults to the declared value and which the host sets through ``MTLCompileOptions.preprocessorMacros`` (or ``-D``
for the offline compiler), exactly where it would set the SPIR-V SpecId:

.. code-block:: c++

    #ifndef CROSSGL_SPECIALIZATION_CONSTANT_ID_1
    #define CROSSGL_SPECIALIZATION_CONSTANT_ID_1 512
    #endif
    constant int size = int(CROSSGL_SPECIALIZATION_CONSTANT_ID_1);
    ...
    threadgroup float4 shared_data[size];

Every other specialization constant stays a function constant.

Two kinds of array never need a constant-expression size, so they do not move a specialization constant to a
macro. A stage-interface array's interface locations are assigned from the default size (a larger specialized size
would overlap the next interface variable), and Metal never declares it as an array: each element becomes its own
``[[attribute(N)]]`` or ``[[user(...)]]`` member, read through constant indices or a switch helper. A uniform or
storage block member keeps the static layout of its default size (see :func:`block_member_array_types`). The
caller passes both kinds as ``exempt_array_types``.
"""

from __future__ import annotations

from ..ast import ArrayType, IdentifierNode
from .ast_walk import walk_ast_values

__all__ = [
    "METAL_FUNCTION_CONSTANT_PREFIX",
    "METAL_SPECIALIZATION_MACRO_PREFIX",
    "block_member_array_types",
    "flattened_stage_io_array_types",
    "metal_function_constant_type_refusal",
    "render_metal_specialization_constant",
    "render_metal_specialization_macro",
    "specialization_constant_roots",
    "specialization_constants_sizing_arrays",
    "specialization_dependent_constants",
]

METAL_FUNCTION_CONSTANT_PREFIX = "__crossgl_function_constant_"
METAL_SPECIALIZATION_MACRO_PREFIX = "CROSSGL_SPECIALIZATION_CONSTANT_ID_"


def metal_function_constant_type_refusal(name, type_name, mapped_type, is_value_type):
    """Return the refusal reason when ``mapped_type`` cannot be a function constant, else ``None``.

    ``is_value_type`` is the emitter's predicate for the scalar and vector types MSL accepts.
    """
    if is_value_type(mapped_type):
        return None
    return (
        f"specialization constant '{name}': Metal function constants are scalar or vector values "
        "of a type the Metal language has (Metal Shading Language Specification, sections 2.1 and "
        f"5.8.1); type '{type_name}' has no function constant form"
    )


def render_metal_specialization_constant(
    qualifier, mapped_type, declaration, name, constant_id, value_code
):
    """Render the function constant and the defaulted program-scope constant that reads it.

    ``qualifier`` is the ``constant`` declaration qualifier, ``declaration`` the declarator for ``name`` and
    ``value_code`` the rendered default.
    """
    raw = f"{METAL_FUNCTION_CONSTANT_PREFIX}{name}"
    return (
        f"{qualifier} {mapped_type} {raw} [[function_constant({constant_id})]];\n"
        f"{qualifier} {declaration} = is_function_constant_defined({raw}) "
        f"? {raw} : {mapped_type}({value_code});\n"
    )


def render_metal_specialization_macro(
    qualifier, mapped_type, declaration, constant_id, value_code
):
    """Render a specialization constant that sizes an array: a program-scope constant read from a macro.

    The macro is named after ``constant_id`` and defaults to ``value_code``; a host that defines it when it compiles
    the source specializes the constant, and every array it sizes, as a SpecId does.
    """
    macro = f"{METAL_SPECIALIZATION_MACRO_PREFIX}{constant_id}"
    return (
        f"#ifndef {macro}\n"
        f"#define {macro} {value_code}\n"
        "#endif\n"
        f"{qualifier} {declaration} = {mapped_type}({macro});\n"
    )


def _identifier_names(value):
    return {
        node.name
        for node in walk_ast_values(value)
        if isinstance(node, IdentifierNode) and isinstance(node.name, str)
    }


def specialization_constant_roots(constants, is_specialization_constant):
    """Map each program constant whose value follows a specialization constant to the ones it follows.

    A specialization constant follows itself; any other constant follows those its initializer reads, through the
    constants declared before it.
    """
    roots = {}
    for node in constants or []:
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            continue
        if is_specialization_constant(node):
            roots[name] = {name}
            continue
        followed = set()
        for read in _identifier_names(getattr(node, "value", None)):
            followed |= roots.get(read, set())
        if followed:
            roots[name] = followed
    return roots


def specialization_dependent_constants(constants, is_specialization_constant):
    """Return the names of the program constants whose value follows a specialization constant."""
    return set(specialization_constant_roots(constants, is_specialization_constant))


def specialization_constants_sizing_arrays(
    ast, is_specialization_constant, exempt_array_types=()
):
    """Return the specialization constants that size an array in ``ast``, directly or through constants.

    ``exempt_array_types`` holds the ``ArrayType`` nodes that never need a constant-expression size in Metal:
    stage-interface members emitted one element per member, and block members laid out from their default size.
    """
    roots = specialization_constant_roots(
        getattr(ast, "constants", None), is_specialization_constant
    )
    if not roots:
        return set()
    exempt = {id(node) for node in exempt_array_types}
    sizing = set()
    for node in walk_ast_values(ast):
        if not isinstance(node, ArrayType) or node.size is None:
            continue
        if id(node) in exempt:
            continue
        if isinstance(node.size, (int, str)):
            names = {node.size} if isinstance(node.size, str) else set()
        else:
            names = _identifier_names(node.size)
        for name in names:
            sizing |= roots.get(name, set())
    return sizing


def block_member_array_types(cbuffers, structs, lowered_block_struct_names):
    """Return the ``ArrayType`` types of uniform and storage block members, whose layout ignores specialization.

    GL_KHR_vulkan_glsl, "Specialization Constants": an array inside a block may be sized with a specialization
    constant, "but the block will have a static layout. Changing the specialized size will not re-layout the block.
    In the absence of explicit offsets, the layout will be based on the default size of the array." Metal lays the
    block out from that default (a ``constant`` struct for a uniform block, byte offsets into the bound buffer for a
    lowered storage block), while ``length()``, loop bounds and every other value use keep reading the function
    constant. Only direct block members qualify; a structure nested inside a block may also be declared as a value
    of its own, where its array would need a constant size.
    """
    found = []
    for cbuffer in cbuffers or []:
        for member in getattr(cbuffer, "members", None) or []:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if isinstance(raw_type, ArrayType):
                found.append(raw_type)
    for struct in structs or []:
        if getattr(struct, "name", None) not in lowered_block_struct_names:
            continue
        for member in getattr(struct, "members", None) or []:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if isinstance(raw_type, ArrayType):
                found.append(raw_type)
    return found


def flattened_stage_io_array_types(structs, stage_io_struct_names, array_info):
    """Return the ``ArrayType`` member types of stage-interface structs that Metal flattens per element.

    ``array_info(member)`` is the code generator's flattening test; it returns ``None`` for a member it keeps as
    a Metal array.
    """
    flattened = []
    for struct in structs or []:
        if getattr(struct, "name", None) not in stage_io_struct_names:
            continue
        for member in getattr(struct, "members", None) or []:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if isinstance(raw_type, ArrayType) and array_info(member) is not None:
                flattened.append(raw_type)
    return flattened
