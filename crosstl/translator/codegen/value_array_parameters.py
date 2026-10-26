"""Copy written ``in`` array parameters into locals for CUDA and HIP.

HLSL passes an array parameter without ``out`` or ``inout`` by value, and the
HLSL importer records it with the explicit ``in`` qualifier. A C++ array
parameter decays to a pointer, so a write in the callee would reach the
caller's array. A written ``in`` array with fixed dimensions is therefore
received under another name and copied element by element into a local array
that keeps the source name, as Metal output does.

Unwritten ``in`` arrays keep pointer passing, ``out`` and ``inout`` arrays keep
reference semantics, and arrays without a qualifier keep their declaration,
because sources such as CUDA treat them as pointers. Arrays of pointers,
references or resources such as structured buffers and textures are not
copied either, as in Metal output: their elements are handles, so a copy would
not change what a write reaches. A struct the program declares is never taken
for a resource, whatever its name.
"""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerType,
    ReferenceType,
    StructNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from ..validation import (
    RESOURCE_BUFFER_TYPE_NAMES,
    SAMPLER_STATE_TYPE_NAMES,
    STORAGE_IMAGE_TYPE_NAMES,
)
from .array_utils import split_array_type_suffix
from .generic_function_utils import iter_function_nodes
from .resource_arrays import split_array_suffix

# Builtins that write through an argument, by argument position.
_WRITING_BUILTIN_ARGUMENTS = {"frexp": (1,), "modf": (1,), "sincos": (1, 2)}
# Unary operators whose operand may be written: increments, and taking the
# address, which lets a pointer write the element.
_WRITING_UNARY_OPERATORS = frozenset({"++", "--", "&"})
# Qualifiers under which a pointer parameter cannot be written through. An
# ``in`` array is also not written through, because the callee copies it.
_READ_ONLY_POINTER_QUALIFIERS = frozenset({"const", "constant", "readonly"})
_READ_ONLY_ARRAY_QUALIFIERS = _READ_ONLY_POINTER_QUALIFIERS | {"in"}
_RESOURCE_ELEMENT_TYPE_NAMES = (
    RESOURCE_BUFFER_TYPE_NAMES | SAMPLER_STATE_TYPE_NAMES | STORAGE_IMAGE_TYPE_NAMES
)
# Lower-case resource type names of HLSL, GLSL, Metal and WGSL, matched whole.
_RESOURCE_ELEMENT_TYPE_PATTERN = re.compile(
    r"(?:rw|rasterizerordered)?texture(?:1d|2d|3d|cube)(?:array)?(?:ms)?(?:array)?"
    r"|[iu]?texture(?:1d|2d|3d|cube|2drect|buffer)(?:array)?(?:ms)?(?:array)?"
    r"|texture(?:1d|2d|3d|cube|buffer)(?:_ms)?(?:_array)?"
    r"|texture_(?:depth_|storage_|multisampled_|external)\w*"
    r"|texture_(?:1d|2d|3d|cube)(?:_array)?"
    r"|depth(?:2d|cube)(?:_ms)?(?:_array)?"
    r"|[iu]?sampler(?:1d|2d|3d|cube|2drect|buffer|externaloes)(?:array)?(?:ms)?"
    r"(?:array)?(?:shadow)?"
    r"|sampler_comparison"
    r"|[iu]?subpassinput(?:ms)?"
    r"|raytracingaccelerationstructure|accelerationstructure(?:ext|nv)?"
    r"|acceleration_structure|instance_acceleration_structure"
    r"|primitive_acceleration_structure"
)


def _parameter_kind(type_node, type_name):
    """``"array"``, ``"pointer"`` (pointers and references) or ``None``."""
    base_type, suffix = split_array_type_suffix(type_name)
    if suffix:
        return "array"
    if isinstance(type_node, (PointerType, ReferenceType)):
        return "pointer"
    if "*" in base_type or "&" in base_type:
        return "pointer"
    return None


def collect_function_parameters_by_name(ast_node, type_name_string):
    """Map each function name to ``(qualifiers, kind)`` per parameter, per overload.

    ``kind`` is ``"array"``, ``"pointer"`` or ``None``, as ``_parameter_kind``.
    """
    parameters_by_name = {}
    for function in iter_function_nodes(ast_node):
        parameters = [
            (
                parameter_qualifier_names(parameter),
                _parameter_kind(
                    getattr(parameter, "param_type", ""),
                    type_name_string(getattr(parameter, "param_type", "")),
                ),
            )
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        ]
        parameters_by_name.setdefault(function.name, []).append(parameters)
    return parameters_by_name


def parameter_qualifier_names(parameter):
    return {
        str(qualifier).strip().lower()
        for qualifier in getattr(parameter, "qualifiers", None) or ()
    }


def fixed_array_dimensions(type_name):
    """The dimension texts of a fixed-size array type, or ``None``."""
    _base_type, suffix = split_array_type_suffix(type_name)
    dimensions = split_array_suffix(suffix)
    if not dimensions or any(not dimension.strip() for dimension in dimensions):
        return None
    return [dimension.strip() for dimension in dimensions]


def _array_element_type(type_node):
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    return type_node


def collect_struct_names(ast_node):
    """The names of the structs a program declares."""
    walk = getattr(ast_node, "walk", None)
    nodes = walk() if walk is not None else ()
    return frozenset(
        node.name
        for node in nodes
        if isinstance(node, StructNode) and isinstance(node.name, str)
    )


def copyable_element_type(type_node, type_name, struct_names=frozenset()):
    """Whether an array of this element type is copied by value.

    Arrays of pointers, references, structured buffers, textures, samplers and
    other resources keep their declaration, as Metal's
    ``metal_value_array_parameter_dimensions`` does. A struct in
    ``struct_names`` is a value whatever its name, such as ``Texture2DParams``.
    """
    element_type = _array_element_type(type_node)
    if isinstance(element_type, (PointerType, ReferenceType)):
        return False
    base_type, _suffix = split_array_type_suffix(type_name)
    if "*" in base_type or "&" in base_type:
        return False
    element_name = getattr(element_type, "name", None)
    if not isinstance(element_name, str):
        element_name = base_type
    element_name = element_name.split("<", 1)[0].strip()
    if element_name in struct_names:
        return True
    element_name = element_name.lower()
    return not (
        element_name in _RESOURCE_ELEMENT_TYPE_NAMES
        or _RESOURCE_ELEMENT_TYPE_PATTERN.fullmatch(element_name)
    )


def _lvalue_root_name(expr):
    """The variable an lvalue or pointer expression reaches, as Metal's
    ``assignment_target_root_name``: through members, ``->``, subscripts,
    swizzles, ``*`` and ``&``, and pointer arithmetic such as ``v + 1``."""
    while True:
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, PointerAccessNode):
            expr = expr.pointer_expr
        elif isinstance(expr, ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, SwizzleNode):
            expr = expr.vector_expr
        elif isinstance(expr, UnaryOpNode) and expr.operator in {"*", "&"}:
            expr = expr.operand
        elif isinstance(expr, BinaryOpNode) and expr.operator in {"+", "-"}:
            return _lvalue_root_name(expr.left) or _lvalue_root_name(expr.right)
        else:
            return None


def _body_nodes(body):
    if isinstance(body, (list, tuple)):
        for statement in body:
            yield from _body_nodes(statement)
        return
    walk = getattr(body, "walk", None)
    if walk is not None:
        yield from walk()


def _call_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, IdentifierNode):
        function = function.name
    if isinstance(function, str):
        # An explicitly specialised generic call is named ``fill<int>``.
        return function.split("<", 1)[0].strip()
    return None


def _call_writes_argument(call, index, parameters_by_name):
    name = _call_name(call)
    if name in _WRITING_BUILTIN_ARGUMENTS:
        return index in _WRITING_BUILTIN_ARGUMENTS[name]
    for parameters in parameters_by_name.get(name, ()):
        if index >= len(parameters):
            continue
        qualifiers, kind = parameters[index]
        if qualifiers & {"out", "inout"}:
            return True
        # A non-const pointer or reference, or an array parameter without a
        # qualifier, lets the callee write the caller's elements.
        if kind == "pointer" and not qualifiers & _READ_ONLY_POINTER_QUALIFIERS:
            return True
        if kind == "array" and not qualifiers & _READ_ONLY_ARRAY_QUALIFIERS:
            return True
    return False


def written_parameter_names(body, parameters_by_name):
    """Names the body assigns, increments, takes the address of or passes to a writer."""
    names = set()
    for node in _body_nodes(body):
        if isinstance(node, AssignmentNode):
            names.add(_lvalue_root_name(node.target))
        elif (
            isinstance(node, UnaryOpNode) and node.operator in _WRITING_UNARY_OPERATORS
        ):
            names.add(_lvalue_root_name(node.operand))
        elif isinstance(node, FunctionCallNode):
            for index, argument in enumerate(node.arguments or ()):
                if _call_writes_argument(node, index, parameters_by_name):
                    names.add(_lvalue_root_name(argument))
    names.discard(None)
    return names


def plan_value_array_copies(
    function, parameters, type_name_string, parameters_by_name, struct_names=frozenset()
):
    """Map each written ``in`` array parameter to the name it is received under.

    ``struct_names`` holds the structs the program declares
    (``collect_struct_names``), which are never taken for resources.
    """
    candidates = []
    for parameter in parameters:
        if "in" not in parameter_qualifier_names(parameter):
            continue
        type_node = getattr(parameter, "param_type", "")
        type_name = type_name_string(type_node)
        if fixed_array_dimensions(type_name) is None:
            continue
        if not copyable_element_type(type_node, type_name, struct_names):
            continue
        candidates.append(parameter)
    if not candidates:
        return {}
    body = getattr(function, "body", None)
    written = written_parameter_names(body, parameters_by_name)
    used_names = {getattr(parameter, "name", None) for parameter in parameters}
    for node in _body_nodes(body):
        if isinstance(node, (IdentifierNode, VariableNode)):
            used_names.add(node.name)
    copies = {}
    for parameter in candidates:
        if parameter.name not in written:
            continue
        incoming = f"{parameter.name}_crosstl_input"
        suffix = 2
        while incoming in used_names:
            incoming = f"{parameter.name}_crosstl_input{suffix}"
            suffix += 1
        used_names.add(incoming)
        copies[parameter.name] = incoming
    return copies


def value_array_copy_lines(declaration, name, incoming, dimensions):
    """Declare the local array ``name`` and copy ``incoming`` into it."""
    indices = [f"crosstl_{name}_i{level}" for level in range(len(dimensions))]
    subscript = "".join(f"[{index}]" for index in indices)
    statement = f"{name}{subscript} = {incoming}{subscript};"
    for level in reversed(range(len(dimensions))):
        index = indices[level]
        statement = (
            f"for (int {index} = 0; {index} < int({dimensions[level]}); "
            f"++{index}) {{ {statement} }}"
        )
    return [f"{declaration};", statement]
