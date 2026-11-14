"""Scalar brace initializers: ``float one = {};`` becomes ``float one = 0.0;``.

C and C++ value-initialize a scalar from an empty brace list, so ``float one = {};`` and ``int n = {};`` are zero.
CUDA and HIP input import them as ``var one: f32 = {};``. Only Metal reads an empty brace list as a scalar's zero:
HLSL copied it, which DXC rejects, WGSL wrote ``array()``, and GLSL refused it as an aggregate.
:func:`zero_empty_scalar_initializers` writes the scalar's zero literal in place of the empty list, for every target,
in variable and constant declarations whose type is ``float``, ``double``, ``half``, ``int``, ``uint``, ``bool``, an
8-, 16- or 64-bit integer (``i16 s = {};`` becomes ``i16 s = 0;``, as written by hand), or an alias of one
(``typedef float real_t; real_t one = {};``).

An empty brace list whose destination is a scalar is that scalar's zero outside a scalar declaration as well:
assigned (``x = {};``), returned (``return {};`` from a function returning ``float``), or a member or element of a
brace list (``S s = {{}, 1.0};``, ``float a[2] = {{}, 1.0};``). The destination is the declared type of the variable,
member or element assigned to, the function's return type, or the structure member or array element the list
initializes. A destination whose type the program does not determine, such as a name declared with different types,
keeps its braces.

A singleton scalar list becomes its sole value, converted to the destination type.
Nested singleton lists are unwrapped too; an empty inner list is zero. More than one
value is refused with :class:`ScalarBraceInitializerError` before code generation.

An enumeration is value-initialized to zero as well. For a plain enumeration, whose variants carry no data,
``Mode m = {};`` becomes ``Mode m = 0;``: the generators write a variable of a plain enumeration as ``int`` (``i32``
in WGSL), whatever type the enumeration names (``enum Mode : uint``), so the zero is ``int``'s. Two targets are left
alone: CUDA output declares the variable with the enumeration's own type, for which ``Mode m = {};`` is already zero
and ``Mode m = 0;`` is an error, and a Rust enum has no integer value. An enumeration whose variants carry data is a
structure on most targets and keeps its braces.

A singleton list of a plain enumeration becomes its sole value on every target, unconverted, as if
``Mode m = Mode::B;`` were written; nested singleton lists are unwrapped too. An empty inner list is zero where the
enumeration is written as ``int``, and stays an empty list on CUDA and Rust, which value-initialize it themselves.
More than one value is refused with :class:`ScalarBraceInitializerError`.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.enum_utils import collect_plain_enums
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["ScalarBraceInitializerError", "zero_empty_scalar_initializers"]

_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})
# CrossGL output keeps the program as written.
_CARRYING = frozenset({"cgl", "crossgl", ""})
# Targets whose generators declare an enumeration-typed variable with the enumeration's own type.
_ENUM_TYPED_TARGETS = frozenset({"cuda", "rust"})
# Scalar spellings, CrossGL and WGSL, by the CrossGL name of their zero's type. An 8-, 16- or 64-bit integer's zero is
# the 32-bit literal of its signedness, as in ``i64 a = 0;`` and ``u8 c = 0u;``.
_SCALAR_NAMES = {
    "float": "float",
    "f32": "float",
    "double": "double",
    "f64": "double",
    "half": "half",
    "f16": "half",
    "int": "int",
    "i32": "int",
    "i8": "int",
    "i16": "int",
    "i64": "int",
    "uint": "uint",
    "u32": "uint",
    "u8": "uint",
    "u16": "uint",
    "u64": "uint",
    "bool": "bool",
}
_ZERO_LITERALS = {
    "float": 0.0,
    "double": 0.0,
    "half": 0.0,
    "int": 0,
    "uint": 0,
    "bool": False,
}


class ScalarBraceInitializerError(TranslationDiagnostic):
    """A scalar declaration has more than one brace-initializer value."""

    # The program is invalid C and C++ whatever the target, so the code names no target.
    code = CODE_PREFIX + "scalar-brace-initializer-invalid"

    def __init__(self, target, name, count, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"scalar brace initializer for '{name}' requires at most one value, got {count}",
            target=target,
        )


def zero_empty_scalar_initializers(program, target) -> int:
    """Normalize scalar declaration brace lists to zero or one converted value; return how many."""
    target = str(target or "").lower()
    if target in _CARRYING:
        return 0
    nodes = list(_nodes(program))
    aliases = {
        node.name: node.var_type
        for node in nodes
        if isinstance(node, A.VariableNode)
        and getattr(node, "is_type_alias", False)
        and isinstance(node.name, str)
    }
    enums = {
        enum.name for enum in collect_plain_enums(nodes) if isinstance(enum.name, str)
    }
    empty_aliases = dict(aliases)
    if target not in _ENUM_TYPED_TARGETS:
        for name in enums:
            empty_aliases.setdefault(name, "int")
    count = 0
    for node in nodes:
        if isinstance(node, A.VariableNode) and not getattr(
            node, "is_type_alias", False
        ):
            type_field, value_field = "var_type", "initial_value"
        elif isinstance(node, A.ConstantNode):
            type_field, value_field = "const_type", "value"
        else:
            continue
        value = getattr(node, value_field, None)
        if not isinstance(value, A.ArrayLiteralNode):
            continue
        declared = getattr(node, type_field, None)
        if value.elements and _enum_name(declared, aliases, enums) is not None:
            setattr(node, value_field, _enum_value(target, node, value, enums))
            count += 1
            continue
        # Only empty enum lists are integers on enum-erasing targets.
        scalar_type = _scalar_type(
            declared, aliases if value.elements else empty_aliases
        )
        if scalar_type is None:
            continue
        scalar = _SCALAR_NAMES[scalar_type.name]
        while isinstance(value, A.ArrayLiteralNode):
            if len(value.elements) > 1:
                raise ScalarBraceInitializerError(
                    target,
                    node.name,
                    len(value.elements),
                    source_location=getattr(value, "source_location", None),
                )
            if not value.elements:
                value = A.LiteralNode(_ZERO_LITERALS[scalar], A.PrimitiveType(scalar))
                break
            value = value.elements[0]
        else:
            # A constructor preserves the destination's width and signedness;
            # later target legalization handles half and narrow integer types.
            value_type = _scalar_type(getattr(value, "literal_type", None), {})
            if (
                not isinstance(value_type, A.PrimitiveType)
                or value_type.name != scalar_type.name
            ):
                value = A.FunctionCallNode(A.IdentifierNode(scalar_type.name), [value])
        setattr(node, value_field, value)
        count += 1
    return count + _zero_empty_scalar_slots(nodes, aliases, empty_aliases)


def _zero_empty_scalar_slots(nodes, aliases, empty_aliases):
    """Zero the empty brace lists assigned, returned, or nested in a brace list whose destination is a scalar."""
    found = {}
    for node in nodes:
        if (
            isinstance(node, A.StructNode)
            and isinstance(node.name, str)
            and not node.generic_params
        ):
            found.setdefault(node.name, []).append(node)
    structs = {name: same[0] for name, same in found.items() if len(same) == 1}
    context = (structs, aliases, empty_aliases)

    functions = []
    local_declarations = set()
    for function in nodes:
        if not isinstance(function, A.FunctionNode) or function.body is None:
            continue
        declared = {}
        for parameter in function.parameters or []:
            if isinstance(parameter, A.ParameterNode):
                declared.setdefault(parameter.name, []).append(parameter.param_type)
        body = list(_nodes(function.body))
        for node in body:
            declaration = _declaration(node)
            if declaration is not None:
                declared.setdefault(node.name, []).append(declaration[0])
                local_declarations.add(id(node))
        functions.append((function, declared, body))
    module = {}
    for node in nodes:
        declaration = _declaration(node)
        if declaration is not None and id(node) not in local_declarations:
            module.setdefault(node.name, []).append(declaration[0])

    count = 0
    for node in nodes:
        declaration = _declaration(node)
        if declaration is not None:
            declared_type, value_field = declaration
            value = getattr(node, value_field, None)
            if isinstance(value, A.ArrayLiteralNode) and value.elements:
                count += _zero_elements(value, declared_type, context)
    for function, declared, body in functions:

        def declared_type(name, declared=declared):
            types = declared.get(name) or module.get(name) or []
            return types[0] if len(types) == 1 else None

        for node in body:
            if isinstance(node, A.AssignmentNode) and node.operator == "=":
                destination = _expression_type(node.target, declared_type, context)
            elif isinstance(node, A.ReturnNode):
                destination = function.return_type
            else:
                continue
            value, zeroed = _zero_value(node.value, destination, context)
            if value is not node.value:
                node.value = value
                if isinstance(node, A.AssignmentNode):
                    node.right = value
            count += zeroed
    return count


def _declaration(node):
    """The declared type and the initializer field of a variable or constant declaration, else ``None``."""
    if isinstance(node, A.VariableNode) and not getattr(node, "is_type_alias", False):
        return node.var_type, "initial_value"
    if isinstance(node, A.ConstantNode):
        return node.const_type, "value"
    return None


def _zero_value(value, destination, context):
    """``value`` with its empty brace lists whose destination is a scalar made zero, and how many were."""
    if not isinstance(value, A.ArrayLiteralNode):
        return value, 0
    if value.elements:
        return value, _zero_elements(value, destination, context)
    scalar_type = _scalar_type(destination, context[2])
    if scalar_type is None:
        return value, 0
    scalar = _SCALAR_NAMES[scalar_type.name]
    return A.LiteralNode(_ZERO_LITERALS[scalar], A.PrimitiveType(scalar)), 1


def _zero_elements(value, destination, context):
    """Zero the empty lists among ``value``'s elements whose structure member or array element is a scalar."""
    destinations = _element_types(destination, len(value.elements), context)
    if destinations is None or any(
        not isinstance(element, A.ArrayLiteralNode)
        and _is_aggregate(element_type, context)
        for element, element_type in zip(value.elements, destinations)
    ):
        # A value for an array or structure starts brace elision, which moves the later elements' destinations.
        return 0
    count = 0
    for index, element_type in enumerate(destinations):
        value.elements[index], zeroed = _zero_value(
            value.elements[index], element_type, context
        )
        count += zeroed
    return count


def _element_types(type_node, count, context):
    """The destinations of the ``count`` elements of a brace list initializing ``type_node``, else ``None``."""
    structs, aliases, _ = context
    type_node = _resolved_type(type_node, aliases)
    if isinstance(type_node, A.ArrayType):
        return [type_node.element_type] * count
    struct = structs.get(_type_name(type_node))
    if struct is None or count > len(struct.members):
        return None
    members = struct.members[:count]
    if not all(isinstance(member, A.StructMemberNode) for member in members):
        return None
    return [member.member_type for member in members]


def _is_aggregate(type_node, context):
    """Whether ``type_node`` is an array or a structure of the program."""
    structs, aliases, _ = context
    type_node = _resolved_type(type_node, aliases)
    return isinstance(type_node, A.ArrayType) or _type_name(type_node) in structs


def _expression_type(expression, declared_type, context):
    """The declared type of the variable, structure member or element ``expression`` names, else ``None``."""
    structs, aliases, _ = context
    if isinstance(expression, A.IdentifierNode):
        return declared_type(expression.name)
    if isinstance(expression, A.MemberAccessNode):
        owner = _resolved_type(
            _expression_type(expression.object_expr, declared_type, context), aliases
        )
        struct = structs.get(_type_name(owner))
        if struct is None:
            return None
        members = [
            member.member_type
            for member in struct.members
            if isinstance(member, A.StructMemberNode)
            and member.name == expression.member
        ]
        return members[0] if len(members) == 1 else None
    if isinstance(expression, A.ArrayAccessNode):
        owner = _resolved_type(
            _expression_type(expression.array_expr, declared_type, context), aliases
        )
        if isinstance(owner, A.ArrayType):
            return owner.element_type
        if isinstance(owner, A.PointerType):
            return owner.pointee_type
    return None


def _resolved_type(type_node, aliases):
    """``type_node`` with type aliases followed to the type they name."""
    seen = set()
    while True:
        name = _type_name(type_node)
        if name is None or name in seen or name not in aliases:
            return type_node
        seen.add(name)
        type_node = aliases[name]


def _type_name(type_node):
    """The name of a type written as a plain name, else ``None``."""
    if isinstance(type_node, str):
        return type_node
    if isinstance(type_node, (A.PrimitiveType, A.NamedType)) and not getattr(
        type_node, "generic_args", None
    ):
        return type_node.name if isinstance(type_node.name, str) else None
    return None


def _enum_value(target, node, value, enums):
    """Unwrap a plain enumeration's singleton list to its value, which keeps its own type."""
    while isinstance(value, A.ArrayLiteralNode):
        if len(value.elements) > 1:
            raise ScalarBraceInitializerError(
                target,
                node.name,
                len(value.elements),
                source_location=getattr(value, "source_location", None),
            )
        if not value.elements:
            # CUDA and Rust value-initialize the enumeration from ``{}``; elsewhere it is ``int``.
            if target in _ENUM_TYPED_TARGETS:
                return value
            return A.LiteralNode(0, A.PrimitiveType("int"))
        value = value.elements[0]
    return value


def _enum_name(type_node, aliases, enums):
    """Resolve a plain enumeration's name through aliases."""
    seen = set()
    while True:
        if isinstance(type_node, (A.PrimitiveType, A.NamedType)) and not getattr(
            type_node, "generic_args", None
        ):
            type_node = type_node.name
        if not isinstance(type_node, str) or type_node in seen:
            return None
        if type_node in enums:
            return type_node
        seen.add(type_node)
        type_node = aliases.get(type_node)


def _scalar_type(type_node, aliases):
    """Resolve a scalar through aliases without losing its width or signedness."""
    seen = set()
    while True:
        if isinstance(type_node, str):
            name = type_node
        elif isinstance(type_node, (A.PrimitiveType, A.NamedType)) and not getattr(
            type_node, "generic_args", None
        ):
            name = type_node.name
        else:
            return None
        if not isinstance(name, str):
            return None
        if name in _SCALAR_NAMES:
            return A.PrimitiveType(
                name
                if name in {"i8", "i16", "i64", "u8", "u16", "u64"}
                else _SCALAR_NAMES[name]
            )
        if name in seen or name not in aliases:
            return None
        seen.add(name)
        type_node = aliases[name]


def _nodes(program):
    seen = set()
    stack = [program]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for field, value in vars(node).items():
            if field in _SKIPPED_FIELDS:
                continue
            if isinstance(value, A.StageMap):
                stack.extend(value.values())
            elif isinstance(value, A.ASTNode):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(item for item in value if isinstance(item, A.ASTNode))
            elif isinstance(value, dict):
                stack.extend(
                    item for item in value.values() if isinstance(item, A.ASTNode)
                )
