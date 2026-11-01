"""WGSL value constructors for CrossGL structure brace initializers.

CrossGL uses brace initializers for structure values, as GLSL does: ``Pair p = {x, 2.0};`` (GLSL 4.60 section 4.1.11,
"Initializers"). WGSL has no initializer lists. Instead, it uses the structure's value constructor, ``Pair(x, 2.0)``,
which takes one value for each member, in member order (WGSL, "Value Constructor Built-in Functions"). WGSL output
therefore writes the initializer as that constructor. Brace initializers for structure or array members, and elements
in brace-initialized arrays of structures, are written using their respective types.

WGSL has no implicit conversion between concrete types (WGSL section 15.1). ``convert_struct_constructor_arguments``
makes conversions for positional constructors explicit before code generation. For example, ``Pair(x, k)`` with an
``int`` named ``k`` has already become ``Pair(x, float(k))``. That function does not process brace initializers, so
values here are converted in the same way: a scalar or vector of another type, with the same number of lanes as the
member, is written using the member type's conversion constructor, such as ``f32(k)``.

As in GLSL output, members after the final supplied value are zero-initialized. Each is written using its zero value,
``T()`` (WGSL, "Zero Value Expressions"), so ``{}`` produces the structure's zero value. WGSL cannot construct a
structure that holds a runtime-sized array, an atomic, or a resource. Such initializers, and initializers with more
values than the structure has members, raise :class:`WGSLStructInitializerError`.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode, ArrayType, NamedType
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .wgsl_array_constructors import wgsl_array_element

__all__ = [
    "WGSLStructInitializerError",
    "is_struct_braced_initializer",
    "wgsl_struct_array_initializer",
    "wgsl_struct_initializer",
]


class WGSLStructInitializerError(TranslationDiagnostic):
    """A structure brace initializer that WGSL cannot write as the structure's value constructor."""

    def __init__(self, struct_name, reason):
        self.struct_name = struct_name
        self.missing_capabilities = ("wgsl.struct-initializer",)
        super().__init__(
            f"WGSL target cannot write the brace initializer of structure {struct_name}: {reason}",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-struct-initializer-unsupported",
        )


def _struct_members(codegen, struct_name):
    struct = codegen._structs_by_name.get(struct_name)
    return [
        (member.name, member.member_type)
        for member in getattr(struct, "members", None) or []
        if hasattr(member, "member_type")
    ]


def _unconstructible_reason(codegen, vtype, seen=()):
    """The reason WGSL has no value constructor for ``vtype``, or ``None`` if it has one.

    WGSL, "Constructible Types".
    """
    if isinstance(vtype, ArrayType):
        if vtype.size is None:
            return "it holds a runtime-sized array"
        return _unconstructible_reason(codegen, vtype.element_type, seen)
    if "atomic" in str(codegen.type_name_string(vtype)):
        return "it holds an atomic"
    struct_name = codegen.struct_type_name(vtype)
    if struct_name not in codegen._structs_by_name or struct_name in seen:
        return None
    if codegen._struct_resource_paths.get(struct_name):
        return "it holds a resource"
    for _name, member_type in _struct_members(codegen, struct_name):
        reason = _unconstructible_reason(
            codegen, member_type, tuple(seen) + (struct_name,)
        )
        if reason is not None:
            return reason
    return None


def is_struct_braced_initializer(codegen, expr, target_type):
    """Whether ``expr`` is a brace initializer for a structure or an array of structures."""
    if not isinstance(expr, ArrayLiteralNode):
        return False
    if isinstance(target_type, ArrayType):
        target_type = target_type.element_type
    return codegen.struct_type_name(target_type) in codegen._structs_by_name


def wgsl_struct_initializer(codegen, expr, target_type):
    """Return WGSL ``S(...)`` when the brace initializer targets structure ``S``; otherwise, return ``None``."""
    if not isinstance(expr, ArrayLiteralNode) or not isinstance(target_type, NamedType):
        return None
    struct_name = codegen.struct_type_name(target_type)
    if struct_name not in codegen._structs_by_name:
        return None
    reason = _unconstructible_reason(codegen, target_type)
    if reason is not None:
        raise WGSLStructInitializerError(struct_name, reason)
    members = _struct_members(codegen, struct_name)
    values = list(expr.elements or [])
    if len(values) > len(members):
        raise WGSLStructInitializerError(
            struct_name,
            f"it has {len(members)} members and the initializer gives {len(values)} values",
        )
    rendered = [
        _member_value(codegen, value, member_type)
        for value, (_name, member_type) in zip(values, members)
    ]
    rendered += [
        f"{codegen.type_name_string(member_type)}()"
        for _name, member_type in members[len(values) :]
    ]
    return f"{codegen.type_name_string(target_type)}({', '.join(rendered)})"


def wgsl_struct_array_initializer(codegen, expr, target_type):
    """Return WGSL ``array<S, N>(S(...), ...)`` for a brace-initialized array of structures; otherwise, ``None``.

    The size may be a constant, as in ``Pair ps[COUNT]``. As with a short array initializer, elements after the
    final supplied value are zero-initialized as ``S()``. An initializer longer than the array, or one targeting an
    array of unknown size, is left to the array literal path.
    """
    if not isinstance(expr, ArrayLiteralNode) or not isinstance(target_type, ArrayType):
        return None
    element_type = target_type.element_type
    if codegen.struct_type_name(element_type) not in codegen._structs_by_name:
        return None
    size = codegen.resolve_integer_expression(target_type.size)
    if size is None or not 0 < len(expr.elements) <= size:
        return None
    element = codegen.type_name_string(element_type)
    values = [
        wgsl_array_element(codegen, value, element_type) for value in expr.elements
    ]
    values += [f"{element}()"] * (size - len(values))
    return f"array<{element}, {size}>({', '.join(values)})"


def _lanes(codegen, vtype):
    shape = codegen.vector_shape(vtype)
    if shape is not None:
        return shape[1]
    return 1 if codegen.scalar_type_name(vtype) is not None else None


def _member_value(codegen, value, member_type):
    """Return WGSL for ``value`` converted to the ``member_type`` member it initializes."""
    if not isinstance(value, ArrayLiteralNode):
        lanes = _lanes(codegen, member_type)
        source_type = codegen.expression_type(value)
        if lanes is not None and lanes == _lanes(codegen, source_type):
            member = codegen.type_name_string(member_type)
            if codegen.type_name_string(source_type) != member:
                return f"{member}({codegen.generate_expression(value)})"
    return codegen.generate_expression_for_target(value, member_type)
