"""Scalar private-pointer views of member lvalues.

``&value.member`` (or a deeper member chain such as ``&value.inner.field``)
rooted at a private value names exactly one non-array object. The HLSL and
GLSL private pointer lowerings pass such a view as an ``inout`` argument of the
member lvalue, so the callee reads and writes the member in place.

The view keeps the root variable as its identity, so the alias checks that
reject two views of one object also reject ``&value`` together with
``&value.member``.

A union member qualifies only when the rendered member *is* the canonical
storage word, which holds for 32-bit unsigned (``uint32``) members. Any other
union member is a reinterpretation of the storage (``asfloat``, sign
conversion or packed lanes), not an lvalue, and raises the backend's union
diagnostic instead of being copied out through a temporary.
"""

from ..ast import IdentifierNode, MemberAccessNode, VariableNode
from .array_utils import split_array_type_suffix

UNION_WORD_MEMBER_KIND = "uint32"


def member_view_chain(operand):
    """Return ``(root_name, members)`` for a pure member-access lvalue.

    ``members`` lists the member-access nodes from the root outwards. Any other
    shape (array elements, calls, pointer members) returns ``(None, ())``.
    """
    members = []
    current = operand
    while isinstance(current, MemberAccessNode):
        members.append(current)
        current = getattr(current, "object_expr", getattr(current, "object", None))
    if not members or not isinstance(current, (IdentifierNode, VariableNode)):
        return None, ()
    name = getattr(current, "name", None)
    if not name:
        return None, ()
    return name, tuple(reversed(members))


def is_single_object_type(type_name):
    """True when ``type_name`` is a known non-array, non-pointer value type."""
    text = str(type_name or "").strip()
    if not text or text.endswith(("*", "&")):
        return False
    _base, array_suffix = split_array_type_suffix(text)
    return not array_suffix


def hlsl_member_view_root(generator, operand, excluded_names):
    """Root name of an HLSL scalar member view of ``operand``, or ``None``.

    ``excluded_names`` holds roots that already have a pointer or array view
    (pointer parameters and fixed local arrays); a member of those is not a
    single private object.
    """
    root, members = member_view_chain(operand)
    if root is None or root in excluded_names:
        return None
    for member in members:
        info = generator.hlsl_union_member_access_info(member)
        if info is not None and info["member"]["kind"] != UNION_WORD_MEMBER_KIND:
            raise generator.hlsl_union_usage_error(
                info,
                member,
                "interpreted-member-address-unsupported",
                "only a 32-bit unsigned union member is the storage word, so "
                "only its address can be passed as a private pointer",
            )
    result_type = generator.type_name_string(generator.expression_result_type(operand))
    if not is_single_object_type(result_type):
        return None
    return root


def glsl_member_view_root(generator, operand, root_binding):
    """Root binding of a GLSL scalar member view of ``operand``, or ``None``.

    ``root_binding`` is the lexical binding of the chain's root; only a
    private value (``scalar_value``) can own a member view.
    """
    root, members = member_view_chain(operand)
    if root is None or root_binding is None:
        return None
    if root_binding.get("root_kind") != "scalar_value":
        return None
    for member in members:
        info = generator.union_storage.member(member)
        if info is not None and info["member"]["kind"] != UNION_WORD_MEMBER_KIND:
            generator.union_storage.reject(
                info,
                "interpreted-member-address-unsupported",
                "only a 32-bit unsigned union member is the storage word, so "
                "only its address can be passed as a private pointer",
            )
    source_type = generator.glsl_source_expression_type(operand)
    if not is_single_object_type(generator.type_name_string(source_type)):
        return None
    return {
        "root_kind": "scalar",
        "root": root,
        "extent": 1,
        "offset": 0,
        "offset_interval": (0, 0),
        "side_effecting": False,
        "element_type": generator.map_type(source_type),
        "source_type": source_type,
        "member_view": True,
    }


def hlsl_aliased_scalar_view(view_calls, scalar_parameters):
    """First call that binds two scalar ``inout`` views to one object.

    HLSL passes a scalar private pointer as an ``inout`` copy, so two views of
    one object are copied in and out independently and lose each other's
    writes. Returns ``(call, first_parameter, second_parameter, root)`` for the
    first such call in deterministic order, or ``None``.
    """
    for caller_name in sorted(view_calls):
        caller_scalars = scalar_parameters.get(caller_name, set())
        for call in view_calls[caller_name]:
            callee_scalars = scalar_parameters.get(call["callee"], set())
            seen = {}
            for parameter_name, binding in call["bindings"].items():
                if parameter_name not in callee_scalars:
                    continue
                root_kind = binding.get("root_kind")
                if root_kind == "parameter":
                    if binding.get("root") not in caller_scalars:
                        continue
                elif root_kind != "scalar":
                    continue
                root = binding.get("root")
                if root in seen:
                    return call, seen[root], parameter_name, root
                seen[root] = parameter_name
    return None
