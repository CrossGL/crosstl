"""The memory qualifiers of a Slang resource declaration.

GLSL's storage-image and storage-block memory qualifiers (GLSL 4.60 section 4.10: ``coherent``, ``volatile``,
``restrict``, ``readonly`` and ``writeonly``) and MSL's ``volatile`` and scoped ``coherent(device)`` resources are
type modifiers on a Slang resource. slangc 2026.20 accepts all five on a module-scope read-write texture or buffer,
and the first three on the read-only ``StructuredBuffer``, ``ByteAddressBuffer`` and ``Buffer``. It decorates the
SPIR-V variable ``Coherent``, ``Volatile``, ``Restrict``, ``NonWritable`` and ``NonReadable``, and its HLSL target
keeps the coherence as ``globallycoherent``.

The Slang generator wrote ``globallycoherent`` alone, on read-write resources alone, and only when the qualifier was
spelled ``coherent`` or ``globallycoherent``. ``volatile``, ``restrict`` and ``writeonly`` were dropped from every
resource, MSL's ``coherent(device)`` and ``coherent(threadgroup)`` were dropped, and GLSL storage blocks lowered to
a structured or byte-address buffer lost every qualifier, coherence included.

:func:`slang_resource_memory_prefix` gives a declaration's modifiers in a fixed order. Coherence at device or
threadgroup scope is written ``globallycoherent``, which is exact for the device and stronger than a thread group
needs. A read-only resource type implies ``readonly``, and an append or consume buffer implies its access, so the
access modifier is written only on the other read-write types.

What Slang cannot declare is refused through the generator's lowering refusals:

- coherence at system or host scope (MSL ``coherent(system)``): Slang's coherence covers the device, and Vulkan has
  no wider coherence;
- a coherent or volatile parameter: slangc allows neither modifier on a parameter (E31201);
- a coherent or volatile resource passed to a function: slangc rejects an argument whose memory qualifiers the
  parameter lacks (E30048), and the parameter cannot declare them.

``restrict``, ``readonly`` and ``writeonly`` only promise how a resource is used. They are left off a parameter, and
``readonly`` and ``writeonly`` off a resource passed to a function (which slangc would reject, E30048), without
changing behaviour.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, FunctionCallNode, IdentifierNode
from .memory_qualifier_names import memory_qualifier_name

__all__ = [
    "helper_resource_argument_names",
    "slang_resource_memory_prefix",
    "slang_resource_parameter_memory_prefix",
]

_ACCESS_TYPES = (
    "RWTexture",
    "RWBuffer",
    "RWStructuredBuffer",
    "RWByteAddressBuffer",
    "RasterizerOrdered",
)
_QUEUE_TYPES = ("AppendStructuredBuffer", "ConsumeStructuredBuffer")
_READ_ONLY_TYPES = frozenset({"StructuredBuffer", "ByteAddressBuffer", "Buffer"})
_COHERENCE = frozenset({"coherent", "globallycoherent"})
_SYSTEM_COHERENCE = frozenset({"coherent(system)", "coherent(host)"})
_SYSTEM_COHERENCE_REASON = (
    "is coherent at system scope: Slang's globallycoherent makes a resource coherent across the device only, "
    "and Vulkan has no wider coherence"
)


def _resource_kind(base_type):
    """``"access"``, ``"queue"``, ``"read"`` or ``None`` for the Slang resource type ``base_type``."""
    if not isinstance(base_type, str):
        return None
    if base_type.startswith(_ACCESS_TYPES):
        return "access"
    if base_type.startswith(_QUEUE_TYPES):
        return "queue"
    if base_type.split("<", 1)[0].strip() in _READ_ONLY_TYPES:
        return "read"
    return None


def _declared_memory_qualifiers(node):
    """The lower-case memory qualifier names ``node`` declares, through qualifiers or attributes."""
    values = list(getattr(node, "qualifiers", None) or [])
    values.extend(getattr(node, "resource_qualifiers", None) or [])
    declared_type = getattr(node, "var_type", None) or getattr(node, "param_type", None)
    while getattr(declared_type, "element_type", None) is not None:
        declared_type = declared_type.element_type
    values.extend(getattr(declared_type, "resource_qualifiers", None) or [])
    names = {memory_qualifier_name(value) for value in values}
    for attribute in getattr(node, "attributes", None) or []:
        name = str(getattr(attribute, "name", "")).lower()
        arguments = getattr(attribute, "arguments", None) or []
        if name == "coherent" and len(arguments) == 1:
            # ``@coherent(device)`` is the attribute spelling of the scoped qualifier ``coherent(device)``.
            scope = str(getattr(arguments[0], "name", arguments[0])).strip()
            name = memory_qualifier_name(f"coherent({scope})")
        names.add(name)
    return names


def _scope_refusal(names, subject):
    """The refusal reason for a coherence scope Slang cannot declare, or ``None``."""
    if names & _SYSTEM_COHERENCE:
        return f"{subject} {_SYSTEM_COHERENCE_REASON}"
    unknown = sorted(name for name in names if name.startswith("coherent("))
    if unknown:
        return f"{subject} declares {unknown[0]}, a coherence scope Slang has no modifier for"
    return None


def _coherence_and_volatility(names):
    words = []
    if names & _COHERENCE:
        words.append("coherent")
    if "volatile" in names:
        words.append("volatile")
    return " and ".join(words)


def _argument_resource_name(argument):
    while isinstance(argument, ArrayAccessNode):
        argument = argument.array_expr
    if isinstance(argument, IdentifierNode):
        return argument.name
    return None


def helper_resource_argument_names(ast, user_function_names):
    """The names passed, whole or indexed, as an argument of a call to one of ``user_function_names``."""
    names = set()
    for node in ast.walk():
        if not isinstance(node, FunctionCallNode):
            continue
        callee = node.function
        if isinstance(callee, IdentifierNode):
            callee = callee.name
        if not isinstance(callee, str) or callee not in user_function_names:
            continue
        for argument in node.arguments or []:
            name = _argument_resource_name(argument)
            if name:
                names.add(name)
    return names


def slang_resource_memory_prefix(codegen, mapped_type, node):
    """The Slang memory modifiers, each followed by a space, for resource ``node`` of Slang type ``mapped_type``.

    ``codegen`` is the Slang generator: its ``resource_base_type`` strips array suffixes, its
    ``explicit_resource_access`` reads the declared access, its ``slang_helper_resource_arguments`` names the
    resources passed to functions, and its ``refuse_lowering`` records a refusal.
    """
    if node is None:
        return ""
    kind = _resource_kind(codegen.resource_base_type(mapped_type))
    if kind is None:
        return ""

    names = _declared_memory_qualifiers(node)
    resource = getattr(node, "name", None) or "<unnamed>"
    scope_refusal = _scope_refusal(names, f"resource '{resource}'")
    if scope_refusal is not None:
        return codegen.refuse_lowering(scope_refusal)

    passed_to_function = resource in getattr(
        codegen, "slang_helper_resource_arguments", ()
    )
    unrepresentable = _coherence_and_volatility(names)
    if passed_to_function and unrepresentable:
        return codegen.refuse_lowering(
            f"resource '{resource}' is {unrepresentable} and is passed to a function: slangc rejects an "
            "argument whose memory qualifiers the parameter lacks (E30048), and allows no coherent or volatile "
            "parameter (E31201)"
        )

    modifiers = []
    if names & _COHERENCE:
        modifiers.append("globallycoherent")
    if "volatile" in names:
        modifiers.append("volatile")
    if "restrict" in names:
        modifiers.append("restrict")
    if kind == "access" and not passed_to_function:
        access = codegen.explicit_resource_access(node)
        if access in {"readonly", "writeonly"}:
            modifiers.append(access)
    return "".join(f"{modifier} " for modifier in modifiers)


def slang_resource_parameter_memory_prefix(codegen, mapped_type, param):
    """The memory modifiers of resource parameter ``param``: none, since Slang declares no coherent parameter.

    slangc rejects ``coherent``, ``globallycoherent`` and ``volatile`` on a parameter (E31201), and ``readonly``,
    ``writeonly`` and ``restrict`` on a buffer parameter (E31206). The last three only promise how the resource is
    used, so leaving them off keeps the parameter's behaviour; a coherent or volatile parameter is refused.
    """
    if param is None or _resource_kind(codegen.resource_base_type(mapped_type)) is None:
        return ""
    names = _declared_memory_qualifiers(param)
    resource = getattr(param, "name", None) or "<unnamed>"
    scope_refusal = _scope_refusal(names, f"resource parameter '{resource}'")
    if scope_refusal is not None:
        return codegen.refuse_lowering(scope_refusal)
    unrepresentable = _coherence_and_volatility(names)
    if unrepresentable:
        return codegen.refuse_lowering(
            f"resource parameter '{resource}' is {unrepresentable}: slangc allows no coherent or volatile "
            "parameter (E31201), so its accesses would lose that behaviour"
        )
    return ""
