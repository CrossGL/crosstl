"""Device-memory references to structures, bound as entry-point buffers.

MSL binds a buffer to an entry-point argument declared as a reference, ``device T& p [[buffer(n)]]`` or ``const
device T& p [[buffer(n)]]``. The argument is the single ``T`` at the start of the buffer, which the shader reads and,
without ``const``, writes (MSL 4.1 section 4.1, "device Address Space", and section 5.2.1). The CrossGL importer
keeps the reference. The generators lower a reference to a scalar or vector into element 0 of a buffer, but they took
a reference to a structure for a stage input. HLSL declared ``const Params p`` with ``TEXCOORD`` semantics and
``inout Params q``, GLSL declared the members as vertex inputs, and SPIR-V declared them as ``Input`` variables. The
shader then read vertex attributes, or nothing, instead of the buffer.

:func:`lower_device_struct_references` rewrites each such parameter of a stage entry into the structured buffer that a
pointer argument ``device T* p`` imports as (``RWStructuredBuffer<T>``, or ``StructuredBuffer<T>`` for a ``const``
reference), and every use of it in the entry into the buffer's element 0. Every generator lowers element accesses
of structured buffers, writes included. Metal output keeps the reference. Only structures of plain data qualify: a
structure with resource members (``texture2d``, ``sampler``, pointers) or ``[[id(n)]]`` members is an argument buffer
(MSL 4.1 section 2.13), which the generators flatten or refuse themselves. An entry that also declares a local of the
parameter's name is refused with :class:`DeviceReferenceUnsupportedError`, because a use could then name either.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import type_from_node
from .rewrite import child_slots, get_child, set_child
from .storage_blocks import _walk

__all__ = ["DeviceReferenceUnsupportedError", "lower_device_struct_references"]

_KEEPING = frozenset({"metal", "cgl", "crossgl", ""})
_BUFFER_ATTRIBUTES = frozenset({"buffer", "binding"})


class DeviceReferenceUnsupportedError(TranslationDiagnostic):
    """A device reference argument that the pass cannot rewrite into a buffer element unambiguously."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        super().__init__(
            f"the device reference argument '{name}' is also the name of a local in its entry point, so its uses "
            "cannot be rewritten into the buffer's element",
            target=target,
            code=f"{CODE_PREFIX}{target}-device-reference-unsupported",
        )


def _structs(program):
    return {
        node.name: node
        for node in _walk(program)
        if isinstance(node, A.StructNode) and isinstance(node.name, str)
    }


def _plain_type(type_node, structs, seen):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, (A.VectorType, A.MatrixType)):
        return True
    if not isinstance(type_node, (A.PrimitiveType, A.NamedType)):
        return False
    if getattr(type_node, "generic_args", None):
        return False
    if type_from_node(type_node) is not None or re.fullmatch(
        r"(?:d|f16)?mat[234](?:x[234])?", str(type_node.name)
    ):
        return True
    return _plain_struct(type_node.name, structs, seen)


def _plain_struct(name, structs, seen=None):
    """Whether structure ``name`` holds only data: no resources, pointers or argument-buffer ``[[id(n)]]`` members."""
    struct = structs.get(name)
    seen = set() if seen is None else seen
    if struct is None or name in seen:
        return False
    seen.add(name)
    for member in struct.members or []:
        if not isinstance(member, A.StructMemberNode):
            return False
        attributes = {
            str(getattr(attribute, "name", attribute)).lower()
            for attribute in getattr(member, "attributes", None) or []
        }
        if "id" in attributes or not _plain_type(member.member_type, structs, seen):
            return False
    return True


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def _device_struct_reference(parameter, structs):
    declared = getattr(parameter, "param_type", None)
    if not isinstance(declared, A.ReferenceType):
        return None
    if str(getattr(declared, "address_space", "") or "").lower() != "device":
        return None
    element = declared.referenced_type
    if not isinstance(element, A.NamedType) or element.generic_args:
        return None
    if not _plain_struct(element.name, structs):
        return None
    attributes = {
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(parameter, "attributes", None) or []
    }
    return declared if attributes & _BUFFER_ATTRIBUTES else None


def _element_uses(body, name):
    """Rewrite each ``name`` under ``body`` into ``name[0]``; callee names are left alone."""

    def visit(node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if (
                isinstance(child, A.IdentifierNode)
                and child.name == name
                and not (isinstance(node, A.FunctionCallNode) and field == "function")
            ):
                element = A.ArrayAccessNode(
                    A.IdentifierNode(name), A.LiteralNode(0, A.PrimitiveType("uint"))
                )
                element.source_location = child.source_location
                set_child(node, field, key, element)
            elif isinstance(child, A.ASTNode):
                visit(child)

    visit(body)


def lower_device_struct_references(program, target) -> int:
    """Bind ``device T&`` structure arguments of stage entries as structured buffers; returns how many."""
    target = str(target or "").lower()
    if target in _KEEPING or not isinstance(program, A.ShaderNode):
        return 0
    structs = None
    count = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if not isinstance(entry, A.FunctionNode) or entry.body is None:
            continue
        for parameter in entry.parameters or []:
            if structs is None:
                structs = _structs(program)
            reference = _device_struct_reference(parameter, structs)
            if reference is None:
                continue
            locals_ = {
                node.name
                for node in _walk(entry.body)
                if isinstance(node, A.VariableNode)
            }
            if parameter.name in locals_:
                raise DeviceReferenceUnsupportedError(
                    target,
                    parameter.name,
                    source_location=getattr(parameter, "source_location", None),
                )
            qualifiers = [str(q) for q in getattr(parameter, "qualifiers", None) or []]
            read_only = not reference.is_mutable or "const" in qualifiers
            buffer = "StructuredBuffer" if read_only else "RWStructuredBuffer"
            parameter.param_type = A.NamedType(buffer, [reference.referenced_type])
            parameter.qualifiers = [
                q for q in qualifiers if q not in ("const", "device")
            ]
            _element_uses(entry.body, parameter.name)
            count += 1
    return count
