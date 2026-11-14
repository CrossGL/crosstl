"""Slang specialization constants imported as CrossGL ``@constant_id`` constants.

Slang declares a specialization constant as a global ``const`` with ``[vk::constant_id(N)]``, or with
``[SpecializationConstant]`` (also spelled ``[vk::specialization_constant]``) to let the compiler choose the id
(Slang user guide, "Specialization constants"). CrossGL writes both as ``const T name @constant_id(N) = value;``, so
every target that has specialization constants (SPIR-V ``SpecId``, GLSL ``layout(constant_id)``, Metal
``[[function_constant]]``, WGSL ``override``) keeps the pipeline's override.

slangc numbers the automatic ids itself: it first reserves every explicit ``vk::constant_id`` in the module, then
gives each automatic constant, in declaration order and whether or not the entry point uses it, the lowest id
that is still free. :func:`specialization_ids` reproduces that numbering so the CrossGL program carries the ids the
pipeline already uses.
"""

from __future__ import annotations

import re

from ..common_ast import AssignmentNode
from ..import_refusals import SlangImportRefused

__all__ = [
    "specialization_attribute",
    "specialization_ids",
    "specialized_workgroup_sizes",
]

_EXPLICIT = "vk::constant_id"
_AUTOMATIC = frozenset({"specializationconstant", "vk::specialization_constant"})
# SpecId is a 32-bit literal; GLSL and CrossGL write it as a signed int literal.
_MAX_SPECIALIZATION_ID = 2**31 - 1


def _declarations(node):
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _declarations(item)
        return
    yield node.left if isinstance(node, AssignmentNode) else node


def specialization_attribute(node):
    """``("explicit", argument)`` or ``("automatic", None)`` for a specialization constant, else ``None``."""
    found = None
    for attribute in getattr(node, "attributes", None) or []:
        name = str(attribute.get("name", "")).lower()
        if name == _EXPLICIT:
            kind = ("explicit", tuple(attribute.get("arguments", None) or ()))
        elif name in _AUTOMATIC:
            kind = ("automatic", None)
        else:
            continue
        if found is not None:
            raise SlangImportRefused(
                [
                    f"specialization constant {node.name} has more than one "
                    "vk::constant_id or SpecializationConstant attribute"
                ]
            )
        found = kind
    return found


def _explicit_id(node, arguments):
    text = str(arguments[0]).strip() if len(arguments) == 1 else ""
    if not text.isdigit() or int(text) > _MAX_SPECIALIZATION_ID:
        raise SlangImportRefused(
            [
                f"specialization constant {node.name} has vk::constant_id"
                f"({', '.join(map(str, arguments))}); CrossGL takes one integer id "
                f"from 0 to {_MAX_SPECIALIZATION_ID}"
            ]
        )
    return int(text)


def specialization_ids(global_nodes, exported=()):
    """Map ``id(declaration)`` to ``(declaration, SpecId)`` for every specialization constant, in source order.

    ``global_nodes`` are the module's global declarations in source order: variables, initialized assignments or
    lists of them, as the Slang parser returns multi-declarator statements. ``exported`` are the ``export``
    declarations, which the parser keeps apart, so their source position among the globals is unknown.
    """
    constants = []
    for node in _declarations(list(global_nodes)):
        kind = specialization_attribute(node)
        if kind is not None:
            constants.append((node, kind))
    for node in _declarations(list(exported)):
        kind = specialization_attribute(node)
        if kind is None:
            continue
        if kind[0] == "automatic":
            raise SlangImportRefused(
                [
                    f"exported specialization constant {node.name} takes its id from "
                    "its position among the module's declarations; give it vk::constant_id"
                ]
            )
        constants.append((node, kind))

    for node, _kind in constants:
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        if "const" not in qualifiers or "static" in qualifiers:
            raise SlangImportRefused(
                [
                    f"specialization constant {node.name} is not a non-static const "
                    "global, which Slang requires of a specialization constant"
                ]
            )

    reserved = set()
    for node, (form, arguments) in constants:
        if form == "explicit":
            reserved.add(_explicit_id(node, arguments))

    ids = {}
    next_free = 0
    for node, (form, arguments) in constants:
        if form == "explicit":
            ids[id(node)] = (node, _explicit_id(node, arguments))
            continue
        while next_free in reserved:
            next_free += 1
        ids[id(node)] = (node, next_free)
        reserved.add(next_free)
    return ids


def specialized_workgroup_sizes(numthreads, specialized_names):
    """Refuse a ``[numthreads]`` dimension that names a specialization constant or a global derived from one.

    Slang sizes the SPIR-V workgroup from the constant (``LocalSizeId``), so the pipeline's value sets the size.
    CrossGL's ``layout(local_size_x = ...)`` takes a constant expression fixed at translation time, which would keep
    only the default.
    """
    if not specialized_names:
        return
    for axis, argument in zip("xyz", numthreads):
        names = sorted(
            set(re.findall(r"[A-Za-z_]\w*", str(argument))) & set(specialized_names)
        )
        if names:
            raise SlangImportRefused(
                [
                    f"numthreads {axis} dimension {argument} depends on specialization "
                    f"constant {', '.join(names)}; CrossGL fixes the workgroup size at "
                    "translation time"
                ]
            )
