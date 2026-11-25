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
    "specialization_constant_defaults",
    "specialized_workgroup_layout",
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


def specialization_constant_defaults(global_nodes, ids, exported=()):
    """``{name: (SpecId, initializer)}`` of each specialization constant of ``ids`` that has an initializer.

    ``ids`` is :func:`specialization_ids` of the same declarations.
    """
    defaults = {}
    for node in _assignments(list(global_nodes) + list(exported)):
        entry = ids.get(id(node.left))
        if entry is not None and entry[0] is node.left:
            defaults[node.left.name] = (entry[1], node.right)
    return defaults


def _assignments(node):
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _assignments(item)
    elif isinstance(node, AssignmentNode):
        yield node


def specialized_workgroup_layout(numthreads, specialized_names, constants):
    """The ``local_size_*`` and ``local_size_*_id`` entries of CrossGL's ``layout(...) in;`` for ``[numthreads]``.

    A dimension that names a specialization constant, ``[numthreads(size, 1, 1)]`` with ``[vk::constant_id(0)] const
    int size = 64;``, is ``local_size_x_id = 0`` with the default ``local_size_x = 64``: the pipeline's value sets the
    size, as slangc's ``LocalSizeId`` does. ``constants`` maps such a constant's name to its id and its default, the
    integer literal it is initialized with. A dimension computed from a specialization constant (or from a global
    that follows one), or a constant whose default is not an integer literal, is refused: CrossGL fixes every other
    workgroup size at translation time.
    """
    sizes, ids = [], []
    for axis, argument in zip("xyz", numthreads):
        text = str(argument).strip()
        if text in constants:
            spec_id, default = constants[text]
            literal = re.fullmatch(r"\+?([0-9]+)[uU]?", str(default).strip())
            if literal is not None:
                sizes.append(f"local_size_{axis} = {int(literal.group(1))}")
                ids.append(f"local_size_{axis}_id = {spec_id}")
                continue
        names = sorted(
            set(re.findall(r"[A-Za-z_]\w*", text)) & set(specialized_names or ())
        )
        if names:
            raise SlangImportRefused(
                [
                    f"numthreads {axis} dimension {argument} depends on specialization "
                    f"constant {', '.join(names)}; CrossGL specializes a workgroup dimension that "
                    "names an integer specialization constant with a literal default, and fixes any "
                    "other at translation time"
                ]
            )
        sizes.append(f"local_size_{axis} = {argument}")
    return ", ".join(sizes + ids)
