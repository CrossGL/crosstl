"""Coherence of Metal storage textures.

MSL 3.2 states the coherence of a texture as its third template argument,
``texture2d<T, access::A, memory_coherence::memory_coherence_device>``. The default is
``memory_coherence_threadgroup`` and there is no system scope. The ``coherent`` qualifier that MSL buffers take is
rejected on texture parameters, and a ``volatile`` texture has no ``read`` or ``write`` members.

The coherence argument is part of the texture type, so a helper parameter must carry the coherence of every
texture passed to it. Device coherence only narrows the set of values a shader may observe, so a group of textures
and helper parameters that calls connect becomes device coherent as soon as one of its members asks for it.
"""

from __future__ import annotations

import re

from ..ast import ArrayAccessNode, FunctionCallNode, IdentifierNode

DEVICE_TEXTURE_COHERENCE = "memory_coherence::memory_coherence_device"

STORAGE_TEXTURE_TYPE = re.compile(
    r"[iu]?image(?:1D|1DArray|2D|2DArray|2DMS|2DMSArray|3D|Cube|CubeArray|Buffer)"
)

# MSL texture kinds whose default access is ``access::read`` rather than ``access::sample``.
_READ_DEFAULT_TEXTURES = {"texture_buffer", "texture2d_ms", "texture2d_ms_array"}

_TEXTURE_TEMPLATE = re.compile(r"\b(texture\w*)<")


def is_storage_texture_type_name(type_name):
    return STORAGE_TEXTURE_TYPE.fullmatch(str(type_name or "").strip()) is not None


def texture_requires_device_coherence(name, contracts, refuse):
    """Return whether ``(kind, scope)`` contracts ask for device coherence.

    Contracts that an MSL texture cannot state are passed to ``refuse``.
    """
    device = False
    for kind, scope in contracts:
        if kind == "volatile":
            refuse(
                f"volatile storage texture '{name}': MSL has no volatile texture types "
                "(a volatile texture has no read or write members)"
            )
        elif kind != "coherent" or scope == "threadgroup":
            continue
        elif scope in {None, "device"}:
            device = True
        elif scope == "system":
            refuse(
                f"storage texture '{name}' is coherent at system scope: MSL texture "
                "coherence ends at memory_coherence_device"
            )
        else:
            refuse(f"storage texture '{name}' has unknown coherence scope '{scope}'")
    return device


def with_device_texture_coherence(type_text):
    """Add ``memory_coherence_device`` to the first MSL texture type in ``type_text``."""
    match = _TEXTURE_TEMPLATE.search(type_text)
    if match is None:
        return type_text
    start = match.end() - 1
    depth = 0
    for index in range(start, len(type_text)):
        character = type_text[index]
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
            if depth == 0:
                arguments = type_text[start + 1 : index]
                if DEVICE_TEXTURE_COHERENCE in arguments:
                    return type_text
                if "access::" not in arguments:
                    default = (
                        "read" if match.group(1) in _READ_DEFAULT_TEXTURES else "sample"
                    )
                    arguments = f"{arguments}, access::{default}"
                return (
                    f"{type_text[: start + 1]}{arguments}, "
                    f"{DEVICE_TEXTURE_COHERENCE}{type_text[index:]}"
                )
    return type_text


class _Groups:
    """Union-find over texture keys, kept in insertion order."""

    def __init__(self):
        self.parent = {}

    def add(self, key):
        self.parent.setdefault(key, key)

    def find(self, key):
        self.add(key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, first, second):
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _texture_root_name(expression):
    while isinstance(expression, ArrayAccessNode):
        expression = expression.array_expr
    if isinstance(expression, IdentifierNode):
        return expression.name
    return None


def collect_device_coherent_textures(
    global_textures,
    functions,
    *,
    is_storage_texture,
    contracts_of,
    call_name,
    iter_nodes,
    refuse,
):
    """Return the texture keys whose MSL declarations need ``memory_coherence_device``.

    A key is ``(None, name)`` for a program-scope texture and ``(function name, parameter name)`` for a helper
    parameter. ``is_storage_texture(node)`` and ``contracts_of(node)`` classify a global or parameter node,
    ``call_name(call)`` names a call's callee and ``iter_nodes(body)`` walks a function body.
    """
    groups = _Groups()
    device = set()
    global_names = set()
    for node in global_textures:
        name = getattr(node, "name", None)
        if not name or not is_storage_texture(node):
            continue
        global_names.add(name)
        groups.add((None, name))
        if texture_requires_device_coherence(name, contracts_of(node), refuse):
            device.add((None, name))

    texture_parameters = {}
    overloads = {}
    for function in functions:
        function_name = getattr(function, "name", None)
        if not function_name:
            continue
        overloads.setdefault(function_name, []).append(function)
        names = set()
        for parameter in getattr(function, "parameters", None) or []:
            if not is_storage_texture(parameter):
                continue
            key = (function_name, parameter.name)
            names.add(parameter.name)
            groups.add(key)
            if texture_requires_device_coherence(
                parameter.name, contracts_of(parameter), refuse
            ):
                device.add(key)
        texture_parameters[id(function)] = names

    for function in functions:
        body = getattr(function, "body", None)
        if body is None:
            continue
        local_parameters = texture_parameters.get(id(function), set())
        for node in iter_nodes(body):
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = getattr(node, "arguments", None) or []
            for callee in overloads.get(call_name(node), []):
                callee_parameters = getattr(callee, "parameters", None) or []
                callee_textures = texture_parameters.get(id(callee), set())
                for argument, parameter in zip(arguments, callee_parameters):
                    if parameter.name not in callee_textures:
                        continue
                    source = _texture_root_name(argument)
                    if source in local_parameters:
                        source_key = (function.name, source)
                    elif source in global_names:
                        source_key = (None, source)
                    else:
                        continue
                    groups.union(source_key, (callee.name, parameter.name))

    device_roots = {groups.find(key) for key in device}
    return frozenset(key for key in groups.parent if groups.find(key) in device_roots)
