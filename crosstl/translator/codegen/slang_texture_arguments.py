"""Slang texture parameters and the arguments passed to them declare the same kind of texture.

Slang declares a CrossGL texture sampled with a separate sampler as a separate ``Texture2D<T>`` and one sampled
on its own as a combined ``Sampler2D<T>``. The generator decides by the texture's name, from the sampling calls
that name it. A texture that is only passed to a function, which samples its parameter with a separate sampler,
was declared ``Sampler2D<T>`` while the parameter was ``Texture2D<T>``, and slangc rejects the call ("expected an
expression of type 'Texture2D<vector<float,4>>', got 'Sampler2D<vector<float,4>>'").

:func:`link_texture_arguments` joins each argument of a call of a program function with the parameter it binds,
so every name of one texture takes the same kind: separate when one of them is sampled with a separate sampler and
none of them on its own.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, FunctionNode

__all__ = ["link_texture_arguments"]


def _callee_name(call):
    callee = getattr(call, "function", None)
    if callee is None:
        callee = getattr(call, "name", None)
    if isinstance(callee, str):
        return callee
    return getattr(callee, "name", None)


def link_texture_arguments(nodes, root_name, is_texture_name, explicit, implicit):
    """Return the explicit and implicit texture name sets closed over the program's calls.

    ``nodes`` is every node of the program, ``root_name`` names the identifier an argument expression reads, and
    ``is_texture_name`` says whether a name is declared with a texture type. Names a call joins share the kind of
    sampling any of them sees.
    """
    nodes = list(nodes)
    parameters = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            names = [
                getattr(parameter, "name", None)
                for parameter in getattr(node, "parameters", None) or []
            ]
            parameters.setdefault(node.name, []).append(names)

    parent = {}

    def find(name):
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for node in nodes:
        if not isinstance(node, FunctionCallNode):
            continue
        overloads = parameters.get(_callee_name(node))
        if not overloads:
            continue
        arguments = getattr(node, "args", None) or []
        for names in overloads:
            if len(names) != len(arguments):
                continue
            for argument, parameter in zip(arguments, names):
                argument_name = root_name(argument)
                if (
                    argument_name
                    and parameter
                    and is_texture_name(argument_name)
                    and is_texture_name(parameter)
                ):
                    parent[find(argument_name)] = find(parameter)

    if not parent:
        return set(explicit), set(implicit)
    explicit_roots = {find(name) for name in explicit}
    implicit_roots = {find(name) for name in implicit}
    linked_explicit = set(explicit) | {
        name for name in parent if find(name) in explicit_roots
    }
    linked_implicit = set(implicit) | {
        name for name in parent if find(name) in implicit_roots
    }
    return linked_explicit, linked_implicit
