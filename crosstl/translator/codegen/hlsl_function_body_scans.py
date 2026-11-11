"""One body scan per function and render for the DirectX generator's whole-body name checks.

Two HLSL generator checks scan whole function bodies:

- ``DispatchMesh`` validation follows the call names of an entry point and of every helper it reaches, and collects
  their ``DispatchMesh`` argument lists;
- helper-name reservation collects the parameter and declared variable names of every function, once for each
  helper family that needs a fresh name, and once more for each function it renders.

A unit with thousands of entry points that share helpers rescanned each helper once per entry point, and each
function once per reservation. :class:`HlslFunctionBodyScans` scans a function the first time one of these checks
asks for it and keeps the result for the rest of the render. No pass rewrites a function body between the start of
a render attempt and its output, and the generator creates a new instance at the start of each attempt. A result is
reused only for the same function object while it still holds the same body and parameter list, so a function whose
body or parameters are replaced is scanned again.

A scan walks the body once with the generator's own walk and records what the two checks read from it, in the
walk's order: call names (as ``function_call_name`` spells them), ``DispatchMesh`` argument lists (from calls and
mesh operations), and declared names (parameters, ``VariableNode`` declarations and ``for``-in patterns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..ast import ForInNode, FunctionCallNode, MeshOpNode, VariableNode

__all__ = ["HlslFunctionBodyScan", "HlslFunctionBodyScans", "scan_function_body"]


@dataclass(frozen=True)
class HlslFunctionBodyScan:
    """What one walk of a function body found, in walk order."""

    call_names: tuple
    dispatch_mesh_args: tuple
    identifier_names: frozenset


def _parameters(func):
    return getattr(func, "parameters", getattr(func, "params", []))


def scan_function_body(
    func, walk: Callable[[Any], Any], call_name: Callable[[Any], Any]
):
    """Scan ``func``'s body once with ``walk``; ``call_name`` spells a call's callee."""
    names, dispatch_mesh_args = [], []
    identifiers = {
        getattr(param, "name", None)
        for param in _parameters(func) or []
        if getattr(param, "name", None)
    }
    for node in walk(getattr(func, "body", [])):
        if isinstance(node, FunctionCallNode):
            name = call_name(node)
            if name:
                names.append(name)
            if name == "DispatchMesh":
                dispatch_mesh_args.append(
                    getattr(node, "arguments", getattr(node, "args", []))
                )
        elif isinstance(node, MeshOpNode):
            if getattr(node, "operation", None) == "DispatchMesh":
                dispatch_mesh_args.append(getattr(node, "arguments", []))
        elif isinstance(node, VariableNode):
            name = getattr(node, "name", None)
            if name:
                identifiers.add(name)
        elif isinstance(node, ForInNode):
            pattern = getattr(node, "pattern", None)
            if pattern:
                identifiers.add(pattern)
    return HlslFunctionBodyScan(
        tuple(names), tuple(dispatch_mesh_args), frozenset(identifiers)
    )


class HlslFunctionBodyScans:
    """The body scans of one render, keyed by function object, body and parameter list."""

    def __init__(self, generator):
        self._generator = generator
        self._scans = {}

    def scan(self, func) -> HlslFunctionBodyScan:
        body = getattr(func, "body", None)
        parameters = _parameters(func)
        entry = self._scans.get(id(func))
        if (
            entry is not None
            and entry[0] is func
            and entry[1] is body
            and entry[2] is parameters
        ):
            return entry[3]
        generator = self._generator
        result = scan_function_body(
            func, generator.walk_ast, generator.function_call_name
        )
        # The entry keeps ``func`` alive, so its id cannot be reused by another object during the render.
        self._scans[id(func)] = (func, body, parameters, result)
        return result
