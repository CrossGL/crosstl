"""Calls to Metal functions that the translation unit declares but does not define.

MSL compiles each source file separately and links the functions of a library, so a file may call a function it only
declares (``half3 tonemap(half3 color);``) and that another file defines. The Metal parser drops such a prototype,
and the importer used to emit the call as is: every target then called an undeclared function. CrossGL has no
external function declaration, so a call to a function that is declared but defined nowhere in the unit is refused
with :class:`MetalExternalFunctionError`; translating the files that define it together resolves it.
"""

from __future__ import annotations

__all__ = ["MetalExternalFunctionError", "refuse_external_function_calls"]


class MetalExternalFunctionError(ValueError):
    """A call to a function the Metal translation unit declares without defining it."""

    project_diagnostic_code = "project.translate.metal-external-function"
    missing_capabilities = ("crossgl.external-function-declarations",)

    def __init__(self, name):
        self.function_name = name
        super().__init__(
            f"Metal function '{name}' is declared but not defined in this translation unit; CrossGL has no external "
            "function declaration, so translate it together with the source that defines it"
        )


def _short(name):
    return str(name).lstrip(":").rsplit("::", 1)[-1].split("<", 1)[0]


def _nodes(root):
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if not hasattr(node, "__dict__") or isinstance(node, type):
            continue
        yield node
        for field, value in vars(node).items():
            if field in ("parent", "source_location"):
                continue
            if isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                stack.append(value)


def refuse_external_function_calls(ast):
    """Raise :class:`MetalExternalFunctionError` for the first call to a declared function no definition provides.

    ``ast.function_prototypes`` holds the names of the prototypes the parser dropped. A name counts as defined when
    any function of the unit, in any namespace or as a template, has the same unqualified name.
    """
    prototypes = getattr(ast, "function_prototypes", None) or {}
    if not prototypes:
        return
    defined = {
        _short(function.name)
        for function in getattr(ast, "functions", None) or []
        if getattr(function, "name", None)
    }
    external = {_short(name) for name in prototypes} - defined
    if not external:
        return
    for node in _nodes(getattr(ast, "functions", None) or []):
        if type(node).__name__ != "FunctionCallNode":
            continue
        name = getattr(node, "name", None)
        if isinstance(name, str) and _short(name) in external:
            raise MetalExternalFunctionError(_short(name))
