"""Generated code that contains the printed form of an AST node, refused instead of returned.

An emitter that meets a node it has no rule for sometimes falls back to ``str(node)``, which is the node's Python
``repr``: ``ConstructorNode(constructor_type=NamedType(name=vec<T, 2>, generic_args=[]), arguments=2)`` or
``<crosstl.translator.ast.SwitchNode object at 0x...>``. No target compiler accepts that text, and a caller that does
not compile the output sees a translation that seems to have succeeded. :func:`refuse_ast_text` raises
:class:`GeneratedAstTextError`, naming the node, for any such text in the code generated for a C-family target.

The check looks for the class name of an AST node of :mod:`crosstl.translator.ast` or :mod:`crosstl.backend.common_ast`
followed by ``(`` and a keyword argument, the shape every node ``repr`` has, or for Python's default object form.
SPIR-V assembly is not checked, since its ``OpSource`` can carry the source program's own text, and neither is
Mojo or CrossGL, where ``Name(field=value)`` is ordinary syntax.
"""

from __future__ import annotations

import inspect
import re
from functools import lru_cache

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["CHECKED_TARGETS", "GeneratedAstTextError", "refuse_ast_text"]

CHECKED_TARGETS = frozenset(
    {"cuda", "directx", "hip", "metal", "opengl", "rust", "slang", "webgl", "wgsl"}
)

_DEFAULT_REPR = re.compile(r"<crosstl\.[\w.]+ object at 0x[0-9a-fA-F]+>")


class GeneratedAstTextError(TranslationDiagnostic):
    """The emitter printed an AST node it could not lower instead of target code."""

    def __init__(self, target, node_text):
        self.node_text = node_text
        self.missing_capabilities = (f"{target}.lowering.{_node_name(node_text)}",)
        super().__init__(
            f"the {target} emitter has no lowering for an AST node and printed it instead of {target} code: "
            f"{node_text}",
            target=target,
            code=f"{CODE_PREFIX}{target}-ast-node-unlowered",
        )


def _node_name(node_text):
    match = re.match(r"<?([\w.]+)", node_text)
    return match.group(1).rsplit(".", 1)[-1] if match else "node"


@lru_cache(maxsize=1)
def _node_repr_pattern():
    from ...backend import common_ast
    from .. import ast as translator_ast

    names = sorted(
        {
            name
            for module in (translator_ast, common_ast)
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and value.__module__ == module.__name__
            and (name.endswith("Node") or name.endswith("Type"))
        },
        key=lambda name: (-len(name), name),
    )
    return re.compile(
        r"\b(?:" + "|".join(map(re.escape, names)) + r")\(\s*[a-z_]\w*\s*="
    )


def refuse_ast_text(code, target):
    """Raise :class:`GeneratedAstTextError` when ``code``, generated for ``target``, prints an AST node."""
    target = str(target or "").lower()
    if target not in CHECKED_TARGETS or not isinstance(code, str):
        return
    match = _DEFAULT_REPR.search(code) or _node_repr_pattern().search(code)
    if match is None:
        return
    line_end = code.find("\n", match.start())
    snippet = code[match.start() : line_end if line_end != -1 else len(code)]
    raise GeneratedAstTextError(target, snippet.strip()[:200])
