"""Default arguments of generic structures, spelled out at every use.

``generic<T, U = vec2> struct Pair`` used as ``Pair<vec2>`` means ``Pair<vec2, vec2>``, and ``generic<T = float>
struct Cell`` used as ``Cell<>`` or ``Cell`` means ``Cell<float>``. The generators specialize a generic structure for
the arguments a use spells, so a use that left a default out matched no specialization: GLSL, SPIR-V and WGSL refused
it, and Metal and HLSL emitted ``Pair<float2>`` verbatim, naming a template that the output never declares.

:func:`fill_generic_struct_defaults` completes every such use, in types and in constructor calls, before the
generators read the program. A default may name an earlier parameter (``generic<T, U = T>``). A use that leaves out a
parameter without a default, or gives too many arguments, raises :class:`GenericStructArityError`: Metal, HLSL and
GLSL emitted ``Pair<float>`` verbatim for it.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_structs

__all__ = ["GenericStructArityError", "fill_generic_struct_defaults"]


class GenericStructArityError(TranslationDiagnostic):
    """A use of a generic structure with more arguments than parameters, or without one that has no default."""

    def __init__(self, target, name, expected, given):
        self.struct_name = name
        super().__init__(
            f"generic structure '{name}' takes {expected} argument{'s' if expected != 1 else ''}, "
            f"but a use gives {given}",
            target=target,
            code=f"{CODE_PREFIX}generic-struct-arity",
        )


_SKIPPED = frozenset({"parent", "source_location", "annotations"})


def _definitions(program):
    definitions = {}
    for struct in iter_structs(program):
        parameters = getattr(struct, "generic_params", None) or []
        if isinstance(struct, A.StructNode) and parameters:
            definitions[struct.name] = parameters
    return definitions


def _substitute(type_node, bound):
    """``type_node`` with each named parameter in ``bound`` replaced by its argument."""
    if (
        isinstance(type_node, (A.NamedType, A.PrimitiveType))
        and not getattr(type_node, "generic_args", None)
        and type_node.name in bound
    ):
        return deepcopy(bound[type_node.name])
    if hasattr(type_node, "__dict__"):
        for field, value in list(vars(type_node).items()):
            if field in _SKIPPED:
                continue
            if isinstance(value, list):
                setattr(type_node, field, [_substitute(item, bound) for item in value])
            elif isinstance(value, A.ASTNode):
                setattr(type_node, field, _substitute(value, bound))
    return type_node


def _completed(parameters, arguments):
    """``arguments`` completed with defaults, or ``None`` when a missing parameter has none."""
    arguments = list(arguments)
    for parameter in parameters[len(arguments) :]:
        default = getattr(parameter, "default_type", None)
        if default is None:
            return None
        bound = dict(zip((item.name for item in parameters), arguments))
        arguments.append(_substitute(deepcopy(default), bound))
    return arguments


def _spelling(type_node, bound):
    """The CrossGL spelling of a default argument for a constructor's callee, or ``None``.

    A bare parameter name in ``bound`` is spelled as that parameter's argument, also inside a default such as
    ``vec2<T>`` or ``Box<T>``.
    """
    if (
        isinstance(type_node, (A.NamedType, A.PrimitiveType))
        and not getattr(type_node, "generic_args", None)
        and type_node.name in bound
    ):
        return bound[type_node.name]
    if isinstance(type_node, A.PrimitiveType):
        return type_node.name
    if isinstance(type_node, A.NamedType):
        arguments = [_spelling(item, bound) for item in type_node.generic_args or []]
        if any(item is None for item in arguments):
            return None
        return (
            f"{type_node.name}<{', '.join(arguments)}>" if arguments else type_node.name
        )
    if isinstance(type_node, A.VectorType):
        element = _spelling(type_node.element_type, bound)
        if element is None:
            return None
        return (
            f"vec{type_node.size}"
            if element == "float"
            else f"vec{type_node.size}<{element}>"
        )
    if (
        isinstance(type_node, A.MatrixType)
        and _spelling(type_node.element_type, bound) == "float"
    ):
        rows, cols = type_node.rows, type_node.cols
        return f"mat{cols}" if rows == cols else f"mat{cols}x{rows}"
    return None


def _split(text):
    """``("Pair", ["vec2"])`` for ``Pair<vec2>``, else ``None``."""
    text = text.strip()
    if "<" not in text or not text.endswith(">"):
        return None
    base, inner = text.split("<", 1)
    arguments, depth, current = [], 0, ""
    for character in inner[:-1]:
        if character in "<([":
            depth += 1
        elif character in ">)]":
            depth -= 1
        if character == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        arguments.append(current.strip())
    return base.strip(), arguments


class _Filler:
    def __init__(self, definitions, target):
        self.definitions = definitions
        self.target = target
        self.filled = 0
        self.seen = set()

    def _refuse(self, name, given):
        raise GenericStructArityError(
            self.target, name, len(self.definitions[name]), given
        )

    def visit(self, value):
        """Complete the uses under ``value`` in place; a stage map is walked without being rebuilt."""
        if isinstance(value, (list, tuple)):
            for item in value:
                self.visit(item)
            return
        if isinstance(value, dict):
            for item in list(value.values()):
                self.visit(item)
            return
        if not isinstance(value, A.ASTNode) or id(value) in self.seen:
            return
        self.seen.add(id(value))
        for field, item in list(vars(value).items()):
            if field not in _SKIPPED and isinstance(
                item, (list, tuple, dict, A.ASTNode)
            ):
                self.visit(item)
        if isinstance(value, A.NamedType) and value.name in self.definitions:
            parameters = self.definitions[value.name]
            arguments = list(value.generic_args or [])
            if len(arguments) > len(parameters):
                self._refuse(value.name, len(arguments))
            if len(arguments) < len(parameters):
                completed = _completed(parameters, arguments)
                if completed is None:
                    # A bare name may be the definition's own (``Pair`` in a generic body); arguments must fit.
                    if arguments:
                        self._refuse(value.name, len(arguments))
                else:
                    value.generic_args = completed
                    self.filled += 1
        if isinstance(value, A.FunctionCallNode):
            self._fill_callee(value)

    def _fill_callee(self, call):
        callee = getattr(call, "function", None)
        if not isinstance(callee, A.IdentifierNode) or not isinstance(callee.name, str):
            return
        parts = _split(callee.name)
        if parts is None or parts[0] not in self.definitions:
            return
        base, arguments = parts
        parameters = self.definitions[base]
        if len(arguments) > len(parameters):
            self._refuse(base, len(arguments))
        if len(arguments) == len(parameters):
            return
        spelled = list(arguments)
        for parameter in parameters[len(arguments) :]:
            default = getattr(parameter, "default_type", None)
            if default is None:
                self._refuse(base, len(arguments))
            bound = dict(zip((item.name for item in parameters), spelled))
            text = _spelling(default, bound)
            if text is None:
                return
            spelled.append(text)
        name = f"{base}<{', '.join(spelled)}>"
        callee.name = name
        if getattr(callee, "identifier", None) is not None:
            callee.identifier = name
        self.filled += 1


def fill_generic_struct_defaults(program, target=None) -> int:
    """Spell out the default arguments of every generic structure use in ``program``; returns how many."""
    definitions = _definitions(program)
    if not definitions:
        return 0
    filler = _Filler(definitions, target)
    filler.visit(program)
    return filler.filled
