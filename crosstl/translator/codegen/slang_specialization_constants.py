"""Slang declarations for CrossGL specialization constants.

Slang declares a specialization constant as ``[vk::constant_id(N)] const T name = default;``, which slangc compiles to
an ``OpSpecConstant`` decorated ``SpecId N`` (Slang User's Guide, "SPIR-V specific functionalities: specialization
constants"). The generator used to print every CrossGL constant as ``static const``, which froze the default and
dropped the id.

A constant whose initializer reads a specialization constant follows its specialized value. slangc folds an integer
or Boolean one built from operators and integer or Boolean conversions into ``OpSpecConstantOp`` (the operations the
SPIR-V ``Shader`` capability allows there), so it stays ``static const`` and may still size an array. Anything else,
such as ``float(scale)``, a vector or a built-in call, is not a compile-time constant to slangc ("static const global
initializer must be a compile-time constant"), so it becomes a ``static`` global with the same initializer, which
slangc evaluates when the entry point starts. Such a constant cannot size an array, and no constant that follows a
specialization constant can label a ``switch`` case, so both are refused.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from ..ast import (
    ArrayType,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    TernaryOpNode,
    TypeNode,
    UnaryOpNode,
)
from .specialization_constants import (
    SpecializationConstantError,
    array_sizes_reading,
    case_labels_reading,
    specialization_constant_ids,
    specialization_dependent_constants,
    walk_nodes,
)

__all__ = ["SlangSpecializationPlan", "plan_slang_specialization_constants"]

_TARGET = "Slang"
# GLSL writes SpecId as a signed int literal, so ids up to 2^31 - 1 are portable.
_MAX_SLANG_ID = 2**31 - 1
_INTEGER_TYPES = frozenset(
    {
        "bool",
        "int",
        "uint",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int64_t",
        "uint64_t",
    }
)
_SCALAR_TYPES = _INTEGER_TYPES | {"float", "half", "double"}
# The CrossGL spellings of the integer and Boolean conversions slangc folds.
_INTEGER_CONVERSIONS = frozenset(
    {
        "bool",
        "int",
        "uint",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    }
    | _INTEGER_TYPES
)
_OPERATOR_NODES = (BinaryOpNode, UnaryOpNode, TernaryOpNode, LiteralNode, TypeNode)


class SlangSpecializationPlan:
    """How each specialization-dependent constant of one program is declared in Slang."""

    def __init__(
        self,
        ids: dict[str, int],
        folded: set[str],
        evaluated: set[str],
        defaults: dict[str, int] | None = None,
    ):
        self.ids = ids
        self.folded = folded
        self.evaluated = evaluated
        self.defaults = defaults or {}

    def interface_array_type(self, member_type):
        """``member_type`` sized by its default when it is a stage-interface array sized by a specialization constant.

        A stage-interface array spans the interface locations of its default size, as every target assigns them,
        and slangc cannot lay out an interface array whose size is a specialization constant. ``None`` when
        ``member_type`` needs no change.
        """
        if not isinstance(member_type, ArrayType):
            return None
        size = member_type.size
        name = size.name if isinstance(size, IdentifierNode) else size
        if not isinstance(name, str) or name not in self.folded:
            return None
        if name not in self.defaults:
            raise SpecializationConstantError(
                f"a stage-interface array is sized by '{name}', which follows a specialization constant; its "
                "interface spans the default size, and Slang needs that size as a literal",
                target=_TARGET,
                constant_name=name,
                specialization_id=self.ids.get(name),
                kind="interface-array-size",
            )
        resized = copy.copy(member_type)
        resized.size = LiteralNode(self.defaults[name], PrimitiveType("int"))
        return resized

    def declaration(self, name: str, declarator: str, value_code):
        """The Slang declaration of constant ``name``, or ``None`` when it is an ordinary ``static const``.

        ``declarator`` is the type and name as declared and ``value_code`` the rendered initializer, or ``None``.
        """
        initializer = "" if value_code is None else f" = {value_code}"
        if name in self.ids:
            return (
                f"[vk::constant_id({self.ids[name]})] const {declarator}{initializer};"
            )
        if name in self.evaluated:
            return f"static {declarator}{initializer};"
        return None


def _conversion_name(node) -> str | None:
    if isinstance(node, FunctionCallNode):
        function = node.function
        return getattr(
            function, "name", function if isinstance(function, str) else None
        )
    if isinstance(node, CastNode):
        return getattr(node.target_type, "name", None)
    return None


def _folds(value, folded: set[str], dependent: set[str]) -> bool:
    """Whether slangc folds ``value``, an integer or Boolean initializer, to specialization constant operations.

    It may hold literals, operators, integer or Boolean conversions and names; a name that follows a specialization
    constant must itself fold.
    """
    callees = set()
    for node in walk_nodes(value):
        if isinstance(node, (FunctionCallNode, CastNode)):
            if _conversion_name(node) not in _INTEGER_CONVERSIONS:
                return False
            callees.add(id(getattr(node, "function", None)))
        elif isinstance(node, IdentifierNode):
            if id(node) not in callees and node.name in dependent - folded:
                return False
        elif not isinstance(node, _OPERATOR_NODES):
            return False
    return True


def plan_slang_specialization_constants(
    ast, type_name_of: Callable[[object], str]
) -> SlangSpecializationPlan:
    """Plan the Slang declarations of ``ast``'s specialization constants and refuse what slangc cannot compile.

    ``type_name_of`` renders a CrossGL type as its Slang spelling.
    """
    constants = list(getattr(ast, "constants", None) or [])
    ids = specialization_constant_ids(constants, _TARGET, _MAX_SLANG_ID)
    if not ids:
        return SlangSpecializationPlan({}, set(), set())
    dependent = specialization_dependent_constants(
        constants, lambda node: getattr(node, "name", None) in ids
    )
    folded: set[str] = set()
    evaluated: set[str] = set()
    for node in constants:
        name = getattr(node, "name", None)
        if name not in dependent:
            continue
        type_name = type_name_of(node.const_type)
        if name in ids:
            if type_name not in _SCALAR_TYPES:
                raise SpecializationConstantError(
                    f"specialization constant '{name}' has type {type_name}; a SPIR-V specialization constant "
                    "is a Boolean, integer or floating-point scalar (SPIR-V, OpSpecConstant)",
                    target=_TARGET,
                    constant_name=name,
                    specialization_id=ids[name],
                    kind="type",
                )
            if type_name in _INTEGER_TYPES:
                folded.add(name)
            continue
        if type_name in _INTEGER_TYPES and _folds(
            getattr(node, "value", None), folded, dependent
        ):
            folded.add(name)
        else:
            evaluated.add(name)
    for _array, name in array_sizes_reading(ast, dependent - folded):
        raise SpecializationConstantError(
            f"an array is sized by '{name}', which follows a specialization constant through a value slangc "
            "evaluates at run time; a Slang array size is a compile-time constant",
            target=_TARGET,
            constant_name=name,
            specialization_id=ids.get(name),
            kind="array-size",
        )
    for _case, name in case_labels_reading(ast, dependent):
        raise SpecializationConstantError(
            f"a switch case label reads '{name}', which follows a specialization constant; slangc takes only "
            "compile-time constants as case labels",
            target=_TARGET,
            constant_name=name,
            specialization_id=ids.get(name),
            kind="case-label",
        )
    defaults = {
        node.name: node.value.value
        for node in constants
        if node.name in folded
        and isinstance(getattr(node, "value", None), LiteralNode)
        and type(node.value.value) is int
    }
    return SlangSpecializationPlan(ids, folded, evaluated, defaults)
