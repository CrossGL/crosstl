"""WGSL ``override`` declarations for CrossGL specialization constants.

WGSL's specialization constants are pipeline-overridable constants: ``@id(N) override name: T = default;`` is set by
the WebGPU pipeline through ``GPUProgrammableStage.constants`` under the key ``N`` (WGSL, "override Declarations" and
the ``id`` attribute). A CrossGL ``const T name @constant_id(N) = value;`` becomes exactly that, and
a constant whose initializer reads one becomes an override without an id, whose initializer is an override-expression
(``override doubled: i32 = scale * 2 + 1;``), so it follows the specialized value as ``OpSpecConstantOp`` does.

The generator used to print every constant as ``const``, which froze the default and dropped the id.

WGSL limits overrides, and each limit is refused rather than frozen:

* an override has a concrete scalar type (``bool``, ``i32``, ``u32``, ``f32`` or ``f16``), so a specialization constant
  of any other type is refused. A dependent constant of another type, such as ``vec2(scale)``, becomes a
  ``var<private>`` whose initializer may be an override-expression (WGSL, "var Declarations"); it may read only
  overrides and constants, since one module-scope variable's initializer cannot read another;
* ids are below 65536 and unique within the shader;
* an array's element count may be an override-expression only for the store type of a ``var<workgroup>`` (WGSL, "Array
  Types"), so any other array sized by a dependent constant is refused;
* a ``switch`` case selector is a const-expression (WGSL, "Switch Statement"), so one that reads a dependent
  constant is refused.
"""

from __future__ import annotations

from collections.abc import Callable

from ..ast import ArrayType, VariableNode
from .specialization_constants import (
    SpecializationConstantError,
    array_sizes_reading,
    case_labels_reading,
    identifier_names,
    specialization_constant_ids,
    specialization_dependent_constants,
    walk_nodes,
)

__all__ = ["WGSLSpecializationPlan", "plan_wgsl_specialization_constants"]

_TARGET = "WGSL"
_MAX_WGSL_ID = 65535
_OVERRIDE_TYPES = frozenset({"bool", "i32", "u32", "f32", "f16"})
_WORKGROUP_QUALIFIERS = frozenset({"workgroup", "shared"})


class WGSLSpecializationPlan:
    """How each specialization-dependent constant of one program is declared in WGSL."""

    def __init__(self, ids: dict[str, int], overrides: set[str], private: set[str]):
        self.ids = ids
        self.overrides = overrides
        self.private = private

    @property
    def dependent(self) -> set[str]:
        return self.overrides | self.private

    def declaration(self, name: str, emitted_name: str, type_name: str, value_code):
        """The WGSL declaration of constant ``name``, or ``None`` when it is an ordinary ``const``.

        ``emitted_name`` is the module identifier it is declared as and ``value_code`` its rendered initializer,
        or ``None`` when it has none: the pipeline must then set it.
        """
        initializer = "" if value_code is None else f" = {value_code}"
        if name in self.private:
            return f"var<private> {emitted_name}: {type_name}{initializer};"
        if name not in self.overrides:
            return None
        declaration = f"override {emitted_name}: {type_name}{initializer};"
        if name in self.ids:
            # The id goes on its own line, so the declaration itself starts with ``override``.
            return f"@id({self.ids[name]})\n{declaration}"
        return declaration


def _workgroup_array_types(ast) -> list[ArrayType]:
    arrays = []
    for node in walk_nodes(ast):
        if not isinstance(node, VariableNode):
            continue
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        var_type = getattr(node, "var_type", None)
        if qualifiers & _WORKGROUP_QUALIFIERS and isinstance(var_type, ArrayType):
            arrays.append(var_type)
    return arrays


def _classify(constants, ids, type_name_of: Callable[[object], str]):
    dependent = specialization_dependent_constants(
        constants, lambda node: getattr(node, "name", None) in ids
    )
    overrides: set[str] = set()
    private: set[str] = set()
    for node in constants:
        name = getattr(node, "name", None)
        if name not in dependent:
            continue
        type_name = type_name_of(node.const_type)
        reads = identifier_names(getattr(node, "value", None))
        if name in ids and type_name not in _OVERRIDE_TYPES:
            raise SpecializationConstantError(
                f"specialization constant '{name}' has type {type_name}; a WGSL override is a bool, i32, u32, "
                "f32 or f16 scalar (WGSL, override Declarations)",
                target=_TARGET,
                constant_name=name,
                specialization_id=ids[name],
                kind="type",
            )
        if reads & private:
            reader = sorted(reads & private)[0]
            raise SpecializationConstantError(
                f"constant '{name}' reads '{reader}', a {_TARGET} module-scope variable that follows a "
                "specialization constant; an override or variable initializer cannot read a variable "
                "(WGSL, override and var Declarations)",
                target=_TARGET,
                constant_name=name,
                kind="dependent-initializer",
            )
        if type_name in _OVERRIDE_TYPES:
            overrides.add(name)
        else:
            private.add(name)
    return overrides, private


def plan_wgsl_specialization_constants(
    ast, type_name_of: Callable[[object], str], flattened_array_types=()
) -> WGSLSpecializationPlan:
    """Plan the WGSL declarations of ``ast``'s specialization constants and refuse what WGSL cannot express.

    ``type_name_of`` renders a CrossGL type as its WGSL spelling. ``flattened_array_types`` are the array types of
    stage-interface members that WGSL declares one member per element; they never reach a WGSL array type.
    """
    constants = list(getattr(ast, "constants", None) or [])
    ids = specialization_constant_ids(constants, _TARGET, _MAX_WGSL_ID)
    if not ids:
        return WGSLSpecializationPlan({}, set(), set())
    overrides, private = _classify(constants, ids, type_name_of)
    dependent = overrides | private
    allowed = [*_workgroup_array_types(ast), *flattened_array_types]
    for _array, name in array_sizes_reading(ast, dependent, allowed):
        raise SpecializationConstantError(
            f"an array is sized by '{name}', which follows a specialization constant; WGSL takes an "
            "override-expression element count only for the store type of a var<workgroup> (WGSL, Array Types)",
            target=_TARGET,
            constant_name=name,
            specialization_id=ids.get(name),
            kind="array-size",
        )
    for _case, name in case_labels_reading(ast, dependent):
        raise SpecializationConstantError(
            f"a switch case selector reads '{name}', which follows a specialization constant; a WGSL case "
            "selector is a const-expression (WGSL, Switch Statement)",
            target=_TARGET,
            constant_name=name,
            specialization_id=ids.get(name),
            kind="case-selector",
        )
    return WGSLSpecializationPlan(ids, overrides, private)
