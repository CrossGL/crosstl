"""Workgroup sizes and unnamed specialization constants of a SPIR-V module, as the importer writes them.

A pipeline can set a compute workgroup dimension through a specialization constant in two ways (SPIR-V 1.6
section 3.21, BuiltIn WorkgroupSize, and section 3.6, LocalSizeId):

* a constant composite decorated ``BuiltIn WorkgroupSize``, built from ``OpSpecConstant`` values, which takes
  precedence over the ``LocalSize`` and ``LocalSizeId`` execution modes of every entry point. glslang writes this
  for ``layout(local_size_x_id = N) in;`` before SPIR-V 1.6;
* the ``LocalSizeId`` execution mode (SPIR-V 1.2), whose operands are constant ids. glslang writes this from
  SPIR-V 1.6, where the decoration is deprecated, and reads ``gl_WorkGroupSize`` through an undecorated composite.

CrossGL spells such a dimension as GLSL does (GLSL 4.60 section 4.4.1.4): ``local_size_x = 64, local_size_x_id = 4``
gives the default and the SpecId, and ``gl_WorkGroupSize`` reads the specialized size. The importer used to write
the defaults only, so the translated shader could no longer be specialized.

A specialization constant without an ``OpName``, as in a module stripped of debug information, was read as its
default value wherever the module used it, which fixed it as well. The importer now names it
``spec_constant_<SpecId>``; constants that share a SpecId, a type and a default are one CrossGL declaration, and one
that only sizes a workgroup is not declared at all, since ``local_size_<d>_id`` declares it.

A dimension sized by anything but a constant or a single specialization constant, such as an ``OpSpecConstantOp``,
has no CrossGL or GLSL form and is refused with :class:`SpirvWorkgroupSizeUnsupportedError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..common_ast import MemberAccessNode

__all__ = [
    "SpirvWorkgroupSizeUnsupportedError",
    "WorkgroupDimension",
    "SpirvWorkgroupSizes",
    "workgroup_size_component",
    "workgroup_size_layout_qualifiers",
]

WORKGROUP_SIZE_BUILTIN_NAME = "gl_WorkGroupSize"
_AXES = ("x", "y", "z")
_SCALAR_SPEC_CONSTANTS = {"OpSpecConstant", "OpSpecConstantTrue", "OpSpecConstantFalse"}
_COMPOSITES = {"OpConstantComposite", "OpSpecConstantComposite"}
_NON_USES = {
    "OpName",
    "OpMemberName",
    "OpDecorate",
    "OpDecorateId",
    "OpDecorateString",
    "OpMemberDecorate",
    "OpGroupDecorate",
}


class SpirvWorkgroupSizeUnsupportedError(TranslationDiagnostic):
    """A workgroup dimension that neither a literal nor one specialization constant sizes."""

    def __init__(self, entry, axis, constant_id, opcode):
        self.missing_capabilities = ("spirv.import.derived_workgroup_size",)
        self.entry = entry
        self.axis = axis
        super().__init__(
            f"SPIR-V entry point '{entry}' sizes its workgroup {axis} dimension with {constant_id}, "
            f"an {opcode or 'undefined id'}; CrossGL sizes a dimension by a literal or by one specialization "
            f"constant (local_size_{axis}_id), as GLSL does",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-workgroup-size-unsupported",
        )


@dataclass(frozen=True)
class WorkgroupDimension:
    """One workgroup dimension: its size, or default size, and the SpecId that sets it, if any."""

    size: str
    spec_id: str | None = None


def workgroup_size_layout_qualifiers(
    dimensions: Sequence[WorkgroupDimension],
) -> list[tuple[str, str]]:
    """``local_size_{x,y,z}`` for every dimension, then ``local_size_<d>_id`` for the specialized ones."""
    qualifiers = [
        (f"local_size_{axis}", dimension.size)
        for axis, dimension in zip(_AXES, dimensions)
    ]
    qualifiers.extend(
        (f"local_size_{axis}_id", dimension.spec_id)
        for axis, dimension in zip(_AXES, dimensions)
        if dimension.spec_id is not None
    )
    return qualifiers


def workgroup_size_component(expression, index) -> MemberAccessNode | None:
    """``gl_WorkGroupSize.x`` for ``OpSpecConstantOp CompositeExtract`` of the specialized built-in, else ``None``.

    glslang sizes shared arrays this way (``shared uint keys[gl_WorkGroupSize.x]``); CrossGL writes the component
    as GLSL does, which an array size can hold.
    """
    if expression != WORKGROUP_SIZE_BUILTIN_NAME:
        return None
    try:
        axis = _AXES[int(str(index), 0)]
    except (IndexError, ValueError):
        return None
    return MemberAccessNode(expression, axis)


def _spec_id(decorations: Iterable[Sequence[str]]) -> str | None:
    for decoration in decorations:
        if len(decoration) >= 3 and decoration[1] == "SpecId":
            return decoration[2]
    return None


class SpirvWorkgroupSizes:
    """What the importer needs to name specialization constants and to size the workgroups of a module.

    Built from the raw instructions before the importer walks them, since constant composites and
    ``OpSpecConstantOp`` values read the names of the constants they use as the importer defines them.
    """

    def __init__(self, instructions: Sequence[tuple]):
        self.definitions: dict[str, tuple[str, list[str]]] = {}
        self._decorations: dict[str, list[list[str]]] = {}
        named: set[str] = set()
        taken: set[str] = set()
        local_size_ids: dict[str, list[str]] = {}
        for result_id, opcode, operands, _line in instructions:
            if opcode == "OpName" and len(operands) >= 2 and operands[1]:
                named.add(operands[0])
                taken.add(operands[1])
            elif opcode == "OpDecorate" and len(operands) >= 2:
                self._decorations.setdefault(operands[0], []).append(operands)
            elif opcode == "OpExecutionModeId" and operands[1:2] == ["LocalSizeId"]:
                local_size_ids[operands[0]] = operands[2:5]
            elif result_id and opcode.startswith(("OpConstant", "OpSpecConstant")):
                self.definitions[result_id] = (opcode, list(operands))
        self.local_size_ids = local_size_ids
        self.builtin_id = self._workgroup_size_builtin()
        self.names: dict[str, str] = {}
        self._fallback_groups: dict[str, list[str]] = {}
        self._name_unnamed_spec_constants(named, taken)
        if self._is_specialized_builtin():
            self.names[self.builtin_id] = WORKGROUP_SIZE_BUILTIN_NAME
        self._uses = self._non_workgroup_uses(instructions)

    def _workgroup_size_builtin(self) -> str | None:
        for constant_id, (opcode, _operands) in self.definitions.items():
            if opcode not in _COMPOSITES:
                continue
            for decoration in self._decorations.get(constant_id, []):
                if decoration[1:3] == ["BuiltIn", "WorkgroupSize"]:
                    return constant_id
        return None

    def _is_specialized_builtin(self) -> bool:
        if self.builtin_id is None:
            return False
        return self.definitions[self.builtin_id][0] == "OpSpecConstantComposite"

    def _name_unnamed_spec_constants(self, named: set[str], taken: set[str]) -> None:
        by_value: dict[tuple, str] = {}
        for constant_id, (opcode, operands) in self.definitions.items():
            if opcode not in _SCALAR_SPEC_CONSTANTS or constant_id in named:
                continue
            spec_id = _spec_id(self._decorations.get(constant_id, []))
            if spec_id is None:
                continue
            key = (spec_id, opcode, tuple(operands))
            name = by_value.get(key)
            if name is None:
                name = base = f"spec_constant_{spec_id}"
                suffix = 1
                while name in taken:
                    name = f"{base}_{suffix}"
                    suffix += 1
                taken.add(name)
                by_value[key] = name
            self.names[constant_id] = name
            self._fallback_groups.setdefault(name, []).append(constant_id)

    def _workgroup_positions(self) -> set[tuple[str, str]]:
        """(instruction key, operand) pairs that size a workgroup, which CrossGL writes as layout qualifiers."""
        positions = {
            ("LocalSizeId", operand) for operand in self._local_size_operands()
        }
        if self._is_specialized_builtin():
            positions.update(
                (self.builtin_id, operand)
                for operand in self.definitions[self.builtin_id][1][1:]
            )
        return positions

    def _local_size_operands(self) -> set[str]:
        return {
            operand for operands in self.local_size_ids.values() for operand in operands
        }

    def _non_workgroup_uses(self, instructions: Sequence[tuple]) -> set[str]:
        positions = self._workgroup_positions()
        uses: set[str] = set()
        for result_id, opcode, operands, _line in instructions:
            if opcode in _NON_USES:
                continue
            if opcode == "OpExecutionModeId" and operands[1:2] == ["LocalSizeId"]:
                key, values = "LocalSizeId", operands[2:]
            else:
                key, values = result_id, operands
            uses.update(
                operand
                for operand in values
                if isinstance(operand, str)
                and operand.startswith("%")
                and (key, operand) not in positions
            )
        return uses

    def declared_spec_constants(self, spec_constant_ids: Sequence[str]) -> list[str]:
        """The specialization constants CrossGL declares, in module order.

        A named constant is declared as before. Of unnamed constants that share a name, the first is declared,
        unless the module reads them only as workgroup dimensions, which ``local_size_<d>_id`` declares. A
        specialized ``WorkgroupSize`` composite is the ``gl_WorkGroupSize`` built-in and is not declared.
        """
        positions = {operand for _key, operand in self._workgroup_positions()}
        declared = []
        for constant_id in spec_constant_ids:
            if constant_id == self.builtin_id and self._is_specialized_builtin():
                continue
            group = self._fallback_groups.get(self.names.get(constant_id, ""))
            if group is None:
                declared.append(constant_id)
            elif constant_id == group[0] and (
                any(member in self._uses for member in group)
                or not any(member in positions for member in group)
            ):
                declared.append(constant_id)
        return declared

    def entry_dimensions(
        self, entry_id: str, entry_name: str
    ) -> tuple[WorkgroupDimension, ...] | None:
        """The workgroup size of a workgroup entry point, or ``None`` when only ``LocalSize`` literals size it.

        A ``WorkgroupSize`` composite takes precedence over the entry's ``LocalSize`` and ``LocalSizeId`` modes.
        """
        if self.builtin_id is not None:
            components = self.definitions[self.builtin_id][1][1:4]
        elif entry_id in self.local_size_ids:
            components = self.local_size_ids[entry_id]
        else:
            return None
        return tuple(
            self._dimension(entry_name, axis, component)
            for axis, component in zip(_AXES, components)
        )

    def _dimension(self, entry: str, axis: str, component: str) -> WorkgroupDimension:
        opcode, operands = self.definitions.get(component, (None, []))
        if opcode == "OpConstant" and len(operands) >= 2:
            return WorkgroupDimension(operands[1])
        if opcode == "OpSpecConstant" and len(operands) >= 2:
            spec_id = _spec_id(self._decorations.get(component, []))
            return WorkgroupDimension(operands[1], spec_id)
        raise SpirvWorkgroupSizeUnsupportedError(entry, axis, component, opcode)
