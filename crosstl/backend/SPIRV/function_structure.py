"""Per-function state that turns an imported SPIR-V function's blocks into a structured CrossGL body.

:class:`SpirvFunctionStructure` is created before the importer walks a function's instructions. It analyses the
CFG once (phi kinds, values that must be evaluated where they are defined), and the importer calls it back while
walking: :meth:`~SpirvFunctionStructure.block_started` at each ``OpLabel``,
:meth:`~SpirvFunctionStructure.import_phi` for each ``OpPhi`` and :meth:`~SpirvFunctionStructure.settle` after each
instruction. :meth:`~SpirvFunctionStructure.structured_body` then assembles the statement tree.
"""

from __future__ import annotations

from .scoped_declarations import hoist_out_of_scope_declarations
from .structured_cfg import StructuredEmitter, build_function_cfg, classify_phis
from .value_ordering import values_to_materialize
from .VulkanAst import AssignmentNode, TernaryOpNode, VariableNode

__all__ = ["SpirvFunctionStructure"]

_OPAQUE_KINDS = frozenset(
    {
        "pointer",
        "function",
        "image",
        "sampler",
        "sampled_image",
        "runtime_array",
        "acceleration_structure",
        "ray_query",
        "named_barrier",
    }
)


class SpirvFunctionStructure:
    def __init__(
        self, parser, raw_instructions, names, decorations, types, variables, constants
    ):
        self.parser = parser
        self.names = names
        self.decorations = decorations
        self.types = types
        self.constants = constants
        self.cfg = build_function_cfg(raw_instructions)
        self.phi_kinds = classify_phis(self.cfg)
        parameters = [
            (result_id, operands[0])
            for result_id, opcode, operands, _line in raw_instructions
            if opcode == "OpFunctionParameter" and result_id and operands
        ]
        self.forced = values_to_materialize(
            self.cfg,
            self.phi_kinds,
            types,
            decorations,
            variables,
            self.can_be_local,
            parameters,
        )
        self.variable_phis = [
            (phi_id, type_id)
            for label in self.cfg.order
            for phi_id, type_id, _incoming in self.cfg.blocks[label].phis
            if self.phi_kinds.get(phi_id, ("variable",))[0] == "variable"
        ]
        self._taken_names = {str(name) for name in names.values() if name}
        self._phi_names = {}
        self._temporaries = 0
        self._settled = set()
        self._block_starts = []
        self._expressions = None

    def can_be_local(self, type_id):
        info = self.types.get(type_id)
        if not info or info.get("kind") in _OPAQUE_KINDS:
            return False
        return info.get("name") != "void"

    def block_started(self, label, statements):
        self._block_starts.append((label, len(statements)))

    def _operand(self, value_id, expressions):
        return self.parser.spirv_assembly_operand_expression(
            value_id, expressions, self.names, self.decorations, self.constants
        )

    def _type_parts(self, type_id):
        type_name, suffix = self.parser.spirv_type_name_and_suffix(
            type_id, self.types, self.constants, names=self.names
        )
        return type_name, suffix

    def phi_name(self, phi_id):
        if phi_id not in self._phi_names:
            base = self.parser.spirv_assembly_value_name(
                phi_id, self.names, self.decorations, prefix="phi"
            )
            name = base
            if self.names.get(phi_id) != base and base in self._taken_names:
                suffix = 1
                while f"{base}_{suffix}" in self._taken_names:
                    suffix += 1
                name = f"{base}_{suffix}"
            self._taken_names.add(name)
            self._phi_names[phi_id] = name
        return self._phi_names[phi_id]

    def import_phi(self, result_id, operands, expressions, expression_type_ids, record):
        kind = self.phi_kinds.get(result_id, ("variable",))
        if kind[0] == "same":
            record(result_id, operands[0], self._operand(kind[1], expressions))
            return
        if kind[0] == "select":
            _kind, condition, true_value, false_value = kind
            record(
                result_id,
                operands[0],
                TernaryOpNode(
                    self._operand(condition, expressions),
                    self._operand(true_value, expressions),
                    self._operand(false_value, expressions),
                ),
            )
            return
        expressions[result_id] = VariableNode(
            "", self.phi_name(result_id), spirv_id=result_id
        )
        expression_type_ids[result_id] = operands[0]

    def settle(self, result_id, expressions, expression_type_ids, statements):
        """Assign ``result_id`` to a local now if it must not be evaluated where it is used."""
        if not result_id or result_id not in self.forced or result_id in self._settled:
            return
        self._settled.add(result_id)
        expression = expressions.get(result_id)
        type_id = expression_type_ids.get(result_id)
        if expression is None or type_id is None:
            return
        if (
            isinstance(expression, VariableNode)
            and getattr(expression, "spirv_id", None) == result_id
        ):
            return
        type_name, suffix = self._type_parts(type_id)
        if not type_name or type_name == "void":
            return
        name = self.parser.spirv_assembly_value_name(
            result_id, self.names, self.decorations, prefix="value"
        )
        statements.append(
            AssignmentNode(
                VariableNode(
                    type_name,
                    f"{name}{suffix}",
                    spirv_id=result_id,
                    spirv_type_id=type_id,
                ),
                expression,
            )
        )
        expressions[result_id] = VariableNode("", name, spirv_id=result_id)

    # Hooks of StructuredEmitter.
    def statements(self, label):
        return list(self._block_statements.get(label, []))

    def expression(self, value_id):
        return self._operand(value_id, self._expressions)

    def phi_variable(self, phi_id):
        return VariableNode("", self.phi_name(phi_id), spirv_id=phi_id)

    def phi_temporary(self, phi_id):
        self._temporaries += 1
        name = f"{self.phi_name(phi_id)}_next{self._temporaries}"
        while name in self._taken_names:
            self._temporaries += 1
            name = f"{self.phi_name(phi_id)}_next{self._temporaries}"
        self._taken_names.add(name)
        type_id = dict(self.variable_phis).get(phi_id)
        type_name, suffix = self._type_parts(type_id)
        declaration = VariableNode(
            type_name or "", f"{name}{suffix}", spirv_type_id=type_id
        )
        return declaration, VariableNode("", name)

    def structured_body(self, statements, expressions):
        if not self._block_starts:
            return statements
        self._expressions = expressions
        self._block_statements = {}
        for index, (label, start) in enumerate(self._block_starts):
            end = (
                self._block_starts[index + 1][1]
                if index + 1 < len(self._block_starts)
                else len(statements)
            )
            self._block_statements[label] = statements[start:end]
        prefix = statements[: self._block_starts[0][1]]
        body = StructuredEmitter(self.cfg, self.phi_kinds, self).emit_function()
        declarations = []
        for phi_id, type_id in self.variable_phis:
            type_name, suffix = self._type_parts(type_id)
            declarations.append(
                VariableNode(
                    type_name or "",
                    f"{self.phi_name(phi_id)}{suffix}",
                    spirv_id=phi_id,
                    spirv_type_id=type_id,
                )
            )
        return hoist_out_of_scope_declarations([*prefix, *declarations, *body])
