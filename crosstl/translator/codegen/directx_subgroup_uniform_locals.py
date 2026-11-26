"""Software-subgroup loop proofs in the HLSL generator for flow-uniform locals.

Software subgroups need every invocation of a workgroup to reach the same
barriers, so each loop around a collective must run the same trips in every
invocation. These members prove loops whose start or bound reads locals that
the flow analysis knows to be uniform where the loop starts: strided loops
that start from a uniform value plus the subgroup index, helpers that read
program-scope constants, and counter loops entered with a counter assigned
before them.
"""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import BinaryOpNode, IdentifierNode, VariableNode
from .array_utils import _UnsignedLiteralInt
from .counter_loops import any_start_counter_loop_terminates


class DirectXSubgroupUniformLocalMixin:
    """Uniform-local loop members of ``HLSLCodeGen``."""

    def hlsl_software_subgroup_program_constants(self, ast):
        """Program-scope names every invocation reads with one value.

        These are the program's constants, immutable globals (``const`` or
        Metal ``constant``, never ``volatile``), specialization constants and
        the integer constants already resolved to literals. A helper that
        reads one adds no dependency to its uniform return summary.
        """
        names = {
            const.name
            for const in getattr(ast, "constants", None) or []
            if getattr(const, "name", None)
        }
        names.update(getattr(self, "literal_int_constants", {}) or {})
        names.update(
            {
                node.name
                for node in getattr(ast, "global_variables", None) or []
                if isinstance(node, VariableNode)
                and set(getattr(node, "qualifiers", None) or []) & {"const", "constant"}
                and "volatile" not in (getattr(node, "qualifiers", None) or [])
            }
        )
        names.update(getattr(self, "directx_specialization_constant_types", {}) or {})
        return names

    # Compound assignments whose result is a function of the old value and
    # the operand alone.
    HLSL_SOFTWARE_SUBGROUP_UNIFORM_COMPOUND_OPERATORS = frozenset(
        {"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
    )

    def hlsl_software_subgroup_uniform_compound_write(self, node, operator, names):
        """Whether ``name op= value`` keeps a uniform local uniform.

        The new value is ``name op value``, so with ``name`` uniform before
        the write and ``value`` uniform (the caller checks), every invocation
        computes the same result, as ``i += off`` does when ``off`` is.
        Element and member targets are excluded: their index or base may
        differ between invocations.
        """
        target = getattr(node, "target", getattr(node, "left", None))
        return (
            operator in self.HLSL_SOFTWARE_SUBGROUP_UNIFORM_COMPOUND_OPERATORS
            and isinstance(target, IdentifierNode)
            and target.name in names
        )

    def hlsl_software_subgroup_strided_start(
        self, expression, uniform_names, uniform_components, loop_name
    ):
        """Split a strided loop's start into a uniform base and a subgroup index.

        ``simd_gid`` starts from no base; ``lo + simd_gid`` and ``simd_gid +
        lo`` start from ``lo`` when ``lo`` is uniform across the workgroup
        where the loop starts. Each subgroup then visits ``lo + k * step +
        simd_gid``, so a loop over ``lo + k * step`` runs the same trips in
        every invocation. The base is read once, when the loop starts, so
        later writes to its names do not matter. Returns ``(base, index)``,
        with ``base`` ``None`` for a start of the index alone, or ``None`` for
        any other start.
        """
        if self.hlsl_software_subgroup_subgroup_id_expression(expression):
            return None, expression
        if not isinstance(expression, BinaryOpNode) or (
            self.map_operator(
                getattr(expression, "op", getattr(expression, "operator", None))
            )
            != "+"
        ):
            return None
        for index, base in (
            (expression.right, expression.left),
            (expression.left, expression.right),
        ):
            if (
                self.hlsl_software_subgroup_subgroup_id_expression(index)
                and loop_name
                not in self.hlsl_software_subgroup_expression_identifier_names(base)
                and self.hlsl_software_subgroup_uniform_expression(
                    base, uniform_names, uniform_components
                )
            ):
                return base, index
        return None

    def hlsl_software_subgroup_strided_header(self, node, plan, base_name, bound):
        """The base loop header and counter assignment of a strided loop.

        The base runs from the start's uniform part, or zero, in steps of the
        subgroup count; each invocation adds its subgroup index. Compared with
        the source loop, the base takes only values some subgroup's counter
        takes or tests: subgroup ``g`` starts at ``lo + g``, and the subgroup
        whose counter reaches ``hi - 1`` steps it to ``hi - 1 + step``, past
        every base step. So the rewrite adds no overflow the source lacks.
        """
        initializer = node.init
        loop_type = plan["loopType"]
        expected = getattr(initializer, "var_type", None)
        suffix_u = "u" if loop_type == "uint" else ""
        start_base, start_index = plan.get("start") or (None, None)
        if start_base is None:
            first = f"0{suffix_u}"
            offset = self.generate_expression_with_expected(
                initializer.initial_value, expected
            )
        else:
            first = self.generate_expression_with_expected(start_base, expected)
            offset = self.generate_expression_with_expected(start_index, expected)
        header = (
            f"for ({loop_type} {base_name} = {first}; "
            f"{base_name} < {bound}; {base_name} += {plan['step']}{suffix_u}) {{"
        )
        return header, f"({base_name} + {offset})"

    def hlsl_rebased_offset_counter_loop_terminates(
        self, type_name, offset, stops, step, operator
    ):
        """Prove ``for (; c + offset OP stop; c += step)`` from any start in HLSL.

        DXC compiles signed ``int`` addition without wrapping (``add nsw``),
        so the GLSL argument that ``c + offset`` steps like ``c`` does not
        carry over, and an unknown start could overflow the first sum. The
        loop is instead emitted as ``c OP stop - offset``, which every start
        value the source admits without overflow passes and fails alike.
        ``stop - offset`` must fit the type for every extreme ``stop``; the
        offset-free loop on it must then terminate from any start, so no
        test or step overflows.
        """
        if not isinstance(type_name, str) or not type_name.isidentifier():
            return False
        values = (offset, step, *stops)
        if not stops or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            return False
        if int(offset) <= 0 or operator not in {"<", "<="}:
            return False
        # Only signed counters against signed values: an unsigned bound or
        # offset would compare ``c + offset`` as unsigned, which ``c`` alone
        # does not, and unsigned counters wrap (see wrapping_counter_loop_uniform).
        shape = source_integer_shape(type_name)
        if (
            shape is None
            or shape[0] != ArithmeticScalarKind.SIGNED_INTEGER
            or any(isinstance(value, _UnsignedLiteralInt) for value in values)
        ):
            return False
        rebased = [int(stop) - int(offset) for stop in stops]
        # any_start_counter_loop_terminates checks each rebased stop fits.
        return any_start_counter_loop_terminates(type_name, 0, rebased, step, operator)

    def hlsl_rebased_offset_condition(self, condition, offset_node, bound):
        """``c OP stop - offset`` for the source test ``c + offset OP stop``."""
        counter = (
            condition.left.left
            if condition.left.right is offset_node
            else condition.left.right
        )
        return BinaryOpNode(
            counter,
            condition.operator,
            BinaryOpNode(bound, "-", offset_node),
        )
