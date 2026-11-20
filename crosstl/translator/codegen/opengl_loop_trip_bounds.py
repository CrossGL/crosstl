"""Trip counts and counter bounds of counted GLSL for loops, for ``GLSLControlFlowIntervalAnalysis.bounded_loop_bounds``."""

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    target_arithmetic_type,
)
from ..ast import (
    ArrayNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    VariableNode,
)


class OpenGLLoopTripBoundMixin:
    """Trip-count members of ``GLSLControlFlowIntervalAnalysis.bounded_loop_bounds``."""

    def bounded_loop_trip_bounds(
        self,
        node,
        loop_intervals,
        condition,
        counter,
        dependencies,
        domain,
        increment,
        init,
        start,
    ):
        """Return the trip count and counter bounds of a checked loop, or None when a bound cannot be proved."""
        declarations = {
            child.name
            for child in self.generator.walk_ast(node.body)
            if isinstance(child, (VariableNode, ArrayNode))
        }
        if declarations & dependencies:
            return None
        bound = self.checked_integer_interval(condition.right, loop_intervals)
        if bound is None or bound[0] < 0:
            return None
        initial_left = self.checked_integer_interval(condition.left, loop_intervals)
        if initial_left is None:
            return None
        inclusive = self.generator.map_operator(condition.op) == "<="
        step = self.checked_integer_interval(increment, loop_intervals)
        if step is None or step[0] <= 0:
            return None
        distance = max(0, bound[1] - initial_left[0] + int(inclusive))
        count = (distance + step[0] - 1) // step[0]
        terminal = start[1] + count * step[1]
        if terminal > domain.maximum:
            return None
        counter_bounds = (start[0], start[1] + max(0, count - 1) * step[1])
        if count:
            update_bounds = self.checked_integer_interval(
                BinaryOpNode(counter, "+", increment),
                {**loop_intervals, init.name: counter_bounds},
            )
            if (
                update_bounds is None
                or not domain.minimum
                <= update_bounds[0]
                <= update_bounds[1]
                <= domain.maximum
            ):
                return None
        if condition.left is not counter:
            for call in self.generator.walk_ast(node.body):
                if not isinstance(call, FunctionCallNode):
                    continue
                name = self.generator.function_call_name(call)
                if (
                    not self.generator.glsl_side_effect_free_expression(call)
                    and name not in self.generator.function_definitions
                    and name not in self.generator.glsl_function_overloads_by_name
                    and not (
                        name in {"buffer_load", "buffer_store"}
                        and name not in self.generator.function_return_types
                    )
                ):
                    return None
            # Include the terminal condition evaluation. Source additions and
            # usual arithmetic conversions must preserve these integer values.
            frame = {**loop_intervals, init.name: (start[0], terminal)}
            source_types = {}
            for child in self.generator.walk_ast(condition):
                if isinstance(child, IdentifierNode):
                    source_type = self.stable_lexical_type(child)
                    if source_type is not None:
                        source_types[child.name] = source_type
            with self.generator.glsl_lexical_source_type_scope(source_types):
                for expression in self.generator.walk_ast(condition):
                    if not isinstance(expression, BinaryOpNode):
                        continue
                    plan = self.generator.glsl_arithmetic_conversion_plan(
                        self.generator.glsl_source_expression_type(expression.left),
                        self.generator.glsl_source_expression_type(expression.right),
                        self.generator.map_operator(expression.op),
                    )
                    if plan is None:
                        return None
                    for operand, target in (
                        (expression.left, plan.left_target_type),
                        (expression.right, plan.right_target_type),
                    ):
                        bounds = self.checked_integer_interval(operand, frame)
                        info = target_arithmetic_type(target)
                        if (
                            bounds is None
                            or info is None
                            or info.lanes != 1
                            or info.kind not in {
                                ArithmeticScalarKind.SIGNED_INTEGER,
                                ArithmeticScalarKind.UNSIGNED_INTEGER,
                            }
                        ):
                            return None
                        signed = info.kind == ArithmeticScalarKind.SIGNED_INTEGER
                        minimum = -(1 << (info.bits - 1)) if signed else 0
                        maximum = (1 << (info.bits - int(signed))) - 1
                        if not minimum <= bounds[0] <= bounds[1] <= maximum:
                            return None
        return count, counter_bounds
