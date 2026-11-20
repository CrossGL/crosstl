"""Exact iteration replay of counted GLSL for loops over pointer arrays and aggregates, for ``GLSLControlFlowIntervalAnalysis.visit_for``."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
)
from ..unhandled import (
    UNHANDLED,
)


class OpenGLLoopReplayIntervalMixin:
    """Loop-replay members of ``GLSLControlFlowIntervalAnalysis.visit_for``."""

    def replay_exact_for_iterations(
        self,
        value,
        intervals,
        count,
        loop_intervals,
        restore_counter,
    ):
        """Visit each iteration of a loop whose exact counter values are known, else return ``UNHANDLED``."""
        pointer_array_loop = any(
            isinstance(child, ArrayAccessNode)
            and self.pointer_array_extent(child.array) is not None
            for child in self.generator.walk_ast(value.body)
        )
        aggregate_loop_names = {
            child.name
            for child in self.generator.walk_ast(value.body)
            if isinstance(child, IdentifierNode)
            and child.name not in self.escaped_names
            and self.generator.glsl_struct_member_types(
                self.generator.type_name_string(self.stable_lexical_type(child))
            )
            and any(key.startswith(f"{child.name}.") for key in intervals)
        }
        self.annotate_expression(value.condition, loop_intervals)
        exact_values = (
            self.generator.glsl_storage_pointer_array_loop_iteration_values(
                value,
                counter_mutated=value.init.name in self.mutation_interval_keys(
                    value.body, id(value.init)
                ),
            )
            if count is not None and (pointer_array_loop or aggregate_loop_names)
            else None
        )
        if (
            exact_values is not None
            and len(exact_values) == count
            and count <= self.replay_budget
        ):
            # Replay proven iterations on source state, joining facts at
            # each repeated occurrence rather than keeping the last visit.
            replay = dict(intervals)
            self.visit(value.init, replay)
            self.replay_budget -= count
            self.replay_depth += 1
            try:
                for counter in exact_values:
                    replay[value.init.name] = (counter, counter)
                    self.visit(value.condition, replay)
                    self.visit_sequence(value.body, replay)
                    self.visit(value.update, replay)
            finally:
                self.replay_depth -= 1
            restore_counter(replay)
            for key in list(intervals):
                if key in replay:
                    intervals[key] = replay[key]
                else:
                    intervals.pop(key, None)
            for key, bounds in replay.items():
                if ".[" in key or key.partition(".")[0] in aggregate_loop_names:
                    intervals[key] = bounds
            return
        return UNHANDLED
