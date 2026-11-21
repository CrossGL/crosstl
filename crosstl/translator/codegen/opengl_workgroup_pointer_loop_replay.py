"""Exact iteration replay of counted GLSL for loops over workgroup pointers, for ``GLSLWorkgroupPointerAccessAnalysis.visit_for``."""


class OpenGLWorkgroupPointerLoopReplayMixin:
    """Loop-replay members of ``GLSLWorkgroupPointerAccessAnalysis.visit_for``."""

    def replay_exact_for_iterations(
        self,
        value,
        entry_aliases,
        entry_intervals,
        exact_values,
        loop_name,
    ):
        """Visit each iteration of a loop whose exact counter values are known, and return the merged post-loop state."""
        loop_aliases = dict(entry_aliases)
        loop_intervals = dict(entry_intervals)
        for iteration_value in exact_values:
            if loop_name:
                loop_intervals[loop_name] = (
                    iteration_value,
                    iteration_value,
                )
            self.visit(
                getattr(value, "condition", None),
                loop_aliases,
                loop_intervals,
            )
            self.visit_scoped_sequence(
                getattr(value, "body", None),
                loop_aliases,
                loop_intervals,
            )
            self.visit(
                getattr(value, "update", None),
                loop_aliases,
                loop_intervals,
            )
        self.visit(
            getattr(value, "condition", None),
            loop_aliases,
            loop_intervals,
        )
        post_aliases = loop_aliases
        post_intervals = loop_intervals
        return post_aliases, post_intervals
