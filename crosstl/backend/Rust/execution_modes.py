"""The execution modes a rust-gpu entry attribute names, as CrossGL stage layout qualifiers.

``#[spirv(fragment(early_fragment_tests))]`` asks for the depth and stencil tests before the fragment shader runs
(SPIR-V ``ExecutionMode EarlyFragmentTests``); CrossGL writes it as GLSL does, ``layout(early_fragment_tests) in;``
in the fragment stage. ``depth_replacing`` is implied by a write of the fragment depth and needs no qualifier.
Any other mode inside the stage's parentheses has no CrossGL spelling here and is refused by name, because
dropping it would change what the stage does.
"""

from __future__ import annotations

# CrossGL stage-scope layout lines of each execution mode, per stage.
_STAGE_MODES = {
    "fragment": {
        "early_fragment_tests": "layout(early_fragment_tests) in;",
        "depth_replacing": None,
    },
}


def _stage_mode_names(tokens):
    """The bare names before the ``)`` that closes the stage's parentheses, skipping ``key = value`` options.

    The attribute parser keeps no commas, so ``entry_point_name = "fs", origin_upper_left`` arrives as
    ``entry_point_name``, ``=``, ``"fs"``, ``origin_upper_left``.
    """
    names, depth, index = [], 0, 0
    while index < len(tokens):
        token = tokens[index]
        if token == ")" and depth == 0:
            break
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0 and index + 1 < len(tokens) and tokens[index + 1] == "=":
            index += 3
            continue
        elif depth == 0 and token not in {",", "="}:
            names.append(token)
        index += 1
    return names


class RustExecutionModeMixin:
    """Read the execution modes of an entry's ``#[spirv(stage(...))]`` attribute."""

    def stage_execution_modes(self, attributes, shader_type):
        """The execution-mode names inside ``#[spirv(<shader_type>(...))]``, in source order.

        ``key = value`` entries (``entry_point_name = "..."``) are options of the entry, not modes.
        """
        modes = []
        for attr in self.effective_attributes(attributes or []):
            if getattr(attr, "name", None) != "spirv":
                continue
            args = [str(token) for token in attr.args or []]
            for index, token in enumerate(args[:-1]):
                if token != shader_type or args[index + 1] != "(":
                    continue
                modes.extend(_stage_mode_names(args[index + 2 :]))
        return modes

    def stage_execution_mode_lines(self, func, shader_type):
        """The CrossGL layout lines of ``func``'s execution modes; refuse a mode with no CrossGL spelling."""
        if shader_type == "compute":
            # ``compute(threads(...))`` is the workgroup size, written as ``@numthreads``.
            return []
        known = _STAGE_MODES.get(shader_type, {})
        lines = []
        for mode in self.stage_execution_modes(func.attributes, shader_type):
            if mode not in known:
                self.refuse_import(
                    f"#[spirv({shader_type}({mode}))] on '{func.name}': the execution mode has no CrossGL "
                    "spelling in the Rust importer"
                )
                continue
            line = known[mode]
            if line is not None and line not in lines:
                lines.append(line)
        return lines
