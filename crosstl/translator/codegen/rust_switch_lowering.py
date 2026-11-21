"""Rust lowering of C-style ``switch`` statements.

A C ``switch`` (HLSL, GLSL, MSL, CrossGL) jumps to a label inside one statement list: consecutive labels share the
body after them, a body that does not end in a jump falls through into the next one, and ``break`` anywhere in a
body, outside a nested loop, leaves the switch. A Rust ``match`` arm runs one block and cannot be the target of
``break``. The lowering therefore builds one arm per body with :func:`.wgsl_switch_clauses` (every label that leads
to the body, plus the bodies it falls into) and wraps the ``match`` in a labeled block, so a ``break`` that leaves the
switch becomes ``break 'label`` (The Rust Reference, "Labeled block expressions").

Case values are written as ``match`` patterns only when every one is an integer literal. Any other selector, such as a
constant name, would be read by Rust as a new binding that matches everything, so those switches compare the value
once per arm in an ``if`` chain instead.
"""

from __future__ import annotations

from ..ast import BreakNode, LiteralNode, UnaryOpNode
from .wgsl_switch_clauses import wgsl_switch_clauses


def _is_literal_selector(value) -> bool:
    if isinstance(value, UnaryOpNode) and getattr(value, "operator", None) == "-":
        value = value.operand
    return isinstance(value, LiteralNode)


class RustSwitchLoweringMixin:
    """Write ``SwitchNode`` statements as labeled ``match`` blocks or ``if`` chains."""

    def active_switch_label(self):
        """The label a ``break`` leaves through, when the innermost breakable construct is a switch."""
        contexts = getattr(self, "switch_contexts", None)
        if contexts and contexts[-1]["loop_depth"] == self.loop_depth:
            return contexts[-1]["label"]
        return None

    def enter_loop_label(self):
        """Push the label of a loop being written; it is printed only if a jump inside a switch names it."""
        if not hasattr(self, "loop_labels"):
            self.loop_labels = []
        self.loop_label_counter = getattr(self, "loop_label_counter", 0) + 1
        record = {"label": f"'crosstl_loop{self.loop_label_counter}", "used": False}
        self.loop_labels.append(record)
        return record

    def exit_loop_label(self, record):
        """Pop ``record`` and return the ``'label: `` prefix its loop needs, or an empty string."""
        self.loop_labels.pop()
        return f"{record['label']}: " if record["used"] else ""

    def loop_jump_label(self):
        """`` 'label`` for a ``break``/``continue`` of the innermost loop written from inside a switch's block.

        Rust rejects an unlabeled jump inside a labeled block (E0695), so a jump that crosses a switch names its loop.
        """
        contexts = getattr(self, "switch_contexts", None)
        labels = getattr(self, "loop_labels", None)
        if not contexts or not labels:
            return ""
        if contexts[-1]["loop_depth"] != self.loop_depth:
            return ""
        labels[-1]["used"] = True
        return f" {labels[-1]['label']}"

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        inner = indent + 1
        inner_str = "    " * inner
        arm_str = "    " * (inner + 1)
        if not hasattr(self, "switch_contexts"):
            self.switch_contexts = []
        label = f"'crosstl_switch{len(self.switch_contexts)}"
        clauses = wgsl_switch_clauses(node)
        selectors = [s for clause in clauses for s in clause.selectors if s is not None]
        use_match = all(_is_literal_selector(s) for s in selectors)

        self.switch_contexts.append({"loop_depth": self.loop_depth, "label": label})
        try:
            subject = self.generate_expression(getattr(node, "expression", ""))
            code = f"{indent_str}{label}: {{\n"
            if use_match:
                code += f"{inner_str}match {subject} {{\n"
                for clause in clauses:
                    if clause.is_default:
                        pattern = "_"
                    else:
                        pattern = " | ".join(
                            self.generate_expression(s) for s in clause.selectors
                        )
                    code += f"{arm_str}{pattern} => {{\n"
                    code += self.generate_switch_clause_body(
                        clause.statements, inner + 2
                    )
                    code += f"{arm_str}}}\n"
                code += f"{inner_str}}}\n"
            else:
                value = f"crosstl_switch_value{len(self.switch_contexts) - 1}"
                code += f"{inner_str}let {value} = {subject};\n"
                branches = [c for c in clauses if not c.is_default]
                default = [c for c in clauses if c.is_default]
                keyword = f"{inner_str}if"
                for clause in branches:
                    condition = " || ".join(
                        f"{value} == {self.generate_expression(s)}"
                        for s in clause.selectors
                    )
                    code += f"{keyword} {condition} {{\n"
                    code += self.generate_switch_clause_body(
                        clause.statements, inner + 1
                    )
                    code += f"{inner_str}}}"
                    keyword = " else if"
                for clause in default:
                    code += " else {\n" if branches else f"{inner_str}{{\n"
                    code += self.generate_switch_clause_body(
                        clause.statements, inner + 1
                    )
                    code += f"{inner_str}}}"
                code += "\n"
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.switch_contexts.pop()

    def generate_switch_clause_body(self, statements, indent):
        """The statements of one arm; a trailing top-level ``break`` is the end of the arm."""
        statements = list(statements)
        while statements and isinstance(statements[-1], BreakNode):
            statements.pop()
        return "".join(self.generate_statement(stmt, indent) for stmt in statements)
