"""Block scopes for the sizes the OpenCL importer folds ``sizeof`` to.

The importer replaces ``sizeof(x)`` with the size of ``x``'s declared type, which it records in
``opencl_sizeof_symbols`` when it visits the declaration. A declaration is in scope until the end of the block that
holds it, and OpenCL C follows C here: a compound statement is a block, a selection or iteration statement is a
block, and so is each of its substatements (C17 6.2.1p4, 6.8.4p3, 6.8.5p5). The importer recorded every declaration
for the rest of the function, so after ``{ char c; }`` an outer ``double c`` was sized as 1 byte, and a ``for``
statement's own declaration was not recorded at all.

:class:`OpenCLSizeofScopeMixin` restores the recorded sizes when each block ends. It wraps the statement visitors
of the HIP converter that the OpenCL converter inherits, so their output is unchanged. An ``if`` statement's
branches are handed to the HIP visitor as :class:`OpenCLScopedBranch` statements, which emit the branch's
statements in a scope of their own, so that a declaration in one branch is not in scope in the other.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

from crosstl.backend.HIP.HipAst import VariableNode

__all__ = ["OpenCLScopedBranch", "OpenCLSizeofScopeMixin"]


class OpenCLScopedBranch:
    """The statements of one branch of an ``if`` statement, which form a block of their own."""

    def __init__(self, statements):
        self.statements = statements


class OpenCLSizeofScopeMixin:
    """Give each OpenCL C block its own ``sizeof`` symbols."""

    @contextmanager
    def opencl_sizeof_scope(self):
        """Restore ``opencl_sizeof_symbols`` when the block the context covers ends."""
        saved = dict(getattr(self, "opencl_sizeof_symbols", None) or {})
        try:
            yield
        finally:
            self.opencl_sizeof_symbols = saved

    def visit_OpenCLScopedBranch(self, node):
        with self.opencl_sizeof_scope():
            self.emit_statement(node.statements)
        return ""

    def visit_BlockNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_BlockNode(node)

    def visit_IfNode(self, node):
        scoped = copy.copy(node)
        for branch in ("if_body", "else_body"):
            body = getattr(node, branch, None)
            if body:
                setattr(scoped, branch, OpenCLScopedBranch(body))
        with self.opencl_sizeof_scope():
            return super().visit_IfNode(scoped)

    def visit_ForNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_ForNode(node)

    def visit_RangeForNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_RangeForNode(node)

    def visit_WhileNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_WhileNode(node)

    def visit_DoWhileNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_DoWhileNode(node)

    def visit_SwitchNode(self, node):
        with self.opencl_sizeof_scope():
            return super().visit_SwitchNode(node)

    def format_statement_fragment(self, stmt):
        # A ``for`` statement's own declaration is written as a fragment of the loop header.
        if isinstance(stmt, VariableNode):
            self.register_opencl_sizeof_symbol(stmt)
        return super().format_statement_fragment(stmt)
