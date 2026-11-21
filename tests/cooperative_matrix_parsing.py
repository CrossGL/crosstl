"""Parse CrossGL into a tree that keeps cooperative matrix operations the parser cannot type.

The parser refuses a cooperative matrix operation whose operands have no contract, or whose shapes or components
disagree. Each generator still checks the operations it is handed, since a tree can also come from a semantic pass
rather than the parser. These helpers build such a tree, so a test can reach a generator's own check.
"""

from contextlib import contextmanager

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

__all__ = [
    "keeping_untyped_cooperative_matrices",
    "parse_keeping_untyped_cooperative_matrices",
]


@contextmanager
def keeping_untyped_cooperative_matrices():
    """While active, the parser keeps an operation it cannot type instead of raising a syntax error."""
    infer = Parser.infer_cooperative_matrix_operation
    constrain = Parser.constrain_cooperative_matrix_result

    def keep_operation(self, operation_node):
        try:
            return infer(self, operation_node)
        except SyntaxError:
            return operation_node

    def keep_constraint(self, expression, expected_type, context):
        try:
            constrain(self, expression, expected_type, context)
        except SyntaxError:
            pass

    Parser.infer_cooperative_matrix_operation = keep_operation
    Parser.constrain_cooperative_matrix_result = keep_constraint
    try:
        yield
    finally:
        Parser.infer_cooperative_matrix_operation = infer
        Parser.constrain_cooperative_matrix_result = constrain


def parse_keeping_untyped_cooperative_matrices(source):
    """The CrossGL tree of ``source``, keeping the cooperative matrix operations the parser cannot type."""
    with keeping_untyped_cooperative_matrices():
        return Parser(Lexer(source).tokens).parse()
