import unittest

from crosstl.src.backend.Opengl.OpenglParser.py import GLSLParser
from crosstl.src.backend.Opengl.OpenglAst.py import IfNode, BinaryOpNode, VariableNode


class TestGLSLParser(unittest.TestCase):

    def test_if_else_if_else(self):
        parser = GLSLParser()
        code = """
        if (a > 1) {
            doSomething();
        } else if (a < 0) {
            doSomethingElse();
        } else {
            fallback();
        }
        """
        ast = parser.parse(code)

        expected_ast = IfNode(
            condition=BinaryOpNode(left=VariableNode("a"), op=">", right=1),
            if_body=[FunctionCallNode("doSomething", [])],
            else_body=IfNode(
                condition=BinaryOpNode(left=VariableNode("a"), op="<", right=0),
                if_body=[FunctionCallNode("doSomethingElse", [])],
                else_body=[FunctionCallNode("fallback", [])],
            ),
        )

        self.assertEqual(ast, expected_ast)


class FunctionCallNode:
    def __init__(self, function_name, arguments):
        self.function_name = function_name
        self.arguments = arguments

    def __eq__(self, other):
        return (
            self.function_name == other.function_name
            and self.arguments == other.arguments
        )

    def __repr__(self):
        return f"FunctionCallNode({self.function_name}, {self.arguments})"


class IfNode:
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __eq__(self, other):
        return (
            self.condition == other.condition
            and self.if_body == other.if_body
            and self.else_body == other.else_body
        )

    def __repr__(self):
        return f"IfNode({self.condition}, {self.if_body}, {self.else_body})"


class BinaryOpNode:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __eq__(self, other):
        return (
            self.left == other.left
            and self.op == other.op
            and self.right == other.right
        )

    def __repr__(self):
        return f"BinaryOpNode({self.left}, {self.op}, {self.right})"


class VariableNode:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"VariableNode({self.name})"


if __name__ == "__main__":
    unittest.main()
