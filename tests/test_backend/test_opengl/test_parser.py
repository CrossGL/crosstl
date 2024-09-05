import unittest
from glsl_parser import GLSLParser
from ast import IfNode, BinaryOpNode, VariableNode

class TestGLSLParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = GLSLParser()

    def test_simple_if_else_if(self):
        glsl_code = """
        if (a > 1) {
            // do something
        } else if (a < 0) {
            // do something else
        } else {
            // fallback
        }
        """
        
        ast = self.parser.parse(glsl_code)
        
        # Check the structure of the IfNode
        self.assertIsInstance(ast, IfNode)
        self.assertIsInstance(ast.condition, BinaryOpNode)
        self.assertEqual(ast.condition.op, ">")
        self.assertEqual(ast.condition.left, VariableNode("a"))
        self.assertEqual(ast.condition.right, 1)
        
        # Check the else if part is another IfNode
        self.assertIsInstance(ast.else_body, IfNode)
        self.assertEqual(ast.else_body.condition.op, "<")
        self.assertEqual(ast.else_body.condition.left, VariableNode("a"))
        self.assertEqual(ast.else_body.condition.right, 0)
        
        # Check the else part is correctly parsed
        self.assertIsNotNone(ast.else_body.else_body)

    def test_nested_else_if(self):
        glsl_code = """
        if (x == 10) {
            // do something
        } else if (x > 5) {
            if (y < 3) {
                // nested if
            }
        } else {
            // fallback
        }
        """

        ast = self.parser.parse(glsl_code)

        self.assertIsInstance(ast, IfNode)
        self.assertEqual(ast.condition.op, "==")
        self.assertEqual(ast.condition.left, VariableNode("x"))
        self.assertEqual(ast.condition.right, 10)

        else_if_node = ast.else_body
        self.assertIsInstance(else_if_node, IfNode)
        self.assertEqual(else_if_node.condition.op, ">")
        self.assertEqual(else_if_node.condition.left, VariableNode("x"))
        self.assertEqual(else_if_node.condition.right, 5)

        nested_if = else_if_node.if_body[0]  # Assuming the body returns a list of statements
        self.assertIsInstance(nested_if, IfNode)
        self.assertEqual(nested_if.condition.op, "<")
        self.assertEqual(nested_if.condition.left, VariableNode("y"))
        self.assertEqual(nested_if.condition.right, 3)

    def test_invalid_else_if(self):
        glsl_code = """
        if (a == 1) {
            // do something
        } else if {
            // missing condition
        }
        """
        
        with self.assertRaises(SyntaxError):
            self.parser.parse(glsl_code)

if __name__ == '__main__':
    unittest.main()

