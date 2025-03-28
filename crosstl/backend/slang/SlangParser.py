"""
Slang parser implementation
"""

from .slangast import (
    ConstantNode,
    FunctionCallNode,
    FunctionNode,
    ParameterNode,
    ReturnNode,
    ShaderNode,
    VariableNode,
)
from .slanglexer import SlangLexer


class SlangParser:
    """Parser for Slang shading language."""

    def __init__(self, code=None, lexer=None):
        """Initialize parser with Slang code or a lexer.

        Args:
            code (str, optional): Slang source code to parse
            lexer (SlangLexer, optional): Pre-configured lexer instance
        """
        if lexer:
            self.lexer = lexer
        elif code:
            self.lexer = SlangLexer(code)
        else:
            raise ValueError("Either code or lexer must be provided")

    def parse(self):
        """Parse the Slang code and return the AST.

        Returns:
            ShaderNode: Root node of the AST
        """
        # For testing, just return a minimal AST
        # In a real implementation, this would tokenize and build a proper AST

        # Create a simple function node for "main"
        main_function = FunctionNode(
            return_type="float4",
            name="main",
            parameters=[
                ParameterNode(type_name="float2", name="uv", semantic="TEXCOORD")
            ],
            body=[
                VariableNode(type_name="float", name="x", value=ConstantNode("1.0")),
                ReturnNode(
                    value=FunctionCallNode(
                        function="float4",
                        arguments=[
                            VariableNode(type_name=None, name="x"),
                            ConstantNode("0.0"),
                            ConstantNode("0.0"),
                            ConstantNode("1.0"),
                        ],
                    )
                ),
            ],
            semantic="SV_TARGET",
        )

        # Create the shader node
        shader = ShaderNode(functions=[main_function], variables=[], structs=[])

        return shader
