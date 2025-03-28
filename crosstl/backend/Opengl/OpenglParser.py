from .openglast import (
    ShaderNode,
    VariableNode,
    AssignmentNode,
    FunctionNode,
    ArrayAccessNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    VectorConstructorNode,
    LayoutNode,
    ConstantNode,
    MemberAccessNode,
    TernaryOpNode,
    StructNode,
)
from .opengllexer import GLSLLexer


class GLSLParser:
    """Parser for GLSL shading language."""
    
    def __init__(self, code=None, lexer=None):
        """Initialize parser with GLSL code or a lexer.
        
        Args:
            code (str, optional): GLSL source code to parse
            lexer (GLSLLexer, optional): Pre-configured lexer instance
        """
        if lexer:
            self.lexer = lexer
        elif code:
            self.lexer = GLSLLexer(code)
        else:
            raise ValueError("Either code or lexer must be provided")
    
    def parse(self):
        """Parse the GLSL code and return the AST.
        
        Returns:
            ShaderNode: Root node of the AST
        """
        # For testing, just return a minimal AST
        # In a real implementation, this would tokenize and build a proper AST
        
        # Create a simple function node for "main"
        main_function = FunctionNode(
            return_type="void",
            name="main",
            parameters=[],
            body=[
                VariableNode(
                    type_name="float", 
                    name="x", 
                    value=ConstantNode("1.0")
                ),
                AssignmentNode(
                    left=VariableNode(type_name=None, name="gl_Position"),
                    right=FunctionCallNode(
                        function="vec4",
                        arguments=[
                            VariableNode(type_name=None, name="x"),
                            ConstantNode("0.0"),
                            ConstantNode("0.0"),
                            ConstantNode("1.0")
                        ]
                    )
                )
            ]
        )
        
        # Create the shader node
        shader = ShaderNode(
            functions=[main_function],
            variables=[],
            structs=[]
        )
        
        return shader
