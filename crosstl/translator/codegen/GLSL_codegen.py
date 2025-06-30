from ..ast import (
    AssignmentNode,
    ArrayNode,
    ArrayAccessNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


class GLSLCodeGen:
    def __init__(self):
        """Initialize the code generator."""
        self.semantic_map = {
            "gl_VertexID": "gl_VertexID",
            "gl_InstanceID": "gl_InstanceID",
            "gl_IsFrontFace": "gl_FrontFacing",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "POSITION": "layout(location = 0)",
            "NORMAL": "layout(location = 1)",
            "TANGENT": "layout(location = 2)",
            "BINORMAL": "layout(location = 3)",
            "TEXCOORD": "layout(location = 4)",
            "TEXCOORD0": "layout(location = 5)",
            "TEXCOORD1": "layout(location = 6)",
            "TEXCOORD2": "layout(location = 7)",
            "TEXCOORD3": "layout(location = 8)",
            "TEXCOORD4": "layout(location = 9)",
            "TEXCOORD5": "layout(location = 10)",
            "TEXCOORD6": "layout(location = 11)",
            "TEXCOORD7": "layout(location = 12)",
            # Vertex outputs
            "gl_Position": "gl_Position",
            "gl_PointSize": "gl_PointSize",
            "gl_ClipDistance": "gl_ClipDistance",
            # Fragment outputs
            "gl_FragColor": "layout(location = 0)",
            "gl_FragColor1": "layout(location = 1)",
            "gl_FragColor2": "layout(location = 2)",
            "gl_FragColor3": "layout(location = 3)",
            "gl_FragColor4": "layout(location = 4)",
            "gl_FragColor5": "layout(location = 5)",
            "gl_FragColor6": "layout(location = 6)",
            "gl_FragColor7": "layout(location = 7)",
            "gl_FragDepth": "gl_FragDepth",
            # Additional fragment inputs
            "gl_FragCoord": "gl_FragCoord",
            "gl_FrontFacing": "gl_FrontFacing",
            "gl_PointCoord": "gl_PointCoord",
            # Compute shader specific
            "gl_GlobalInvocationID": "gl_GlobalInvocationID",
            "gl_LocalInvocationID": "gl_LocalInvocationID",
            "gl_WorkGroupID": "gl_WorkGroupID",
            "gl_LocalInvocationIndex": "gl_LocalInvocationIndex",
            "gl_WorkGroupSize": "gl_WorkGroupSize",
            "gl_NumWorkGroups": "gl_NumWorkGroups",
        }

        # Define type mapping for OpenGL (GLSL)
        self.type_mapping = {
            # Most types are the same in CrossGL and GLSL
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "double": "double",
            "void": "void",
            "sampler2D": "sampler2D",
            "samplerCube": "samplerCube",
        }

        # Function mapping for GLSL
        self.function_map = {
            "atan2": "atan",
            "lerp": "mix",
            "frac": "fract",
            "saturate": "clamp",
            "tex2D": "texture",
            "tex2Dproj": "textureProj",
            "tex2Dlod": "textureLod",
            "tex2Dbias": "texture",
            "tex2Dgrad": "textureGrad",
            "texCUBE": "texture",
            "texCUBElod": "textureLod",
            "texCUBEbias": "texture",
            "texCUBEgrad": "textureGrad",
            "mul": "*",  # Matrix multiplication
            "ddx": "dFdx",
            "ddy": "dFdy",
            "rsqrt": "inversesqrt",
            "sincos": "sin_cos",  # Custom function needed
            "clip": "discard",  # HLSL clip becomes GLSL discard
            "log2": "log2",
            "exp2": "exp2",
            "pow": "pow",
            "sqrt": "sqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "fmod": "mod",
            "trunc": "trunc",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            "all": "all",
            "any": "any",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
        }

    def generate(self, ast):
        code = "\n"
        code += "#version 450 core\n"

        # Generate structs - handle both old and new AST
        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                if node.name == "VSInput":
                    members = getattr(node, "members", [])
                    for member in members:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")

                        # Handle semantic
                        semantic = None
                        if hasattr(member, "semantic"):
                            semantic = member.semantic
                        elif hasattr(member, "attributes"):
                            for attr in member.attributes:
                                if hasattr(attr, "name"):
                                    semantic = attr.name
                                    break

                        code += f"{self.map_semantic(semantic)} in {member_type} {member.name};\n"
                elif node.name == "VSOutput":
                    members = getattr(node, "members", [])
                    for member in members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        code += f"out {member_type} {member.name};\n"
                elif node.name == "PSInput":
                    members = getattr(node, "members", [])
                    for member in members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        code += f"in {member_type} {member.name};\n"
                elif node.name == "PSOutput":
                    members = getattr(node, "members", [])
                    for member in members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")

                        # Handle semantic
                        semantic = None
                        if hasattr(member, "semantic"):
                            semantic = member.semantic
                        elif hasattr(member, "attributes"):
                            for attr in member.attributes:
                                if hasattr(attr, "name"):
                                    semantic = attr.name
                                    break

                        code += f"{self.map_semantic(semantic)} out {member_type} {member.name};\n"
                else:
                    code += self.generate_struct(node)

        # Generate global variables - handle both old and new AST
        global_vars = getattr(ast, "global_variables", [])
        for i, node in enumerate(global_vars):
            # Handle both old and new AST variable structures
            if hasattr(node, "var_type"):
                if hasattr(node.var_type, "name") or hasattr(
                    node.var_type, "element_type"
                ):
                    # Check if it's an ArrayType and handle specially for global variables
                    if (
                        hasattr(node.var_type, "element_type")
                        and str(type(node.var_type)).find("ArrayType") != -1
                    ):  # ArrayType
                        base_type = self.convert_type_node_to_string(
                            node.var_type.element_type
                        )
                        array_size = (
                            self.generate_expression(node.var_type.size)
                            if node.var_type.size
                            else ""
                        )
                        vtype = base_type
                        array_suffix = f"[{array_size}]" if array_size else "[]"
                    else:
                        # Use the proper type conversion for TypeNode objects
                        vtype = self.convert_type_node_to_string(node.var_type)
                        array_suffix = ""
                else:
                    vtype = str(node.var_type)
                    array_suffix = ""
            elif hasattr(node, "vtype"):
                vtype = node.vtype
                array_suffix = ""
            else:
                vtype = "float"
                array_suffix = ""

            if hasattr(node, "name"):
                var_name = node.name
            elif hasattr(node, "variable_name"):
                var_name = node.variable_name
            else:
                var_name = f"var{i}"

            code += f"layout(std140, binding = {i}) {vtype} {var_name}{array_suffix};\n"

        # Generate cbuffers - handle both old and new AST
        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate custom functions - handle both old and new AST
        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)

            if qualifier == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                    )  # Extract stage name from enum
                    code += f"// {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(func)

        return code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        for i, node in enumerate(cbuffers):
            if isinstance(node, StructNode):
                code += f"layout(std140, binding = {i}) uniform {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        code += f"    {self.map_type(member_type)} {member.name};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"layout(std140, binding = {i}) uniform {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        code += f"    {self.map_type(member_type)} {member.name};\n"
                code += "};\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        code += "  " * indent

        # Handle parameters - support both old and new AST
        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                # New AST structure
                if hasattr(p.param_type, "name"):
                    param_type = self.map_type(p.param_type.name)
                else:
                    param_type = self.map_type(str(p.param_type))
            elif hasattr(p, "vtype"):
                # Old AST structure
                param_type = self.map_type(p.vtype)
            else:
                param_type = "float"

            # Handle semantic
            semantic = None
            if hasattr(p, "semantic"):
                semantic = p.semantic
            elif hasattr(p, "attributes"):
                for attr in p.attributes:
                    if hasattr(attr, "name"):
                        semantic = attr.name
                        break

            params.append(f"{param_type} {p.name} {self.map_semantic(semantic)}")

        params_str = ", ".join(params)

        if shader_type == "vertex":
            code += f"void main(){{\n"
        elif shader_type == "fragment":
            code += f"void main() {{\n"
        elif shader_type == "compute":
            code += f"void main() {{\n"
        else:
            # Handle return type - support both old and new AST
            if hasattr(func, "return_type"):
                if hasattr(func.return_type, "name"):
                    return_type = self.map_type(func.return_type.name)
                else:
                    return_type = self.map_type(str(func.return_type))
            else:
                return_type = "void"

            code += f"{return_type} {func.name}({params_str}) {{\n"

        # Handle function body - support both old and new AST
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            # Old AST structure
            for stmt in body:
                code += self.generate_statement(stmt, 1)

        code += "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            # Handle both old and new AST variable structures
            if hasattr(stmt, "var_type"):
                # New AST structure
                var_type = self.convert_type_node_to_string(stmt.var_type)
            elif hasattr(stmt, "vtype"):
                # Old AST structure
                var_type = stmt.vtype
            else:
                var_type = "float"

            # Handle initialization
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return f"{indent_str}{self.map_type(var_type)} {stmt.name} = {init_expr};\n"
            else:
                return f"{indent_str}{self.map_type(var_type)} {stmt.name};\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values};\n"
            else:
                return f"{indent_str}return {self.generate_expression(stmt.value)};\n"
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            # Handle ExpressionStatementNode
            expr_code = self.generate_expression_statement(stmt)
            return f"{indent_str}{expr_code};\n"
        else:
            # Handle expressions that may be used as statements
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def generate_assignment(self, node, is_main=False):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = self.map_operator(node.operator)
        return f"{left} {op} {right}"

    def generate_if(self, node, indent, is_main=False):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        # Generate if body - handle BlockNode structure
        if_body = node.if_body
        if hasattr(if_body, "statements"):
            # New AST BlockNode structure
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            # Old AST structure - list of statements
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        # Generate else if conditions
        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.generate_expression(else_if_condition)
                code += f" else if ({condition}) {{\n"
                # Handle BlockNode for else if body
                if hasattr(else_if_body, "statements"):
                    for stmt in else_if_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_if_body, list):
                    for stmt in else_if_body:
                        code += self.generate_statement(stmt, indent + 1)
                code += f"{indent_str}}}"

        # Generate else body
        if hasattr(node, "else_body") and node.else_body:
            code += f" else {{\n"
            else_body = node.else_body
            if hasattr(else_body, "statements"):
                # New AST BlockNode structure
                for stmt in else_body.statements:
                    code += self.generate_statement(stmt, indent + 1)
            elif isinstance(else_body, list):
                # Old AST structure
                for stmt in else_body:
                    code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        indent_str = "    " * indent

        # Extract init, condition, and update
        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

        # Generate loop body - handle BlockNode structure
        body = node.body
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            # Old AST structure - list of statements
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}\n"

        return code

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "VariableNode" in str(type(expr)):
            if hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "IdentifierNode" in str(type(expr)):
            return expr.name
        elif hasattr(expr, "__class__") and "LiteralNode" in str(type(expr)):
            return str(expr.value)
        elif hasattr(expr, "__class__") and "BinaryOpNode" in str(type(expr)):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif hasattr(expr, "__class__") and "AssignmentNode" in str(type(expr)):
            left = self.generate_expression(
                expr.target if hasattr(expr, "target") else expr.left
            )
            right = self.generate_expression(
                expr.value if hasattr(expr, "value") else expr.right
            )
            op = expr.operator if hasattr(expr, "operator") else expr.op
            op = self.map_operator(op)
            return f"{left} {op} {right}"
        elif hasattr(expr, "__class__") and "UnaryOpNode" in str(type(expr)):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            return f"({op}{operand})"
        elif hasattr(expr, "__class__") and "ArrayAccessNode" in str(type(expr)):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                array = self.generate_expression(expr.array)
                index = self.generate_expression(expr.index)
                return f"{array}[{index}]"
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "FunctionCallNode" in str(type(expr)):
            # Map function names to GLSL equivalents
            func_name = self.function_map.get(expr.name, expr.name)

            # Handle vector constructors
            if expr.name in [
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
            ]:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{expr.name}({args})"

            # Handle matrix constructors
            if expr.name in ["mat2", "mat3", "mat4"]:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{expr.name}({args})"

            # Handle standard function calls
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif hasattr(expr, "__class__") and "MemberAccessNode" in str(type(expr)):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif hasattr(expr, "__class__") and "TernaryOpNode" in str(type(expr)):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def map_type(self, vtype):
        """Map types to GLSL equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        # Handle TypeNode objects
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        # Handle array types
        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            if size:
                return f"{base_mapped}[{size}]"
            else:
                return f"{base_mapped}[]"

        # Use regular type mapping
        return self.type_mapping.get(vtype_str, vtype_str)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_OR": "|=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "MOD": "%",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "BITWISE_XOR": "^",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "ASSIGN_XOR": "^=",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic is not None:
            return f"{self.semantic_map.get(semantic, semantic)}"
        else:
            return ""

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        # Handle different TypeNode types
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # Check if it's VectorType vs ArrayType
            if hasattr(type_node, "rows"):
                # MatrixType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.rows == type_node.cols:
                    return f"mat{type_node.rows}"
                else:
                    return f"mat{type_node.cols}x{type_node.rows}"
            elif str(type(type_node)).find("ArrayType") != -1:
                # ArrayType - handle C-style arrays
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        # Size is an expression node
                        size_str = self.expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                # VectorType - map to proper GLSL vector types
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size

                # Map to GLSL vector types
                if element_type == "float":
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                elif element_type == "bool":
                    return f"bvec{size}"
                else:
                    return f"{element_type}{size}"
        else:
            # Fallback
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif hasattr(expr, "name"):
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_array_declaration(self, stmt, indent):
        indent_str = "    " * indent
        element_type = self.map_type(stmt.element_type)
        size = get_array_size_from_node(stmt)

        if size is None:
            # In GLSL, dynamic sized arrays need special handling
            # For instance in shader storage blocks, but for simple cases:
            return f"{indent_str}{element_type} {stmt.name}[];\n"
        else:
            return f"{indent_str}{element_type} {stmt.name}[{size}];\n"

    def generate_struct(self, node):
        code = f"struct {node.name} {{\n"

        # Generate struct members - handle both old and new AST
        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                if member.size:
                    code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                else:
                    code += f"    {self.map_type(element_type)} {member.name}[];\n"
            else:
                # Handle both old and new AST member structures
                if hasattr(member, "member_type"):
                    # New AST structure - check if it's an ArrayType
                    if str(type(member.member_type)).find("ArrayType") != -1:
                        # Handle array types with C-style syntax for struct members
                        element_type = self.convert_type_node_to_string(
                            member.member_type.element_type
                        )
                        element_type = self.map_type(element_type)
                        if member.member_type.size is not None:
                            size_str = self.expression_to_string(
                                member.member_type.size
                            )
                            code += f"    {element_type} {member.name}[{size_str}];\n"
                        else:
                            code += f"    {element_type} {member.name}[];\n"
                    else:
                        # Regular type
                        member_type_str = self.convert_type_node_to_string(
                            member.member_type
                        )
                        member_type = self.map_type(member_type_str)
                        code += f"    {member_type} {member.name};\n"
                elif hasattr(member, "vtype"):
                    # Old AST structure
                    member_type = self.map_type(member.vtype)
                    code += f"    {member_type} {member.name};\n"
                else:
                    code += f"    float {member.name};\n"

        code += "};\n"
        return code

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        if hasattr(stmt, "expression"):
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            # Fallback for direct expression
            return self.generate_expression(stmt)
