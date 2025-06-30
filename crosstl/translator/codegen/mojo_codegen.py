from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CbufferNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


class MojoCodeGen:
    def __init__(self):
        self.current_shader = None
        self.type_mapping = {
            # Scalar Types
            "void": "None",
            "int": "Int32",
            "short": "Int16",
            "long": "Int64",
            "uint": "UInt32",
            "ushort": "UInt16",
            "ulong": "UInt64",
            "float": "Float32",
            "double": "Float64",
            "half": "Float16",
            "bool": "Bool",
            # Vector Types
            "vec2": "SIMD[DType.float32, 2]",
            "vec3": "SIMD[DType.float32, 3]",
            "vec4": "SIMD[DType.float32, 4]",
            "ivec2": "SIMD[DType.int32, 2]",
            "ivec3": "SIMD[DType.int32, 3]",
            "ivec4": "SIMD[DType.int32, 4]",
            "uvec2": "SIMD[DType.uint32, 2]",
            "uvec3": "SIMD[DType.uint32, 3]",
            "uvec4": "SIMD[DType.uint32, 4]",
            "bvec2": "SIMD[DType.bool, 2]",
            "bvec3": "SIMD[DType.bool, 3]",
            "bvec4": "SIMD[DType.bool, 4]",
            # Matrix Types
            "mat2": "Matrix[DType.float32, 2, 2]",
            "mat3": "Matrix[DType.float32, 3, 3]",
            "mat4": "Matrix[DType.float32, 4, 4]",
            # Texture Types (Mojo equivalents)
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
            "sampler": "Sampler",
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment attributes
            "gl_FragColor": "color(0)",
            "gl_FragColor0": "color(0)",
            "gl_FragColor1": "color(1)",
            "gl_FragColor2": "color(2)",
            "gl_FragColor3": "color(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord0",
            "TEXCOORD1": "texcoord1",
            "TEXCOORD2": "texcoord2",
            "TEXCOORD3": "texcoord3",
            "COLOR": "color",
            "COLOR0": "color0",
            "COLOR1": "color1",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "normalize": "normalize",
            "dot": "dot_product",
            "cross": "cross_product",
            "length": "magnitude",
            "reflect": "reflect",
            "refract": "refract",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "sqrt": "sqrt",
            "pow": "power",
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
        }

    def generate(self, ast):
        code = "# Generated Mojo Shader Code\n"
        code += "from math import *\n"
        code += "from simd import *\n"
        code += "from gpu import *\n\n"

        # Generate structs - handle both old and new AST
        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += self.generate_struct(node)

        # Generate global variables - handle both old and new AST
        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                # Handle both old and new AST variable structures
                if hasattr(node, "var_type"):
                    if hasattr(node.var_type, "name"):
                        vtype = node.var_type.name
                    else:
                        vtype = str(node.var_type)
                elif hasattr(node, "vtype"):
                    vtype = node.vtype
                else:
                    vtype = "float"
                code += f"var {node.name}: {self.map_type(vtype)}\n"

        # Generate cbuffers as structs - handle both old and new AST
        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate functions - handle both old and new AST
        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)

            if qualifier == "vertex":
                code += "# Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "# Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "# Compute Shader\n"
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
                    code += f"# {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(func)

        return code

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        # Handle different TypeNode types
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # VectorType - map to proper Mojo vector types
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size

            # Map to Mojo vector types
            if element_type == "float":
                return f"vec{size}"  # This will be mapped to SIMD[DType.float32, {size}] later
            elif element_type == "int":
                return f"ivec{size}"  # This will be mapped to SIMD[DType.int32, {size}] later
            elif element_type == "uint":
                return f"uvec{size}"  # This will be mapped to SIMD[DType.uint32, {size}] later
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            # MatrixType
            element_type = self.convert_type_node_to_string(type_node.element_type)
            return f"mat{type_node.rows}x{type_node.cols}"  # Will be mapped later
        else:
            # Fallback
            return str(type_node)

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

    def generate_struct(self, node):
        code = f"@value\nstruct {node.name}:\n"

        # Generate struct members - handle both old and new AST
        members = getattr(node, "members", [])
        for member in members:
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                if member.size:
                    code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                else:
                    code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
            else:
                # Handle both old and new AST member structures
                if hasattr(member, "member_type"):
                    # New AST structure
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    # Old AST structure
                    member_type = member.vtype
                else:
                    member_type = "float"

                # Handle semantic - get from attributes in new AST
                semantic = None
                if hasattr(member, "semantic"):
                    semantic = member.semantic
                elif hasattr(member, "attributes"):
                    semantic = self.extract_semantic_from_attributes(member.attributes)

                semantic_comment = (
                    f"  # {self.map_semantic(semantic)}" if semantic else ""
                )
                code += f"    var {member.name}: {self.map_type(member_type)}{semantic_comment}\n"

        code += "\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"@value\nstruct {node.name}:\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    var {member.name}: {member_type}\n"
                code += "\n"
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += f"@value\nstruct {node.name}:\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    var {member.name}: StaticTuple[{self.map_type(element_type)}, {member.size}]\n"
                        else:
                            code += f"    var {member.name}: DynamicVector[{self.map_type(element_type)}]\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    var {member.name}: {member_type}\n"
                code += "\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        "    " * indent

        # Handle parameters - support both old and new AST
        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        for p in param_list:
            if hasattr(p, "param_type"):
                # New AST structure
                param_type = self.convert_type_node_to_string(p.param_type)
            elif hasattr(p, "vtype"):
                # Old AST structure
                param_type = p.vtype
            else:
                param_type = "float"

            # Handle semantic
            semantic = None
            if hasattr(p, "semantic"):
                semantic = p.semantic
            elif hasattr(p, "attributes"):
                semantic = self.extract_semantic_from_attributes(p.attributes)

            param_semantic = f"  # {self.map_semantic(semantic)}" if semantic else ""
            params.append(f"{p.name}: {self.map_type(param_type)}{param_semantic}")

        params_str = ", ".join(params) if params else ""

        # Handle return type - support both old and new AST
        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"

        # Add shader type decorators for Mojo GPU programming
        if shader_type == "vertex":
            code += f"@vertex_shader\n"
        elif shader_type == "fragment":
            code += f"@fragment_shader\n"
        elif shader_type == "compute":
            code += f"@compute_shader\n"

        code += f"fn {func.name}({params_str}) -> {self.map_type(return_type)}:\n"

        # Handle function body - support both old and new AST
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            # Old AST structure
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += "    pass\n"

        code += "\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            # Handle variable declarations with initialization
            if hasattr(stmt, "var_type"):
                # New AST structure
                var_type = self.convert_type_node_to_string(stmt.var_type)
            elif hasattr(stmt, "vtype") and stmt.vtype:
                # Old AST structure - check if this is actually an array declaration disguised as a variable
                vtype_str = str(stmt.vtype)
                if (
                    "ArrayAccessNode" in vtype_str
                    and "array=" in vtype_str
                    and "index=" in vtype_str
                ):
                    # This is likely an array declaration
                    import re

                    array_match = re.search(r"array=(\w+).*?index=(\w+)", vtype_str)
                    if array_match:
                        array_match.group(1)
                        size = array_match.group(2)
                        base_type = "Float32"  # Default, could be improved
                        return f"{indent_str}var {stmt.name}: StaticTuple[{base_type}, {size}]\n"
                var_type = stmt.vtype
            else:
                var_type = "float"

            # Handle initialization
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return f"{indent_str}var {stmt.name}: {self.map_type(var_type)} = {init_expr}\n"
            else:
                return f"{indent_str}var {stmt.name}: {self.map_type(var_type)}\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)}\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values}\n"
            else:
                return f"{indent_str}return {self.generate_expression(stmt.value)}\n"
        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode should not appear as a statement by itself - it's likely a misclassified array declaration
            # Try to handle it gracefully
            return f"{indent_str}# Unhandled ArrayAccessNode: {stmt}\n"
        else:
            # Handle expressions that may be used as statements
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result}\n"
            else:
                return f"{indent_str}# Unhandled statement: {type(stmt).__name__}\n"

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type = self.map_type(node.element_type)
        size = get_array_size_from_node(node)

        if size is None:
            return f"{indent_str}var {node.name}: DynamicVector[{element_type}]\n"
        else:
            return f"{indent_str}var {node.name}: StaticTuple[{element_type}, {size}]\n"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = self.map_operator(node.operator)
        return f"{left} {op} {right}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition}:\n"

        # Generate if body - handle BlockNode structure
        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            # New AST BlockNode structure
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            # Old AST structure - list of statements
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        # Handle else branch - check if it's another if statement (else-if chain)
        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            # Check if else branch is another IfNode (else-if chain)
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate elif by recursively generating the nested if with elif prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f"{indent_str}elif {elif_condition}:\n"

                # Generate elif body
                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                # Recursively handle any remaining else-if chain
                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    # Create a temporary node to continue the chain
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another elif
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "elif"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", "elif ", 1
                            )
                        code += "\n".join(remaining_lines)
                    else:
                        # Final else clause
                        code += f"{indent_str}else:\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
            else:
                # Regular else clause
                code += f"{indent_str}else:\n"
                if hasattr(else_branch, "statements"):
                    # New AST BlockNode structure
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    # Old AST structure
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    # Single statement
                    code += self.generate_statement(else_branch, indent + 1)

        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # Extract init, condition, and update
        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        # In Mojo, we'll use a while loop for C-style for loops
        code = f"{indent_str}{init}\n"
        code += f"{indent_str}while {condition}:\n"

        # Generate loop body
        # Handle BlockNode structure
        if hasattr(node.body, "statements"):
            # New AST BlockNode structure
            for stmt in node.body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(node.body, list):
            # Old AST list structure
            for stmt in node.body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            # Single statement
            code += self.generate_statement(node.body, indent + 1)

        # Add update at the end of the loop
        code += f"{indent_str}    {update}\n"

        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            return str(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype and expr.name:
                return f"{expr.name}"
            elif hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            op = self.map_operator(expr.op)
            return f"({op}{operand})"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                array = self.generate_expression(expr.array)
                index = self.generate_expression(expr.index)
                return f"{array}[{index}]"
            else:
                # Fallback for malformed ArrayAccessNode
                return str(expr)
        elif isinstance(expr, FunctionCallNode):
            # Extract function name properly (might be IdentifierNode)
            func_name = expr.name
            if hasattr(func_name, "name"):
                # It's an IdentifierNode, extract the name
                func_name = func_name.name
            elif not isinstance(func_name, str):
                # Convert to string if it's some other type
                func_name = str(func_name)

            # Map function names to Mojo equivalents
            func_name = self.function_map.get(func_name, func_name)

            # Handle vector constructors
            if func_name in [
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
            ]:
                mojo_type = self.map_type(func_name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{mojo_type}({args})"

            # Handle standard function calls
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({true_expr} if {condition} else {false_expr})"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
            if hasattr(expr, "value"):
                value = expr.value
                if isinstance(value, str) and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    return f'"{value}"'  # Add quotes for string literals
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            # Handle IdentifierNode
            return getattr(expr, "name", str(expr))
        elif hasattr(expr, "__class__") and "ExpressionStatement" in str(
            expr.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(expr, "expression"):
                return self.generate_expression(expr.expression)
            else:
                return self.generate_expression(expr)
        else:
            # For unknown expression types, handle special cases
            expr_str = str(expr)
            # Check if this looks like an array declaration being misinterpreted
            if (
                "ArrayAccessNode" in expr_str
                and "array=" in expr_str
                and "index=" in expr_str
            ):
                # Try to extract array name and size for array declarations
                import re

                array_match = re.search(r"array=(\w+).*?index=(\w+)", expr_str)
                if array_match:
                    array_name = array_match.group(1)
                    array_match.group(2)
                    return f"{array_name}"  # Just return the array name for now
            return expr_str

    def map_type(self, vtype):
        if vtype is None:
            return "Float32"

        # Handle TypeNode objects by converting to string first
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        # Handle array types first
        if "[" in vtype_str and "]" in vtype_str:
            base_type, size = parse_array_type(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            if size:
                return f"StaticTuple[{base_mapped}, {size}]"
            else:
                return f"DynamicVector[{base_mapped}]"

        # Use regular type mapping
        return self.type_mapping.get(vtype_str, vtype_str)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "ASSIGN_OR": "|=",
            "ASSIGN_AND": "&=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "and",
            "OR": "or",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "LOGICAL_AND": "and",
            "LOGICAL_OR": "or",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
            "NOT": "not",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""
