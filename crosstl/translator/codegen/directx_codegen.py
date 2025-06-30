from ..ast import (
    AssignmentNode,
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
    ArrayAccessNode,
    ArrayNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


class HLSLCodeGen:
    def __init__(self):
        self.type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "float": "float",
            "double": "double",
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
            "sampler": "SamplerState",
        }

        self.semantic_map = {
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_IsFrontFace": "FRONT_FACE",
            "gl_PrimitiveID": "PRIMITIVE_ID",
            "InstanceID": "INSTANCE_ID",
            "VertexID": "VERTEX_ID",
            "gl_Position": "SV_POSITION",
            "gl_PointSize": "SV_POINTSIZE",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_FragColor": "SV_TARGET",
            "gl_FragColor0": "SV_TARGET0",
            "gl_FragColor1": "SV_TARGET1",
            "gl_FragColor2": "SV_TARGET2",
            "gl_FragColor3": "SV_TARGET3",
            "gl_FragColor4": "SV_TARGET4",
            "gl_FragColor5": "SV_TARGET5",
            "gl_FragColor6": "SV_TARGET6",
            "gl_FragColor7": "SV_TARGET7",
            "gl_FragDepth": "SV_DEPTH",
        }

    def generate(self, ast):
        code = "\n"

        # Generate structs - handle both old and new AST
        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in HLSL
                            code += (
                                f"    {self.map_type(element_type)}[] {member.name};\n"
                            )
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
                                    array_syntax = f"[{size_str}]"
                                else:
                                    array_syntax = "[]"
                                member_type = element_type
                            else:
                                # Regular type - pass TypeNode directly to map_type
                                member_type = self.map_type(member.member_type)
                                array_syntax = ""
                        elif hasattr(member, "vtype"):
                            # Old AST structure
                            member_type = self.map_type(member.vtype)
                            array_syntax = ""
                        else:
                            member_type = "float"
                            array_syntax = ""

                        # Handle semantic - get from attributes in new AST
                        semantic = None
                        if hasattr(member, "semantic"):
                            semantic = member.semantic
                        elif hasattr(member, "attributes"):
                            for attr in member.attributes:
                                if hasattr(attr, "name") and attr.name in [
                                    "position",
                                    "color",
                                    "texcoord",
                                    "normal",
                                ]:
                                    semantic = attr.name
                                    break

                        code += f"    {member_type} {member.name}{array_syntax}{self.map_semantic(semantic)};\n"
                code += "};\n"

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

            code += f"{vtype} {var_name}{array_suffix};\n"

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
                code += f"cbuffer {node.name} : register(b{i}) {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported, so we'll make it fixed size
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[1];\n"
                            )
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    {member_type} {member.name};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # Generic cbuffer handling
                code += f"cbuffer {node.name} : register(b{i}) {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[1];\n"
                            )
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    {member_type} {member.name};\n"
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
        shader_map = {"vertex": "VSMain", "fragment": "PSMain", "compute": "CSMain"}

        # Handle return type - support both old and new AST
        if hasattr(func, "return_type"):
            if hasattr(func.return_type, "name"):
                return_type = self.map_type(func.return_type.name)
            else:
                return_type = self.map_type(str(func.return_type))
        else:
            return_type = "void"

        # Handle qualifier
        if hasattr(func, "qualifiers") and func.qualifiers:
            qualifier = func.qualifiers[0] if func.qualifiers else None
        else:
            qualifier = getattr(func, "qualifier", None)

        if qualifier in shader_map:
            code += f"// {qualifier.capitalize()} Shader\n"
            code += f"{return_type} {shader_map[qualifier]}({params_str}) {{\n"
        else:
            code += f"{return_type} {func.name}({params_str}) {{\n"

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

        code += "  " * indent + "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            # Handle both old and new AST variable structures
            if hasattr(stmt, "var_type"):
                vtype = stmt.var_type
            elif hasattr(stmt, "vtype"):
                vtype = stmt.vtype
            else:
                vtype = "float"

            # Handle initialization
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return (
                    f"{indent_str}{self.map_type(vtype)} {stmt.name} = {init_expr};\n"
                )
            else:
                return f"{indent_str}{self.map_type(vtype)} {stmt.name};\n"

        elif isinstance(stmt, ArrayNode):
            # Improved array node handling
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)

            if size is None:
                # HLSL dynamic arrays need a size, but can be accessed with buffer types
                # For basic shaders, use a fixed size as fallback
                return f"{indent_str}{element_type}[1024] {stmt.name};\n"
            else:
                return f"{indent_str}{element_type}[{size}] {stmt.name};\n"

        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"

        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)

        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                # Handle both single values and lists
                if isinstance(stmt.value, list):
                    # Multiple return values
                    code = ""
                    for i, return_stmt in enumerate(stmt.value):
                        code += f"{self.generate_expression(return_stmt)}"
                        if i < len(stmt.value) - 1:
                            code += ", "
                    return f"{indent_str}return {code};\n"
                else:
                    # Single return value
                    return (
                        f"{indent_str}return {self.generate_expression(stmt.value)};\n"
                    )
            else:
                # Void return
                return f"{indent_str}return;\n"

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(stmt, "expression"):
                return f"{indent_str}{self.generate_expression(stmt.expression)};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        else:
            # Try to generate as expression
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_expression(node.target)
            rhs = self.generate_expression(node.value)
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_expression(node.left)
            rhs = self.generate_expression(node.right)
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent

        # Handle both old and new AST if structures
        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        then_body = getattr(node, "then_branch", getattr(node, "if_body", []))
        else_body = getattr(node, "else_branch", getattr(node, "else_body", []))

        code = f"{indent_str}if ({self.generate_expression(condition)}) {{\n"

        # Handle then body
        if hasattr(then_body, "statements"):
            # BlockNode structure
            for stmt in then_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(then_body, list):
            # List of statements
            for stmt in then_body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            # Single statement
            code += self.generate_statement(then_body, indent + 1)

        code += f"{indent_str}}}"

        # Handle else if conditions (old AST)
        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
                for stmt in else_if_body:
                    code += self.generate_statement(stmt, indent + 1)
                code += f"{indent_str}}}"

        # Handle else body
        if else_body:
            code += " else {\n"
            if hasattr(else_body, "statements"):
                # BlockNode structure
                for stmt in else_body.statements:
                    code += self.generate_statement(stmt, indent + 1)
            elif isinstance(else_body, list):
                # List of statements
                for stmt in else_body:
                    code += self.generate_statement(stmt, indent + 1)
            else:
                # Single statement
                code += self.generate_statement(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # Handle for loop components
        init = ""
        condition = ""
        update = ""

        if hasattr(node, "init") and node.init:
            if isinstance(node.init, str):
                init = node.init
            else:
                init = self.generate_expression(node.init).strip().rstrip(";")

        if hasattr(node, "condition") and node.condition:
            if isinstance(node.condition, str):
                condition = node.condition
            else:
                condition = self.generate_expression(node.condition).strip().rstrip(";")

        if hasattr(node, "update") and node.update:
            if isinstance(node.update, str):
                update = node.update
            else:
                update = self.generate_expression(node.update).strip().rstrip(";")

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

        # Handle body
        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            # BlockNode structure
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            # List of statements
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            # Single statement
            code += self.generate_statement(body, indent + 1)

        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr):
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float)):
            return str(expr)
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
        elif isinstance(expr, VariableNode):
            # Variable reference, just return the name
            return expr.name
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            # Handle BinaryOpNode
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({left} {self.map_operator(op)} {right})"
        elif isinstance(expr, AssignmentNode):
            # Handle assignment as expression
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            # Handle UnaryOpNode
            operand = self.generate_expression(getattr(expr, "operand", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"{self.map_operator(op)}{operand}"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            # Handle ArrayAccessNode
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]"
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            # Handle FunctionCallNode
            func_name = getattr(expr, "function", getattr(expr, "name", "unknown"))
            if hasattr(func_name, "name"):
                func_name = func_name.name
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            # Handle special vector constructor calls
            if func_name in ["vec2", "vec3", "vec4"]:
                mapped_type = self.map_type(func_name)
                args_str = ", ".join(self.generate_expression(arg) for arg in args)
                return f"{mapped_type}({args_str})"
            # Standard function call
            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{func_name}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            # Handle MemberAccessNode
            obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
            member = getattr(expr, "member", "")
            obj = self.generate_expression(obj_expr)
            return f"{obj}.{member}"
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            # Handle TernaryOpNode
            condition = self.generate_expression(getattr(expr, "condition", ""))
            true_expr = self.generate_expression(getattr(expr, "true_expr", ""))
            false_expr = self.generate_expression(getattr(expr, "false_expr", ""))
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            # Fallback - return string representation
            return str(expr)

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
                    return f"float{type_node.rows}x{type_node.rows}"
                else:
                    return f"float{type_node.cols}x{type_node.rows}"
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
                # VectorType - map to proper DirectX vector types
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size

                # Map to DirectX vector types
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type == "uint":
                    return f"uint{size}"
                elif element_type == "bool":
                    return f"bool{size}"
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

    def map_type(self, vtype):
        """Map types to DirectX equivalents, handling both strings and TypeNode objects."""
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
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "ASSIGN_OR": "|=",
            "ASSIGN_XOR": "^=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic:
            return f": {self.semantic_map.get(semantic, semantic)}"
        else:
            return ""  # Handle None by returning an empty string
