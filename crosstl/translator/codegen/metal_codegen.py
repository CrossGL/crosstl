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


class CharTypeMapper:
    def map_char_type(self, vtype):
        char_type_mapping = {
            "char": "int",
            "signed char": "int",
            "unsigned char": "uint",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
        }
        return char_type_mapping.get(vtype, vtype)


class MetalCodeGen:
    def __init__(self):
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None
        self.gl_position = False
        self.char_mapper = CharTypeMapper()
        self.texture_variables = []
        self.sampler_variables = []
        self.type_mapping = {
            # Scalar Types
            "void": "void",
            "short": "int",
            "signed short": "int",
            "unsigned short": "uint",
            "int": "int",
            "signed int": "int",
            "unsigned int": "uint",
            "long": "int64_t",
            "signed long": "int64_t",
            "unsigned long": "uint64_t",
            "float": "float",
            "half": "half",
            "bool": "bool",
            # Vector Types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "short2": "int2",
            "short3": "int3",
            "short4": "int4",
            "ushort2": "uint2",
            "ushort3": "uint3",
            "ushort4": "uint4",
            "int2": "int2",
            "int3": "int3",
            "int4": "int4",
            "uint2": "uint2",
            "uint3": "uint3",
            "uint4": "uint4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "float2": "float2",
            "float3": "float3",
            "float4": "float4",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "bool2": "bool2",
            "bool3": "bool3",
            "bool4": "bool4",
            "sampler2D": "texture2d<float>",
            "samplerCube": "texturecube<float>",
            # Matrix Types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "half2x2": "half2x2",
            "half3x3": "half3x3",
            "half4x4": "half4x4",
        }

        self.semantic_map = {
            # Vertex inputs
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_IsFrontFace": "is_front_facing",
            "gl_PrimitiveID": "primitive_id",
            "POSITION": "attribute(0)",
            "NORMAL": "attribute(1)",
            "TANGENT": "attribute(2)",
            "BINORMAL": "attribute(3)",
            "TEXCOORD": "attribute(4)",
            "TEXCOORD0": "attribute(5)",
            "TEXCOORD1": "attribute(6)",
            "TEXCOORD2": "attribute(7)",
            "TEXCOORD3": "attribute(8)",
            "TEXCOORD4": "attribute(9)",
            "TEXCOORD5": "attribute(10)",
            "TEXCOORD6": "attribute(11)",
            "TEXCOORD7": "attribute(12)",
            # Vertex outputs
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            # Fragment inputs
            "gl_FragColor": "[[color(0)]]",
            "gl_FragColor0": "[[color(0)]]",
            "gl_FragColor1": "[[color(1)]]",
            "gl_FragColor2": "[[color(2)]]",
            "gl_FragColor3": "[[color(3)]]",
            "gl_FragColor4": "[[color(4)]]",
            "gl_FragColor5": "[[color(5)]]",
            "gl_FragColor6": "[[color(6)]]",
            "gl_FragColor7": "[[color(7)]]",
            "gl_FragDepth": "depth(any)",
            # Additional Metal-specific attributes
            "gl_FragCoord": "position",
            "gl_FrontFacing": "is_front_facing",
            "gl_PointCoord": "point_coord",
            # Compute shader specific
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "gl_WorkGroupSize": "threadgroup_size",
            "gl_NumWorkGroups": "threads_per_grid",
        }

    def generate(self, ast):
        code = "\n"
        code += "#include <metal_stdlib>\n"
        code += "using namespace metal;\n"
        code += "\n"

        # Generate structs - handle both old and new AST
        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        # Handle array types in structs
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in Metal use array<type>
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
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
                                    # For Metal, use C-style array syntax: type name[size]
                                    semantic_attr = (
                                        self.map_semantic(semantic) if semantic else ""
                                    )
                                    code += f"    {element_type} {member.name}[{size_str}]{semantic_attr};\n"
                                else:
                                    # Dynamic arrays - use array<type> syntax
                                    semantic_attr = (
                                        self.map_semantic(semantic) if semantic else ""
                                    )
                                    code += f"    array<{element_type}> {member.name}{semantic_attr};\n"
                                continue  # Skip the normal member_type handling
                            else:
                                # Regular type - use convert_type_node_to_string instead of str()
                                member_type_str = self.convert_type_node_to_string(
                                    member.member_type
                                )
                                member_type = self.map_type(member_type_str)
                        elif hasattr(member, "vtype"):
                            # Old AST structure
                            member_type = self.map_type(member.vtype)
                        else:
                            member_type = "float"

                        # Handle semantic - get from attributes in new AST
                        semantic = None
                        if hasattr(member, "semantic"):
                            semantic = member.semantic
                        elif hasattr(member, "attributes"):
                            # Extract semantic from attributes - handle all semantic types
                            for attr in member.attributes:
                                if hasattr(attr, "name"):
                                    semantic = attr.name
                                    break

                        semantic_attr = self.map_semantic(semantic) if semantic else ""
                        code += f"    {member_type} {member.name}{semantic_attr};\n"
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

            if vtype in ["sampler2D", "samplerCube"]:
                self.texture_variables.append((node, i))
            elif vtype in ["sampler"]:
                self.sampler_variables.append((node, i))
            else:
                code += f"{self.map_type(vtype)} {node.name}{array_suffix};\n"

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
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += f"{node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in buffer blocks
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    {member_type} {member.name};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"{node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in buffer blocks
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(str(member.member_type))
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"    {member_type} {member.name};\n"
                code += "};\n"

        for i, node in enumerate(cbuffers):
            if isinstance(node, StructNode) or hasattr(node, "name"):
                code += f"constant {node.name} &{node.name} [[buffer({i})]];\n"
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

            # Extract semantic from parameter attributes
            semantic = None
            if hasattr(p, "attributes") and p.attributes:
                for attr in p.attributes:
                    if hasattr(attr, "name"):
                        semantic = attr.name
                        break

            # Use semantic-specific attribute or default to [[stage_in]]
            param_attr = self.map_semantic(semantic) if semantic else " [[stage_in]]"
            params.append(f"{param_type} {p.name}{param_attr}")

        params_str = ", ".join(params)

        # Handle return type - support both old and new AST
        if hasattr(func, "return_type"):
            if hasattr(func.return_type, "name"):
                return_type = self.map_type(func.return_type.name)
            else:
                # Use convert_type_node_to_string instead of str() for TypeNode objects
                return_type_str = self.convert_type_node_to_string(func.return_type)
                return_type = self.map_type(return_type_str)
        else:
            return_type = "void"

        if shader_type == "vertex":
            code += f"vertex {return_type} vertex_{func.name}({params_str}) {{\n"
        elif shader_type == "fragment":
            if self.texture_variables:
                for texture_variable, i in self.texture_variables:
                    params_str += (
                        f" , texture2d<float> {texture_variable.name} [[texture({i})]]"
                    )
            if self.sampler_variables:
                for sampler_variable, i in self.sampler_variables:
                    params_str += f" , sampler {sampler_variable.name} [[sampler({i})]]"
            code += f"fragment {return_type} fragment_{func.name}({params_str}) {{\n"
        elif shader_type == "compute":
            code += f"kernel {return_type} kernel_{func.name}({params_str}) {{\n"
        else:
            # Handle semantic - get from attributes in new AST
            semantic = None
            if hasattr(func, "semantic"):
                semantic = func.semantic
            elif hasattr(func, "attributes"):
                for attr in func.attributes:
                    if hasattr(attr, "name"):
                        semantic = attr.name
                        break
            code += f"{return_type} {func.name}({params_str}) {self.map_semantic(semantic)} {{\n"

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
            # Improved array node handling
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)

            if size is None:
                # Dynamic arrays in Metal need a size, use a large enough buffer
                return f"{indent_str}device array<{element_type}, 1024> {stmt.name};\n"
            else:
                return f"{indent_str}array<{element_type}, {size}> {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
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
                return f"{indent_str}return {self.generate_expression(stmt.value)};\n"
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            # Handle ExpressionStatementNode
            expr_code = self.generate_expression_statement(stmt)
            return f"{indent_str}{expr_code};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        if hasattr(stmt, "expression"):
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            # Fallback for direct expression
            return self.generate_expression(stmt)

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
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

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

        code += f"{indent_str}}}"

        # Handle else branch - check if it's another if statement (else-if chain)
        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            # Check if else branch is another IfNode (else-if chain)
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate else if by recursively generating the nested if with else if prefix
                elif_condition = self.generate_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if ({elif_condition}) {{\n"

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

                code += f"{indent_str}}}"

                # Recursively handle any remaining else-if chain
                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another else if - recursively handle
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "else if"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if ("):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if (", " else if (", 1
                            )
                        code += "\n".join(
                            remaining_lines[1:]
                        )  # Skip first line as we already handled it
                    else:
                        # Final else clause
                        code += " else {\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
                        code += f"{indent_str}}}"
            else:
                # Regular else clause
                code += " else {\n"
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
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()[
            :-1
        ]  # Remove trailing semicolon

        condition = self.generate_expression(node.condition)

        update = self.generate_expression(node.update)

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

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

        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr):
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, VariableNode):
            # Fix infinite recursion - directly return the name
            if hasattr(expr, "name"):
                return expr.name
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"{left} {self.map_operator(expr.op)} {right}"
        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"{left} {self.map_operator(expr.operator)} {right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"{self.map_operator(expr.op)}{operand}"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, FunctionCallNode):
            # Extract function name properly (might be IdentifierNode)
            func_name = expr.name
            if hasattr(func_name, "name"):
                # It's an IdentifierNode, extract the name
                func_name = func_name.name
            elif not isinstance(func_name, str):
                # Convert to string if it's some other type
                func_name = str(func_name)

            # Special handling for texture sampling
            if func_name == "texture":
                # texture() is used for sampling in GLSL, but in Metal we need to use sample() method
                if len(expr.args) >= 2:
                    texture_name = self.generate_expression(expr.args[0])
                    coord = self.generate_expression(expr.args[1])

                    # Handle texture sampling in Metal
                    if isinstance(expr.args[0], str) and expr.args[0] in [
                        v[0].name for v in self.texture_variables
                    ]:
                        # Check if we have a sampler with the same name
                        sampler_arg = ""
                        for s in self.sampler_variables:
                            if s[0].name == texture_name + "Sampler":
                                sampler_arg = s[0].name
                                break

                        # If no explicit sampler, use the default sampler
                        if not sampler_arg:
                            return f"{texture_name}.sample(sampler(mag_filter::linear, min_filter::linear), {coord})"
                        else:
                            return f"{texture_name}.sample({sampler_arg}, {coord})"
                    else:
                        # Fallback to standard texture function if not a texture variable
                        args = ", ".join(
                            self.generate_expression(arg) for arg in expr.args
                        )
                        return f"{texture_name}.sample(sampler(mag_filter::linear, min_filter::linear), {coord})"
                else:
                    # Handle incomplete texture call more gracefully
                    args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                    return f"texture({args})"
            # Special handling for common GLSL functions
            elif func_name == "normalize":
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"normalize({args})"
            elif func_name in ["mix", "clamp", "smoothstep", "step", "dot", "cross"]:
                # These function names are the same in GLSL and Metal
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"
            # Vector constructors
            elif func_name in ["vec2", "vec3", "vec4"]:
                # Map to Metal's float2, float3, float4
                metal_type = self.map_type(func_name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{metal_type}({args})"
            else:
                # Standard function call
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition)} ? {self.generate_expression(expr.true_expr)} : {self.generate_expression(expr.false_expr)}"
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
        else:
            return str(expr)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        # Handle different TypeNode types
        if hasattr(type_node, "name"):
            # PrimitiveType or NamedType
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            # MatrixType - check this first before element_type + size check
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"float{type_node.rows}x{type_node.rows}"
            else:
                return f"float{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # Check if it's VectorType vs ArrayType
            if str(type(type_node)).find("ArrayType") != -1:
                # ArrayType - handle C-style arrays
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        # Size is an expression node - prevent infinite recursion
                        size_str = self.safe_expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                # VectorType - map to proper Metal vector types
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size

                # Map to Metal vector types
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

    def safe_expression_to_string(self, expr):
        """Convert an expression node to a string representation safely (avoid infinite recursion)."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif hasattr(expr, "name"):
            return str(expr.name)
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, str):
            return expr
        else:
            # Fallback - avoid calling generate_expression to prevent infinite recursion
            return str(expr)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        return self.safe_expression_to_string(expr)

    def map_type(self, vtype):
        """Map types to Metal equivalents, handling both strings and TypeNode objects."""
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
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic is not None:
            mapped_semantic = self.semantic_map.get(semantic, semantic)
            # If the mapped semantic already has brackets, use it as-is
            if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
                return f" {mapped_semantic}"
            else:
                # Add brackets for Metal attribute syntax
                return f" [[{mapped_semantic}]]"
        else:
            return ""
