from ..ast import (
    AssignmentNode,
    ArrayNode,
    ArrayAccessNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    MeshOpNode,
    PreprocessorNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .array_utils import (
    parse_array_type,
    format_array_type,
    format_c_style_array_declaration,
    split_array_type_suffix,
    get_array_size_from_node,
    evaluate_literal_int_expression,
    collect_literal_int_constants,
)


class GLSLCodeGen:
    def __init__(self):
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.function_sampler_parameter_indices = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
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
            "sampler": "sampler",
            "sampler1D": "sampler1D",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCube": "samplerCube",
            "sampler2DArray": "sampler2DArray",
            "samplerCubeArray": "samplerCubeArray",
            "sampler2DMS": "sampler2DMS",
            "sampler2DMSArray": "sampler2DMSArray",
            "sampler2DShadow": "sampler2DShadow",
            "sampler2DArrayShadow": "sampler2DArrayShadow",
            "samplerCubeShadow": "samplerCubeShadow",
            "samplerCubeArrayShadow": "samplerCubeArrayShadow",
            "iimage2D": "iimage2D",
            "iimage3D": "iimage3D",
            "iimage2DArray": "iimage2DArray",
            "uimage2D": "uimage2D",
            "uimage3D": "uimage3D",
            "uimage2DArray": "uimage2DArray",
            "image2D": "image2D",
            "image3D": "image3D",
            "imageCube": "imageCube",
            "image2DArray": "image2DArray",
        }

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
            "tex2Doffset": "textureOffset",
            "texCUBE": "texture",
            "texCUBElod": "textureLod",
            "texCUBEbias": "texture",
            "texCUBEgrad": "textureGrad",
            "textureOffset": "textureOffset",
            "textureProj": "textureProj",
            "textureGatherOffset": "textureGatherOffset",
            "textureGatherOffsets": "textureGatherOffsets",
            "textureQueryLevels": "textureQueryLevels",
            "textureQueryLod": "textureQueryLod",
            "texelFetch": "texelFetch",
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "atomicCounterIncrement": "atomicCounterIncrement",
            "atomicCounterDecrement": "atomicCounterDecrement",
            "atomicCounter": "atomicCounter",
            "atomicCounterAdd": "atomicCounterAdd",
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
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.function_sampler_parameter_indices = (
            self.collect_function_sampler_parameter_indices(ast)
        )
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        version_line = None
        extra_lines = []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                if directive.directive == "precision":
                    line = (
                        f"precision {directive.content};"
                        if directive.content
                        else "precision;"
                    )
                else:
                    line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line.startswith("#version") and version_line is None:
                version_line = line
            elif line:
                extra_lines.append(line)
        if version_line is None:
            version_line = "#version 450 core"
        code += f"{version_line}\n"
        if extra_lines:
            code += "\n".join(extra_lines) + "\n"
        code += self.generate_constants(ast)

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                if node.name == "VSInput":
                    members = getattr(node, "members", [])
                    for member in members:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )

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
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"out {member_type} {member.name};\n"
                elif node.name == "PSInput":
                    members = getattr(node, "members", [])
                    for member in members:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        code += f"in {member_type} {member.name};\n"
                elif node.name == "PSOutput":
                    members = getattr(node, "members", [])
                    for member in members:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )

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

        global_vars = getattr(ast, "global_variables", [])
        binding = 0
        for index, node in enumerate(global_vars):
            # Handle both old and new AST variable structures
            resource_count = 1
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
                            else (
                                self.resource_array_size_hints.get(node.name, "")
                                if self.is_inferable_resource_array_type(base_type)
                                else ""
                            )
                        )
                        vtype = base_type
                        array_suffix = f"[{array_size}]" if array_size else "[]"
                        resource_count = self.resource_array_count(
                            node.var_type.size if node.var_type.size else array_size
                        )
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
                var_name = f"var{index}"

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                self.sampler_variables.add(var_name)
                continue
            if self.is_opaque_resource_type(mapped_type):
                self.texture_variable_types[var_name] = mapped_type
                explicit_format = self.explicit_image_format_qualifier(node)
                if explicit_format:
                    self.image_variable_formats[var_name] = explicit_format
            declaration = format_c_style_array_declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            if self.is_opaque_resource_type(mapped_type):
                layout = self.opaque_resource_layout(mapped_type, binding, node)
                code += f"{layout} uniform {declaration};\n"
            else:
                code += f"layout(std140, binding = {binding}) {declaration};\n"
            binding += (
                resource_count if self.is_opaque_resource_type(mapped_type) else 1
            )

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

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

    def generate_constants(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            code += f"const {self.map_type(const_type)} {name} = {value_code};\n"

        return f"{code}\n" if code else ""

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

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

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        for p in param_list:
            if hasattr(p, "param_type"):
                if hasattr(p.param_type, "name"):
                    raw_param_type = p.param_type.name
                else:
                    raw_param_type = p.param_type
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                continue

            param_type = self.map_resource_parameter_type_with_hint(
                raw_param_type, p, getattr(func, "name", None)
            )
            if self.is_opaque_resource_type(
                self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                explicit_format = self.explicit_image_format_qualifier(p)
                if explicit_format:
                    image_format_parameters[p.name] = explicit_format

            semantic = self.semantic_from_node(p)

            declaration = format_c_style_array_declaration(param_type, p.name)
            semantic_attr = self.map_semantic(semantic)
            params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )

        params_str = ", ".join(params)

        stage_entry_types = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
        }

        if shader_type in stage_entry_types:
            code += "void main() {\n"
        else:
            if hasattr(func, "return_type"):
                if hasattr(func.return_type, "name"):
                    return_type = self.map_type(func.return_type.name)
                else:
                    return_type = self.map_type(func.return_type)
            else:
                return_type = "void"

            code += f"{return_type} {func.name}({params_str}) {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_image_format_parameters = self.current_image_format_parameters
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = texture_parameters
        self.current_image_format_parameters = image_format_parameters
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_image_format_parameters = previous_image_format_parameters

        code += "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if hasattr(stmt, "var_type"):
                var_type = self.convert_type_node_to_string(stmt.var_type)
            elif hasattr(stmt, "vtype"):
                var_type = stmt.vtype
            else:
                var_type = "float"

            declaration = format_c_style_array_declaration(
                self.map_type(var_type), stmt.name
            )
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression(stmt.initial_value)
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
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

        if_body = node.if_body
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.generate_expression(else_if_condition)
                code += f" else if ({condition}) {{\n"
                if hasattr(else_if_body, "statements"):
                    for stmt in else_if_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_if_body, list):
                    for stmt in else_if_body:
                        code += self.generate_statement(stmt, indent + 1)
                code += f"{indent_str}}}"

        if hasattr(node, "else_body") and node.else_body:
            code += f" else {{\n"
            else_body = node.else_body
            if hasattr(else_body, "statements"):
                for stmt in else_body.statements:
                    code += self.generate_statement(stmt, indent + 1)
            elif isinstance(else_body, list):
                for stmt in else_body:
                    code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

        body = node.body
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
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
        elif isinstance(expr, WaveOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayTracingOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, MeshOpNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{expr.operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args = ", ".join(self.generate_expression(arg) for arg in expr.arguments)
            return f"{query}.{expr.operation}({args})"
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
            func_expr = getattr(expr, "function", getattr(expr, "name", expr))
            func_name = None
            if hasattr(func_expr, "name"):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            original_func_name = func_name
            func_name = self.function_map.get(func_name, func_name)

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call

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
                "bvec2",
                "bvec3",
                "bvec4",
            ]:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"

            if func_name in ["mat2", "mat3", "mat4"]:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"

            call_args = self.filter_sampler_arguments(original_func_name, expr.args)
            args = ", ".join(self.generate_expression(arg) for arg in call_args)
            return f"{func_name or callee}({args})"
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

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        return (
            sampler_name in self.sampler_variables
            or sampler_name in self.current_sampler_parameters
        )

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, coord, extra_args

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def image_resource_format(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_image_format_parameters.get(
            texture_name, self.image_variable_formats.get(texture_name)
        )

    def is_integer_image_type(self, texture_type):
        return texture_type in {
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
        }

    def is_scalar_image_format(self, image_format):
        return image_format in {
            "r8",
            "r8_snorm",
            "r16",
            "r16_snorm",
            "r16f",
            "r32f",
            "r8i",
            "r16i",
            "r32i",
            "r8ui",
            "r16ui",
            "r32ui",
        }

    def is_two_component_image_format(self, image_format):
        return image_format in {
            "rg8",
            "rg8_snorm",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg8i",
            "rg16i",
            "rg8ui",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
        }

    def is_scalar_integer_image_resource(self, texture_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(texture_type)

    def image_load_component_suffix(self, texture_type, image_format):
        if self.is_scalar_integer_image_resource(texture_type, image_format):
            return ".x"
        if self.is_two_component_image_format(image_format):
            return ".xy"
        return ""

    def image_format_store_constructor(self, image_format):
        return {
            "r8": "vec4",
            "r8_snorm": "vec4",
            "r16": "vec4",
            "r16_snorm": "vec4",
            "r16f": "vec4",
            "r32f": "vec4",
            "r8i": "ivec4",
            "r16i": "ivec4",
            "r32i": "ivec4",
            "r8ui": "uvec4",
            "r16ui": "uvec4",
            "r32ui": "uvec4",
        }.get(image_format)

    def integer_image_store_constructor(self, texture_type):
        if texture_type in {"iimage2D", "iimage3D", "iimage2DArray"}:
            return "ivec4"
        if texture_type in {"uimage2D", "uimage3D", "uimage2DArray"}:
            return "uvec4"
        return None

    def two_component_image_store_expression(self, image_format, value):
        constructors = {
            "rg8": ("vec4", "0.0"),
            "rg8_snorm": ("vec4", "0.0"),
            "rg16": ("vec4", "0.0"),
            "rg16_snorm": ("vec4", "0.0"),
            "rg16f": ("vec4", "0.0"),
            "rg8i": ("ivec4", "0"),
            "rg16i": ("ivec4", "0"),
            "rg8ui": ("uvec4", "0u"),
            "rg16ui": ("uvec4", "0u"),
            "rg32f": ("vec4", "0.0"),
            "rg32i": ("ivec4", "0"),
            "rg32ui": ("uvec4", "0u"),
        }
        constructor = constructors.get(image_format)
        if constructor is None:
            return None
        type_name, zero_value = constructor
        return f"{type_name}({value}, {zero_value}, {zero_value})"

    def image_store_value_expression(self, texture_type, image_format, value):
        two_component_value = self.two_component_image_store_expression(
            image_format, value
        )
        if two_component_value is not None:
            return two_component_value

        constructor = None
        if self.is_scalar_integer_image_resource(texture_type, image_format):
            constructor = self.integer_image_store_constructor(texture_type)
            if constructor is None:
                constructor = self.image_format_store_constructor(image_format)
        if constructor:
            return f"{constructor}({value})"
        return value

    def generate_texture_call(self, func_name, args):
        if not func_name or len(args) < 2:
            return None

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            load_expr = f"imageLoad({image_name}, {coord})"
            image_format = self.image_resource_format(args[0])
            return f"{load_expr}{self.image_load_component_suffix(texture_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            texture_type = self.texture_resource_type(args[0])
            image_format = self.image_resource_format(args[0])
            value = self.image_store_value_expression(texture_type, image_format, value)
            return f"imageStore({image_name}, {coord}, {value})"

        if func_name == "textureCompare":
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, extra_args = parts
            if not extra_args:
                return None
            compare = self.generate_expression(extra_args[0])
            texture_type = self.texture_resource_type(args[0])
            if texture_type == "samplerCubeArrayShadow":
                return f"texture({texture_name}, {coord}, {compare})"
            compare_constructor = (
                "vec4"
                if texture_type
                in {
                    "sampler2DArrayShadow",
                    "samplerCubeShadow",
                }
                else "vec3"
            )
            return f"texture({texture_name}, {compare_constructor}({coord}, {compare}))"

        if func_name == "textureQueryLod" and self.is_explicit_sampler_argument(args):
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, _ = parts
            return f"textureQueryLod({texture_name}, {coord})"

        texture_funcs = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureGather",
            "textureOffset",
            "textureProj",
            "textureGatherOffset",
            "textureGatherOffsets",
        }
        if func_name not in texture_funcs or not self.is_explicit_sampler_argument(
            args
        ):
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None
        texture_name, coord, extra_args = parts
        mapped_args = [texture_name, coord] + [
            self.generate_expression(arg) for arg in extra_args
        ]
        return f"{func_name}({', '.join(mapped_args)})"

    def collect_function_sampler_parameter_indices(self, root):
        sampler_indices = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            name = getattr(value, "name", None)
            params = getattr(value, "parameters", getattr(value, "params", []))
            if name and params:
                indices = []
                for index, param in enumerate(params):
                    param_type = getattr(
                        param, "param_type", getattr(param, "vtype", None)
                    )
                    if self.is_sampler_type(param_type):
                        indices.append(index)
                if indices:
                    sampler_indices[name] = set(indices)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_indices

    def filter_sampler_arguments(self, func_name, args):
        sampler_indices = self.function_sampler_parameter_indices.get(func_name, set())
        if not sampler_indices:
            return args
        return [arg for index, arg in enumerate(args) if index not in sampler_indices]

    def collect_resource_array_size_hints(self, ast):
        global_arrays = self.collect_unsized_sampled_texture_globals(ast)
        function_arrays = self.collect_unsized_sampled_texture_parameters(ast)
        global_hints = {name: 0 for name in global_arrays}
        function_hints = {
            func_name: {param_name: 0 for param_name in params}
            for func_name, params in function_arrays.items()
        }
        functions = {
            getattr(func, "name", None): func for func in self.collect_functions(ast)
        }
        functions = {name: func for name, func in functions.items() if name}

        for func_name, func in functions.items():
            visible_constants = self.visible_literal_int_constants(func)
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, ArrayAccessNode):
                    continue
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                array_name = self.expression_name(array_expr)
                index = self.literal_int_value(index_expr, visible_constants)
                if array_name is None or index is None or index < 0:
                    continue
                required_size = index + 1
                if array_name in global_hints:
                    global_hints[array_name] = max(
                        global_hints[array_name], required_size
                    )
                if array_name in function_hints.get(func_name, {}):
                    function_hints[func_name][array_name] = max(
                        function_hints[func_name][array_name], required_size
                    )

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions.items():
                caller_param_hints = function_hints.get(caller_name, {})
                for call in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee_param_hints = function_hints.get(callee_name)
                    if not callee_param_hints:
                        continue
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_params = getattr(callee, "parameters", [])
                    args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            continue
                        required_size = callee_param_hints.get(
                            getattr(callee_params[index], "name", None)
                        )
                        if not required_size:
                            continue
                        arg_name = self.expression_name(arg)
                        if (
                            arg_name in global_hints
                            and required_size > global_hints[arg_name]
                        ):
                            global_hints[arg_name] = required_size
                            changed = True
                        if (
                            arg_name in caller_param_hints
                            and required_size > caller_param_hints[arg_name]
                        ):
                            caller_param_hints[arg_name] = required_size
                            changed = True

        return (
            {
                name: str(size) if size > 1 else ""
                for name, size in global_hints.items()
            },
            {
                func_name: {
                    param_name: str(size) if size > 1 else ""
                    for param_name, size in param_hints.items()
                }
                for func_name, param_hints in function_hints.items()
            },
        )

    def collect_unsized_sampled_texture_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_sampled_texture_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_unsized_sampled_texture_parameters(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_sampled_texture_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def is_unsized_sampled_texture_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_inferable_resource_array_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_inferable_resource_array_type(base_type)

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def walk_ast(self, root):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    yield from walk(child)

        yield from walk(root)

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) == "sampler"

    def is_sampled_texture_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return (
            mapped_type != "sampler"
            and self.is_opaque_resource_type(mapped_type)
            and not mapped_type.startswith(("image", "iimage", "uimage"))
        )

    def is_storage_image_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return self.is_opaque_resource_type(mapped_type) and mapped_type.startswith(
            ("image", "iimage", "uimage")
        )

    def is_inferable_resource_array_type(self, vtype):
        return self.is_sampled_texture_type(vtype) or self.is_storage_image_type(vtype)

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def visible_literal_int_constants(self, func):
        if not self.literal_int_constants:
            return {}

        shadowed = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", []) or []
        }
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                shadowed.add(getattr(node, "name", None))

        shadowed.discard(None)
        if not shadowed:
            return self.literal_int_constants
        return {
            name: value
            for name, value in self.literal_int_constants.items()
            if name not in shadowed
        }

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None

    def map_type(self, vtype):
        """Map types to GLSL equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.type_mapping.get(base_type, base_type)
            return f"{base_mapped}{array_suffix}"

        return self.type_mapping.get(vtype_str, vtype_str)

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_inferable_resource_array_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return (
                    f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"
                )

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_inferable_resource_array_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return (
                            f"{mapped_type}[{array_size}]"
                            if array_size
                            else f"{mapped_type}[]"
                        )
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = (
            self.explicit_image_format_qualifier(node) if node is not None else None
        )
        if explicit_format in {
            "r8",
            "r8_snorm",
            "r16",
            "r16_snorm",
            "r16f",
            "r32f",
            "r8i",
            "r16i",
            "r32i",
            "r8ui",
            "r16ui",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg8i",
            "rg16i",
            "rg8ui",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba32f",
            "rgba8i",
            "rgba16i",
            "rgba32i",
            "rgba8ui",
            "rgba16ui",
            "rgba32ui",
        }:
            if explicit_format in {
                "r8",
                "r8_snorm",
                "r16",
                "r16_snorm",
                "r16f",
                "rg8",
                "rg8_snorm",
                "rg16",
                "rg16_snorm",
                "rg16f",
                "rg32f",
                "rgba8",
                "rgba8_snorm",
                "rgba16",
                "rgba16_snorm",
                "rgba16f",
                "rgba32f",
            }:
                format_class = "r32f"
            elif explicit_format.endswith("ui"):
                format_class = "r32ui"
            elif explicit_format.endswith("i"):
                format_class = "r32i"
            else:
                format_class = explicit_format
            format_type_map = {
                "image2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "iimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "uimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "image3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "iimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "uimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "image2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "iimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "uimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "imageCube": {"r32f": "imageCube"},
            }
            mapped_type = format_type_map.get(base_type, {}).get(format_class)
            if mapped_type:
                return mapped_type

        return self.map_type(vtype)

    def is_opaque_resource_type(self, vtype):
        return vtype in {
            "sampler1D",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "sampler2DRect",
            "samplerBuffer",
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler2D",
            "usampler2D",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
            "imageBuffer",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "atomic_uint",
        }

    def supported_image_formats(self):
        return {
            "r8",
            "r8_snorm",
            "r8i",
            "r8ui",
            "r16",
            "r16_snorm",
            "r16f",
            "r16i",
            "r16ui",
            "r32f",
            "r32i",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg8i",
            "rg8ui",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg16i",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba8i",
            "rgba8ui",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba16i",
            "rgba16ui",
            "rgba32f",
            "rgba32i",
            "rgba32ui",
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name"):
            return str(value.name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def explicit_image_format_qualifier(self, node):
        if not hasattr(node, "attributes"):
            return None
        supported_formats = self.supported_image_formats()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name
            if attr_name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                format_name = self.attribute_value_to_string(arguments[0])
                if format_name is None:
                    continue
                format_name = str(format_name).lower()
                if format_name in supported_formats:
                    return format_name
        return None

    def is_image_format_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        attr_name = str(attr_name).lower()
        return attr_name == "format" or attr_name in self.supported_image_formats()

    def semantic_from_node(self, node):
        if hasattr(node, "semantic"):
            return node.semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if self.is_image_format_attribute(attr):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def image_format_qualifier(self, vtype, node=None):
        explicit_format = self.explicit_image_format_qualifier(node)
        if explicit_format:
            return explicit_format
        if vtype in {
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
        }:
            return "rgba32f"
        if vtype in {
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
        }:
            return "r32i"
        if vtype in {
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
        }:
            return "r32ui"
        return None

    def opaque_resource_layout(self, vtype, binding, node=None):
        image_format = self.image_format_qualifier(vtype, node)
        if image_format:
            return f"layout({image_format}, binding = {binding})"
        return f"layout(binding = {binding})"

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
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.rows == type_node.cols:
                    return f"mat{type_node.rows}"
                else:
                    return f"mat{type_node.cols}x{type_node.rows}"
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        size_str = self.expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
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
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
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
                if hasattr(member, "member_type"):
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
                        member_type_str = self.convert_type_node_to_string(
                            member.member_type
                        )
                        member_type = self.map_type(member_type_str)
                        code += f"    {member_type} {member.name};\n"
                elif hasattr(member, "vtype"):
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
