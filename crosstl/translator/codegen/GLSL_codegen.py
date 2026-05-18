"""CrossGL-to-GLSL code generator."""

from ..ast import (
    AssignmentNode,
    ArrayNode,
    ArrayAccessNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PreprocessorNode,
    RayQueryOpNode,
    RayTracingOpNode,
    RangeNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import (
    parse_array_type,
    format_array_type,
    format_c_style_array_declaration,
    split_array_type_suffix,
    get_array_size_from_node,
    evaluate_literal_int_expression,
    collect_literal_int_constants,
    collect_struct_member_types,
)
from .stage_utils import (
    compute_local_size,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .resource_arrays import collect_resource_array_size_hints


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
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.flattened_stage_variables = set()
        self.structs_by_name = {}
        self.vertex_input_struct_names = set()
        self.vertex_output_struct_names = set()
        self.fragment_input_struct_names = set()
        self.vertex_input_member_names = set()
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = {}
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
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)

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
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.flattened_stage_variables = set()
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = collect_struct_member_types(
            getattr(ast, "structs", []), self.type_name_string
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
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        self.vertex_input_struct_names = self.stage_parameter_struct_names(
            ast, "vertex"
        )
        self.vertex_output_struct_names = self.stage_return_struct_names(ast, "vertex")
        self.fragment_input_struct_names = self.stage_parameter_struct_names(
            ast, "fragment"
        )
        self.vertex_input_member_names = self.struct_member_names(
            self.vertex_input_struct_names
        )
        emit_vertex_io = target_stage in {None, "vertex"}
        emit_fragment_io = target_stage in {None, "fragment"}
        emit_graphics_io = target_stage in {None, "vertex", "fragment"}
        for node in structs:
            if isinstance(node, StructNode):
                if (
                    node.name == "VSInput"
                    or node.name in self.vertex_input_struct_names
                ):
                    if emit_vertex_io:
                        code += self.generate_stage_input_declarations(node)
                    elif not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name == "VSOutput":
                    emitted_io = False
                    if node.name in self.vertex_output_struct_names and emit_vertex_io:
                        code += self.generate_vertex_output_declarations(node)
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_input_declarations(node)
                        emitted_io = True
                    if not emitted_io:
                        if emit_graphics_io:
                            code += self.generate_legacy_output_declarations(node)
                        else:
                            code += self.generate_struct(node)
                elif node.name in self.vertex_output_struct_names:
                    if emit_vertex_io:
                        code += self.generate_vertex_output_declarations(node)
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.generate_fragment_input_declarations(node)
                    code += self.generate_struct(node)
                elif node.name == "PSInput":
                    if emit_fragment_io:
                        code += self.generate_fragment_input_declarations(node)
                    elif not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name in self.fragment_input_struct_names:
                    if emit_fragment_io:
                        code += self.generate_fragment_input_declarations(node)
                    if not emit_graphics_io:
                        code += self.generate_struct(node)
                elif node.name == "PSOutput":
                    if emit_graphics_io:
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
            qualifier_name = normalize_stage_name(qualifier)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if qualifier_name == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier_name == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier_name == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = normalize_stage_name(stage_type)
                    if not stage_matches(target_stage, stage_name):
                        continue
                    code += f"// {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                    )
                if hasattr(stage, "local_functions"):
                    stage_name = normalize_stage_name(stage_type)
                    if not stage_matches(target_stage, stage_name):
                        continue
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

    def generate_compute_layout(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return (
            f"layout(local_size_x = {x}, "
            f"local_size_y = {y}, "
            f"local_size_z = {z}) in;\n"
        )

    def generate_function(
        self, func, indent=0, shader_type=None, execution_config=None
    ):
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {}
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
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)

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

        stage_output = self.fragment_stage_output(func, shader_type)
        if stage_output and stage_output["declaration"]:
            code += f"{stage_output['declaration']}\n"

        if shader_type == "compute":
            code += self.generate_compute_layout(execution_config)

        if shader_type in stage_entry_types:
            code += "void main() {\n"
            self.current_function_return_type = "void"
        else:
            raw_return_type = self.type_name_string(getattr(func, "return_type", None))
            self.current_function_return_type = raw_return_type or "void"
            return_type = self.map_type(self.current_function_return_type)
            code += f"{return_type} {func.name}({params_str}) {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_image_format_parameters = self.current_image_format_parameters
        previous_stage_output = self.current_stage_output
        previous_stage_inputs = self.current_stage_inputs
        previous_stage_outputs = self.current_stage_outputs
        previous_flattened_stage_variables = self.flattened_stage_variables
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = texture_parameters
        self.current_image_format_parameters = image_format_parameters
        self.current_stage_output = stage_output
        self.current_stage_inputs = self.stage_input_member_maps(func, shader_type)
        self.current_stage_outputs = self.stage_output_member_maps(func, shader_type)
        self.flattened_stage_variables = set(self.current_stage_outputs)
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
        self.current_stage_output = previous_stage_output
        self.current_stage_inputs = previous_stage_inputs
        self.current_stage_outputs = previous_stage_outputs
        self.flattened_stage_variables = previous_flattened_stage_variables
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types

        code += "}\n\n"
        return code

    def stage_functions(self, ast, stage_name):
        functions = []

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", []) or []
            qualifier = (
                qualifiers[0] if qualifiers else getattr(func, "qualifier", None)
            )
            if normalize_stage_name(qualifier) == stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            current_stage = normalize_stage_name(stage_type)
            if current_stage == stage_name and hasattr(stage, "entry_point"):
                functions.append(stage.entry_point)

        return functions

    def stage_parameter_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            for param in parameters:
                type_name = self.type_node_name(getattr(param, "param_type", None))
                if type_name in self.structs_by_name:
                    struct_names.add(type_name)
        return struct_names

    def stage_return_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            type_name = self.type_node_name(getattr(func, "return_type", None))
            if type_name in self.structs_by_name:
                struct_names.add(type_name)
        return struct_names

    def struct_member_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                names.add(member.name)
        return names

    def type_node_name(self, type_node):
        if type_node is None:
            return None
        if hasattr(type_node, "name"):
            return type_node.name
        return str(type_node)

    def generate_stage_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            code += f"{prefix}in {member_type} {member.name};\n"
        return code

    def generate_legacy_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            code += f"out {self.member_type_name(member)} {member.name};\n"
        return code

    def generate_vertex_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                continue

            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            code += f"{prefix}out {self.member_type_name(member)} {output_name};\n"
        return code

    def generate_fragment_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            input_name = self.fragment_input_member_name(member, node.name)
            if input_name is None:
                continue

            semantic = self.semantic_from_node(member)
            layout = self.map_semantic(semantic)
            prefix = f"{layout} " if layout.startswith("layout(") else ""
            code += f"{prefix}in {self.member_type_name(member)} {input_name};\n"
        return code

    def member_type_name(self, member):
        if hasattr(member, "member_type"):
            return self.map_type(member.member_type)
        return self.map_type(getattr(member, "vtype", "float"))

    def vertex_output_member_name(self, member):
        semantic = self.semantic_from_node(member)
        mapped_semantic = self.map_semantic(semantic)
        if self.is_vertex_builtin_output(mapped_semantic):
            return mapped_semantic
        if member.name in self.vertex_input_member_names:
            return f"out_{member.name}"
        return member.name

    def fragment_input_member_name(self, member, struct_name):
        if struct_name in self.vertex_output_struct_names:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                return None
            return output_name
        return member.name

    def is_vertex_builtin_output(self, name):
        return name in {"gl_Position", "gl_PointSize", "gl_ClipDistance"}

    def stage_input_member_maps(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        maps = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            type_name = self.type_node_name(getattr(param, "param_type", None))
            struct = self.structs_by_name.get(type_name)
            if struct is None:
                continue
            if shader_type == "fragment":
                member_map = {}
                for member in getattr(struct, "members", []) or []:
                    input_name = self.fragment_input_member_name(member, type_name)
                    if input_name is not None:
                        member_map[member.name] = input_name
                maps[param.name] = member_map
            else:
                maps[param.name] = {
                    member.name: member.name
                    for member in getattr(struct, "members", [])
                }
        return maps

    def stage_output_member_maps(self, func, shader_type):
        if shader_type != "vertex":
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        struct = self.structs_by_name.get(type_name)
        if struct is None:
            return {}

        member_map = {
            member.name: self.vertex_output_member_name(member)
            for member in getattr(struct, "members", []) or []
        }
        maps = {}
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        for stmt in statements:
            if not isinstance(stmt, VariableNode):
                continue
            if self.type_node_name(getattr(stmt, "var_type", None)) == type_name:
                maps[stmt.name] = member_map
        return maps

    def function_return_type(self, func):
        return_type = getattr(func, "return_type", None)
        if return_type is None:
            return "void"
        return self.map_type(return_type)

    def fragment_stage_output(self, func, shader_type):
        if shader_type != "fragment":
            return None

        output_type = self.function_return_type(func)
        if output_type == "void":
            return None

        semantic = self.semantic_from_node(func) or "gl_FragColor"
        if semantic == "gl_FragDepth":
            return {
                "name": "gl_FragDepth",
                "declaration": "",
            }

        layout = self.map_semantic(semantic)
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"

        output_name = self.fragment_output_name(semantic)
        return {
            "name": output_name,
            "declaration": f"{layout} out {output_type} {output_name};",
        }

    def fragment_output_name(self, semantic):
        if semantic and semantic.startswith("gl_FragColor"):
            suffix = semantic.removeprefix("gl_FragColor")
            return f"fragColor{suffix}"
        return "fragColor"

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if stmt.name in self.flattened_stage_variables:
                return ""
            if hasattr(stmt, "var_type"):
                var_type = self.convert_type_node_to_string(stmt.var_type)
            elif hasattr(stmt, "vtype"):
                var_type = stmt.vtype
            else:
                var_type = "float"
            self.local_variable_types[stmt.name] = var_type

            declaration = format_c_style_array_declaration(
                self.map_type(var_type), stmt.name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression_with_expected(
                    stmt.initial_value, var_type
                )
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
        elif isinstance(stmt, BreakNode):
            return f"{indent_str}break;\n"
        elif isinstance(stmt, ContinueNode):
            return f"{indent_str}continue;\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)
        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)
        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)
        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if getattr(stmt, "value", None) is None:
                return f"{indent_str}return;\n"
            return_value_name = self.expression_name(stmt.value)
            if return_value_name in self.flattened_stage_variables:
                return f"{indent_str}return;\n"
            if self.current_stage_output is not None:
                if isinstance(stmt.value, list):
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    value = values
                else:
                    value = self.generate_expression_with_expected(
                        stmt.value, self.current_function_return_type
                    )
                return (
                    f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                    f"{indent_str}return;\n"
                )
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values};\n"
            else:
                return (
                    f"{indent_str}return "
                    f"{self.generate_expression_with_expected(stmt.value, self.current_function_return_type)};\n"
                )
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

    def local_variable_qualifier(self, node):
        return "const " if "const" in getattr(node, "qualifiers", []) else ""

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("dvec"):
            return "double"
        if mapped_type.startswith("uvec"):
            return "uint"
        if mapped_type.startswith("ivec"):
            return "int"
        if mapped_type.startswith("bvec"):
            return "bool"
        if mapped_type.startswith("vec"):
            return "float"
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            return self.local_variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, AssignmentNode):
            target = getattr(expr, "target", getattr(expr, "left", None))
            return self.expression_result_type(target)
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            return None
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            if func_name in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
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
            }:
                return str(func_name)
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

    def generate_assignment(self, node, is_main=False):
        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        left = self.generate_expression(left_node)
        right = self.generate_expression_with_expected(
            right_node, self.expression_result_type(left_node)
        )
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
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

        init = self.generate_for_initializer(getattr(node, "init", None))
        condition = (
            self.generate_expression(node.condition)
            if getattr(node, "condition", None)
            else ""
        )
        update = (
            self.generate_expression(node.update)
            if getattr(node, "update", None)
            else ""
        )

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

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")

        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            comparator = "<=" if iterable_node.inclusive else "<"
            code = (
                f"{indent_str}for (int {pattern} = {start}; "
                f"{pattern} {comparator} {end}; ++{pattern}) {{\n"
            )
        else:
            iterable = self.generate_expression(iterable_node)
            code = (
                f"{indent_str}for (int {pattern} = 0; {pattern} < {iterable}; "
                f"++{pattern}) {{\n"
            )

        code += self.generate_statement_body(getattr(node, "body", []), indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_statement_body(getattr(node, "body", []), indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_statement_body(getattr(node, "body", []), indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for case in getattr(node, "cases", []) or []:
            value = getattr(case, "value", None)
            if value is None:
                code += f"{indent_str}    default:\n"
            else:
                code += f"{indent_str}    case {self.generate_expression(value)}:\n"
            code += self.generate_statement_body(
                getattr(case, "statements", []), indent + 2
            )

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            code += f"{indent_str}    default:\n"
            code += self.generate_statement_body(default_case, indent + 2)

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for arm in getattr(node, "arms", []) or []:
            pattern = getattr(arm, "pattern", None)
            if not self.is_supported_switch_match_arm(arm):
                raise ValueError(
                    "Unsupported match arm for GLSL codegen; only unguarded "
                    "literal and wildcard patterns can be lowered to switch"
                )

            if isinstance(pattern, WildcardPatternNode):
                code += f"{indent_str}    default:\n"
            else:
                code += (
                    f"{indent_str}    case "
                    f"{self.generate_expression(pattern.literal)}:\n"
                )
            body = getattr(arm, "body", [])
            code += self.generate_statement_body(body, indent + 2)
            if not self.statement_body_terminates(body):
                code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        return code

    def is_supported_switch_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def statement_body_terminates(self, body):
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]

        return bool(statements) and isinstance(
            statements[-1], (BreakNode, ContinueNode, ReturnNode)
        )

    def generate_statement_body(self, body, indent):
        code = ""
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent)
        elif body is not None:
            code += self.generate_statement(body, indent)
        return code

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            return self.generate_statement(init, 0).strip().rstrip(";")
        return self.generate_expression(init).strip().rstrip(";")

    def generate_expression(self, expr, is_main=False):
        if expr is None:
            return ""
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
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if (
                literal_type == "uint"
                and isinstance(expr.value, int)
                and not isinstance(expr.value, bool)
            ):
                return f"{expr.value}u"
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
            flattened_member = self.flattened_stage_member_name(expr)
            if flattened_member is not None:
                return flattened_member
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif hasattr(expr, "__class__") and "TernaryOpNode" in str(type(expr)):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def flattened_stage_member_name(self, expr):
        object_name = self.expression_name(getattr(expr, "object", None))
        if object_name in self.current_stage_inputs:
            return self.current_stage_inputs[object_name].get(expr.member)
        if object_name in self.current_stage_outputs:
            return self.current_stage_outputs[object_name].get(expr.member)
        return None

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

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type in {"sampler2DArray", "sampler2DArrayShadow"}:
            return self.vector_component(coord, "xy")
        if texture_type in {"samplerCubeArray", "samplerCubeArrayShadow"}:
            return self.vector_component(coord, "xyz")
        return coord

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_component_value(self, component_arg):
        if component_arg is None:
            return None
        return self.literal_int_value(component_arg, self.literal_int_constants)

    def texture_gather_call_expression(
        self, function_name, texture_name, coord, offset_arg=None, component=None
    ):
        args = [texture_name, coord]
        if offset_arg is not None:
            args.append(offset_arg)
        if component is not None:
            args.append(str(component))
        return f"{function_name}({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, coord, offset_args, component
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            gather = self.texture_gather_call_expression(
                "textureGatherOffset",
                texture_name,
                coord,
                self.generate_expression(offset_arg),
                component,
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"vec4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(self, build_expression, component):
        component_calls = [build_expression(index) for index in range(4)]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return (
            f"/* unsupported GLSL texture gather: "
            f"{func_name} {reason} */ vec4(0.0)"
        )

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = parts
        offset_args = []
        component_arg = None

        if func_name == "textureGather":
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, "accepts at most one component argument"
                )
            if extra_args:
                component_arg = extra_args[0]
        elif func_name == "textureGatherOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, "requires offset and optional component arguments"
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

        component = self.texture_gather_component_value(component_arg)
        if component is not None:
            if component not in {0, 1, 2, 3}:
                return self.unsupported_texture_gather_call(
                    func_name, "component literal must be 0, 1, 2, or 3"
                )
            if func_name == "textureGatherOffsets":
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, component
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if func_name == "textureGatherOffset"
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, component
            )

        if component_arg is None:
            if func_name == "textureGatherOffsets":
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, None
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if func_name == "textureGatherOffset"
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg
            )

        component_expr = self.generate_expression(component_arg)
        if func_name == "textureGatherOffsets":
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, option
                ),
                component_expr,
            )

        offset_arg = (
            self.generate_expression(offset_args[0]) if offset_args else None
        )
        function_name = (
            "textureGatherOffset"
            if func_name == "textureGatherOffset"
            else "textureGather"
        )
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, option
            ),
            component_expr,
        )

    def texture_compare_coordinate(self, texture_type, coord, compare):
        texture_type = self.resource_base_type(texture_type)
        if texture_type == "samplerCubeArrayShadow":
            return None
        constructor = (
            "vec4"
            if texture_type
            in {
                "sampler2DArrayShadow",
                "samplerCubeShadow",
            }
            else "vec3"
        )
        return f"{constructor}({coord}, {compare})"

    def texture_compare_offset_supported(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }

    def texture_compare_lod_supported(self, texture_type):
        return self.resource_base_type(texture_type) == "sampler2DShadow"

    def texture_compare_grad_supported(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
        }

    def texture_compare_lod_offset_supported(self, texture_type):
        return self.resource_base_type(texture_type) == "sampler2DShadow"

    def texture_compare_grad_offset_supported(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }

    def texture_compare_projected_coordinate(
        self, texture_type, coord_arg, coord, compare
    ):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "sampler2DShadow":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
            return f"vec3({projected_coord}, {compare})"

        if texture_type != "sampler2DArrayShadow" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        layer = self.vector_component(coord, "z")
        return f"vec4({projected_coord}, {layer}, {compare})"

    def unsupported_texture_compare_call(self, func_name, reason):
        return f"/* unsupported GLSL texture compare: {func_name} {reason} */ 0.0"

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, "requires a compare argument"
            )

        compare = self.generate_expression(extra_args[0])
        texture_type = self.texture_resource_type(args[0])
        if func_name in {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }:
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            compare_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow vec4 projection coordinates",
                )

            if func_name == "textureCompareProj":
                if len(extra_args) != 1:
                    return self.unsupported_texture_compare_call(
                        func_name, "accepts no extra arguments"
                    )
                return f"texture({texture_name}, {compare_coord})"

            if func_name == "textureCompareProjOffset":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and offset arguments"
                    )
                offset = self.generate_expression(extra_args[1])
                return f"textureOffset({texture_name}, {compare_coord}, {offset})"

            if func_name == "textureCompareProjLod":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and lod arguments"
                    )
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "projected explicit LOD is not supported for sampler2DArrayShadow",
                    )
                lod = self.generate_expression(extra_args[1])
                return f"textureLod({texture_name}, {compare_coord}, {lod})"

            if func_name == "textureCompareProjLodOffset":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare, lod, and offset arguments"
                    )
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "projected explicit LOD is not supported for sampler2DArrayShadow",
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                return (
                    f"textureLodOffset({texture_name}, {compare_coord}, "
                    f"{lod}, {offset})"
                )

            if func_name == "textureCompareProjGrad":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "requires compare, gradient x, and gradient y arguments",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"textureGradOffset({texture_name}, {compare_coord}, "
                f"{ddx}, {ddy}, {offset})"
            )

        if func_name == "textureCompare":
            if texture_type == "samplerCubeArrayShadow":
                return f"texture({texture_name}, {coord}, {compare})"
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            return f"texture({texture_name}, {compare_coord})"

        if func_name == "textureCompareOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and offset arguments"
                )
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            offset = self.generate_expression(extra_args[1])
            return f"textureOffset({texture_name}, {compare_coord}, {offset})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and lod arguments"
                )
            if not self.texture_compare_lod_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD requires 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            lod = self.generate_expression(extra_args[1])
            return f"textureLod({texture_name}, {compare_coord}, {lod})"

        if func_name == "textureCompareLodOffset":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare, lod, and offset arguments"
                )
            if not self.texture_compare_lod_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD offsets require 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return f"textureLodOffset({texture_name}, {compare_coord}, {lod}, {offset})"

        if func_name == "textureCompareGrad":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare, gradient x, and gradient y arguments"
                )
            if not self.texture_compare_grad_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

        if func_name == "textureCompareGradOffset":
            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            if not self.texture_compare_grad_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradient offsets require 2D or 2D-array shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, "requires supported shadow texture coordinates"
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"textureGradOffset({texture_name}, {compare_coord}, "
                f"{ddx}, {ddy}, {offset})"
            )

        return None

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return (
            f"/* unsupported GLSL texture gather compare: "
            f"{func_name} {reason} */ vec4(0.0)"
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires a compare argument"
            )

        compare = self.generate_expression(extra_args[0])
        if func_name == "textureGatherCompare":
            if len(extra_args) != 1:
                return self.unsupported_texture_gather_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"textureGather({texture_name}, {coord}, {compare})"

        if len(extra_args) != 2:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires compare and offset arguments"
            )
        if not self.texture_gather_compare_offset_supported(
            self.texture_resource_type(args[0])
        ):
            return self.unsupported_texture_gather_compare_call(
                func_name, "offsets require 2D or 2D-array shadow samplers"
            )
        offset = self.generate_expression(extra_args[1])
        return f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"

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

    def is_float_image_resource(self, texture_type):
        return texture_type in {"image2D", "image3D", "image2DArray"}

    def image_load_component_suffix(self, texture_type, image_format):
        if self.is_scalar_integer_image_resource(texture_type, image_format):
            return ".x"
        if self.is_float_image_resource(texture_type) and self.is_scalar_value_type(
            self.current_expression_expected_type
        ):
            return ".x"
        if self.is_two_component_image_format(image_format):
            if self.is_scalar_value_type(self.current_expression_expected_type):
                return ".x"
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

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
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
        if self.is_scalar_value_type(value_type):
            return f"{type_name}({value}, {zero_value}, {zero_value}, {zero_value})"
        return f"{type_name}({value}, {zero_value}, {zero_value})"

    def image_store_value_expression(
        self, texture_type, image_format, value, value_type=None
    ):
        two_component_value = self.two_component_image_store_expression(
            image_format, value, value_type
        )
        if two_component_value is not None:
            return two_component_value

        constructor = None
        if self.is_scalar_integer_image_resource(texture_type, image_format):
            constructor = self.integer_image_store_constructor(texture_type)
            if constructor is None:
                constructor = self.image_format_store_constructor(image_format)
        elif self.is_float_image_resource(texture_type) and self.is_scalar_value_type(
            value_type
        ):
            constructor = "vec4"
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
            value = self.image_store_value_expression(
                texture_type, image_format, value, self.expression_result_type(args[2])
            )
            return f"imageStore({image_name}, {coord}, {value})"

        if func_name in {
            "textureCompare",
            "textureCompareOffset",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }:
            return self.generate_texture_compare_call(func_name, args)

        if func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            return self.generate_texture_gather_compare_call(func_name, args)

        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return self.generate_texture_gather_call(func_name, args)

        if func_name == "textureQueryLod":
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, _ = parts
            coord = self.texture_query_lod_coordinate(
                self.texture_resource_type(args[0]), coord
            )
            return f"textureQueryLod({texture_name}, {coord})"

        texture_funcs = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
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
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_sampled_texture_globals(ast),
            function_arrays=self.collect_unsized_sampled_texture_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=self.collect_functions(ast),
            walk_nodes=self.walk_ast,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=0,
            format_size=lambda size: str(size) if size > 1 else "",
        )

    def collect_unsized_sampled_texture_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_sampled_texture_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

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

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                size = self.fixed_resource_array_size(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_inferable_resource_array_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_inferable_resource_array_type(base_type):
            return None
        return max(size, 1)

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
        visible_constants = dict(self.literal_int_constants)

        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if not name:
                    continue

                visible_constants.pop(name, None)
                if "const" not in getattr(node, "qualifiers", []):
                    continue

                value = self.literal_int_value(
                    getattr(node, "initial_value", None), visible_constants
                )
                if value is not None:
                    visible_constants[name] = value

        return visible_constants

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
