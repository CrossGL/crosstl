"""CrossGL-to-Metal code generator."""

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
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .resource_arrays import collect_resource_array_size_hints


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
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = {}
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
            "sampler1D": "texture1d<float>",
            "sampler2D": "texture2d<float>",
            "sampler3D": "texture3d<float>",
            "samplerCube": "texturecube<float>",
            "sampler2DArray": "texture2d_array<float>",
            "samplerCubeArray": "texturecube_array<float>",
            "sampler2DMS": "texture2d_ms<float>",
            "sampler2DMSArray": "texture2d_ms_array<float>",
            "sampler2DShadow": "depth2d<float>",
            "sampler2DArrayShadow": "depth2d_array<float>",
            "samplerCubeShadow": "depthcube<float>",
            "samplerCubeArrayShadow": "depthcube_array<float>",
            "iimage2D": "texture2d<int, access::read_write>",
            "iimage3D": "texture3d<int, access::read_write>",
            "iimage2DArray": "texture2d_array<int, access::read_write>",
            "uimage2D": "texture2d<uint, access::read_write>",
            "uimage3D": "texture3d<uint, access::read_write>",
            "uimage2DArray": "texture2d_array<uint, access::read_write>",
            "image2D": "texture2d<float, access::read_write>",
            "image3D": "texture3d<float, access::read_write>",
            "imageCube": "texture2d_array<float, access::read_write>",
            "image2DArray": "texture2d_array<float, access::read_write>",
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
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            # Ray tracing / payload semantics
            "payload": "payload",
            "hit_attribute": "hit_attribute",
            "callable_data": "callable_data",
            "shader_record": "shader_record",
        }

    def generate(self, ast):
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)

        self.texture_variables = []
        self.sampler_variables = []
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.required_image_atomic_compare_helpers = set()
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = collect_struct_member_types(
            getattr(ast, "structs", []), self.type_name_string
        )
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        pre_lines = []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line:
                pre_lines.append(line)
        if pre_lines:
            code += "\n".join(pre_lines) + "\n"
        if not any("metal_stdlib" in line for line in pre_lines):
            code += "#include <metal_stdlib>\n"
        code += "using namespace metal;\n"
        code += "\n"
        code += self.generate_constants(ast)

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
                        semantic = None
                        if hasattr(member, "semantic"):
                            semantic = member.semantic
                        elif hasattr(member, "attributes"):
                            for attr in member.attributes:
                                if hasattr(attr, "name"):
                                    semantic = attr.name
                                    break

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
                                member_type_str = self.convert_type_node_to_string(
                                    member.member_type
                                )
                                member_type = self.map_type(member_type_str)
                        elif hasattr(member, "vtype"):
                            member_type = self.map_type(member.vtype)
                        else:
                            member_type = "float"

                        semantic_attr = self.map_semantic(semantic) if semantic else ""
                        code += f"    {member_type} {member.name}{semantic_attr};\n"
                code += "};\n"

        global_vars = getattr(ast, "global_variables", [])
        texture_register = 0
        sampler_register = 0
        for i, node in enumerate(global_vars):
            # Handle both old and new AST variable structures
            resource_count = 1
            array_size = None
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
                            self.expression_to_string(node.var_type.size)
                            if node.var_type.size
                            else self.resource_array_size_hints.get(node.name, "")
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

            if vtype in [
                "sampler1D",
                "sampler2D",
                "sampler3D",
                "samplerCube",
                "sampler2DArray",
                "samplerCubeArray",
                "sampler2DMS",
                "sampler2DMSArray",
                "sampler2DShadow",
                "sampler2DArrayShadow",
                "samplerCubeShadow",
                "samplerCubeArrayShadow",
                "iimage2D",
                "iimage3D",
                "iimage2DArray",
                "uimage2D",
                "uimage3D",
                "uimage2DArray",
                "image2D",
                "image3D",
                "imageCube",
                "image2DArray",
            ]:
                mapped_type = self.map_resource_type_with_format(vtype, node)
                self.texture_variables.append(
                    (node, texture_register, mapped_type, array_size)
                )
                self.texture_variable_types[node.name] = mapped_type
                explicit_format = self.explicit_image_format(node)
                if explicit_format:
                    self.image_variable_formats[node.name] = explicit_format
                texture_register += resource_count
            elif vtype in ["sampler"]:
                self.sampler_variables.append((node, sampler_register, array_size))
                sampler_register += resource_count
            else:
                code += f"{self.map_type(vtype)} {node.name}{array_suffix};\n"

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        functions = getattr(ast, "functions", [])
        functions_code = ""
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
                functions_code += "// Vertex Shader\n"
                functions_code += self.generate_function(func, shader_type="vertex")
            elif qualifier_name == "fragment":
                functions_code += "// Fragment Shader\n"
                functions_code += self.generate_function(func, shader_type="fragment")
            elif qualifier_name == "compute":
                functions_code += "// Compute Shader\n"
                functions_code += self.generate_function(func, shader_type="compute")
            else:
                functions_code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = normalize_stage_name(stage_type)
                    if not stage_matches(target_stage, stage_name):
                        continue
                    functions_code += f"// {stage_name.title()} Shader\n"
                    functions_code += self.generate_function(
                        stage.entry_point, shader_type=stage_name
                    )
                if hasattr(stage, "local_functions"):
                    stage_name = normalize_stage_name(stage_type)
                    if not stage_matches(target_stage, stage_name):
                        continue
                    for func in stage.local_functions:
                        functions_code += self.generate_function(func)

        code += self.generate_image_atomic_compare_helpers()
        code += functions_code
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
            code += f"constant {self.map_type(const_type)} {name} = {value_code};\n"

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

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        self.current_function_name = getattr(func, "name", None)
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
            elif self.is_resource_parameter_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                explicit_format = self.explicit_image_format(p)
                if explicit_format:
                    image_format_parameters[p.name] = explicit_format
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)

            param_attr = self.parameter_attribute(raw_param_type, semantic, shader_type)
            declaration = self.format_parameter_declaration(
                raw_param_type, param_type, p.name, p
            )
            params.append(f"{declaration}{param_attr}")

        if shader_type == "compute":
            existing_param_names = {getattr(p, "name", None) for p in param_list}
            for name, param_type, attribute in self.required_compute_builtin_parameters(
                func
            ):
                if name not in existing_param_names:
                    params.append(f"{param_type} {name} [[{attribute}]]")

        params_str = ", ".join(params)

        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        self.current_function_return_type = raw_return_type

        if shader_type == "vertex":
            code += f"vertex {return_type} vertex_{func.name}({params_str}) {{\n"
        elif shader_type == "fragment":
            params_str = self.append_global_resource_parameters(params_str)
            code += f"fragment {return_type} fragment_{func.name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(params_str)
            code += f"kernel {return_type} kernel_{func.name}({params_str}) {{\n"
        elif shader_type in ["mesh", "object", "task", "amplification"]:
            stage_keyword = "mesh" if shader_type == "mesh" else "object"
            code += f"{stage_keyword} {return_type} {stage_keyword}_{func.name}({params_str}) {{\n"
        elif shader_type in [
            "ray_intersection",
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        ]:
            rt_stage_map = {
                "ray_intersection": "intersection",
                "ray_any_hit": "anyhit",
                "ray_closest_hit": "closesthit",
                "ray_miss": "miss",
                "ray_callable": "callable",
                "intersection": "intersection",
                "anyhit": "anyhit",
                "closesthit": "closesthit",
                "miss": "miss",
                "callable": "callable",
            }
            stage_keyword = rt_stage_map.get(shader_type, shader_type)
            code += f"{stage_keyword} {return_type} {stage_keyword}_{func.name}({params_str}) {{\n"
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
        self.current_function_name = previous_function_name
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types

        code += "}\n\n"
        return code

    def required_compute_builtin_parameters(self, func):
        used_names = self.used_compute_builtin_names(getattr(func, "body", []))
        builtin_parameters = [
            ("gl_GlobalInvocationID", "uint3", "thread_position_in_grid"),
            ("gl_LocalInvocationID", "uint3", "thread_position_in_threadgroup"),
            ("gl_WorkGroupID", "uint3", "threadgroup_position_in_grid"),
            ("gl_LocalInvocationIndex", "uint", "thread_index_in_threadgroup"),
            ("gl_WorkGroupSize", "uint3", "threads_per_threadgroup"),
            ("gl_NumWorkGroups", "uint3", "threadgroups_per_grid"),
        ]
        return [
            parameter for parameter in builtin_parameters if parameter[0] in used_names
        ]

    def used_compute_builtin_names(self, body):
        builtin_names = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        used_names = set()
        for node in self.iter_ast_nodes(body):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", "")
                base_name = name.split(".", 1)[0]
                if base_name in builtin_names:
                    used_names.add(base_name)
        return used_names

    def append_global_resource_parameters(self, params_str):
        resource_params = []
        if self.texture_variables:
            for (
                texture_variable,
                i,
                texture_type,
                array_size,
            ) in self.texture_variables:
                declaration = self.format_resource_parameter(
                    texture_type, texture_variable.name, array_size
                )
                resource_params.append(f"{declaration} [[texture({i})]]")
        if self.sampler_variables:
            for sampler_variable, i, array_size in self.sampler_variables:
                declaration = self.format_resource_parameter(
                    "sampler", sampler_variable.name, array_size
                )
                resource_params.append(f"{declaration} [[sampler({i})]]")
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
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
            return f"{indent_str}{self.generate_expression(stmt)};\n"

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
            "half",
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
            "float2",
            "float3",
            "float4",
            "half2",
            "half3",
            "half4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("bool"):
            return "bool"
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
            return self.expression_result_type(
                getattr(expr, "target", getattr(expr, "left", None))
            )
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_result_type(expr.array)
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
                "half",
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
                "float2",
                "float3",
                "float4",
                "int2",
                "int3",
                "int4",
                "uint2",
                "uint3",
                "uint4",
                "bool2",
                "bool3",
                "bool4",
            }:
                return str(func_name)
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            value = getattr(expr, "value", None)
            if isinstance(value, float):
                return "float"
            if isinstance(value, int):
                return "int"
            if isinstance(value, str):
                return "float" if "." in value else "int"
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

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
            rhs = self.generate_expression_with_expected(
                node.value, self.expression_result_type(node.target)
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_expression(node.left)
            rhs = self.generate_expression_with_expected(
                node.right, self.expression_result_type(node.left)
            )
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
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

        if hasattr(node.body, "statements"):
            for stmt in node.body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(node.body, list):
            for stmt in node.body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += self.generate_statement(node.body, indent + 1)

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
                    "Unsupported match arm for Metal codegen; only unguarded "
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
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, FunctionCallNode):
            # Resolve callee expression (can be Identifier/Member/Array access)
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            func_name = None
            if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call
            # Special handling for common GLSL functions
            elif func_name == "normalize":
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"normalize({args})"
            elif func_name in ["mix", "clamp", "smoothstep", "step", "dot", "cross"]:
                # These function names are the same in GLSL and Metal
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{func_name}({args})"
            # Vector constructors
            elif func_name in [
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
                # Map to Metal's float2, float3, float4
                metal_type = self.map_type(func_name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{metal_type}({args})"
            else:
                # Standard function call
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition)} ? {self.generate_expression(expr.true_expr)} : {self.generate_expression(expr.false_expr)}"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            # Handle LiteralNode
            if hasattr(expr, "value"):
                value = expr.value
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                if (
                    literal_type == "uint"
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    return f"{value}u"
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

    def default_sampler_expression(self):
        return "sampler(mag_filter::linear, min_filter::linear)"

    def sampler_variable_names(self):
        return {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) == "sampler"

    def is_resource_parameter_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler",
            "sampler1D",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler2DMS",
            "sampler2DMSArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
        }

    def parameter_attribute(self, raw_param_type, semantic, shader_type):
        if semantic:
            return self.map_semantic(semantic)
        if self.is_resource_parameter_type(raw_param_type):
            return ""
        if shader_type in {"vertex", "fragment"}:
            return " [[stage_in]]"
        return ""

    def format_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None
    ):
        array_type = self.resource_array_parameter(raw_param_type, node)
        if array_type is not None:
            resource_type, array_size = array_type
            return self.format_resource_parameter(resource_type, name, array_size)
        return format_c_style_array_declaration(mapped_type, name)

    def format_resource_parameter(self, resource_type, name, array_size):
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{resource_type}, {array_size}> {name}"
        return f"{resource_type} {name}"

    def resource_array_parameter(self, vtype, node=None):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_parameter_type(base_type):
                return None
            array_size = (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else self.function_resource_array_size_hints.get(
                    self.current_function_name, {}
                ).get(node.name, "")
            )
            return self.map_resource_type_with_format(base_type, node), array_size

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, array_size = parse_array_type(type_string)
        if not self.is_resource_parameter_type(base_type):
            return None
        return self.map_resource_type_with_format(base_type, node), (
            self.function_resource_array_size_hints.get(
                self.current_function_name, {}
            ).get(node.name, "")
            if array_size is None
            else array_size
        )

    def collect_resource_array_size_hints(self, ast):
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_resource_globals(ast),
            function_arrays=self.collect_unsized_resource_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=self.all_functions(ast),
            walk_nodes=self.iter_ast_nodes,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=1,
            format_size=str,
        )

    def collect_unsized_resource_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_resource_array_type(vtype):
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

    def collect_unsized_resource_parameters(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
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
            if not self.is_resource_parameter_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_parameter_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_parameter_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_parameter_type(base_type)

    def all_functions(self, ast):
        functions = list(getattr(ast, "functions", []) or [])
        for stage in getattr(ast, "stages", {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
            functions.extend(getattr(stage, "local_functions", []) or [])
        return functions

    def iter_ast_nodes(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple, set)):
            for item in node:
                yield from self.iter_ast_nodes(item)
            return
        if not hasattr(node, "__dict__"):
            return
        yield node
        for key, value in vars(node).items():
            if key in {"parent", "annotations"}:
                continue
            yield from self.iter_ast_nodes(value)

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def visible_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)

        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
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

    def scalar_image_format_components(self):
        return {
            "r8": "float",
            "r8_snorm": "float",
            "r16": "float",
            "r16_snorm": "float",
            "r16f": "float",
            "r32f": "float",
            "r8i": "int",
            "r16i": "int",
            "r32i": "int",
            "r8ui": "uint",
            "r16ui": "uint",
            "r32ui": "uint",
        }

    def vector_image_format_components(self):
        return {
            "rg8": "float",
            "rg8_snorm": "float",
            "rg16": "float",
            "rg16_snorm": "float",
            "rg16f": "float",
            "rg8i": "int",
            "rg16i": "int",
            "rg8ui": "uint",
            "rg16ui": "uint",
            "rg32f": "float",
            "rg32i": "int",
            "rg32ui": "uint",
            "rgba8": "float",
            "rgba8_snorm": "float",
            "rgba16": "float",
            "rgba16_snorm": "float",
            "rgba16f": "float",
            "rgba32f": "float",
            "rgba8i": "int",
            "rgba16i": "int",
            "rgba32i": "int",
            "rgba8ui": "uint",
            "rgba16ui": "uint",
            "rgba32ui": "uint",
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

    def explicit_image_format(self, node):
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
        explicit_format = self.explicit_image_format(node) if node is not None else None
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image2D": "texture2d",
            "iimage2D": "texture2d",
            "uimage2D": "texture2d",
            "image3D": "texture3d",
            "iimage3D": "texture3d",
            "uimage3D": "texture3d",
            "image2DArray": "texture2d_array",
            "iimage2DArray": "texture2d_array",
            "uimage2DArray": "texture2d_array",
            "imageCube": "texture2d_array",
        }
        texture_type = texture_types.get(base_type)
        if component_type and texture_type:
            return f"{texture_type}<{component_type}, access::read_write>"
        return self.map_type(vtype)

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

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_name(expr.array)
        return None

    def texture_sampler_expression(self, texture_name):
        sampler_arg = ""
        for sampler_variable, _, _ in self.sampler_variables:
            if sampler_variable.name == texture_name + "Sampler":
                sampler_arg = sampler_variable.name
                break
        return sampler_arg or self.default_sampler_expression()

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        return sampler_name in self.sampler_variable_names()

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        sampler_arg = (
            self.generate_expression(args[1])
            if explicit_sampler
            else self.texture_sampler_expression(texture_base_name)
        )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_arg, coord, extra_args

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

    def is_array_texture_resource(self, texture_type):
        return texture_type in {
            "texture2d_array<float>",
            "depth2d_array<float>",
            "texturecube_array<float>",
            "depthcube_array<float>",
        }

    def is_multisample_texture_resource(self, texture_type):
        return texture_type in {
            "texture2d_ms<float>",
            "texture2d_ms_array<float>",
        }

    def is_storage_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            texture_type.startswith("texture2d<")
            or texture_type.startswith("texture3d<")
            or texture_type.startswith("texture2d_array<")
        ) and "access::read_write" in texture_type

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def array_texture_coordinate_parts(self, coord):
        coord_xy = self.vector_component(coord, "xy")
        layer = f"uint({self.vector_component(coord, 'z')})"
        return coord_xy, layer

    def cube_array_texture_coordinate_parts(self, coord):
        coord_xyz = self.vector_component(coord, "xyz")
        layer = f"uint({self.vector_component(coord, 'w')})"
        return coord_xyz, layer

    def texture_coordinate_parts(self, texture_type, coord):
        if texture_type in {"texturecube_array<float>", "depthcube_array<float>"}:
            return self.cube_array_texture_coordinate_parts(coord)
        return self.array_texture_coordinate_parts(coord)

    def texture_gradient_options(self, texture_type, ddx, ddy):
        if texture_type in {
            "texturecube<float>",
            "depthcube<float>",
            "texturecube_array<float>",
            "depthcube_array<float>",
        }:
            return f"gradientcube({ddx}, {ddy})"
        if texture_type == "texture3d<float>":
            return f"gradient3d({ddx}, {ddy})"
        return f"gradient2d({ddx}, {ddy})"

    def texture_gather_supports_offset(self, texture_type):
        return texture_type in {"texture2d<float>", "texture2d_array<float>"}

    def texture_gather_supported(self, texture_type):
        return texture_type in {
            "texture2d<float>",
            "texture2d_array<float>",
            "texturecube<float>",
            "texturecube_array<float>",
        }

    def texture_sample_supports_offset(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {"texture2d<float>", "texture2d_array<float>"}

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return (
            f"/* unsupported Metal texture offset: {func_name} {reason} */ float4(0.0)"
        )

    def texture_sample_offset_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            return self.texture_coordinate_parts(texture_type, coord)
        return (coord,)

    def generate_texture_sample_offset_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not self.texture_sample_supports_offset(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, "offsets require 2D or 2D-array textures"
            )

        coord_args = self.texture_sample_offset_coord_args(texture_type, coord)

        if func_name == "textureOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_sample_offset_call(
                    func_name, "requires one offset argument"
                )
            offset = self.generate_expression(extra_args[0])
            args = [sampler_arg] + list(coord_args) + [offset]
            return f"{texture_name}.sample({', '.join(args)})"

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_sample_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [f"level({lod})", offset]
            return f"{texture_name}.sample({', '.join(args)})"

        if func_name == "textureGradOffset":
            if len(extra_args) != 3:
                return self.unsupported_texture_sample_offset_call(
                    func_name,
                    "requires gradient x, gradient y, and offset arguments",
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [gradient_options, offset]
            return f"{texture_name}.sample({', '.join(args)})"

        return self.unsupported_texture_sample_offset_call(
            func_name, "is not a supported texture offset operation"
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return f"/* unsupported Metal projected texture: {func_name} {reason} */ float4(0.0)"

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "texture1d<float>": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "texture2d<float>": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture2d_array<float>": {
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "texture3d<float>": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        texture_specs = specs.get(texture_type)
        if texture_specs is None:
            return None
        coord_spec = texture_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        projected_coord = (
            f"{self.vector_component(coord, numerator)} / "
            f"{self.vector_component(coord, divisor)}"
        )
        if texture_type == "texture2d_array<float>":
            return f"{projected_coord}, " f"uint({self.vector_component(coord, 'z')})"
        return projected_coord

    def projected_texture_offset_supported(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {"texture2d<float>", "texture2d_array<float>"}

    def generate_texture_projected_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args,
    ):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        projected_coord = self.projected_texture_coord(
            args[0], args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires 1D, 2D, or 3D projection coordinates"
            )

        if func_name == "textureProj":
            if not extra_args:
                return f"{texture_name}.sample({sampler_arg}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}))"
                )
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "offsets require 2D textures"
                )
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.sample("
                    f"{sampler_arg}, {projected_coord}, bias({bias}), {offset})"
                )
            return self.unsupported_texture_projected_call(
                func_name, "requires offset and optional bias arguments"
            )

        if func_name == "textureProjLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_projected_call(
                    func_name, "requires one lod argument"
                )
            lod = self.generate_expression(extra_args[0])
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, level({lod}))"
            )

        if func_name == "textureProjLodOffset":
            if not self.projected_texture_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, "offsets require 2D textures"
                )
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, level({lod}), {offset})"
            )

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            return (
                f"{texture_name}.sample("
                f"{sampler_arg}, {projected_coord}, {gradient_options})"
            )

        if not self.projected_texture_offset_supported(texture_type):
            return self.unsupported_texture_projected_call(
                func_name, "offsets require 2D textures"
            )
        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
        return (
            f"{texture_name}.sample("
            f"{sampler_arg}, {projected_coord}, {gradient_options}, {offset})"
        )

    def is_array_expression(self, node):
        type_name = self.expression_result_type(node)
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

    def texture_gather_component_option(self, component_arg):
        if component_arg is None:
            return None

        components = {
            0: "component::x",
            1: "component::y",
            2: "component::z",
            3: "component::w",
        }
        return components.get(self.literal_int_value(component_arg))

    def texture_gather_coord_args(self, texture_type, coord):
        if self.is_array_texture_resource(texture_type):
            coord_part, layer = self.texture_coordinate_parts(texture_type, coord)
            return [coord_part, layer]
        return [coord]

    def texture_gather_call_expression(
        self,
        texture_name,
        sampler_arg,
        coord_args,
        offset_arg=None,
        component=None,
        default_offset_for_component=False,
    ):
        args = [sampler_arg] + coord_args
        if offset_arg is not None:
            args.append(offset_arg)
        elif component is not None and default_offset_for_component:
            args.append("int2(0)")
        if component is not None:
            args.append(component)
        return f"{texture_name}.gather({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, sampler_arg, coord_args, offset_args, component
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            gather = self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                self.generate_expression(offset_arg),
                component,
                default_offset_for_component=True,
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"float4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(
        self, build_expression, component_expr
    ):
        component_options = (
            "component::x",
            "component::y",
            "component::z",
            "component::w",
        )
        component_calls = [
            build_expression(component) for component in component_options
        ]
        return (
            f"({component_expr} == 0 ? {component_calls[0]} : "
            f"{component_expr} == 1 ? {component_calls[1]} : "
            f"{component_expr} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return (
            f"/* unsupported Metal texture gather: {func_name} {reason} */ float4(0.0)"
        )

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        return (
            f"/* unsupported Metal multisample texture call: "
            f"{func_name} on {texture_type} */ float4(0.0)"
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        return (
            "/* unsupported Metal multisample texture query: "
            f"textureQueryLod on {texture_type} */ float2(0.0)"
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            "/* unsupported Metal texture query: "
            f"textureQueryLevels on {texture_type} */ 0"
        )

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            "/* unsupported Metal texture query: "
            f"textureQueryLod on {texture_type} */ float2(0.0)"
        )

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_resource(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
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
            return (
                "/* unsupported Metal storage image texture comparison: "
                f"{func_name} on {texture_type} */ 0.0"
            )

        if func_name in {
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
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "texelFetch",
            "texelFetchOffset",
        }:
            return (
                "/* unsupported Metal storage image texture operation: "
                f"{func_name} on {texture_type} */ float4(0.0)"
            )

        return None

    def is_cube_texture_resource(self, texture_type):
        return texture_type in {
            "texturecube<float>",
            "texturecube_array<float>",
            "depthcube<float>",
            "depthcube_array<float>",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        return (
            f"/* unsupported Metal texel fetch: {func_name} on "
            f"{texture_type} */ float4(0.0)"
        )

    def generate_texture_gather_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if func_name == "textureGather" and not self.texture_gather_supported(
            texture_type
        ):
            return self.unsupported_texture_gather_call(
                func_name, "requires 2D, 2D-array, cube, or cube-array textures"
            )

        coord_args = self.texture_gather_coord_args(texture_type, coord)
        supports_offset = self.texture_gather_supports_offset(texture_type)
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
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, "offsets require 2D or 2D-array textures"
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            if not supports_offset:
                return self.unsupported_texture_gather_call(
                    func_name, "offsets require 2D or 2D-array textures"
                )
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

        component = self.texture_gather_component_option(component_arg)
        if component is not None or component_arg is None:
            if func_name == "textureGatherOffsets":
                return self.texture_gather_offsets_expression(
                    texture_name, sampler_arg, coord_args, offset_args, component
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            return self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                component,
                default_offset_for_component=supports_offset,
            )

        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, "component literal must be 0, 1, 2, or 3"
            )

        component_expr = self.generate_expression(component_arg)
        if func_name == "textureGatherOffsets":
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, sampler_arg, coord_args, offset_args, option
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                texture_name,
                sampler_arg,
                coord_args,
                offset_arg,
                option,
                default_offset_for_component=supports_offset,
            ),
            component_expr,
        )

    def texture_compare_offset_supported(self, texture_type):
        return texture_type in {"depth2d<float>", "depth2d_array<float>"}

    def unsupported_texture_compare_call(self, func_name, reason):
        return f"/* unsupported Metal texture compare: {func_name} {reason} */ 0.0"

    def texture_compare_projected_coord_args(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "depth2d<float>":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            return [f"{self.vector_component(coord, 'xy')} / {divisor}"]

        if texture_type != "depth2d_array<float>" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        layer = f"uint({self.vector_component(coord, 'z')})"
        return [projected_coord, layer]

    def generate_texture_compare_call(
        self,
        func_name,
        texture_name,
        sampler_arg,
        coord,
        extra_args,
        texture_type,
        args=None,
    ):
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, "requires a compare argument"
            )

        compare = self.generate_expression(extra_args[0])
        if func_name in {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }:
            coord_index = 2 if self.is_explicit_sampler_argument(args or []) else 1
            coord_arg = (args or [None, None])[coord_index]
            coord_args = self.texture_compare_projected_coord_args(
                texture_type, coord_arg, coord
            )
            if coord_args is None:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates",
                )
            projected_args = [sampler_arg] + coord_args + [compare]

            if func_name == "textureCompareProj":
                if len(extra_args) != 1:
                    return self.unsupported_texture_compare_call(
                        func_name, "accepts no extra arguments"
                    )
                return f"{texture_name}.sample_compare({', '.join(projected_args)})"

            if func_name == "textureCompareProjOffset":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and offset arguments"
                    )
                offset = self.generate_expression(extra_args[1])
                args = projected_args + [offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if func_name == "textureCompareProjLod":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and lod arguments"
                    )
                lod = self.generate_expression(extra_args[1])
                args = projected_args + [f"level({lod})"]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if func_name == "textureCompareProjLodOffset":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare, lod, and offset arguments"
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                args = projected_args + [f"level({lod})", offset]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if func_name == "textureCompareProjGrad":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "requires compare, gradient x, and gradient y arguments",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
                args = projected_args + [gradient_options]
                return f"{texture_name}.sample_compare({', '.join(args)})"

            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            offset = self.generate_expression(extra_args[3])
            args = projected_args + [gradient_options, offset]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        coord_args = (
            self.texture_coordinate_parts(texture_type, coord)
            if self.is_array_texture_resource(texture_type)
            else (coord,)
        )

        if func_name == "textureCompare":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "accepts no extra arguments"
                )
            args = [sampler_arg] + list(coord_args) + [compare]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if func_name == "textureCompareOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and offset arguments"
                )
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array depth textures"
                )
            offset = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, offset]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and lod arguments"
                )
            lod = self.generate_expression(extra_args[1])
            args = [sampler_arg] + list(coord_args) + [compare, f"level({lod})"]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if func_name == "textureCompareLodOffset":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare, lod, and offset arguments"
                )
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array depth textures"
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    f"level({lod})",
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if func_name == "textureCompareGrad":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, and gradient y arguments",
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            args = [sampler_arg] + list(coord_args) + [compare, gradient_options]
            return f"{texture_name}.sample_compare({', '.join(args)})"

        if func_name == "textureCompareGradOffset":
            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array depth textures"
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            offset = self.generate_expression(extra_args[3])
            args = (
                [sampler_arg]
                + list(coord_args)
                + [
                    compare,
                    gradient_options,
                    offset,
                ]
            )
            return f"{texture_name}.sample_compare({', '.join(args)})"

        return self.unsupported_texture_compare_call(
            func_name, "is not a supported shadow compare operation"
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return texture_type in {"depth2d<float>", "depth2d_array<float>"}

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return (
            f"/* unsupported Metal texture gather compare: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_gather_compare_call(
        self, func_name, texture_name, sampler_arg, coord, extra_args, texture_type
    ):
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires a compare argument"
            )

        compare = self.generate_expression(extra_args[0])
        coord_args = self.texture_gather_coord_args(texture_type, coord)
        if func_name == "textureGatherCompare":
            if len(extra_args) != 1:
                return self.unsupported_texture_gather_compare_call(
                    func_name, "accepts no extra arguments"
                )
            args = [sampler_arg] + coord_args + [compare]
            return f"{texture_name}.gather_compare({', '.join(args)})"

        if len(extra_args) != 2:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires compare and offset arguments"
            )
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, "offsets require 2D or 2D-array depth textures"
            )
        offset = self.generate_expression(extra_args[1])
        args = [sampler_arg] + coord_args + [compare, offset]
        return f"{texture_name}.gather_compare({', '.join(args)})"

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
        lod_arg_string = f"uint({lod})"

        if self.is_storage_image_resource(texture_type):
            texture_type = self.resource_base_type(texture_type)
            if texture_type.startswith("texture2d_array<"):
                return (
                    f"int3({texture_name}.get_width(), "
                    f"{texture_name}.get_height(), "
                    f"{texture_name}.get_array_size())"
                )
            if texture_type.startswith("texture3d<"):
                return (
                    f"int3({texture_name}.get_width(), "
                    f"{texture_name}.get_height(), "
                    f"{texture_name}.get_depth())"
                )
            return f"int2({texture_name}.get_width(), " f"{texture_name}.get_height())"

        if texture_type in {"texture1d<float>"}:
            return f"int({texture_name}.get_width({lod_arg_string}))"
        if texture_type in {
            "texture2d<float>",
            "depth2d<float>",
            "texturecube<float>",
            "depthcube<float>",
        }:
            return (
                f"int2({texture_name}.get_width({lod_arg_string}), "
                f"{texture_name}.get_height({lod_arg_string}))"
            )
        if texture_type in {
            "texture2d_array<float>",
            "depth2d_array<float>",
            "texturecube_array<float>",
            "depthcube_array<float>",
        }:
            return (
                f"int3({texture_name}.get_width({lod_arg_string}), "
                f"{texture_name}.get_height({lod_arg_string}), "
                f"{texture_name}.get_array_size())"
            )
        if texture_type in {"texture3d<float>"}:
            return (
                f"int3({texture_name}.get_width({lod_arg_string}), "
                f"{texture_name}.get_height({lod_arg_string}), "
                f"{texture_name}.get_depth({lod_arg_string}))"
            )
        if texture_type == "texture2d_ms<float>":
            return f"int2({texture_name}.get_width(), {texture_name}.get_height())"
        if texture_type == "texture2d_ms_array<float>":
            return (
                f"int3({texture_name}.get_width(), {texture_name}.get_height(), "
                f"{texture_name}.get_array_size())"
            )
        return None

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        if self.is_storage_image_resource(texture_type):
            return self.unsupported_texture_query_levels_call(texture_type)
        if self.is_multisample_texture_resource(texture_type):
            return "1"
        return f"int({texture_name}.get_num_mip_levels())"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        if not self.is_multisample_texture_resource(texture_type):
            return "/* unsupported Metal texture samples query: requires multisample texture */ 0"
        return f"int({texture_name}.get_num_samples())"

    def image_coordinate_expression(self, image_type, coord):
        if image_type in {
            "texture2d_array<float, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            coord_xy = f"uint2({self.vector_component(coord, 'xy')})"
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer
        if image_type in {
            "texture3d<float, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
        }:
            return f"uint3({coord})", None
        return f"uint2({coord})", None

    def is_integer_image_type(self, image_type):
        return image_type in {
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
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

    def is_scalar_integer_image_resource(self, image_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(image_type)

    def is_float_image_resource(self, image_type):
        return image_type in {
            "texture2d<float, access::read_write>",
            "texture3d<float, access::read_write>",
            "texture2d_array<float, access::read_write>",
        }

    def image_load_component_suffix(self, image_type, image_format):
        if self.is_scalar_integer_image_resource(image_type, image_format):
            return ".x"
        if self.is_float_image_resource(image_type) and self.is_scalar_value_type(
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
            "r8": "float4",
            "r8_snorm": "float4",
            "r16": "float4",
            "r16_snorm": "float4",
            "r16f": "float4",
            "r32f": "float4",
            "r8i": "int4",
            "r16i": "int4",
            "r32i": "int4",
            "r8ui": "uint4",
            "r16ui": "uint4",
            "r32ui": "uint4",
        }.get(image_format)

    def integer_image_store_constructor(self, image_type):
        if image_type in {
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
        }:
            return "int4"
        if image_type in {
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "uint4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        constructors = {
            "rg8": ("float4", "0.0"),
            "rg8_snorm": ("float4", "0.0"),
            "rg16": ("float4", "0.0"),
            "rg16_snorm": ("float4", "0.0"),
            "rg16f": ("float4", "0.0"),
            "rg8i": ("int4", "0"),
            "rg16i": ("int4", "0"),
            "rg8ui": ("uint4", "0u"),
            "rg16ui": ("uint4", "0u"),
            "rg32f": ("float4", "0.0"),
            "rg32i": ("int4", "0"),
            "rg32ui": ("uint4", "0u"),
        }
        constructor = constructors.get(image_format)
        if constructor is None:
            return None
        type_name, zero_value = constructor
        if self.is_scalar_value_type(value_type):
            return f"{type_name}({value}, {zero_value}, {zero_value}, {zero_value})"
        return f"{type_name}({value}, {zero_value}, {zero_value})"

    def image_store_value_expression(
        self, image_type, image_format, value, value_type=None
    ):
        two_component_value = self.two_component_image_store_expression(
            image_format, value, value_type
        )
        if two_component_value is not None:
            return two_component_value

        constructor = None
        if self.is_scalar_integer_image_resource(image_type, image_format):
            constructor = self.integer_image_store_constructor(image_type)
            if constructor is None:
                constructor = self.image_format_store_constructor(image_format)
        elif self.is_float_image_resource(image_type) and self.is_scalar_value_type(
            value_type
        ):
            constructor = "float4"
        if constructor:
            return f"{constructor}({value})"
        return value

    def image_atomic_method(self, func_name):
        return {
            "imageAtomicAdd": "atomic_fetch_add",
            "imageAtomicMin": "atomic_fetch_min",
            "imageAtomicMax": "atomic_fetch_max",
            "imageAtomicAnd": "atomic_fetch_and",
            "imageAtomicOr": "atomic_fetch_or",
            "imageAtomicXor": "atomic_fetch_xor",
            "imageAtomicExchange": "atomic_exchange",
        }.get(func_name)

    def image_atomic_compare_helper_name(self, texture_type):
        suffixes = {
            "texture2d<int, access::read_write>": "iimage2D",
            "texture2d<uint, access::read_write>": "uimage2D",
            "texture3d<int, access::read_write>": "iimage3D",
            "texture3d<uint, access::read_write>": "uimage3D",
            "texture2d_array<int, access::read_write>": "iimage2DArray",
            "texture2d_array<uint, access::read_write>": "uimage2DArray",
        }
        suffix = suffixes.get(texture_type)
        if not suffix:
            return None
        return f"imageAtomicCompSwap_{suffix}"

    def image_atomic_compare_return_type(self, texture_type):
        if texture_type in {
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
        }:
            return "int"
        if texture_type in {
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "uint"
        return None

    def image_atomic_compare_vector_type(self, texture_type):
        if texture_type in {
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
        }:
            return "int4"
        if texture_type in {
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "uint4"
        return None

    def image_atomic_compare_coord_type(self, texture_type):
        if texture_type in {
            "texture2d<int, access::read_write>",
            "texture2d<uint, access::read_write>",
        }:
            return "int2"
        if texture_type in {
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "int3"
        return None

    def image_atomic_compare_exchange_expression(self, texture_type):
        if texture_type in {
            "texture2d<int, access::read_write>",
            "texture2d<uint, access::read_write>",
        }:
            return "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
        if texture_type in {
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
        }:
            return "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
        if texture_type in {
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
        return None

    def generate_image_atomic_compare_helpers(self):
        if not self.required_image_atomic_compare_helpers:
            return ""

        helpers = []
        for texture_type in sorted(self.required_image_atomic_compare_helpers):
            helper_name = self.image_atomic_compare_helper_name(texture_type)
            return_type = self.image_atomic_compare_return_type(texture_type)
            vector_type = self.image_atomic_compare_vector_type(texture_type)
            coord_type = self.image_atomic_compare_coord_type(texture_type)
            exchange_expr = self.image_atomic_compare_exchange_expression(texture_type)
            if (
                not helper_name
                or not return_type
                or not vector_type
                or not coord_type
                or not exchange_expr
            ):
                continue
            helpers.append(
                f"{return_type} {helper_name}({texture_type} image, {coord_type} coord, {return_type} compareValue, {return_type} value) {{\n"
                f"    {vector_type} original;\n"
                "    do {\n"
                "        original.x = compareValue;\n"
                f"    }} while (!{exchange_expr} && original.x == compareValue);\n"
                "    return original.x;\n"
                "}\n\n"
            )
        return "".join(helpers)

    def generate_image_call(self, func_name, args):
        if func_name == "imageAtomicCompSwap" and len(args) >= 4:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            image_type = self.texture_resource_type(args[0])
            helper_name = self.image_atomic_compare_helper_name(image_type)
            if not helper_name:
                return None
            self.required_image_atomic_compare_helpers.add(image_type)
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        atomic_method = self.image_atomic_method(func_name)
        if atomic_method and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            image_type = self.texture_resource_type(args[0])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if layer is not None:
                return (
                    f"{image_name}.{atomic_method}({texel_coord}, {layer}, {value}).x"
                )
            return f"{image_name}.{atomic_method}({texel_coord}, {value}).x"

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if layer is not None:
                load_expr = f"{image_name}.read({texel_coord}, {layer})"
            else:
                load_expr = f"{image_name}.read({texel_coord})"
            image_format = self.image_resource_format(args[0])
            return f"{load_expr}{self.image_load_component_suffix(image_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            image_type = self.texture_resource_type(args[0])
            image_format = self.image_resource_format(args[0])
            value = self.image_store_value_expression(
                image_type, image_format, value, self.expression_result_type(args[2])
            )
            texel_coord, layer = self.image_coordinate_expression(image_type, coord)
            if layer is not None:
                return f"{image_name}.write({value}, {texel_coord}, {layer})"
            return f"{image_name}.write({value}, {texel_coord})"

        return None

    def generate_texture_call(self, func_name, args):
        if not func_name:
            return None

        image_call = self.generate_image_call(func_name, args)
        if image_call is not None:
            return image_call

        if func_name in {"textureSize", "imageSize"} and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if func_name == "textureQueryLevels" and args:
            return self.texture_query_levels_expression(args[0])

        if func_name in {"textureSamples", "imageSamples"} and args:
            return self.texture_samples_expression(args[0])

        if len(args) < 2:
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None

        texture_name, sampler_arg, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        is_array_texture = self.is_array_texture_resource(texture_type)
        if is_array_texture:
            coord_xy, layer = self.texture_coordinate_parts(texture_type, coord)

        if func_name in {
            "texture",
            "textureLod",
            "textureGrad",
        } and self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)

        if func_name == "texture":
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer})"
            return f"{texture_name}.sample({sampler_arg}, {coord})"
        if func_name == "textureLod" and extra_args:
            lod = self.generate_expression(extra_args[0])
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, level({lod}))"
            return f"{texture_name}.sample({sampler_arg}, {coord}, level({lod}))"
        if func_name == "textureGrad" and len(extra_args) >= 2:
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            gradient_options = self.texture_gradient_options(texture_type, ddx, ddy)
            if is_array_texture:
                return f"{texture_name}.sample({sampler_arg}, {coord_xy}, {layer}, {gradient_options})"
            return f"{texture_name}.sample({sampler_arg}, {coord}, {gradient_options})"
        if func_name in {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
        }:
            return self.generate_texture_sample_offset_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
            )
        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            return self.generate_texture_projected_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return self.generate_texture_gather_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
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
            return self.generate_texture_compare_call(
                func_name,
                texture_name,
                sampler_arg,
                coord,
                extra_args,
                texture_type,
                args,
            )
        if func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            return self.generate_texture_gather_compare_call(
                func_name, texture_name, sampler_arg, coord, extra_args, texture_type
            )
        if func_name == "textureQueryLod":
            if self.is_multisample_texture_resource(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_resource(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            lod_coord = coord_xy if is_array_texture else coord
            return (
                f"float2({texture_name}.calculate_unclamped_lod({sampler_arg}, {lod_coord}), "
                f"{texture_name}.calculate_clamped_lod({sampler_arg}, {lod_coord}))"
            )
        if func_name == "texelFetch" and len(args) >= 3:
            lod = self.generate_expression(args[2])
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                if texture_type == "texture2d_ms_array<float>":
                    texel_xy, layer = self.array_texture_coordinate_parts(coord)
                    return f"{texture_name}.read({texel_xy}, {layer}, uint({lod}))"
                return f"{texture_name}.read({coord}, uint({lod}))"
            if is_array_texture:
                texel_xy, layer = self.array_texture_coordinate_parts(coord)
                return f"{texture_name}.read({texel_xy}, {layer}, {lod})"
            return f"{texture_name}.read({coord}, {lod})"

        if func_name == "texelFetchOffset" and len(args) >= 4:
            lod = self.generate_expression(args[2])
            offset = self.generate_expression(args[3])
            if self.is_cube_texture_resource(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource(texture_type):
                return "/* unsupported Metal texel fetch offset: multisample textures do not support offsets */ float4(0.0)"
            if is_array_texture:
                texel_xy, layer = self.array_texture_coordinate_parts(coord)
                return f"{texture_name}.read(({texel_xy} + {offset}), {layer}, {lod})"
            return f"{texture_name}.read(({coord} + {offset}), {lod})"

        return None

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"float{type_node.rows}x{type_node.rows}"
            else:
                return f"float{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        size_str = self.safe_expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
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
            return str(type_node)

    def safe_expression_to_string(self, expr):
        """Convert an expression node to a string representation safely (avoid infinite recursion)."""
        return self.safe_expression_to_string_with_precedence(expr)

    def safe_expression_to_string_with_precedence(self, expr, parent_precedence=0):
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, BinaryOpNode):
            operator = self.map_operator(expr.op)
            precedence = self.expression_precedence(operator)
            left = self.safe_expression_to_string_with_precedence(expr.left, precedence)
            right = self.safe_expression_to_string_with_precedence(
                expr.right, precedence + 1
            )
            expression = f"{left} {operator} {right}"
            if precedence < parent_precedence:
                return f"({expression})"
            return expression
        elif isinstance(expr, UnaryOpNode):
            operand = self.safe_expression_to_string_with_precedence(
                expr.operand, self.expression_precedence("unary")
            )
            return f"{self.map_operator(expr.op)}{operand}"
        else:
            # Fallback - avoid calling generate_expression to prevent infinite recursion
            return str(expr)

    def expression_precedence(self, operator):
        return {
            "||": 1,
            "&&": 2,
            "|": 3,
            "^": 4,
            "&": 5,
            "==": 6,
            "!=": 6,
            "<": 7,
            ">": 7,
            "<=": 7,
            ">=": 7,
            "<<": 8,
            ">>": 8,
            "+": 9,
            "-": 9,
            "*": 10,
            "/": 10,
            "%": 10,
            "unary": 11,
        }.get(operator, 0)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        return self.safe_expression_to_string(expr)

    def map_type(self, vtype):
        """Map types to Metal equivalents, handling both strings and TypeNode objects."""
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
