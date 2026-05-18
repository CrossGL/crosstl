"""CrossGL-to-HLSL code generator."""

from ..ast import (
    AssignmentNode,
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
    ArrayAccessNode,
    ArrayNode,
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


class HLSLCodeGen:
    def __init__(self):
        self.texture_variables = set()
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.current_implicit_texture_samplers = {}
        self.current_implicit_texture_query_lod_samplers = {}
        self.global_implicit_texture_query_lod_samplers = {}
        self.required_texture_query_helpers = set()
        self.required_image_atomic_helpers = set()
        self.comparison_sampler_parameters = {}
        self.implicit_texture_sampler_parameters = {}
        self.function_parameter_names = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = {}
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
            "sampler1D": "Texture1D",
            "sampler2D": "Texture2D",
            "sampler3D": "Texture3D",
            "samplerCube": "TextureCube",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "sampler2DMS": "Texture2DMS<float4>",
            "sampler2DMSArray": "Texture2DMSArray<float4>",
            "sampler2DShadow": "Texture2D",
            "sampler2DArrayShadow": "Texture2DArray",
            "samplerCubeShadow": "TextureCube",
            "samplerCubeArrayShadow": "TextureCubeArray",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "imageCube": "RWTextureCube<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "sampler": "SamplerState",
        }

        self.semantic_map = {
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_IsFrontFace": "FRONT_FACE",
            "gl_PrimitiveID": "PRIMITIVE_ID",
            "gl_ViewID": "SV_ViewID",
            "gl_Layer": "SV_RenderTargetArrayIndex",
            "gl_ViewportIndex": "SV_ViewportArrayIndex",
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

        self.texture_variables = set()
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.current_implicit_texture_samplers = {}
        self.current_implicit_texture_query_lod_samplers = {}
        self.global_implicit_texture_query_lod_samplers = {}
        self.required_texture_query_helpers = set()
        self.required_image_atomic_helpers = set()
        self.comparison_sampler_parameters = {}
        self.implicit_texture_sampler_parameters = {}
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.struct_member_types = collect_struct_member_types(
            getattr(ast, "structs", []), self.type_name_string
        )
        self.function_parameter_names = self.collect_function_parameter_names(ast)
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line:
                code += f"{line}\n"

        code += self.generate_constants(ast)

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

                        semantic = self.semantic_from_node(member)
                        if semantic is None and getattr(member, "name", "") in [
                            "view",
                            "layer",
                            "viewport",
                        ]:
                            # Fallback to name-based mapping for common multiview outputs
                            name_semantics = {
                                "view": "gl_ViewID",
                                "layer": "gl_Layer",
                                "viewport": "gl_ViewportIndex",
                            }
                            semantic = name_semantics.get(member.name)

                        code += f"    {member_type} {member.name}{array_syntax}{self.map_semantic(semantic)};\n"
                code += "};\n"

        global_vars = getattr(ast, "global_variables", [])
        comparison_texture_names, comparison_sampler_names = (
            self.collect_comparison_resources(ast)
        )
        self.comparison_sampler_parameters = self.collect_comparison_sampler_parameters(
            ast
        )
        self.implicit_texture_sampler_parameters = (
            self.collect_implicit_texture_sampler_parameters(ast)
        )
        comparison_sampler_names |= self.collect_comparison_sampler_arguments(
            ast, self.comparison_sampler_parameters
        )
        comparison_texture_names |= self.collect_implicit_comparison_texture_arguments(
            ast, self.implicit_texture_sampler_parameters
        )
        sampler_parameter_names = self.collect_sampler_parameter_names(ast)
        declared_sampler_names = set()
        explicit_sampler_names = set()
        for node in global_vars:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_sampler_type(var_type):
                declared_sampler_names.add(var_name)
                explicit_sampler_names.add(var_name)
        global_implicit_sampler_texture_names = (
            self.collect_global_implicit_sampler_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        global_implicit_query_lod_texture_names = (
            self.collect_global_implicit_query_lod_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )

        texture_register = 0
        sampler_register = 0
        uav_register = 0
        for i, node in enumerate(global_vars):
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

            if hasattr(node, "name"):
                var_name = node.name
            elif hasattr(node, "variable_name"):
                var_name = node.variable_name
            else:
                var_name = f"var{i}"

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if var_name in comparison_sampler_names and mapped_type == "SamplerState":
                mapped_type = "SamplerComparisonState"
            declaration = format_c_style_array_declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            register = ""
            if mapped_type.startswith("Texture"):
                self.texture_variables.add(var_name)
                self.texture_variable_types[var_name] = mapped_type
                register = f" : register(t{texture_register})"
                texture_register += resource_count
            elif mapped_type.startswith("RWTexture"):
                self.texture_variable_types[var_name] = mapped_type
                register = f" : register(u{uav_register})"
                uav_register += resource_count
            elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
                self.sampler_variables.add(var_name)
                register = f" : register(s{sampler_register})"
                sampler_register += resource_count

            code += f"{declaration}{register};\n"

            if mapped_type.startswith("Texture"):
                sampler_name = f"{var_name}Sampler"
                needs_implicit_sampler = var_name in global_implicit_sampler_texture_names
                needs_query_lod_sampler = (
                    var_name in global_implicit_query_lod_texture_names
                )
                needs_comparison_sampler = var_name in comparison_texture_names or (
                    self.is_shadow_sampler_type(vtype)
                    and needs_implicit_sampler
                    and not needs_query_lod_sampler
                )
                if (
                    needs_implicit_sampler
                    and sampler_name not in explicit_sampler_names
                    and not self.is_multisample_sampler_type(vtype)
                ):
                    sampler_type = (
                        "SamplerComparisonState"
                        if needs_comparison_sampler
                        else "SamplerState"
                    )
                    self.sampler_variables.add(sampler_name)
                    code += f"{sampler_type} {sampler_name} : register(s{sampler_register});\n"
                    sampler_register += 1
                if needs_query_lod_sampler:
                    query_sampler_name = (
                        f"{var_name}QuerySampler"
                        if needs_comparison_sampler
                        else sampler_name
                    )
                    self.global_implicit_texture_query_lod_samplers[
                        var_name
                    ] = query_sampler_name
                    if (
                        needs_comparison_sampler
                        and query_sampler_name not in explicit_sampler_names
                        and not self.is_multisample_sampler_type(vtype)
                    ):
                        self.sampler_variables.add(query_sampler_name)
                        code += f"SamplerState {query_sampler_name} : register(s{sampler_register});\n"
                        sampler_register += 1

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
                functions_code += self.generate_function(func, shader_type="vertex")
            elif qualifier_name == "fragment":
                functions_code += self.generate_function(func, shader_type="fragment")
            elif qualifier_name == "compute":
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
                    functions_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                    )
                if hasattr(stage, "local_functions"):
                    stage_name = normalize_stage_name(stage_type)
                    if not stage_matches(target_stage, stage_name):
                        continue
                    for func in stage.local_functions:
                        functions_code += self.generate_function(func)

        code += self.generate_texture_query_helpers()
        code += self.generate_image_atomic_helpers()
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
            code += f"static const {self.map_type(const_type)} {name} = {value_code};\n"

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

    def generate_compute_numthreads(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return f"[numthreads({x}, {y}, {z})]\n"

    def generate_function(
        self, func, indent=0, shader_type=None, execution_config=None
    ):
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        comparison_sampler_parameters = self.comparison_sampler_parameters.get(
            getattr(func, "name", None), set()
        )
        implicit_texture_samplers = self.implicit_texture_sampler_parameters.get(
            getattr(func, "name", None), {}
        )
        implicit_existing_comparison_samplers = {
            data["sampler_name"]
            for data in implicit_texture_samplers.values()
            if data["comparison"] and not data["synthetic"]
        }
        param_names = {getattr(param, "name", None) for param in param_list}
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

            param_type = self.map_resource_parameter_type_with_hint(
                raw_param_type, p, getattr(func, "name", None)
            )
            if self.is_texture_type(raw_param_type) or self.is_image_type(
                raw_param_type
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                if (
                    p.name in comparison_sampler_parameters
                    or p.name in implicit_existing_comparison_samplers
                ):
                    param_type = param_type.replace(
                        "SamplerState", "SamplerComparisonState", 1
                    )

            semantic = self.semantic_from_node(p)

            declaration = format_c_style_array_declaration(param_type, p.name)
            semantic_attr = self.map_semantic(semantic)
            params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )
            if p.name in implicit_texture_samplers:
                sampler_info = implicit_texture_samplers[p.name]
                if sampler_info["synthetic"]:
                    sampler_type = (
                        "SamplerComparisonState"
                        if sampler_info["comparison"]
                        else "SamplerState"
                    )
                    sampler_name = sampler_info["sampler_name"]
                    params.append(f"{sampler_type} {sampler_name}")
                    sampler_parameters.add(sampler_name)
                query_sampler_name = self.implicit_query_lod_sampler_name(
                    p.name, sampler_info
                )
                if (
                    sampler_info.get("query_lod")
                    and query_sampler_name != sampler_info["sampler_name"]
                    and query_sampler_name not in param_names
                ):
                    params.append(f"SamplerState {query_sampler_name}")
                    sampler_parameters.add(query_sampler_name)

        params_str = ", ".join(params)
        shader_map = {"vertex": "VSMain", "fragment": "PSMain", "compute": "CSMain"}
        shader_attr_map = {
            "geometry": "geometry",
            "tessellation_control": "hull",
            "tessellation_evaluation": "domain",
            "mesh": "mesh",
            "amplification": "amplification",
            "task": "amplification",
            "object": "amplification",
            "ray_generation": "raygeneration",
            "ray_intersection": "intersection",
            "ray_closest_hit": "closesthit",
            "ray_any_hit": "anyhit",
            "ray_miss": "miss",
            "ray_callable": "callable",
        }

        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        self.current_function_return_type = raw_return_type

        if hasattr(func, "qualifiers") and func.qualifiers:
            qualifier = func.qualifiers[0] if func.qualifiers else None
        else:
            qualifier = getattr(func, "qualifier", None)

        effective_shader_type = shader_type or qualifier

        if effective_shader_type in shader_map:
            return_semantic = self.map_semantic(self.semantic_from_node(func))
            code += f"// {effective_shader_type.capitalize()} Shader\n"
            if effective_shader_type == "compute":
                code += self.generate_compute_numthreads(execution_config)
            code += f"{return_type} {shader_map[effective_shader_type]}({params_str}){return_semantic} {{\n"
        else:
            shader_attr = shader_attr_map.get(effective_shader_type)
            if shader_attr:
                code += f'[shader("{shader_attr}")]\n'
            code += f"{return_type} {func.name}({params_str}) {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_implicit_texture_samplers = self.current_implicit_texture_samplers
        previous_implicit_texture_query_lod_samplers = (
            self.current_implicit_texture_query_lod_samplers
        )
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = texture_parameters
        self.current_implicit_texture_samplers = {
            texture_name: sampler_info["sampler_name"]
            for texture_name, sampler_info in implicit_texture_samplers.items()
        }
        self.current_implicit_texture_query_lod_samplers = {
            texture_name: self.implicit_query_lod_sampler_name(
                texture_name, sampler_info
            )
            for texture_name, sampler_info in implicit_texture_samplers.items()
            if sampler_info.get("query_lod")
        }
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_implicit_texture_samplers = previous_implicit_texture_samplers
        self.current_implicit_texture_query_lod_samplers = (
            previous_implicit_texture_query_lod_samplers
        )
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types

        code += "  " * indent + "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if hasattr(stmt, "var_type"):
                vtype = stmt.var_type
            elif hasattr(stmt, "vtype"):
                vtype = stmt.vtype
            else:
                vtype = "float"
            self.local_variable_types[stmt.name] = self.type_name_string(vtype)

            declaration = format_c_style_array_declaration(
                self.map_type(vtype), stmt.name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                init_expr = self.generate_expression_with_expected(
                    stmt.initial_value, vtype
                )
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"

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
                        f"{indent_str}return "
                        f"{self.generate_expression_with_expected(stmt.value, self.current_function_return_type)};\n"
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
            "float2",
            "float3",
            "float4",
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
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

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

        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        then_body = getattr(node, "then_branch", getattr(node, "if_body", []))
        else_body = getattr(node, "else_branch", getattr(node, "else_body", []))

        code = f"{indent_str}if ({self.generate_expression(condition)}) {{\n"

        if hasattr(then_body, "statements"):
            for stmt in then_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(then_body, list):
            for stmt in then_body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += self.generate_statement(then_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
                for stmt in else_if_body:
                    code += self.generate_statement(stmt, indent + 1)
                code += f"{indent_str}}}"

        if else_body:
            code += " else {\n"
            if hasattr(else_body, "statements"):
                for stmt in else_body.statements:
                    code += self.generate_statement(stmt, indent + 1)
            elif isinstance(else_body, list):
                for stmt in else_body:
                    code += self.generate_statement(stmt, indent + 1)
            else:
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
                init = self.generate_for_initializer(node.init)

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

        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += self.generate_statement(body, indent + 1)

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
                    "Unsupported match arm for HLSL codegen; only unguarded "
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
        elif isinstance(expr, (int, float)):
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
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
                    return f'"{value}"'
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return getattr(expr, "name", str(expr))
        elif isinstance(expr, VariableNode):
            return expr.name
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({left} {self.map_operator(op)} {right})"
        elif isinstance(expr, AssignmentNode):
            # Handle assignment as expression
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand = self.generate_expression(getattr(expr, "operand", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"{self.map_operator(op)}{operand}"
        elif isinstance(expr, WaveOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, RayTracingOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, MeshOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{query}.{expr.operation}({args_str})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]"
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
            func_name = None
            if hasattr(func_expr, "name"):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            texture_call = self.generate_texture_call(func_name, args)
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
                mapped_type = self.map_type(func_name)
                args_str = ", ".join(self.generate_expression(arg) for arg in args)
                return f"{mapped_type}({args_str})"
            args_str = ", ".join(self.generate_call_arguments(func_name, args))
            return f"{callee}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
            member = getattr(expr, "member", "")
            obj = self.generate_expression(obj_expr)
            return f"{obj}.{member}"
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            condition = self.generate_expression(getattr(expr, "condition", ""))
            true_expr = self.generate_expression(getattr(expr, "true_expr", ""))
            false_expr = self.generate_expression(getattr(expr, "false_expr", ""))
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def collect_comparison_resources(self, root):
        texture_names = set()
        sampler_names = set()
        visited = set()
        comparison_funcs = {
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
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }

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

            if isinstance(value, FunctionCallNode):
                func_expr = getattr(value, "function", getattr(value, "name", None))
                func_name = self.expression_name(func_expr)
                args = getattr(value, "arguments", getattr(value, "args", []))
                if func_name in comparison_funcs and len(args) >= 3:
                    texture_name = self.expression_name(args[0])
                    if texture_name:
                        texture_names.add(texture_name)
                    if len(args) >= 4:
                        sampler_name = self.expression_name(args[1])
                        if sampler_name:
                            sampler_names.add(sampler_name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return texture_names, sampler_names

    def collect_global_texture_names(self, root):
        texture_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            mapped_type = self.map_resource_type_with_format(var_type, node)
            if var_name and mapped_type.startswith("Texture"):
                texture_names.add(var_name)
        return texture_names

    def collect_global_implicit_sampler_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        functions = self.collect_functions(root)
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
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureQueryLod",
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
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param in callee_implicit:
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_global_implicit_query_lod_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name != "textureQueryLod":
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info.get("query_lod"):
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_sampler_parameter_names(self, root):
        sampler_names = set()
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

            for param in getattr(value, "parameters", getattr(value, "params", [])):
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_sampler_type(param_type):
                    sampler_names.add(param.name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_names

    def collect_comparison_sampler_parameters(self, root):
        comparison_params = {}
        functions = self.collect_functions(root)

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            sampler_params = {
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not sampler_params:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                if self.expression_name(func_expr) not in {
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
                    "textureGatherCompare",
                    "textureGatherCompareOffset",
                }:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 4:
                    continue
                sampler_name = self.expression_name(args[1])
                if sampler_name in sampler_params:
                    comparison_params.setdefault(func_name, set()).add(sampler_name)

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                sampler_param_names = {
                    param.name
                    for param in params
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not sampler_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, comparison_params, callee_name
                    )
                    if not required_indices:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name in sampler_param_names:
                            current = comparison_params.setdefault(func_name, set())
                            if arg_name not in current:
                                current.add(arg_name)
                                changed = True

        return comparison_params

    def collect_comparison_sampler_arguments(self, root, comparison_params):
        comparison_sampler_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                required_indices = self.comparison_sampler_parameter_indices(
                    functions, comparison_params, callee_name
                )
                if not required_indices:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                for index in required_indices:
                    if index >= len(args):
                        continue
                    arg_name = self.expression_name(args[index])
                    if arg_name:
                        comparison_sampler_names.add(arg_name)

        return comparison_sampler_names

    def comparison_sampler_parameter_indices(
        self, functions, comparison_params, function_name
    ):
        if not function_name or function_name not in comparison_params:
            return set()

        for func in functions:
            if getattr(func, "name", None) != function_name:
                continue
            indices = set()
            params = getattr(func, "parameters", getattr(func, "params", []))
            for index, param in enumerate(params):
                if param.name in comparison_params[function_name]:
                    indices.add(index)
            return indices

        return set()

    def collect_implicit_texture_sampler_parameters(self, root):
        implicit_params = {}
        functions = self.collect_functions(root)
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
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureQueryLod",
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
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue

            params = getattr(func, "parameters", getattr(func, "params", []))
            param_names = {param.name for param in params}
            texture_param_types = {
                param.name: getattr(param, "param_type", getattr(param, "vtype", None))
                for param in params
                if self.is_texture_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            sampler_params = {
                param.name
                for param in params
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not texture_param_types:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                texture_func = self.expression_name(func_expr)
                if texture_func not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue

                texture_name = self.expression_name(args[0])
                if texture_name not in texture_param_types:
                    continue
                if self.has_explicit_sampler_argument(
                    texture_func, args, sampler_params
                ):
                    continue

                query_lod = texture_func == "textureQueryLod"
                comparison = not query_lod and (
                    texture_func
                    in {
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
                        "textureGatherCompare",
                        "textureGatherCompareOffset",
                    }
                    or self.is_shadow_sampler_type(texture_param_types[texture_name])
                )
                self.add_implicit_texture_sampler_parameter(
                    implicit_params,
                    func_name,
                    texture_name,
                    comparison,
                    param_names,
                    query_lod=query_lod,
                )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                param_names = {param.name for param in params}
                texture_param_names = {
                    param.name
                    for param in params
                    if self.is_texture_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not texture_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    callee_implicit = implicit_params.get(callee_name, {})
                    if not callee_implicit:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee_params = self.function_parameter_names.get(callee_name, [])
                    for texture_param, sampler_info in callee_implicit.items():
                        try:
                            texture_index = callee_params.index(texture_param)
                        except ValueError:
                            continue
                        if texture_index >= len(args):
                            continue
                        arg_name = self.expression_name(args[texture_index])
                        if arg_name not in texture_param_names:
                            continue
                        changed |= self.add_implicit_texture_sampler_parameter(
                            implicit_params,
                            func_name,
                            arg_name,
                            sampler_info["comparison"],
                            param_names,
                            query_lod=sampler_info.get("query_lod", False),
                        )

        return implicit_params

    def add_implicit_texture_sampler_parameter(
        self,
        implicit_params,
        func_name,
        texture_name,
        comparison,
        param_names,
        query_lod=False,
    ):
        sampler_name = f"{texture_name}Sampler"
        new_info = {
            "sampler_name": sampler_name,
            "comparison": comparison,
            "synthetic": sampler_name not in param_names,
            "query_lod": query_lod,
        }
        current = implicit_params.setdefault(func_name, {}).get(texture_name)
        if current is None:
            implicit_params[func_name][texture_name] = new_info
            return True
        changed = False
        if comparison and not current["comparison"]:
            current["comparison"] = True
            changed = True
        if query_lod and not current.get("query_lod"):
            current["query_lod"] = True
            changed = True
        return changed

    def has_explicit_sampler_argument(self, func_name, args, sampler_names):
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
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }:
            return len(args) >= 4 and self.expression_name(args[1]) in sampler_names
        return len(args) >= 3 and self.expression_name(args[1]) in sampler_names

    def collect_implicit_comparison_texture_arguments(self, root, implicit_params):
        texture_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info["comparison"]:
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name:
                        texture_names.add(texture_name)

        return texture_names

    def collect_function_parameter_names(self, root):
        parameter_names = {}
        for func in self.collect_functions(root):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_names[func_name] = [
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
            ]
        return parameter_names

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def collect_resource_array_size_hints(self, ast):
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_resource_globals(ast),
            function_arrays=self.collect_unsized_resource_parameters(ast),
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
        for func in self.collect_functions(ast):
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

    def is_shadow_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def is_multisample_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_sampler_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return mapped_type in {"SamplerState", "SamplerComparisonState"}

    def is_texture_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith("Texture")

    def is_image_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith("RWTexture")

    def is_resource_parameter_type(self, vtype):
        return (
            self.is_texture_type(vtype)
            or self.is_sampler_type(vtype)
            or self.is_image_type(vtype)
        )

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

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        sampler_name = (
            self.generate_expression(args[1])
            if explicit_sampler
            else self.current_implicit_texture_samplers.get(
                texture_base_name, f"{texture_base_name}Sampler"
            )
        )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_name, coord, extra_args

    def generate_call_arguments(self, func_name, args):
        generated_args = []
        implicit_samplers = self.implicit_texture_sampler_parameters.get(func_name, {})
        param_names = self.function_parameter_names.get(func_name, [])

        for index, arg in enumerate(args):
            generated_args.append(self.generate_expression(arg))
            if index >= len(param_names):
                continue
            texture_param = param_names[index]
            if texture_param not in implicit_samplers:
                continue
            sampler_info = implicit_samplers[texture_param]
            generated_args.append(self.generate_implicit_sampler_argument(arg))
            query_sampler_name = self.implicit_query_lod_sampler_name(
                texture_param, sampler_info
            )
            if (
                sampler_info.get("query_lod")
                and query_sampler_name != sampler_info["sampler_name"]
            ):
                generated_args.append(
                    self.generate_implicit_query_lod_sampler_argument(arg, sampler_info)
                )

        return generated_args

    def generate_implicit_sampler_argument(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_samplers:
            return self.current_implicit_texture_samplers[texture_name]
        if texture_name:
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        return f"{texture_expr}Sampler"

    def implicit_query_lod_sampler_name(self, texture_name, sampler_info):
        if sampler_info.get("query_lod") and sampler_info.get("comparison"):
            return f"{texture_name}QuerySampler"
        return sampler_info["sampler_name"]

    def generate_implicit_query_lod_sampler_argument(self, texture_arg, sampler_info=None):
        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_query_lod_samplers:
            return self.current_implicit_texture_query_lod_samplers[texture_name]
        if texture_name in self.global_implicit_texture_query_lod_samplers:
            return self.global_implicit_texture_query_lod_samplers[texture_name]
        if texture_name:
            if sampler_info and sampler_info.get("comparison"):
                return f"{texture_name}QuerySampler"
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        if sampler_info and sampler_info.get("comparison"):
            return f"{texture_expr}QuerySampler"
        return f"{texture_expr}Sampler"

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def is_float_vector_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {
            "RWTexture2D<float4>",
            "RWTexture3D<float4>",
            "RWTexture2DArray<float4>",
        }

    def two_component_image_store_constructor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return {
            "RWTexture2D<float2>": ("float2", "0.0"),
            "RWTexture3D<float2>": ("float2", "0.0"),
            "RWTexture2DArray<float2>": ("float2", "0.0"),
            "RWTexture2D<int2>": ("int2", "0"),
            "RWTexture3D<int2>": ("int2", "0"),
            "RWTexture2DArray<int2>": ("int2", "0"),
            "RWTexture2D<uint2>": ("uint2", "0u"),
            "RWTexture3D<uint2>": ("uint2", "0u"),
            "RWTexture2DArray<uint2>": ("uint2", "0u"),
        }.get(texture_type)

    def texture_query_dimension(self, texture_type):
        if texture_type in {"Texture1D"}:
            return 1
        if texture_type in {
            "Texture2D",
            "TextureCube",
            "Texture2DMS<float4>",
        }:
            return 2
        return 3

    def texture_query_helper_key(self, helper_name, texture_type):
        if not texture_type:
            return None
        return helper_name, texture_type

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        key = self.texture_query_helper_key("textureSize", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
            return f"textureSize({texture_name})"
        lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
        return f"textureSize({texture_name}, {lod})"

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        texture_type = self.texture_resource_type(texture_arg)
        key = self.texture_query_helper_key("textureQueryLevels", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        return f"textureQueryLevels({texture_name})"

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type == "Texture2DArray":
            return self.vector_component(coord, "xy")
        if texture_type == "TextureCubeArray":
            return self.vector_component(coord, "xyz")
        return coord

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

    def texture_gather_method(self, component_arg):
        if component_arg is None:
            return "Gather"

        methods = {
            0: "GatherRed",
            1: "GatherGreen",
            2: "GatherBlue",
            3: "GatherAlpha",
        }
        return methods.get(self.literal_int_value(component_arg))

    def texture_gather_component_expression(self, texture_name, method_args, component):
        arg_list = ", ".join(method_args)
        component_calls = [
            f"{texture_name}.{method}({arg_list})"
            for method in (
                "GatherRed",
                "GatherGreen",
                "GatherBlue",
                "GatherAlpha",
            )
        ]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return f"/* unsupported DirectX texture gather: {func_name} {reason} */ float4(0.0)"

    def texture_sample_offset_supported(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {
            "",
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture3D",
        }

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return f"/* unsupported DirectX texture offset: {func_name} {reason} */ float4(0.0)"

    def generate_texture_sample_offset_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_sample_offset_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if not self.texture_sample_offset_supported(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, "offsets require 1D, 2D, 2D-array, or 3D textures"
            )

        if func_name == "textureOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_sample_offset_call(
                    func_name, "requires one offset argument"
                )
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.Sample({sampler_name}, {coord}, {offset})"

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_sample_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleLevel("
                f"{sampler_name}, {coord}, {lod}, {offset})"
            )

        if func_name == "textureGradOffset":
            if len(extra_args) != 3:
                return self.unsupported_texture_sample_offset_call(
                    func_name,
                    "requires gradient x, gradient y, and offset arguments",
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {coord}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_sample_offset_call(
            func_name, "is not a supported texture offset operation"
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return f"/* unsupported DirectX projected texture: {func_name} {reason} */ float4(0.0)"

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "Texture1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "Texture2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "Texture3D": {
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
        return (
            f"{self.vector_component(coord, numerator)} / "
            f"{self.vector_component(coord, divisor)}"
        )

    def generate_texture_projected_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
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
                return f"{texture_name}.Sample({sampler_name}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias})"
                )
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.Sample("
                    f"{sampler_name}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias}, {offset})"
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
                f"{texture_name}.SampleLevel("
                f"{sampler_name}, {projected_coord}, {lod})"
            )

        if func_name == "textureProjLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleLevel("
                f"{sampler_name}, {projected_coord}, {lod}, {offset})"
            )

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {projected_coord}, {ddx}, {ddy})"
            )

        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return (
            f"{texture_name}.SampleGrad("
            f"{sampler_name}, {projected_coord}, {ddx}, {ddy}, {offset})"
        )

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
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

        method_args = [sampler_name, coord] + [
            self.generate_expression(offset_arg) for offset_arg in offset_args
        ]
        method = self.texture_gather_method(component_arg)
        if method is not None:
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, "component literal must be 0, 1, 2, or 3"
            )

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_compare_offset_supported(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {"Texture2D", "Texture2DArray"}

    def unsupported_texture_compare_call(self, func_name, reason):
        return f"/* unsupported DirectX texture compare: {func_name} {reason} */ 0.0"

    def texture_compare_projected_coordinate(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "Texture2D":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            return f"{self.vector_component(coord, 'xy')} / {divisor}"

        if texture_type != "Texture2DArray" or coord_type not in {"vec4", "float4"}:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        return f"float3({projected_coord}, {self.vector_component(coord, 'z')})"

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
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
            texture_type = self.texture_resource_type(args[0])
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            projected_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord
            )
            if projected_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates",
                )

            if func_name == "textureCompareProj":
                if len(extra_args) != 1:
                    return self.unsupported_texture_compare_call(
                        func_name, "accepts no extra arguments"
                    )
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare})"
                )

            if func_name == "textureCompareProjOffset":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and offset arguments"
                    )
                offset = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare}, {offset})"
                )

            if func_name == "textureCompareProjLod":
                if len(extra_args) != 2:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare and lod arguments"
                    )
                lod = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleCmpLevel("
                    f"{sampler_name}, {projected_coord}, {compare}, {lod})"
                )

            if func_name == "textureCompareProjLodOffset":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name, "requires compare, lod, and offset arguments"
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                return (
                    f"{texture_name}.SampleCmpLevel("
                    f"{sampler_name}, {projected_coord}, {compare}, {lod}, {offset})"
                )

            if func_name == "textureCompareProjGrad":
                if len(extra_args) != 3:
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "requires compare, gradient x, and gradient y arguments",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return (
                    f"{texture_name}.SampleCmpGrad("
                    f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy})"
                )

            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy}, {offset})"
            )

        if func_name == "textureCompare":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare})"

        if func_name == "textureCompareOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and offset arguments"
                )
            texture_type = self.texture_resource_type(args[0])
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array textures"
                )
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare}, {offset})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 2:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare and lod arguments"
                )
            lod = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleCmpLevel("
                f"{sampler_name}, {coord}, {compare}, {lod})"
            )

        if func_name == "textureCompareLodOffset":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name, "requires compare, lod, and offset arguments"
                )
            texture_type = self.texture_resource_type(args[0])
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array textures"
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleCmpLevel("
                f"{sampler_name}, {coord}, {compare}, {lod}, {offset})"
            )

        if func_name == "textureCompareGrad":
            if len(extra_args) != 3:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, and gradient y arguments",
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy})"
            )

        if func_name == "textureCompareGradOffset":
            if len(extra_args) != 4:
                return self.unsupported_texture_compare_call(
                    func_name,
                    "requires compare, gradient x, gradient y, and offset arguments",
                )
            texture_type = self.texture_resource_type(args[0])
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "offsets require 2D or 2D-array textures"
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_compare_call(
            func_name, "is not a supported shadow compare operation"
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {"Texture2D", "Texture2DArray"}

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return (
            f"/* unsupported DirectX texture gather compare: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
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
            return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare})"

        if len(extra_args) != 2:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires compare and offset arguments"
            )
        texture_type = self.texture_resource_type(args[0])
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, "offsets require 2D or 2D-array textures"
            )
        offset = self.generate_expression(extra_args[1])
        return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare}, {offset})"

    def generate_texture_query_helpers(self):
        if not self.required_texture_query_helpers:
            return ""

        helpers = []
        for helper_name, texture_type in sorted(self.required_texture_query_helpers):
            if helper_name == "textureSize":
                helpers.append(self.generate_texture_size_helper(texture_type))
            elif helper_name == "textureQueryLevels":
                helpers.append(self.generate_texture_query_levels_helper(texture_type))

        return "".join(helper for helper in helpers if helper)

    def image_atomic_helper_name(self, operation, texture_type):
        suffixes = {
            "RWTexture2D<int>": "iimage2D",
            "RWTexture2D<uint>": "uimage2D",
            "RWTexture3D<int>": "iimage3D",
            "RWTexture3D<uint>": "uimage3D",
            "RWTexture2DArray<int>": "iimage2DArray",
            "RWTexture2DArray<uint>": "uimage2DArray",
        }
        suffix = suffixes.get(texture_type)
        if not suffix:
            return None
        return f"{operation}_{suffix}"

    def image_atomic_helper_return_type(self, texture_type):
        if texture_type in {
            "RWTexture2D<int>",
            "RWTexture3D<int>",
            "RWTexture2DArray<int>",
        }:
            return "int"
        if texture_type in {
            "RWTexture2D<uint>",
            "RWTexture3D<uint>",
            "RWTexture2DArray<uint>",
        }:
            return "uint"
        return None

    def image_atomic_helper_coord_type(self, texture_type):
        if texture_type in {"RWTexture2D<int>", "RWTexture2D<uint>"}:
            return "int2"
        if texture_type in {
            "RWTexture3D<int>",
            "RWTexture3D<uint>",
            "RWTexture2DArray<int>",
            "RWTexture2DArray<uint>",
        }:
            return "int3"
        return None

    def image_atomic_intrinsic(self, operation):
        return {
            "imageAtomicAdd": "InterlockedAdd",
            "imageAtomicMin": "InterlockedMin",
            "imageAtomicMax": "InterlockedMax",
            "imageAtomicAnd": "InterlockedAnd",
            "imageAtomicOr": "InterlockedOr",
            "imageAtomicXor": "InterlockedXor",
            "imageAtomicExchange": "InterlockedExchange",
            "imageAtomicCompSwap": "InterlockedCompareExchange",
        }.get(operation)

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None
        if operation == "imageAtomicCompSwap":
            if len(args) < 4:
                return None
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            image_type = self.texture_resource_type(args[0])
            helper_name = self.image_atomic_helper_name(operation, image_type)
            if not helper_name:
                return None
            self.required_image_atomic_helpers.add((operation, image_type))
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"
        if len(args) < 3:
            return None
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        value = self.generate_expression(args[2])
        image_type = self.texture_resource_type(args[0])
        helper_name = self.image_atomic_helper_name(operation, image_type)
        if not helper_name:
            return None
        self.required_image_atomic_helpers.add((operation, image_type))
        return f"{helper_name}({image_name}, {coord}, {value})"

    def generate_image_atomic_helpers(self):
        if not self.required_image_atomic_helpers:
            return ""

        helpers = []
        for operation, texture_type in sorted(self.required_image_atomic_helpers):
            helper_name = self.image_atomic_helper_name(operation, texture_type)
            return_type = self.image_atomic_helper_return_type(texture_type)
            coord_type = self.image_atomic_helper_coord_type(texture_type)
            intrinsic = self.image_atomic_intrinsic(operation)
            if not helper_name or not return_type or not coord_type or not intrinsic:
                continue
            if operation == "imageAtomicCompSwap":
                helpers.append(
                    f"{return_type} {helper_name}({texture_type} image, {coord_type} coord, {return_type} compareValue, {return_type} value) {{\n"
                    f"    {return_type} original;\n"
                    "    InterlockedCompareExchange(image[coord], compareValue, value, original);\n"
                    "    return original;\n"
                    "}\n\n"
                )
                continue
            helpers.append(
                f"{return_type} {helper_name}({texture_type} image, {coord_type} coord, {return_type} value) {{\n"
                f"    {return_type} original;\n"
                f"    {intrinsic}(image[coord], value, original);\n"
                "    return original;\n"
                "}\n\n"
            )

        return "".join(helpers)

    def generate_texture_size_helper(self, texture_type):
        dimension = self.texture_query_dimension(texture_type)
        return_type = "int" if dimension == 1 else f"int{dimension}"
        if texture_type == "Texture1D":
            return (
                f"{return_type} textureSize({texture_type} tex, int lod) {{\n"
                "    uint width;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(lod, width, levels);\n"
                "    return int(width);\n"
                "}\n\n"
            )
        if texture_type in {"Texture2D", "TextureCube"}:
            return (
                f"{return_type} textureSize({texture_type} tex, int lod) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(lod, width, height, levels);\n"
                "    return int2(width, height);\n"
                "}\n\n"
            )
        if texture_type in {"Texture2DArray", "TextureCubeArray"}:
            return (
                f"{return_type} textureSize({texture_type} tex, int lod) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint elements;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(lod, width, height, elements, levels);\n"
                "    return int3(width, height, elements);\n"
                "}\n\n"
            )
        if texture_type == "Texture3D":
            return (
                f"{return_type} textureSize({texture_type} tex, int lod) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint depth;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(lod, width, height, depth, levels);\n"
                "    return int3(width, height, depth);\n"
                "}\n\n"
            )
        if texture_type == "Texture2DMS<float4>":
            return (
                f"{return_type} textureSize({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint samples;\n"
                "    tex.GetDimensions(width, height, samples);\n"
                "    return int2(width, height);\n"
                "}\n\n"
            )
        if texture_type == "Texture2DMSArray<float4>":
            return (
                f"{return_type} textureSize({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint elements;\n"
                "    uint samples;\n"
                "    tex.GetDimensions(width, height, elements, samples);\n"
                "    return int3(width, height, elements);\n"
                "}\n\n"
            )
        return ""

    def generate_texture_query_levels_helper(self, texture_type):
        if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
            return (
                f"int textureQueryLevels({texture_type} tex) {{\n"
                "    return 1;\n"
                "}\n\n"
            )
        if texture_type == "Texture1D":
            return (
                f"int textureQueryLevels({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(0, width, levels);\n"
                "    return int(levels);\n"
                "}\n\n"
            )
        if texture_type in {"Texture2D", "TextureCube"}:
            return (
                f"int textureQueryLevels({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(0, width, height, levels);\n"
                "    return int(levels);\n"
                "}\n\n"
            )
        if texture_type in {"Texture2DArray", "TextureCubeArray"}:
            return (
                f"int textureQueryLevels({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint elements;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(0, width, height, elements, levels);\n"
                "    return int(levels);\n"
                "}\n\n"
            )
        if texture_type == "Texture3D":
            return (
                f"int textureQueryLevels({texture_type} tex) {{\n"
                "    uint width;\n"
                "    uint height;\n"
                "    uint depth;\n"
                "    uint levels;\n"
                "    tex.GetDimensions(0, width, height, depth, levels);\n"
                "    return int(levels);\n"
                "}\n\n"
            )
        return ""

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
            "rg8": "float2",
            "rg8_snorm": "float2",
            "rg16": "float2",
            "rg16_snorm": "float2",
            "rg16f": "float2",
            "rg8i": "int2",
            "rg16i": "int2",
            "rg8ui": "uint2",
            "rg16ui": "uint2",
            "rg32f": "float2",
            "rg32i": "int2",
            "rg32ui": "uint2",
            "rgba8": "float4",
            "rgba8_snorm": "float4",
            "rgba16": "float4",
            "rgba16_snorm": "float4",
            "rgba16f": "float4",
            "rgba32f": "float4",
            "rgba8i": "int4",
            "rgba16i": "int4",
            "rgba32i": "int4",
            "rgba8ui": "uint4",
            "rgba16ui": "uint4",
            "rgba32ui": "uint4",
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

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_resource_parameter_type(base_type):
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
                if self.is_resource_parameter_type(base_type):
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
        explicit_format = self.explicit_image_format(node) if node is not None else None
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "imageCube": "RWTextureCube",
        }
        texture_type = texture_types.get(base_type)
        if component_type and texture_type:
            return f"{texture_type}<{component_type}>"
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

    def generate_texture_call(self, func_name, args):
        if not func_name:
            return None

        if func_name == "textureSize" and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if func_name == "textureQueryLevels" and args:
            return self.texture_query_levels_expression(args[0])

        if func_name == "textureQueryLod" and len(args) >= 2:
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, sampler_name, coord, _ = parts
            if not self.is_explicit_sampler_argument(args):
                sampler_name = self.generate_implicit_query_lod_sampler_argument(args[0])
            coord = self.texture_query_lod_coordinate(
                self.texture_resource_type(args[0]), coord
            )
            return (
                f"float2({texture_name}.CalculateLevelOfDetailUnclamped({sampler_name}, {coord}), "
                f"{texture_name}.CalculateLevelOfDetail({sampler_name}, {coord}))"
            )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            return self.image_atomic_expression(func_name, args)

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            load_expr = f"{image_name}[{coord}]"
            texture_type = self.texture_resource_type(args[0])
            if self.is_float_vector_image_resource(
                texture_type
            ) and self.is_scalar_value_type(self.current_expression_expected_type):
                return f"{load_expr}.x"
            if self.two_component_image_store_constructor(
                texture_type
            ) and self.is_scalar_value_type(self.current_expression_expected_type):
                return f"{load_expr}.x"
            return load_expr

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            texture_type = self.texture_resource_type(args[0])
            if self.is_float_vector_image_resource(
                texture_type
            ) and self.is_scalar_value_type(self.expression_result_type(args[2])):
                value = f"float4({value})"
            two_component_constructor = self.two_component_image_store_constructor(
                texture_type
            )
            if two_component_constructor and self.is_scalar_value_type(
                self.expression_result_type(args[2])
            ):
                constructor, zero_value = two_component_constructor
                value = f"{constructor}({value}, {zero_value})"
            return f"{image_name}[{coord}] = {value}"

        if len(args) < 2:
            return None

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

        if func_name in {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
        }:
            return self.generate_texture_sample_offset_call(func_name, args)

        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            return self.generate_texture_projected_call(func_name, args)

        texture_ops = {
            "texture": "Sample",
            "textureLod": "SampleLevel",
            "textureGrad": "SampleGrad",
        }
        if func_name in texture_ops:
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, sampler_name, coord, extra_args = parts
            mapped_args = [coord] + [
                self.generate_expression(arg) for arg in extra_args
            ]
            return f"{texture_name}.{texture_ops[func_name]}({sampler_name}, {', '.join(mapped_args)})"

        if func_name == "texelFetch" and len(args) >= 3:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            texture_type = self.texture_resource_type(args[0])
            if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
                return f"{texture_name}.Load({coord}, {lod})"
            load_coord_type = (
                "int4" if texture_type in {"Texture2DArray", "Texture3D"} else "int3"
            )
            return f"{texture_name}.Load({load_coord_type}({coord}, {lod}))"

        if func_name == "texelFetchOffset" and len(args) >= 4:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            offset = self.generate_expression(args[3])
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
                return "/* unsupported DirectX texel fetch offset: multisample textures do not support offsets */ float4(0.0)"
            if texture_type == "Texture2DArray":
                coord_xy = self.vector_component(coord, "xy")
                layer = self.vector_component(coord, "z")
                return (
                    f"{texture_name}.Load("
                    f"int4(({coord_xy} + {offset}), {layer}, {lod}))"
                )
            if texture_type == "Texture3D":
                return f"{texture_name}.Load(int4(({coord} + {offset}), {lod}))"
            if texture_type == "Texture1D":
                return f"{texture_name}.Load(int2(({coord} + {offset}), {lod}))"
            if texture_type == "Texture1DArray":
                coord_x = self.vector_component(coord, "x")
                layer = self.vector_component(coord, "y")
                return (
                    f"{texture_name}.Load("
                    f"int3(({coord_x} + {offset}), {layer}, {lod}))"
                )
            return f"{texture_name}.Load(int3(({coord} + {offset}), {lod}))"

        return None

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.rows == type_node.cols:
                    return f"float{type_node.rows}x{type_node.rows}"
                else:
                    return f"float{type_node.cols}x{type_node.rows}"
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

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def map_type(self, vtype):
        """Map types to DirectX equivalents, handling both strings and TypeNode objects."""
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
