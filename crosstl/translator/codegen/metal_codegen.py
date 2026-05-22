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
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
    floating_coordinate_dimension,
    integer_coordinate_dimension,
    is_floating_scalar_type,
    is_integer_scalar_type,
    is_numeric_scalar_type,
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    OFFSET_DIMENSION_INTRINSIC_NAMES,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gather_component_argument_index,
    texture_gradient_argument_indices,
    texture_intrinsic_allowed_argument_counts,
    texture_intrinsic_max_argument_count,
    texture_intrinsic_min_argument_count,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
    texture_sample_index_argument_index,
)
from .stage_utils import (
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .resource_arrays import collect_resource_array_size_hints
from .glsl_buffer_layout import (
    byte_offset_expression,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_compound_binary_operator,
    matrix_column_offsets,
    vector_component_offsets,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    explicit_image_access,
    explicit_image_format,
    image_format_channel_count,
    image_format_component_type,
    image_access_satisfies_requirement,
    is_image_format_attribute,
    is_metal_float_image_resource,
    is_metal_integer_image_type,
    is_metal_storage_image_resource,
    is_resource_access_attribute,
    is_scalar_image_format,
    is_two_component_image_format,
    metal_storage_image_access_agnostic_type,
    metal_storage_image_component_type,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_value_expression,
    storage_image_two_component_store_expression,
    supported_image_formats,
)


class CharTypeMapper:
    """Normalize CrossGL char-like scalar and vector types for Metal output."""

    def map_char_type(self, vtype):
        """Return the Metal-compatible integer type for a char-like type."""
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
    """Emit Metal Shading Language from the shared CrossGL translator AST."""

    def __init__(self):
        """Initialize Metal type maps and per-generation resource state."""
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None
        self.gl_position = False
        self.char_mapper = CharTypeMapper()
        self.texture_variables = []
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.cbuffer_variables = []
        self.cbuffer_parameter_names = {}
        self.cbuffer_member_references = {}
        self.ambiguous_cbuffer_members = set()
        self.cbuffers_by_name = {}
        self.user_function_names = set()
        self.function_parameter_names = {}
        self.function_image_access_requirements = {}
        self.function_cbuffer_dependencies = {}
        self.function_global_resource_dependencies = {}
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
        self.structs_by_name = {}
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.glsl_buffer_block_variables = []
        self.metal_temp_variable_index = 0
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
            "sampler1DArray": "texture1d_array<float>",
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
            "iimage1D": "texture1d<int, access::read_write>",
            "iimage1DArray": "texture1d_array<int, access::read_write>",
            "iimage2D": "texture2d<int, access::read_write>",
            "iimage3D": "texture3d<int, access::read_write>",
            "iimage2DArray": "texture2d_array<int, access::read_write>",
            "uimage1D": "texture1d<uint, access::read_write>",
            "uimage1DArray": "texture1d_array<uint, access::read_write>",
            "uimage2D": "texture2d<uint, access::read_write>",
            "uimage3D": "texture3d<uint, access::read_write>",
            "uimage2DArray": "texture2d_array<uint, access::read_write>",
            "image1D": "texture1d<float, access::read_write>",
            "image1DArray": "texture1d_array<float, access::read_write>",
            "image2D": "texture2d<float, access::read_write>",
            "image3D": "texture3d<float, access::read_write>",
            "imageCube": "texture2d_array<float, access::read_write>",
            "image2DArray": "texture2d_array<float, access::read_write>",
            # Matrix Types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float3x2",
            "mat2x4": "float4x2",
            "mat3x2": "float2x3",
            "mat3x3": "float3x3",
            "mat3x4": "float4x3",
            "mat4x2": "float2x4",
            "mat4x3": "float3x4",
            "mat4x4": "float4x4",
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
        """Generate complete Metal Shading Language source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        """Generate Metal source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to Metal, optionally filtering stage entry points."""
        target_stage = normalize_stage_name(target_stage)

        self.texture_variables = []
        self.sampler_variables = []
        self.structured_buffer_variables = []
        self.glsl_buffer_block_variables = []
        self.lowered_glsl_buffer_blocks = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.metal_temp_variable_index = 0
        self.cbuffer_variables = getattr(ast, "cbuffers", []) or []
        self.cbuffers_by_name = {
            cbuffer.name: cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None)
        }
        all_functions = self.all_functions(ast)
        self.user_function_names = {
            func.name for func in all_functions if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(all_functions)
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                all_functions,
                self.function_parameter_names,
                self.iter_ast_nodes,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.function_cbuffer_dependencies = self.collect_function_cbuffer_dependencies(
            all_functions
        )
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )
        if not self.cbuffer_variables:
            self.ambiguous_cbuffer_members = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.function_global_resource_dependencies = {}
        self.required_image_atomic_compare_helpers = set()
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.validate_global_resource_shadows(ast)
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.structs_by_name = {
            node.name: node
            for node in getattr(ast, "structs", [])
            if isinstance(node, StructNode)
        }
        global_vars = getattr(ast, "global_variables", [])
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(global_vars)
        )
        (
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        ) = collect_lowered_glsl_buffer_blocks(
            global_vars,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="metal_type",
            unsupported_type_message=(
                "type is not supported by Metal pointer/offset lowering"
            ),
        )
        self.lowered_glsl_buffer_block_struct_names = {
            block["type_name"] for block in self.lowered_glsl_buffer_blocks.values()
        }
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
                if node.name in self.lowered_glsl_buffer_block_struct_names:
                    continue
                if node.name in self.glsl_buffer_block_struct_names:
                    code += self.glsl_buffer_block_diagnostic(
                        "Metal", node.name, None, None
                    )
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
                        semantic = self.semantic_from_node(member)
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

        texture_register = 0
        sampler_register = 0
        buffer_register = 0
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

            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
            if lowered_block is not None:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = buffer_register
                self.glsl_buffer_block_variables.append(
                    (node, binding, lowered_block, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
                continue

            if self.is_glsl_buffer_block_variable(node, vtype):
                code += self.glsl_buffer_block_diagnostic(
                    "Metal", vtype, var_name, node
                )

            if self.is_structured_buffer_type(vtype):
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "buffer"}, ("b", "u", "t")
                )
                if binding is None:
                    binding = buffer_register
                self.structured_buffer_variables.append(
                    (node, binding, vtype, array_size)
                )
                buffer_register = max(buffer_register, binding + resource_count)
            elif vtype in [
                "sampler1D",
                "sampler1DArray",
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
                "iimage1D",
                "iimage1DArray",
                "iimage2D",
                "iimage3D",
                "iimage2DArray",
                "uimage1D",
                "uimage1DArray",
                "uimage2D",
                "uimage3D",
                "uimage2DArray",
                "image1D",
                "image1DArray",
                "image2D",
                "image3D",
                "imageCube",
                "image2DArray",
            ]:
                mapped_type = self.map_resource_type_with_format(vtype, node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture"}, ("t", "u")
                )
                if binding is None:
                    binding = texture_register
                self.texture_variables.append((node, binding, mapped_type, array_size))
                self.texture_variable_types[node.name] = mapped_type
                explicit_format = explicit_image_format(
                    node, self.attribute_value_to_string
                )
                if explicit_format:
                    self.image_variable_formats[node.name] = explicit_format
                texture_register = max(texture_register, binding + resource_count)
            elif vtype in ["sampler"]:
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "sampler"}, ("s",)
                )
                if binding is None:
                    binding = sampler_register
                self.sampler_variables.append((node, binding, array_size))
                sampler_register = max(sampler_register, binding + resource_count)
            else:
                code += f"{self.map_type(vtype)} {node.name}{array_suffix};\n"

        self.function_global_resource_dependencies = (
            self.collect_function_global_resource_dependencies(all_functions)
        )

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
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate cbuffer name(s) in Metal output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise ValueError(
                "Cbuffer name(s) conflict with existing Metal declaration(s): "
                f"{names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise ValueError(
                "Cbuffer member name(s) conflict with Metal global declaration(s): "
                f"{names}"
            )

        for node in cbuffers:
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
                            # Dynamic arrays in buffer blocks
                            code += f"    array<{self.map_type(element_type)}> {member.name};\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"struct {node.name} {{\n"
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
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"

        return code

    def generate_function(self, func, indent=0, shader_type=None):
        """Render a function or stage entry point with Metal attributes."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        reserved_parameter_names = {
            getattr(parameter, "name", None)
            for parameter in param_list
            if getattr(parameter, "name", None)
        }
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        previous_cbuffer_parameter_names = self.cbuffer_parameter_names
        previous_cbuffer_member_references = self.cbuffer_member_references
        previous_ambiguous_cbuffer_members = self.ambiguous_cbuffer_members
        self.current_function_name = getattr(func, "name", None)
        self.local_variable_types = {}
        for p in param_list:
            if hasattr(p, "param_type"):
                raw_param_type = (
                    self.type_name_string(p.param_type)
                    if getattr(p.param_type, "generic_args", None)
                    else p.param_type
                )
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
            elif self.is_texture_or_image_resource_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                explicit_format = explicit_image_format(
                    p, self.attribute_value_to_string
                )
                if explicit_format:
                    image_format_parameters[p.name] = explicit_format
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)

            param_attr = self.parameter_attribute(
                raw_param_type, semantic, shader_type, p
            )
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
                    reserved_parameter_names.add(name)

        reserved_parameter_names.update(self.global_resource_parameter_names())
        self.cbuffer_parameter_names = self.collect_cbuffer_parameter_names(
            self.cbuffer_variables, reserved_names=reserved_parameter_names
        )
        self.cbuffer_member_references = self.collect_cbuffer_member_references(
            self.cbuffer_variables
        )

        params_str = ", ".join(params)
        if shader_type is None:
            params_str = self.append_required_cbuffer_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_global_resource_parameters(
                params_str, self.current_function_name
            )

        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        self.current_function_return_type = raw_return_type

        if shader_type == "vertex":
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
            code += f"vertex {return_type} vertex_{func.name}({params_str}) {{\n"
        elif shader_type == "fragment":
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
            code += f"fragment {return_type} fragment_{func.name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name
            )
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
            semantic = self.semantic_from_node(func)
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
        self.cbuffer_parameter_names = previous_cbuffer_parameter_names
        self.cbuffer_member_references = previous_cbuffer_member_references
        self.ambiguous_cbuffer_members = previous_ambiguous_cbuffer_members

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

    def append_global_resource_parameters(self, params_str, func_name=None):
        resource_params = []
        dependencies = (
            self.function_global_resource_dependencies.get(func_name, set())
            if func_name
            else None
        )
        if self.cbuffer_variables:
            buffer_index = 0
            for cbuffer in self.cbuffer_variables:
                binding = self.explicit_resource_binding_index(
                    cbuffer, {"binding", "buffer"}, ("b",)
                )
                if binding is None:
                    binding = buffer_index
                buffer_index = max(buffer_index, binding + 1)
                parameter_name = self.cbuffer_parameter_name(cbuffer)
                resource_params.append(
                    f"constant {cbuffer.name}& {parameter_name} [[buffer({binding})]]"
                )
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
        if self.structured_buffer_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                declaration = self.format_structured_buffer_parameter(
                    buffer_type, buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.glsl_buffer_block_variables:
            for (
                buffer_variable,
                i,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                declaration = self.format_glsl_buffer_block_parameter(
                    block, buffer_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.sampler_variables:
            for sampler_variable, i, array_size in self.sampler_variables:
                sampler_name = getattr(sampler_variable, "name", None)
                if dependencies is not None and sampler_name not in dependencies:
                    continue
                declaration = self.format_resource_parameter(
                    "sampler", sampler_name, array_size
                )
                resource_params.append(f"{declaration} [[sampler({i})]]")
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def global_resource_parameter_names(self):
        names = set()
        for texture_variable, _, _, _ in self.texture_variables:
            if getattr(texture_variable, "name", None):
                names.add(texture_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for buffer_variable, _, _, _ in self.glsl_buffer_block_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for sampler_variable, _, _ in self.sampler_variables:
            if getattr(sampler_variable, "name", None):
                names.add(sampler_variable.name)
        return names

    def append_required_cbuffer_parameters(self, params_str, func_name):
        cbuffer_params = []
        for cbuffer in self.required_function_cbuffers(func_name):
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            cbuffer_params.append(f"constant {cbuffer.name}& {parameter_name}")
        if not cbuffer_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(cbuffer_params)}"
        return ", ".join(cbuffer_params)

    def append_required_global_resource_parameters(self, params_str, func_name):
        resource_params = []
        for (
            texture_variable,
            texture_type,
            array_size,
        ) in self.required_function_textures(func_name):
            texture_name = getattr(texture_variable, "name", None)
            if texture_name:
                resource_params.append(
                    self.format_resource_parameter(
                        texture_type, texture_name, array_size
                    )
                )
        for (
            buffer_variable,
            buffer_type,
            array_size,
        ) in self.required_function_structured_buffers(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_structured_buffer_parameter(
                        buffer_type, buffer_name, array_size
                    )
                )
        for (
            buffer_variable,
            block,
            array_size,
        ) in self.required_function_glsl_buffer_blocks(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_glsl_buffer_block_parameter(
                        block, buffer_name, array_size
                    )
                )
        for sampler_variable, array_size in self.required_function_samplers(func_name):
            sampler_name = getattr(sampler_variable, "name", None)
            if sampler_name:
                resource_params.append(
                    self.format_resource_parameter("sampler", sampler_name, array_size)
                )
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def cbuffer_parameter_name(self, cbuffer):
        parameter_name = self.cbuffer_parameter_names.get(id(cbuffer))
        if parameter_name:
            return parameter_name
        return self.default_cbuffer_parameter_name(cbuffer)

    def default_cbuffer_parameter_name(self, cbuffer):
        name = getattr(cbuffer, "name", "constants")
        if not name:
            return "constants"
        return name[:1].lower() + name[1:]

    def collect_cbuffer_parameter_names(self, cbuffers, reserved_names=None):
        parameter_names = {}
        used_names = set(reserved_names or [])
        for cbuffer in cbuffers:
            base_name = self.default_cbuffer_parameter_name(cbuffer)
            parameter_name = base_name
            suffix = 1
            while parameter_name in used_names:
                parameter_name = f"{base_name}{suffix}"
                suffix += 1
            used_names.add(parameter_name)
            parameter_names[id(cbuffer)] = parameter_name
        return parameter_names

    def collect_cbuffer_member_references(self, cbuffers):
        references = {}
        ambiguous_members = collect_duplicate_cbuffer_member_names(cbuffers)
        for cbuffer in cbuffers:
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name or member_name in ambiguous_members:
                    continue
                references[member_name] = f"{parameter_name}.{member_name}"
        self.ambiguous_cbuffer_members = ambiguous_members
        return references

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as Metal source."""
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
            return self.generate_statement_code(self.generate_assignment(stmt), indent)
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
            return self.generate_statement_code(expr_code, indent)
        else:
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        if not lines:
            return ""

        result = ""
        for line in lines:
            terminator = "" if line.endswith((";", "}")) else ";"
            result += f"{indent_str}{line}{terminator}\n"
        return result

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
            member_types = {
                self.type_name_string(members[member])
                for members in self.struct_member_types.values()
                if member in members
            }
            if len(member_types) == 1:
                return next(iter(member_types))
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
            if isinstance(stmt.expression, AssignmentNode):
                return self.generate_assignment(stmt.expression)
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            # Fallback for direct expression
            return self.generate_expression(stmt)

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            target = node.target
            rhs = self.generate_expression_with_expected(
                node.value, self.expression_result_type(node.target)
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            target = node.left
            rhs = self.generate_expression_with_expected(
                node.right, self.expression_result_type(node.left)
            )
            op = getattr(node, "operator", "=")

        block_store = self.generate_glsl_buffer_block_store(target, rhs, op)
        if block_store is not None:
            return block_store

        lhs = self.generate_expression(target)
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
        """Render a CrossGL AST expression into Metal expression syntax."""
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
            block_load = self.generate_glsl_buffer_block_array_load(expr)
            if block_load is not None:
                return block_load
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

            buffer_call = self.generate_buffer_call(func_name, expr.args)
            if buffer_call is not None:
                return buffer_call
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
                self.validate_function_image_access_arguments(func_name, expr.args)
                args = [self.generate_expression(arg) for arg in expr.args]
                if func_name in self.user_function_names:
                    args.extend(
                        self.cbuffer_parameter_name(cbuffer)
                        for cbuffer in self.required_function_cbuffers(func_name)
                    )
                    args.extend(
                        self.required_function_resource_argument_names(func_name)
                    )
                args = ", ".join(args)
                return f"{callee}({args})"
        elif isinstance(expr, MemberAccessNode):
            block_load = self.generate_glsl_buffer_block_member_load(expr)
            if block_load is not None:
                return block_load
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
            name = getattr(expr, "name", str(expr))
            if (
                name not in self.local_variable_types
                and name in self.ambiguous_cbuffer_members
            ):
                raise ValueError(
                    f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
                )
            if (
                name not in self.local_variable_types
                and name in self.cbuffer_member_references
            ):
                return self.cbuffer_member_references[name]
            return name
        else:
            return str(expr)

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}[{index}]"
        if func_name == "buffer_store" and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}[{index}] = {value}"
        if func_name == "buffer_append" and len(args) >= 2:
            return (
                "/* unsupported Metal buffer append: requires explicit "
                "counter buffer */"
            )
        if func_name == "buffer_consume" and args:
            return (
                "0 /* unsupported Metal buffer consume: requires explicit "
                "counter buffer */"
            )
        if func_name == "buffer_dimensions" and args:
            diagnostic = (
                "0 /* unsupported Metal buffer dimensions: device buffers do not "
                "carry length */"
            )
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {diagnostic}"
            return diagnostic
        return None

    def default_sampler_expression(self):
        return "sampler(mag_filter::linear, min_filter::linear)"

    def sampler_variable_names(self):
        return {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters

    def structured_buffer_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_structured_buffer_type(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "uint"
        element_type = type_name.split("<", 1)[1][:-1].strip()
        return self.map_type(element_type)

    def structured_buffer_address_space(self, vtype):
        if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
            return "const device"
        return "device"

    def format_structured_buffer_parameter(self, vtype, name, array_size=None):
        element_type = self.structured_buffer_element_type(vtype)
        address_space = self.structured_buffer_address_space(vtype)
        pointer_type = f"{address_space} {element_type}*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def format_glsl_buffer_block_parameter(self, block, name, array_size=None):
        address_space = "const device" if block.get("readonly") else "device"
        pointer_type = f"{address_space} uchar*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) == "sampler"

    def is_resource_parameter_type(self, vtype):
        return self.is_structured_buffer_type(vtype) or self.resource_base_type(vtype) in {
            "sampler",
            "sampler1D",
            "sampler1DArray",
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
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
        }

    def is_texture_or_image_resource_type(self, vtype):
        return self.is_resource_parameter_type(vtype) and not self.is_sampler_type(
            vtype
        ) and not self.is_structured_buffer_type(vtype)

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return base_type in {
            "int",
            "uint",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
        } or mapped_type in {
            "int",
            "uint",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
        }

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = "_ms" in texture_type
        is_storage_image = "access::" in texture_type

        coordinate_dimension = None
        if texture_type and "cube" not in texture_type:
            if texture_type.startswith("texture1d_array<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture1d<"):
                coordinate_dimension = 1
            elif texture_type.startswith("texture2d_ms_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d_ms<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("depth2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("depth2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                coordinate_dimension = 3

        texel_fetch_offset_dimension = None
        if texture_type and "cube" not in texture_type and not is_multisample:
            if texture_type.startswith("texture1d_array<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture1d<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture2d_array<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture2d<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture3d<"):
                texel_fetch_offset_dimension = 3

        sample_offset_dimension = None
        if texture_type.startswith("texture2d_array<"):
            sample_offset_dimension = 2
        elif texture_type.startswith("texture2d<"):
            sample_offset_dimension = 2
        if not sampling["sample_offset"]:
            sample_offset_dimension = None

        gradient_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                gradient_dimension = 2
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                gradient_dimension = 2
            elif texture_type.startswith("texture3d<"):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                gradient_dimension = 3

        query_lod_coordinate_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith("texture1d_array<"):
                query_lod_coordinate_dimension = 2
            elif texture_type.startswith("texture1d<"):
                query_lod_coordinate_dimension = 1
            elif texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                query_lod_coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                query_lod_coordinate_dimension = 4
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                query_lod_coordinate_dimension = 3

        compare_offset_dimension = 2 if sampling["compare_offset"] else None
        return {
            "texture_type": texture_type,
            "coordinate_dimension": coordinate_dimension,
            "offset_dimension": sample_offset_dimension,
            "sample_offset_dimension": sample_offset_dimension,
            "texel_fetch_offset_dimension": texel_fetch_offset_dimension,
            "gather_offset_dimension": 2 if sampling["gather_offset"] else None,
            "compare_offset_dimension": compare_offset_dimension,
            "compare_lod_offset_dimension": compare_offset_dimension,
            "compare_grad_offset_dimension": compare_offset_dimension,
            "gather_compare_offset_dimension": (
                2 if sampling["gather_compare_offset"] else None
            ),
            "gradient_dimension": gradient_dimension,
            "query_lod_coordinate_dimension": query_lod_coordinate_dimension,
        }

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        if func_name == "texelFetchOffset":
            return descriptor["texel_fetch_offset_dimension"]
        if func_name in {"textureGatherOffset", "textureGatherOffsets"}:
            return descriptor["gather_offset_dimension"]
        if func_name == "textureGatherCompareOffset":
            return descriptor["gather_compare_offset_dimension"]
        if func_name in {
            "textureCompareOffset",
            "textureCompareLodOffset",
            "textureCompareGradOffset",
            "textureCompareProjOffset",
            "textureCompareProjLodOffset",
            "textureCompareProjGradOffset",
        }:
            return descriptor["compare_offset_dimension"]
        if func_name in OFFSET_DIMENSION_INTRINSIC_NAMES:
            return descriptor["sample_offset_dimension"]
        return descriptor["offset_dimension"]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def parameter_attribute(self, raw_param_type, semantic, shader_type, node=None):
        if semantic:
            return self.map_semantic(semantic)
        if shader_type in {"vertex", "fragment", "compute"}:
            resource_attr = self.resource_parameter_attribute(raw_param_type, node)
            if resource_attr:
                return resource_attr
        if self.is_resource_parameter_type(raw_param_type):
            return ""
        if shader_type in {"vertex", "fragment"}:
            return " [[stage_in]]"
        return ""

    def resource_parameter_attribute(self, raw_param_type, node=None):
        if node is None:
            return ""
        if self.is_sampler_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "sampler"}, ("s",)
            )
            return f" [[sampler({binding})]]" if binding is not None else ""
        if self.is_structured_buffer_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            return f" [[buffer({binding})]]" if binding is not None else ""
        if self.is_resource_parameter_type(raw_param_type):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "texture"}, ("t", "u")
            )
            return f" [[texture({binding})]]" if binding is not None else ""
        return ""

    def format_parameter_declaration(
        self, raw_param_type, mapped_type, name, node=None
    ):
        if self.is_structured_buffer_type(raw_param_type):
            return self.format_structured_buffer_parameter(raw_param_type, name)
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

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and (
                self.is_resource_parameter_type(var_type)
                or self.glsl_buffer_block_attribute(node) is not None
            ):
                resource_names.add(var_name)
        return resource_names

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_resource_parameter_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Non-resource local declaration(s) shadow Metal global resource(s): "
                f"{names}"
            )

    def collect_function_cbuffer_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_cbuffer_dependencies(func)
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_cbuffer_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        member_to_cbuffer = {}
        for cbuffer in self.cbuffer_variables:
            cbuffer_name = getattr(cbuffer, "name", None)
            if not cbuffer_name:
                continue
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if member_name:
                    member_to_cbuffer[member_name] = cbuffer_name

        dependencies = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", None)
            if not name or name in local_names:
                continue
            cbuffer_name = member_to_cbuffer.get(name)
            if cbuffer_name:
                dependencies.add(cbuffer_name)
        return dependencies

    def called_user_function_names(self, func):
        called_names = set()
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if func_name in self.user_function_names and func_name != getattr(
                func, "name", None
            ):
                called_names.add(func_name)
        return called_names

    def required_function_cbuffers(self, func_name):
        dependencies = self.function_cbuffer_dependencies.get(func_name, set())
        return [
            cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None) in dependencies
        ]

    def collect_function_global_resource_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_global_resource_dependencies(
                func
            )
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_global_resource_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        texture_names = self.global_texture_names()
        buffer_names = (
            self.global_structured_buffer_names()
            | self.global_glsl_buffer_block_names()
        )
        sampler_names = self.global_sampler_names()
        dependencies = set()

        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", None)
                if (
                    name
                    and name not in local_names
                    and (
                        name in texture_names
                        or name in buffer_names
                        or name in sampler_names
                    )
                ):
                    dependencies.add(name)

            if isinstance(node, FunctionCallNode):
                self.add_texture_call_resource_dependencies(
                    node, local_names, texture_names, sampler_names, dependencies
                )
                self.add_buffer_call_resource_dependencies(
                    node, local_names, buffer_names, dependencies
                )

        return dependencies

    def add_texture_call_resource_dependencies(
        self, call, local_names, texture_names, sampler_names, dependencies
    ):
        func_name = self.function_call_name(call)
        if not func_name or not str(func_name).startswith(("texture", "image")):
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return

        texture_name = self.expression_name(args[0])
        if texture_name in texture_names and texture_name not in local_names:
            dependencies.add(texture_name)

        if len(args) >= 3:
            sampler_name = self.expression_name(args[1])
            if sampler_name in sampler_names and sampler_name not in local_names:
                dependencies.add(sampler_name)
                return

        if not self.texture_call_needs_implicit_sampler_dependency(func_name, args):
            return
        implicit_sampler_name = f"{texture_name}Sampler" if texture_name else None
        if (
            implicit_sampler_name in sampler_names
            and implicit_sampler_name not in local_names
        ):
            dependencies.add(implicit_sampler_name)

    def add_buffer_call_resource_dependencies(
        self, call, local_names, buffer_names, dependencies
    ):
        func_name = self.function_call_name(call)
        if func_name not in {
            "buffer_load",
            "buffer_store",
            "buffer_append",
            "buffer_consume",
            "buffer_dimensions",
        }:
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return
        buffer_name = self.expression_name(args[0])
        if buffer_name in buffer_names and buffer_name not in local_names:
            dependencies.add(buffer_name)

    def texture_sampling_uses_implicit_sampler(self, func_name):
        return func_name in {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjLod",
            "textureProjGrad",
            "textureProjOffset",
            "textureProjLodOffset",
            "textureProjGradOffset",
            "textureQueryLod",
        }

    def texture_call_needs_implicit_sampler_dependency(self, func_name, args):
        if not self.texture_sampling_uses_implicit_sampler(func_name):
            return False

        texture_type = self.texture_resource_type(args[0])
        if self.storage_image_texture_operation_expression(func_name, texture_type):
            return False

        if func_name in {"texture", "textureLod", "textureGrad"}:
            return not self.is_multisample_texture_resource(texture_type)

        if func_name == "textureQueryLod":
            return not (
                self.is_multisample_texture_resource(texture_type)
                or self.is_storage_image_resource(texture_type)
            )

        if func_name in {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
        }:
            return self.texture_sample_supports_offset(texture_type)

        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            texture_type = self.resource_base_type(texture_type)
            return (
                texture_type.startswith("texture1d<")
                or texture_type.startswith("texture2d<")
                or texture_type.startswith("texture2d_array<")
                or texture_type.startswith("texture3d<")
            )

        return True

    def global_texture_names(self):
        return {
            texture_variable.name
            for texture_variable, _, _, _ in self.texture_variables
            if getattr(texture_variable, "name", None)
        }

    def global_structured_buffer_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_glsl_buffer_block_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_sampler_names(self):
        return {
            sampler_variable.name
            for sampler_variable, _, _ in self.sampler_variables
            if getattr(sampler_variable, "name", None)
        }

    def required_function_textures(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (texture_variable, texture_type, array_size)
            for texture_variable, _, texture_type, array_size in self.texture_variables
            if getattr(texture_variable, "name", None) in dependencies
        ]

    def required_function_samplers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (sampler_variable, array_size)
            for sampler_variable, _, array_size in self.sampler_variables
            if getattr(sampler_variable, "name", None) in dependencies
        ]

    def required_function_structured_buffers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, buffer_type, array_size)
            for (
                buffer_variable,
                _,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_glsl_buffer_blocks(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, block, array_size)
            for (
                buffer_variable,
                _,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_resource_argument_names(self, func_name):
        return [
            texture_variable.name
            for texture_variable, _, _ in self.required_function_textures(func_name)
        ] + [
            buffer_variable.name
            for buffer_variable, _, _ in self.required_function_structured_buffers(
                func_name
            )
        ] + [
            buffer_variable.name
            for buffer_variable, _, _ in self.required_function_glsl_buffer_blocks(
                func_name
            )
        ] + [
            sampler_variable.name
            for sampler_variable, _ in self.required_function_samplers(func_name)
        ]

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
        return supported_image_formats()

    def scalar_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) == 1
        }

    def vector_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) in {2, 4}
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

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def explicit_resource_binding_index(
        self, node, attribute_names=(), register_prefixes=()
    ):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in attribute_names:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], register_prefixes)
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def semantic_from_node(self, node):
        if hasattr(node, "semantic"):
            return node.semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
            ):
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
        explicit_format = (
            explicit_image_format(node, self.attribute_value_to_string)
            if node is not None
            else None
        )
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "texture1d",
            "iimage1D": "texture1d",
            "uimage1D": "texture1d",
            "image2D": "texture2d",
            "iimage2D": "texture2d",
            "uimage2D": "texture2d",
            "image3D": "texture3d",
            "iimage3D": "texture3d",
            "uimage3D": "texture3d",
            "image1DArray": "texture1d_array",
            "iimage1DArray": "texture1d_array",
            "uimage1DArray": "texture1d_array",
            "image2DArray": "texture2d_array",
            "iimage2DArray": "texture2d_array",
            "uimage2DArray": "texture2d_array",
            "imageCube": "texture2d_array",
        }
        texture_type = texture_types.get(base_type)
        if texture_type:
            if component_type is None:
                component_type = self.default_image_component_type(base_type)
            access = (
                explicit_image_access(node, self.attribute_value_to_string)
                or "read_write"
            )
            return f"{texture_type}<{component_type}, access::{access}>"
        return self.map_type(vtype)

    def default_image_component_type(self, vtype):
        base_type = self.resource_base_type(vtype)
        if base_type.startswith("iimage"):
            return "int"
        if base_type.startswith("uimage"):
            return "uint"
        return "float"

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "restrict",
            "volatile",
        }

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

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        arguments = getattr(attr, "arguments", []) if attr is not None else []
        if arguments:
            layout = self.attribute_value_to_string(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "glsl_buffer_block")

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        if self.glsl_buffer_block_attribute(node) is None:
            return False
        type_name = self.resource_base_type(vtype or getattr(node, "var_type", None))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, global_vars):
        names = set()
        for node in global_vars:
            if not self.is_glsl_buffer_block_variable(node):
                continue
            type_name = self.resource_base_type(getattr(node, "var_type", None))
            names.add(str(type_name))
        return names

    def glsl_buffer_block_lowering_failure_detail(self, type_name, var_name=None):
        if var_name:
            reason = self.glsl_buffer_block_lowering_failures.get(var_name)
            if reason:
                return reason
        type_name = str(self.resource_base_type(type_name))
        return self.glsl_buffer_block_struct_lowering_failures.get(type_name)

    def glsl_buffer_block_member_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        var_name = self.expression_name(object_expr)
        if not var_name:
            return None
        block = self.lowered_glsl_buffer_blocks.get(var_name)
        if block is None:
            return None
        member_name = getattr(expr, "member", None)
        member = block["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": var_name,
            "member": member_name,
            "readonly": block["readonly"],
            **member,
        }

    def glsl_buffer_block_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        member = self.glsl_buffer_block_member_access(array_expr)
        if member is None or not member.get("is_array"):
            return None
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
        index = self.generate_expression(index_expr)
        offset = byte_offset_expression(member["offset"], index, member["stride"])
        return {**member, "offset_expr": offset}

    def metal_scalar_load(self, component_type, buffer_name, offset):
        return (
            f"(*reinterpret_cast<const device {component_type}*>"
            f"({buffer_name} + {offset}))"
        )

    def metal_scalar_store(self, component_type, buffer_name, offset, value):
        return (
            f"(*reinterpret_cast<device {component_type}*>"
            f"({buffer_name} + {offset})) = {value}"
        )

    def metal_buffer_load(self, buffer_name, offset, access):
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                column_access = {
                    "components": access["matrix_rows"],
                    "component_type": "float",
                    "metal_type": f"float{access['matrix_rows']}",
                }
                columns.append(
                    self.metal_buffer_load(buffer_name, column_offset, column_access)
                )
            return f"{access['metal_type']}({', '.join(columns)})"

        if access["components"] == 1:
            return self.metal_scalar_load(access["component_type"], buffer_name, offset)

        values = []
        for _, component_offset in vector_component_offsets(
            offset, access["components"]
        ):
            values.append(
                self.metal_scalar_load(
                    access["component_type"], buffer_name, component_offset
                )
            )
        return f"{access['metal_type']}({', '.join(values)})"

    def next_metal_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.metal_temp_variable_index}"
        self.metal_temp_variable_index += 1
        return name

    def metal_buffer_store(self, buffer_name, offset, value, access):
        if access.get("matrix_columns"):
            return self.metal_matrix_store(buffer_name, offset, value, access)

        if access["components"] == 1:
            return self.metal_scalar_store(
                access["component_type"], buffer_name, offset, value
            )

        temp_name = self.next_metal_temp_variable("buffer_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for component, component_offset in vector_component_offsets(
            offset, access["components"]
        ):
            field = "xyzw"[component]
            lines.append(
                self.metal_scalar_store(
                    access["component_type"],
                    buffer_name,
                    component_offset,
                    f"{temp_name}.{field}",
                )
            )
        return "\n".join(lines)

    def metal_matrix_store(self, buffer_name, offset, value, access):
        temp_name = self.next_metal_temp_variable("matrix_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            for row, element_offset in vector_component_offsets(
                column_offset, access["matrix_rows"]
            ):
                field = "xyzw"[row]
                lines.append(
                    self.metal_scalar_store(
                        "float",
                        buffer_name,
                        element_offset,
                        f"{temp_name}[{column}].{field}",
                    )
                )
        return "\n".join(lines)

    def metal_matrix_compound_store(self, buffer_name, offset, value, op, access):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            return (
                "/* unsupported Metal GLSL buffer block matrix compound store: "
                "requires explicit matrix operation lowering */"
            )

        current = self.metal_buffer_load(buffer_name, offset, access)
        rhs = f"({current} {binary_op} {value})"
        return self.metal_matrix_store(buffer_name, offset, rhs, access)

    def metal_buffer_compound_store_diagnostic(self, op, access):
        return (
            "/* unsupported Metal GLSL buffer block compound store: "
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members */"
        )

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("is_array"):
            return None
        return self.metal_buffer_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        return self.metal_buffer_load(access["buffer"], access["offset_expr"], access)

    def generate_glsl_buffer_block_store(self, target, rhs, op):
        access = self.glsl_buffer_block_array_access(target)
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("is_array"):
                return None
            offset = access["offset"]
        else:
            offset = access["offset_expr"]

        if access.get("readonly"):
            return (
                "/* unsupported Metal GLSL buffer block store: "
                "readonly device buffer cannot be written */"
            )

        if access.get("matrix_columns"):
            if op != "=":
                return self.metal_matrix_compound_store(
                    access["buffer"], offset, rhs, op, access
                )
            return self.metal_matrix_store(access["buffer"], offset, rhs, access)

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return self.metal_buffer_compound_store_diagnostic(op, access)
            current = self.metal_buffer_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        return self.metal_buffer_store(access["buffer"], offset, rhs, access)

    def glsl_buffer_block_diagnostic(self, target, type_name, var_name=None, node=None):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        details = ""
        if node is not None:
            layout = self.glsl_buffer_block_layout(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            details = f" ({layout}"
            if binding is not None:
                details += f", binding = {binding}"
            details += ")"
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(
            type_name, var_name
        )
        if failure_detail:
            details += f"; {failure_detail}"
        return (
            f"// unsupported {target} GLSL buffer block {declaration}{details}: "
            "mixed metadata/runtime-array layout requires explicit pointer/offset "
            "lowering\n"
        )

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
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.expression_name(array_expr)
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
        return self.texture_call_uses_explicit_sampler(args)

    def texture_call_uses_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        if sampler_name in self.sampler_variable_names():
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

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

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_resource_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_texture_or_image_resource_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"Metal texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or func_name not in IMAGE_RESOURCE_INTRINSIC_NAMES:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_resource(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"Metal image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or func_name not in IMAGE_RESOURCE_INTRINSIC_NAMES:
            return
        access = self.storage_image_access_mode(
            self.texture_argument_resource_type(args[0])
        )
        if access is None:
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise ValueError(
                f"Metal image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got access::write"
            )
        if func_name == "imageStore" and access == "read":
            raise ValueError(
                f"Metal image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got access::read"
            )
        if (
            func_name
            in {
                "imageAtomicAdd",
                "imageAtomicMin",
                "imageAtomicMax",
                "imageAtomicAnd",
                "imageAtomicOr",
                "imageAtomicXor",
                "imageAtomicExchange",
                "imageAtomicCompSwap",
            }
            and access != "read_write"
        ):
            raise ValueError(
                f"Metal image operation '{func_name}' requires read_write "
                f"storage image access for {texture_name}: got access::{access}"
            )

    def image_access_requirement_label(self, required_access):
        return {
            "read": "read-capable",
            "write": "write-capable",
            "read_write": "read_write",
        }.get(required_access, str(required_access))

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.storage_image_access_mode(
                self.texture_argument_resource_type(args[index])
            )
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = self.image_access_requirement_label(required_access)
            raise ValueError(
                f"Metal function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got access::{actual_access}"
            )

    def validate_integer_coordinate_argument(self, func_name, args):
        if func_name not in INTEGER_COORDINATE_INTRINSIC_NAMES or len(args) < 2:
            return
        coord_type = self.expression_result_type(args[1])
        if coord_type is None or self.is_integer_coordinate_type(coord_type):
            return
        raise ValueError(
            f"Metal resource operation '{func_name}' requires an integer "
            f"coordinate argument: {expression_debug_name(args[1])} has type "
            f"{self.type_name_string(coord_type)}"
        )

    def validate_coordinate_dimension_argument(self, func_name, args):
        if func_name not in INTEGER_COORDINATE_INTRINSIC_NAMES or len(args) < 2:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[1])
        coord_dimension = integer_coordinate_dimension(
            self.type_name_string(coord_type)
        )
        if coord_dimension is None or coord_dimension == expected_dimension:
            return
        raise ValueError(
            f"Metal resource operation '{func_name}' requires a "
            f"{expected_dimension}D integer coordinate for "
            f"{self.resource_base_type(texture_type)}: "
            f"{expression_debug_name(args[1])} has type "
            f"{self.type_name_string(coord_type)}"
        )

    def validate_offset_dimension_argument(self, func_name, args):
        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not offset_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for offset_index in offset_indices:
            offset_type = self.expression_result_type(args[offset_index])
            if offset_type is None:
                continue
            if not self.is_integer_coordinate_type(offset_type):
                raise ValueError(
                    f"Metal resource operation '{func_name}' requires an integer "
                    f"offset argument: {expression_debug_name(args[offset_index])} "
                    f"has type {self.type_name_string(offset_type)}"
                )
            offset_dimension = integer_coordinate_dimension(
                self.type_name_string(offset_type)
            )
            if offset_dimension is None or offset_dimension == expected_dimension:
                continue
            raise ValueError(
                f"Metal resource operation '{func_name}' requires a "
                f"{expected_dimension}D integer offset for "
                f"{self.resource_base_type(texture_type)}: "
                f"{expression_debug_name(args[offset_index])} has type "
                f"{self.type_name_string(offset_type)}"
            )

    def gradient_argument_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        mapped_type = self.map_type(type_name)
        return floating_coordinate_dimension(
            mapped_type
        ) or floating_coordinate_dimension(type_name)

    def query_lod_coordinate_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        mapped_type = self.map_type(type_name)
        return floating_coordinate_dimension(
            mapped_type
        ) or floating_coordinate_dimension(type_name)

    def validate_query_lod_coordinate_argument(self, func_name, args):
        coord_index = texture_query_lod_coordinate_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if coord_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_query_lod_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[coord_index])
        if coord_type is None:
            return
        coord_dimension = self.query_lod_coordinate_dimension(coord_type)
        if coord_dimension is None:
            raise ValueError(
                f"Metal texture query operation '{func_name}' requires a floating "
                f"coordinate argument: {expression_debug_name(args[coord_index])} "
                f"has type {self.type_name_string(coord_type)}"
            )
        if coord_dimension == expected_dimension:
            return
        raise ValueError(
            f"Metal texture query operation '{func_name}' requires a "
            f"{expected_dimension}D floating coordinate for "
            f"{self.resource_base_type(texture_type)}: "
            f"{expression_debug_name(args[coord_index])} has type "
            f"{self.type_name_string(coord_type)}"
        )

    def validate_gradient_dimension_arguments(self, func_name, args):
        gradient_indices = texture_gradient_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not gradient_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_gradient_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for gradient_index in gradient_indices:
            gradient_type = self.expression_result_type(args[gradient_index])
            if gradient_type is None:
                continue
            gradient_dimension = self.gradient_argument_dimension(gradient_type)
            if gradient_dimension is None:
                raise ValueError(
                    f"Metal resource operation '{func_name}' requires a floating "
                    f"gradient argument: {expression_debug_name(args[gradient_index])} "
                    f"has type {self.type_name_string(gradient_type)}"
                )
            if gradient_dimension == expected_dimension:
                continue
            raise ValueError(
                f"Metal resource operation '{func_name}' requires a "
                f"{expected_dimension}D floating gradient for "
                f"{self.resource_base_type(texture_type)}: "
                f"{expression_debug_name(args[gradient_index])} has type "
                f"{self.type_name_string(gradient_type)}"
            )

    def is_scalar_floating_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name or "[" in str(type_name):
            return False
        mapped_type = self.map_type(type_name)
        return is_floating_scalar_type(mapped_type) or is_floating_scalar_type(
            type_name
        )

    def is_scalar_numeric_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name or "[" in str(type_name):
            return False
        mapped_type = self.map_type(type_name)
        return is_numeric_scalar_type(mapped_type) or is_numeric_scalar_type(type_name)

    def is_scalar_integer_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name or "[" in str(type_name):
            return False
        mapped_type = self.map_type(type_name)
        return is_integer_scalar_type(mapped_type) or is_integer_scalar_type(type_name)

    def texture_argument_diagnostic_type(self, arg):
        texture_type = self.texture_resource_type(arg)
        if texture_type is not None:
            return texture_type
        arg_name = self.expression_name(arg)
        sampler_names = {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        }
        if arg_name in sampler_names or arg_name in self.current_sampler_parameters:
            return "sampler"
        return self.expression_result_type(arg)

    def validate_compare_argument(self, func_name, args):
        compare_index = texture_compare_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if compare_index is None:
            return
        compare_type = self.expression_result_type(args[compare_index])
        if compare_type is None or self.is_scalar_floating_type(compare_type):
            return
        raise ValueError(
            f"Metal texture compare operation '{func_name}' requires a scalar "
            f"floating compare argument: {expression_debug_name(args[compare_index])} "
            f"has type {self.type_name_string(compare_type)}"
        )

    def validate_lod_argument(self, func_name, args):
        lod_index = texture_lod_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if lod_index is None:
            return
        lod_type = self.texture_argument_diagnostic_type(args[lod_index])
        if lod_type is None or self.is_scalar_numeric_type(lod_type):
            return
        raise ValueError(
            f"Metal texture LOD operation '{func_name}' requires a scalar "
            f"numeric lod argument: {expression_debug_name(args[lod_index])} "
            f"has type {self.type_name_string(lod_type)}"
        )

    def validate_bias_argument(self, func_name, args):
        bias_index = texture_bias_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if bias_index is None:
            return
        bias_type = self.texture_argument_diagnostic_type(args[bias_index])
        if bias_type is None or self.is_scalar_numeric_type(bias_type):
            return
        raise ValueError(
            f"Metal texture bias operation '{func_name}' requires a scalar "
            f"numeric bias argument: {expression_debug_name(args[bias_index])} "
            f"has type {self.type_name_string(bias_type)}"
        )

    def validate_mip_level_argument(self, func_name, args):
        level_index = texture_mip_level_argument_index(func_name, len(args))
        if level_index is None:
            return
        level_type = self.texture_argument_diagnostic_type(args[level_index])
        if level_type is None or self.is_scalar_integer_type(level_type):
            return
        raise ValueError(
            f"Metal resource operation '{func_name}' requires a scalar integer "
            f"mip/sample level argument: {expression_debug_name(args[level_index])} "
            f"has type {self.type_name_string(level_type)}"
        )

    def validate_sample_index_argument(self, func_name, args):
        sample_index = texture_sample_index_argument_index(func_name, len(args))
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise ValueError(
            f"Metal multisample texel fetch operation '{func_name}' requires a "
            f"scalar integer sample index argument: "
            f"{expression_debug_name(args[sample_index])} has type "
            f"{self.type_name_string(sample_type)}"
        )

    def validate_gather_component_argument(self, func_name, args):
        component_index = texture_gather_component_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if component_index is None:
            return
        component_type = self.texture_argument_diagnostic_type(args[component_index])
        if component_type is None or self.is_scalar_integer_type(component_type):
            return
        raise ValueError(
            f"Metal texture gather operation '{func_name}' requires a scalar "
            f"integer component argument: "
            f"{expression_debug_name(args[component_index])} has type "
            f"{self.type_name_string(component_type)}"
        )

    def validate_texture_call_arity(self, func_name, args):
        if func_name not in self.texture_resource_operation_names():
            return
        has_explicit_sampler = self.texture_call_uses_explicit_sampler(args)
        min_count = texture_intrinsic_min_argument_count(
            func_name,
            has_explicit_sampler,
        )
        if min_count is not None and len(args) < min_count:
            raise ValueError(
                f"Metal texture operation '{func_name}' requires at least "
                f"{min_count} argument(s), got {len(args)}"
            )
        allowed_counts = texture_intrinsic_allowed_argument_counts(
            func_name,
            has_explicit_sampler,
        )
        if allowed_counts is not None and len(args) not in allowed_counts:
            counts = ", ".join(str(count) for count in allowed_counts)
            raise ValueError(
                f"Metal texture operation '{func_name}' accepts "
                f"{counts} argument(s), got {len(args)}"
            )
        max_count = texture_intrinsic_max_argument_count(
            func_name,
            has_explicit_sampler,
        )
        if max_count is None or len(args) <= max_count:
            return
        raise ValueError(
            f"Metal texture operation '{func_name}' accepts at most "
            f"{max_count} argument(s), got {len(args)}"
        )

    def texture_resource_operation_names(self):
        return {
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
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureQueryLod",
            "textureQueryLevels",
            "textureSize",
            "textureSamples",
            "texelFetch",
            "texelFetchOffset",
            "imageLoad",
            "imageStore",
            "imageSize",
            "imageSamples",
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def image_resource_format(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_image_format_parameters.get(
            texture_name, self.image_variable_formats.get(texture_name)
        )

    def is_array_texture_resource(self, texture_type):
        return texture_type in {
            "texture1d_array<float>",
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
        return is_metal_storage_image_resource(texture_type)

    def storage_image_access_mode(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if "access::read_write>" in texture_type:
            return "read_write"
        if "access::read>" in texture_type:
            return "read"
        if "access::write>" in texture_type:
            return "write"
        return None

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
        if texture_type == "texture1d_array<float>":
            coord_x = self.vector_component(coord, "x")
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
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

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        color_offset_types = {"texture2d<float>", "texture2d_array<float>"}
        depth_offset_types = {"depth2d<float>", "depth2d_array<float>"}
        return {
            "texture_type": texture_type,
            "gather": (
                texture_type
                in {
                    "texture2d<float>",
                    "texture2d_array<float>",
                    "texturecube<float>",
                    "texturecube_array<float>",
                }
            ),
            "gather_offset": texture_type in color_offset_types,
            "sample_offset": texture_type in color_offset_types,
            "projected_offset": texture_type in color_offset_types,
            "compare_offset": texture_type in depth_offset_types,
            "gather_compare_offset": texture_type in depth_offset_types,
        }

    def texture_gather_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_sample_supports_offset(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

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
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_sample_offset_call(
                    func_name, "requires offset and optional bias arguments"
                )
            offset = self.generate_expression(extra_args[0])
            args = [sampler_arg] + list(coord_args)
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                args.append(f"bias({bias})")
            args.append(offset)
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
        return self.texture_sampling_capabilities(texture_type)["projected_offset"]

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

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        return (
            f"/* unsupported Metal multisample texture comparison: "
            f"{func_name} on {texture_type} */ 0.0"
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        return (
            f"/* unsupported Metal multisample texture gather comparison: "
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
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

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

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
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
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

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

        if self.is_multisample_texture_resource(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
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

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_resource(texture_type),
            "multisample": self.is_multisample_texture_resource(texture_type),
            "size_descriptor": self.texture_query_size_descriptor(texture_type),
        }

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        query_descriptor = self.texture_query_resource_descriptor(texture_arg)
        size_descriptor = query_descriptor["size_descriptor"]
        if size_descriptor is None:
            return None

        lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
        lod_arg_string = f"uint({lod})"
        return self.texture_query_size_descriptor_expression(
            texture_name, size_descriptor, lod_arg_string
        )

    def texture_query_size_descriptor_expression(
        self, texture_name, descriptor, lod_arg_string
    ):
        dimension_expressions = [
            self.texture_query_size_dimension_expression(
                texture_name, method, use_lod, lod_arg_string
            )
            for method, use_lod in descriptor["dimensions"]
        ]
        if descriptor["return_type"] == "int":
            return f"int({dimension_expressions[0]})"
        return f"{descriptor['return_type']}({', '.join(dimension_expressions)})"

    def texture_query_size_dimension_expression(
        self, texture_name, method, use_lod, lod_arg_string
    ):
        args = lod_arg_string if use_lod else ""
        return f"{texture_name}.{method}({args})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_storage_image_resource(texture_type):
            return self.storage_image_size_descriptor(texture_type)
        return self.sampled_texture_size_descriptor(texture_type)

    def storage_image_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("texture1d_array<"):
            return {
                "return_type": "int2",
                "dimensions": (("get_width", False), ("get_array_size", False)),
            }
        if texture_type.startswith("texture1d<"):
            return {
                "return_type": "int",
                "dimensions": (("get_width", False),),
            }
        if texture_type.startswith("texture2d_array<"):
            return {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            }
        if texture_type.startswith("texture3d<"):
            return {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_depth", False),
                ),
            }
        return {
            "return_type": "int2",
            "dimensions": (("get_width", False), ("get_height", False)),
        }

    def sampled_texture_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        descriptors = {
            "texture1d<float>": {
                "return_type": "int",
                "dimensions": (("get_width", True),),
            },
            "texture1d_array<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", True), ("get_array_size", False)),
            },
            "texture2d<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", True), ("get_height", True)),
            },
            "depth2d<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", True), ("get_height", True)),
            },
            "texturecube<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", True), ("get_height", True)),
            },
            "depthcube<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", True), ("get_height", True)),
            },
            "texture2d_array<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            },
            "depth2d_array<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            },
            "texturecube_array<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            },
            "depthcube_array<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_array_size", False),
                ),
            },
            "texture3d<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", True),
                    ("get_height", True),
                    ("get_depth", True),
                ),
            },
            "texture2d_ms<float>": {
                "return_type": "int2",
                "dimensions": (("get_width", False), ("get_height", False)),
            },
            "texture2d_ms_array<float>": {
                "return_type": "int3",
                "dimensions": (
                    ("get_width", False),
                    ("get_height", False),
                    ("get_array_size", False),
                ),
            },
        }
        return descriptors.get(texture_type)

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return "1"
        return f"int({texture_name}.get_num_mip_levels())"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if not descriptor["multisample"]:
            return "/* unsupported Metal texture samples query: requires multisample texture */ 0"
        return f"int({texture_name}.get_num_samples())"

    def image_coordinate_expression(self, image_type, coord):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d_array<float, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture1d_array<uint, access::read_write>",
        }:
            coord_x = self.unsigned_coordinate_expression(
                self.vector_component(coord, "x"), 1
            )
            layer = f"uint({self.vector_component(coord, 'y')})"
            return coord_x, layer
        if image_type in {
            "texture1d<float, access::read_write>",
            "texture1d<int, access::read_write>",
            "texture1d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 1), None
        if image_type in {
            "texture2d_array<float, access::read_write>",
            "texture2d_array<int, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            coord_xy = self.unsigned_coordinate_expression(
                self.vector_component(coord, "xy"), 2
            )
            layer = f"uint({self.vector_component(coord, 'z')})"
            return coord_xy, layer
        if image_type in {
            "texture3d<float, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture3d<uint, access::read_write>",
        }:
            return self.unsigned_coordinate_expression(coord, 3), None
        return self.unsigned_coordinate_expression(coord, 2), None

    def unsigned_coordinate_expression(self, coord, dimensions):
        constructor = "uint" if dimensions == 1 else f"uint{dimensions}"
        coord_text = str(coord).strip()
        if coord_text.startswith(f"{constructor}("):
            return coord_text
        return f"{constructor}({coord_text})"

    def storage_image_access_agnostic_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        return metal_storage_image_access_agnostic_type(image_type)

    def is_integer_image_type(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_integer_image_type(image_type)

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, image_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(image_type)

    def is_float_image_resource(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        return is_metal_float_image_resource(image_type)

    def image_store_constructors_by_kind(self):
        return {
            "float": "float4",
            "int": "int4",
            "uint": "uint4",
        }

    def image_store_zero_values_by_kind(self):
        return {
            "float": "0.0",
            "int": "0",
            "uint": "0u",
        }

    def image_load_component_suffix(self, image_type, image_format):
        return storage_image_load_component_suffix(
            image_format,
            expected_scalar=self.is_scalar_value_type(
                self.current_expression_expected_type
            ),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                image_type, image_format
            ),
            float_resource=self.is_float_image_resource(image_type),
        )

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, image_type):
        image_type = self.storage_image_access_agnostic_type(image_type)
        if image_type in {
            "texture1d<int, access::read_write>",
            "texture1d_array<int, access::read_write>",
            "texture2d<int, access::read_write>",
            "texture3d<int, access::read_write>",
            "texture2d_array<int, access::read_write>",
        }:
            return "int4"
        if image_type in {
            "texture1d<uint, access::read_write>",
            "texture1d_array<uint, access::read_write>",
            "texture2d<uint, access::read_write>",
            "texture3d<uint, access::read_write>",
            "texture2d_array<uint, access::read_write>",
        }:
            return "uint4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        return storage_image_two_component_store_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            self.image_store_constructors_by_kind(),
            self.image_store_zero_values_by_kind(),
        )

    def image_store_value_expression(
        self, image_type, image_format, value, value_type=None
    ):
        return storage_image_store_value_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                image_type, image_format
            ),
            float_resource=self.is_float_image_resource(image_type),
            integer_constructor=self.integer_image_store_constructor(image_type),
            float_constructor="float4",
            constructors_by_kind=self.image_store_constructors_by_kind(),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

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

    def image_atomic_compare_descriptor(self, texture_type):
        texture_type = self.storage_image_access_agnostic_type(texture_type)
        component_type = metal_storage_image_component_type(texture_type)
        if component_type not in {"int", "uint"} or "<" not in texture_type:
            return None

        texture_family = texture_type.split("<", 1)[0]
        suffix_family = {
            "texture1d": "image1D",
            "texture1d_array": "image1DArray",
            "texture2d": "image2D",
            "texture3d": "image3D",
            "texture2d_array": "image2DArray",
        }.get(texture_family)
        coordinate_type = {
            "texture1d": "int",
            "texture1d_array": "int2",
            "texture2d": "int2",
            "texture3d": "int3",
            "texture2d_array": "int3",
        }.get(texture_family)
        exchange_expression = {
            "texture1d": (
                "image.atomic_compare_exchange_weak(uint(coord), &original, value)"
            ),
            "texture1d_array": (
                "image.atomic_compare_exchange_weak(uint(coord.x), uint(coord.y), &original, value)"
            ),
            "texture2d": (
                "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
            ),
            "texture3d": (
                "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
            ),
            "texture2d_array": (
                "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
            ),
        }.get(texture_family)
        if (
            suffix_family is None
            or coordinate_type is None
            or exchange_expression is None
        ):
            return None

        return {
            "helper_name": (
                f"imageAtomicCompSwap_"
                f"{'i' if component_type == 'int' else 'u'}{suffix_family}"
            ),
            "return_type": component_type,
            "vector_type": f"{component_type}4",
            "coord_type": coordinate_type,
            "exchange_expr": exchange_expression,
        }

    def image_atomic_compare_helper_name(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["helper_name"] if descriptor else None

    def image_atomic_compare_return_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["return_type"] if descriptor else None

    def image_atomic_compare_vector_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["vector_type"] if descriptor else None

    def image_atomic_compare_coord_type(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["coord_type"] if descriptor else None

    def image_atomic_compare_exchange_expression(self, texture_type):
        descriptor = self.image_atomic_compare_descriptor(texture_type)
        return descriptor["exchange_expr"] if descriptor else None

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

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_integer_coordinate_argument(func_name, args)
        self.validate_coordinate_dimension_argument(func_name, args)
        self.validate_query_lod_coordinate_argument(func_name, args)
        self.validate_compare_argument(func_name, args)
        self.validate_lod_argument(func_name, args)
        self.validate_bias_argument(func_name, args)
        self.validate_sample_index_argument(func_name, args)
        self.validate_mip_level_argument(func_name, args)
        self.validate_gradient_dimension_arguments(func_name, args)
        self.validate_offset_dimension_argument(func_name, args)
        self.validate_gather_component_argument(func_name, args)

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
            if extra_args:
                bias = self.generate_expression(extra_args[0])
                if is_array_texture:
                    return (
                        f"{texture_name}.sample("
                        f"{sampler_arg}, {coord_xy}, {layer}, bias({bias}))"
                    )
                return f"{texture_name}.sample({sampler_arg}, {coord}, bias({bias}))"
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
        generic_args = getattr(type_node, "generic_args", [])
        if hasattr(type_node, "name") and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg) for arg in generic_args
            )
            return f"{type_node.name}<{args}>"
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
        """Map a CrossGL semantic to Metal attribute syntax."""
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
