"""CrossGL-to-WGSL code generator."""

from __future__ import annotations

import copy
import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    GenericType,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    RangeNode,
    ReferenceType,
    ReturnNode,
    StructMemberNode,
    StructNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WhileNode,
)
from .enum_utils import generic_enum_specialization_name, generic_type_parts
from .image_access_contracts import (
    explicit_image_access,
    merge_image_access_requirement,
)
from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name


class WGSLCodeGen:
    """Generate WebGPU WGSL output from CrossGL ASTs."""

    SUPPORTED_STAGE_NAMES = {"vertex", "fragment", "compute"}
    UNSUPPORTED_STAGE_NAMES = {
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "amplification",
        "object",
        "ray_generation",
        "raygen",
        "ray_intersection",
        "intersection",
        "ray_any_hit",
        "any_hit",
        "ray_closest_hit",
        "closest_hit",
        "ray_miss",
        "miss",
        "ray_callable",
        "callable",
    }
    VECTOR_TYPE_RE = re.compile(r"^(?:vec|float|int|uint|bool|ivec|uvec|bvec)([234])$")
    FLOAT16_VECTOR_TYPE_RE = re.compile(r"^(?:f16vec|half)([234])$")
    MATRIX_TYPE_RE = re.compile(r"^(?:mat|float)([234])(?:x([234]))?$")
    TYPE_CONSTRUCTOR_RE = re.compile(
        r"^(?:f16vec[234]|half[234]|vec[234]|float[234]|int[234]|uint[234]|"
        r"bool[234]|ivec[234]|uvec[234]|bvec[234]|mat[234](?:x[234])?|"
        r"float[234]x[234])$"
    )

    PRIMITIVE_TYPE_MAP = {
        "void": "void",
        "bool": "bool",
        "boolean": "bool",
        "int": "i32",
        "i32": "i32",
        "short": "i32",
        "long": "i32",
        "uint": "u32",
        "u32": "u32",
        "unsigned": "u32",
        "float": "f32",
        "f32": "f32",
        "f16": "f32",
        "float16": "f32",
        "half": "f32",
        "double": "f32",
        "f64": "f32",
    }
    BUILTIN_SEMANTICS = {
        "gl_position": "position",
        "gl_fragcoord": "position",
        "position_builtin": "position",
        "sv_position": "position",
        "frag_depth": "frag_depth",
        "sv_depth": "frag_depth",
        "vertex_index": "vertex_index",
        "vertexid": "vertex_index",
        "sv_vertexid": "vertex_index",
        "instance_index": "instance_index",
        "instanceid": "instance_index",
        "sv_instanceid": "instance_index",
        "global_invocation_id": "global_invocation_id",
        "gl_globalinvocationid": "global_invocation_id",
        "sv_dispatchthreadid": "global_invocation_id",
        "local_invocation_id": "local_invocation_id",
        "gl_localinvocationid": "local_invocation_id",
        "sv_groupthreadid": "local_invocation_id",
        "local_invocation_index": "local_invocation_index",
        "gl_localinvocationindex": "local_invocation_index",
        "sv_groupindex": "local_invocation_index",
        "workgroup_id": "workgroup_id",
        "gl_workgroupid": "workgroup_id",
        "sv_groupid": "workgroup_id",
        "num_workgroups": "num_workgroups",
    }
    BUILTIN_IDENTIFIER_ALIASES = {
        "gl_Position": "position",
        "gl_FragCoord": "position",
        "SV_Position": "position",
        "gl_GlobalInvocationID": "global_invocation_id",
        "SV_DispatchThreadID": "global_invocation_id",
        "gl_LocalInvocationID": "local_invocation_id",
        "SV_GroupThreadID": "local_invocation_id",
        "gl_LocalInvocationIndex": "local_invocation_index",
        "SV_GroupIndex": "local_invocation_index",
        "gl_WorkGroupID": "workgroup_id",
        "SV_GroupID": "workgroup_id",
        "gl_NumWorkGroups": "num_workgroups",
    }
    WORKGROUP_SIZE_IDENTIFIER_ALIASES = {"gl_WorkGroupSize"}
    INPUT_BUILTIN_TYPE_MAP = {
        "position": "vec4<f32>",
        "vertex_index": "u32",
        "instance_index": "u32",
        "global_invocation_id": "vec3<u32>",
        "local_invocation_id": "vec3<u32>",
        "local_invocation_index": "u32",
        "workgroup_id": "vec3<u32>",
        "num_workgroups": "vec3<u32>",
    }
    FUNCTION_NAME_MAP = {
        "atan2": "atan2",
        "fract": "fract",
        "inversesqrt": "inverseSqrt",
        "inverseSqrt": "inverseSqrt",
        "lerp": "mix",
        "mix": "mix",
        "rsqrt": "inverseSqrt",
        "saturate": "saturate",
    }
    DERIVATIVE_FUNCTION_NAME_MAP = {
        "dfdx": "dpdx",
        "ddx": "dpdx",
        "dfdxfine": "dpdxFine",
        "ddxfine": "dpdxFine",
        "ddx_fine": "dpdxFine",
        "dfdxcoarse": "dpdxCoarse",
        "ddxcoarse": "dpdxCoarse",
        "ddx_coarse": "dpdxCoarse",
        "dfdy": "dpdy",
        "ddy": "dpdy",
        "dfdyfine": "dpdyFine",
        "ddyfine": "dpdyFine",
        "ddy_fine": "dpdyFine",
        "dfdycoarse": "dpdyCoarse",
        "ddycoarse": "dpdyCoarse",
        "ddy_coarse": "dpdyCoarse",
    }
    STAGE_INPUT_BUILTINS = {
        "vertex": {"instance_index", "vertex_index"},
        "fragment": {"position"},
        "compute": {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "num_workgroups",
            "workgroup_id",
        },
    }
    RESOURCE_TYPE_NAMES = {
        "image1d",
        "image1darray",
        "image2d",
        "image2darray",
        "image2dms",
        "image2dmsarray",
        "image3d",
        "imagebuffer",
        "imagecube",
        "imagecubearray",
        "iimage1d",
        "iimage1darray",
        "iimage2d",
        "iimage2darray",
        "iimage2dms",
        "iimage2dmsarray",
        "iimage3d",
        "iimagebuffer",
        "iimagecube",
        "iimagecubearray",
        "sampler",
        "sampler1d",
        "sampler1darray",
        "sampler2d",
        "sampler2darray",
        "sampler2darrayshadow",
        "sampler2dms",
        "sampler2dmsarray",
        "sampler2dshadow",
        "sampler3d",
        "samplercube",
        "samplercubearray",
        "samplercubearrayshadow",
        "samplercubeshadow",
        "samplerstate",
        "samplercomparisonstate",
        "texture1d",
        "texture1darray",
        "texture2d",
        "texture2darray",
        "texture2dms",
        "texture2dmsarray",
        "texture3d",
        "texturecube",
        "texturecubearray",
        "uimage1d",
        "uimage1darray",
        "uimage2d",
        "uimage2darray",
        "uimage2dms",
        "uimage2dmsarray",
        "uimage3d",
        "uimagebuffer",
        "uimagecube",
        "uimagecubearray",
        "isampler2d",
        "isampler2darray",
        "usampler2d",
        "usampler2darray",
    }
    STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
        "structuredbuffer",
    }
    WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
    }
    CONSTANT_BUFFER_TYPE_NAMES = {
        "constantbuffer",
    }
    UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES = {
        "appendstructuredbuffer",
        "byteaddressbuffer",
        "consumestructuredbuffer",
        "rwbyteaddressbuffer",
    }
    STRUCTURED_BUFFER_FREE_HELPERS = {
        "buffer_load",
        "buffer_store",
    }
    UNSUPPORTED_STORAGE_BUFFER_FREE_HELPERS = {
        "buffer_append",
        "buffer_consume",
        "buffer_dimensions",
    }
    STRUCTURED_BUFFER_MEMBER_HELPERS = {
        "load",
        "store",
    }
    UNSUPPORTED_STRUCTURED_BUFFER_MEMBER_HELPERS = {
        "append",
        "consume",
        "decrementcounter",
        "getdimensions",
        "incrementcounter",
        "load2",
        "load3",
        "load4",
        "store2",
        "store3",
        "store4",
    }
    SUPPORTED_GLSL_BUFFER_BLOCK_LAYOUTS = {"std430"}
    SAMPLED_TEXTURE_TYPE_MAP = {
        "sampler1d": "texture_1d<f32>",
        "sampler2d": "texture_2d<f32>",
        "sampler2darray": "texture_2d_array<f32>",
        "sampler2darrayshadow": "texture_depth_2d_array",
        "sampler2dshadow": "texture_depth_2d",
        "sampler3d": "texture_3d<f32>",
        "samplercube": "texture_cube<f32>",
        "samplercubearray": "texture_cube_array<f32>",
        "samplercubearrayshadow": "texture_depth_cube_array",
        "samplercubeshadow": "texture_depth_cube",
        "texture1d": "texture_1d<f32>",
        "texture2d": "texture_2d<f32>",
        "texture2darray": "texture_2d_array<f32>",
        "texture3d": "texture_3d<f32>",
        "texturecube": "texture_cube<f32>",
        "texturecubearray": "texture_cube_array<f32>",
        "isampler2d": "texture_2d<i32>",
        "isampler2darray": "texture_2d_array<i32>",
        "usampler2d": "texture_2d<u32>",
        "usampler2darray": "texture_2d_array<u32>",
    }
    STORAGE_TEXTURE_DIMENSION_MAP = {
        "image1d": "1d",
        "image2d": "2d",
        "image2darray": "2d_array",
        "image3d": "3d",
        "iimage1d": "1d",
        "iimage2d": "2d",
        "iimage2darray": "2d_array",
        "iimage3d": "3d",
        "uimage1d": "1d",
        "uimage2d": "2d",
        "uimage2darray": "2d_array",
        "uimage3d": "3d",
    }
    STORAGE_TEXTURE_FORMAT_MAP = {
        "r32f": "r32float",
        "r32i": "r32sint",
        "r32ui": "r32uint",
        "rg32f": "rg32float",
        "rg32i": "rg32sint",
        "rg32ui": "rg32uint",
        "rgba8": "rgba8unorm",
        "rgba8snorm": "rgba8snorm",
        "rgba8i": "rgba8sint",
        "rgba8ui": "rgba8uint",
        "rgba16f": "rgba16float",
        "rgba16i": "rgba16sint",
        "rgba16ui": "rgba16uint",
        "rgba32f": "rgba32float",
        "rgba32i": "rgba32sint",
        "rgba32ui": "rgba32uint",
    }
    SAMPLER_TYPE_NAMES = {
        "sampler",
        "samplerstate",
    }
    COMPARISON_SAMPLER_TYPE_NAMES = {
        "samplercomparisonstate",
    }
    TEXTURE_FUNCTION_NAMES = {
        "texture",
        "texturebias",
        "texturecompare",
        "texturecomparegrad",
        "texturecompareoffset",
        "texturecomparelod",
        "texturecomparelodoffset",
        "texturecompareproj",
        "texturecompareprojgrad",
        "texturecompareprojoffset",
        "texturegrad",
        "texturegradoffset",
        "texturegather",
        "texturegathercompare",
        "texturegathercompareoffset",
        "texturegatheroffset",
        "texturegatheroffsets",
        "texturelod",
        "texturelodoffset",
        "textureoffset",
        "textureproj",
        "textureprojgrad",
        "textureprojgradoffset",
        "textureprojlod",
        "textureprojlodoffset",
        "textureprojoffset",
        "texturequerylevels",
        "texturequerylod",
        "texturesize",
        "texelfetch",
    }
    BARRIER_FUNCTION_NAMES = {
        "barrier",
        "groupmemorybarrierwithgroupsync",
        "workgroupbarrier",
    }
    WGSL_RESERVED_IDENTIFIERS = {
        "NULL",
        "Self",
        "abstract",
        "active",
        "alias",
        "alignas",
        "alignof",
        "as",
        "asm",
        "asm_fragment",
        "async",
        "attribute",
        "auto",
        "await",
        "become",
        "break",
        "case",
        "cast",
        "catch",
        "class",
        "co_await",
        "co_return",
        "co_yield",
        "coherent",
        "column_major",
        "common",
        "compile",
        "compile_fragment",
        "concept",
        "const",
        "const_assert",
        "const_cast",
        "consteval",
        "constexpr",
        "constinit",
        "continue",
        "continuing",
        "crate",
        "debugger",
        "decltype",
        "default",
        "delete",
        "demote",
        "demote_to_helper",
        "diagnostic",
        "discard",
        "do",
        "dynamic_cast",
        "else",
        "enable",
        "enum",
        "explicit",
        "export",
        "extends",
        "extern",
        "external",
        "fallthrough",
        "false",
        "filter",
        "final",
        "finally",
        "fn",
        "for",
        "friend",
        "from",
        "fxgroup",
        "get",
        "goto",
        "groupshared",
        "highp",
        "if",
        "impl",
        "implements",
        "import",
        "inline",
        "instanceof",
        "interface",
        "layout",
        "let",
        "loop",
        "lowp",
        "macro",
        "macro_rules",
        "match",
        "mediump",
        "meta",
        "mod",
        "module",
        "move",
        "mut",
        "mutable",
        "namespace",
        "new",
        "nil",
        "noexcept",
        "noinline",
        "nointerpolation",
        "non_coherent",
        "noncoherent",
        "noperspective",
        "null",
        "nullptr",
        "of",
        "operator",
        "override",
        "package",
        "packoffset",
        "partition",
        "pass",
        "patch",
        "pixelfragment",
        "precise",
        "precision",
        "premerge",
        "priv",
        "protected",
        "pub",
        "public",
        "readonly",
        "ref",
        "regardless",
        "register",
        "reinterpret_cast",
        "require",
        "requires",
        "resource",
        "restrict",
        "return",
        "self",
        "set",
        "shared",
        "sizeof",
        "smooth",
        "snorm",
        "static",
        "static_assert",
        "static_cast",
        "std",
        "struct",
        "subroutine",
        "super",
        "switch",
        "target",
        "template",
        "this",
        "thread_local",
        "throw",
        "trait",
        "true",
        "try",
        "type",
        "typedef",
        "typeid",
        "typename",
        "typeof",
        "union",
        "unless",
        "unorm",
        "unsafe",
        "unsized",
        "use",
        "using",
        "var",
        "varying",
        "virtual",
        "volatile",
        "wgsl",
        "where",
        "while",
        "with",
        "writeonly",
        "yield",
    }

    def __init__(self):
        self._current_stage_name = None
        self._current_workgroup_size = None
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._global_binding_index = 0
        self._cbuffer_member_accesses = {}
        self._function_texture_parameters = {}
        self._function_pointer_parameters = {}
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        self._current_expression_expected_type = None
        self._current_function_return_type = None
        self._identifier_scopes = []
        self._identifier_alias_scopes = []
        self._pointer_identifier_scopes = []
        self._value_type_scopes = []
        self._resource_alias_scopes = []
        self._module_identifier_names = {}
        self._function_identifier_names = {}
        self._function_signature_identifier_names = {}
        self._function_overloads_by_name = {}
        self._type_identifier_names = {}
        self._struct_member_identifier_names = {}
        self._structs_by_name = {}
        self._struct_member_types = {}
        self._cbuffer_member_types = {}
        self._struct_resource_paths = {}
        self._glsl_buffer_block_struct_names = set()
        self._uniform_buffer_struct_names = set()
        self._uniform_scalar_array_wrappers = {}
        self._module_variable_types = {}
        self._module_storage_access_modes = {}
        self._module_storage_texture_access_modes = {}
        self._inferred_storage_texture_access_modes = {}
        self._module_resource_bindings = {}
        self._function_resource_member_parameters = {}
        self._reserved_bindings_by_group = {}
        self._allocated_bindings_by_group = {}
        self._reserved_sampler_bindings = {}
        self._explicit_binding_owners = {}
        self._explicit_binding_register_classes = {}
        self._hlsl_register_binding_allocations = {}
        self._hlsl_register_source_owners = {}
        self._stage_output_lowerings = {}
        self._stage_interface_member_locations = {}
        self._builtin_option_specializations = {}

    def generate(self, ast):
        return self.generate_program(ast)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        self.validate_wgsl_stage_support(ast, target_stage)
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._global_binding_index = 0
        self._identifier_scopes = []
        self._identifier_alias_scopes = []
        self._pointer_identifier_scopes = []
        self._value_type_scopes = []
        self._resource_alias_scopes = []
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        self._current_expression_expected_type = None
        self._current_function_return_type = None
        self._module_identifier_names = {}
        self._function_identifier_names = {}
        self._function_signature_identifier_names = {}
        self._function_overloads_by_name = {}
        self._type_identifier_names = {}
        self._struct_member_identifier_names = {}
        self._structs_by_name = {}
        self._struct_member_types = {}
        self._cbuffer_member_types = {}
        self._struct_resource_paths = {}
        self._glsl_buffer_block_struct_names = set()
        self._uniform_buffer_struct_names = set()
        self._uniform_scalar_array_wrappers = {}
        self._module_variable_types = {}
        self._module_storage_access_modes = {}
        self._module_storage_texture_access_modes = {}
        self._inferred_storage_texture_access_modes = {}
        self._module_resource_bindings = {}
        self._function_resource_member_parameters = {}
        self._reserved_bindings_by_group = {}
        self._allocated_bindings_by_group = {}
        self._reserved_sampler_bindings = {}
        self._explicit_binding_owners = {}
        self._explicit_binding_register_classes = {}
        self._hlsl_register_binding_allocations = {}
        self._hlsl_register_source_owners = {}
        self._stage_output_lowerings = {}
        self._stage_interface_member_locations = {}
        self._builtin_option_specializations = {}

        lines = ["// Generated by CrossGL for WebGPU WGSL"]
        emitted_sections = []

        structs = self._collect_structs(ast, target_stage)
        self._stage_output_lowerings = self.stage_output_lowerings(
            ast, target_stage, structs
        )
        if self._stage_output_lowerings:
            structs = [
                *structs,
                *(
                    lowering["struct"]
                    for lowering in self._stage_output_lowerings.values()
                ),
            ]
        self._stage_interface_member_locations = (
            self.stage_interface_member_locations(ast, target_stage, structs)
        )
        cbuffers = self._collect_cbuffers(ast, target_stage)
        constants = list(getattr(ast, "constants", []) or [])
        global_variable_nodes = self._collect_global_variables(ast, target_stage)
        stage_resource_parameters = self._collect_stage_resource_parameters(
            ast, target_stage
        )
        helper_function_nodes = list(self._helper_functions(ast, target_stage))
        all_function_nodes = list(helper_function_nodes)
        all_function_nodes.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        self._builtin_option_specializations = (
            self.collect_builtin_option_specializations(ast)
        )

        self.collect_identifier_metadata(
            structs,
            cbuffers,
            constants,
            global_variable_nodes,
            stage_resource_parameters,
            helper_function_nodes,
        )
        self.collect_struct_type_metadata(structs)
        self._struct_member_types.update(
            self.builtin_option_specialization_member_types(
                self._builtin_option_specializations
            )
        )
        self.collect_cbuffer_member_identifier_metadata(cbuffers)
        self._cbuffer_member_accesses = self.cbuffer_member_accesses(cbuffers)
        self._cbuffer_member_types = self.cbuffer_member_types(cbuffers)
        self._function_texture_parameters = self.function_texture_parameters(
            ast, target_stage
        )
        self._function_pointer_parameters = self.function_buffer_pointer_parameters(
            ast, target_stage
        )
        self._function_resource_member_parameters = (
            self.function_resource_member_parameters(ast, target_stage)
        )
        self._module_variable_types = {
            getattr(node, "name", ""): getattr(node, "var_type", None)
            for node in global_variable_nodes
            if getattr(node, "name", "")
        }
        self._module_variable_types.update(
            {
                getattr(parameter, "name", ""): self.stage_resource_module_type(
                    parameter
                )
                for parameter in stage_resource_parameters
                if getattr(parameter, "name", "")
            }
        )
        self._module_variable_types.update(
            {
                self.cbuffer_instance_name(cbuffer): NamedType(cbuffer.name)
                for cbuffer in cbuffers
            }
        )
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        for function in all_function_nodes:
            function_name = getattr(function, "name", "")
            if not function_name:
                continue
            signature_key = self.function_signature_key(function)
            self._function_return_types_by_signature[signature_key] = getattr(
                function, "return_type", None
            )
            if len(self._function_overloads_by_name.get(function_name, ())) <= 1:
                self._function_return_types[function_name] = getattr(
                    function, "return_type", None
                )
        self._glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                global_variable_nodes, all_function_nodes
            )
        )
        self._uniform_buffer_struct_names = self.uniform_buffer_struct_names(
            cbuffers, global_variable_nodes, stage_resource_parameters
        )
        self._uniform_scalar_array_wrappers = (
            self.uniform_scalar_array_wrapper_names(cbuffers, structs)
        )
        self._module_storage_access_modes = self.module_storage_access_modes(
            global_variable_nodes, stage_resource_parameters
        )
        self._inferred_storage_texture_access_modes = (
            self.inferred_storage_texture_access_modes(
                global_variable_nodes,
                stage_resource_parameters,
                all_function_nodes,
            )
        )
        self.reserve_explicit_resource_bindings(
            cbuffers, global_variable_nodes, stage_resource_parameters
        )
        uniform_scalar_array_wrappers = self.generate_uniform_scalar_array_wrappers()
        if uniform_scalar_array_wrappers:
            emitted_sections.append(uniform_scalar_array_wrappers)

        builtin_option_declarations = self.generate_builtin_option_declarations()
        if builtin_option_declarations:
            emitted_sections.append(builtin_option_declarations)

        if structs:
            emitted_sections.append(
                "\n\n".join(self.generate_struct(node) for node in structs)
            )

        constants = [self.generate_constant(node) for node in constants]
        if constants:
            emitted_sections.append("\n".join(constants))

        if cbuffers:
            emitted_sections.append(
                "\n\n".join(self.generate_cbuffer(node) for node in cbuffers)
            )

        stage_resources = [
            self.generate_stage_resource_parameter(node)
            for node in stage_resource_parameters
        ]
        if stage_resources:
            emitted_sections.append("\n".join(stage_resources))

        global_variables = [
            self.generate_global_variable(node) for node in global_variable_nodes
        ]
        if global_variables:
            emitted_sections.append("\n".join(global_variables))

        helper_functions = [
            self.generate_function(func) for func in helper_function_nodes if func
        ]
        if helper_functions:
            emitted_sections.append("\n\n".join(helper_functions))

        stage_functions = []
        stage_name_counts = {}
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            stage_index = stage_name_counts.get(stage_name, 0)
            stage_name_counts[stage_name] = stage_index + 1
            entry_name = f"{stage_name}_main"
            if stage_index:
                entry_name = f"{entry_name}_{stage_index}"
            stage_functions.append(self.generate_stage(stage_node, entry_name))
        if stage_functions:
            emitted_sections.append("\n\n".join(stage_functions))

        if emitted_sections:
            lines.append("")
            lines.append(
                "\n\n".join(section for section in emitted_sections if section)
            )
        return "\n".join(lines).rstrip() + "\n"

    def validate_wgsl_stage_support(self, ast, target_stage=None):
        stages = set()
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages.add(normalized_target_stage)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name:
                stages.add(stage_name)
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)
            entry_point = getattr(stage_node, "entry_point", None)
            qualifier = (
                entry_point.qualifiers[0]
                if getattr(entry_point, "qualifiers", None)
                else None
            )
            entry_stage = normalize_stage_name(qualifier)
            if entry_stage:
                stages.add(entry_stage)

        unsupported = sorted(
            stage
            for stage in stages
            if stage in self.UNSUPPORTED_STAGE_NAMES
            or (stage and stage not in self.SUPPORTED_STAGE_NAMES)
        )
        if unsupported:
            raise ValueError(
                "WGSL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

    def generate_stage(self, stage_node, entry_name=None):
        stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
        if stage_name not in self.SUPPORTED_STAGE_NAMES:
            raise ValueError(f"WGSL target does not support shader stage: {stage_name}")

        entry_point = self.stage_output_lowered_entry_point(
            stage_node, stage_node.entry_point
        )
        previous_stage = self._current_stage_name
        previous_workgroup_size = self._current_workgroup_size
        previous_function_return_type = self._current_function_return_type
        self._current_stage_name = stage_name
        self._current_workgroup_size = (
            self.stage_workgroup_size_values(stage_node)
            if stage_name == "compute"
            else None
        )
        self._current_function_return_type = getattr(entry_point, "return_type", None)
        try:
            attributes = [f"@{stage_name}"]
            if stage_name == "compute":
                attributes.append(self.generate_workgroup_size_attribute(stage_node))
            stage_resource_parameter_ids = {
                id(parameter)
                for parameter in getattr(entry_point, "parameters", [])
                if self.is_stage_resource_parameter(parameter)
            }
            self.push_identifier_scope(
                getattr(param, "name", "")
                for param in getattr(entry_point, "parameters", [])
            )
            self.register_parameter_value_types(entry_point)
            self.push_pointer_identifier_scope(
                self.buffer_pointer_parameter_names(entry_point)
            )
            try:
                signature = self.generate_function_signature(
                    entry_point,
                    name=entry_name or f"{stage_name}_main",
                    return_attributes=self.stage_return_attributes(
                        stage_name, entry_point
                    ),
                    leading_parameters=self.stage_implicit_builtin_parameters(
                        entry_point, stage_name
                    ),
                    skip_parameter_ids=stage_resource_parameter_ids,
                    parameter_attributes=True,
                )
                body = self.generate_block(entry_point.body, indent=0)
            finally:
                self.pop_pointer_identifier_scope()
                self.pop_identifier_scope()
            return "\n".join(attributes + [f"{signature} {body}"])
        finally:
            self._current_stage_name = previous_stage
            self._current_workgroup_size = previous_workgroup_size
            self._current_function_return_type = previous_function_return_type

    def stage_output_lowerings(self, ast, target_stage, existing_structs):
        used_struct_names = {
            getattr(struct, "name", "")
            for struct in existing_structs
            if getattr(struct, "name", "")
        }
        lowerings = {}
        for stage_node in self._stage_nodes(ast, target_stage):
            output_variable = self.vertex_position_output_variable(stage_node)
            entry_point = getattr(stage_node, "entry_point", None)
            if output_variable is None or entry_point is None:
                continue
            if not self.is_void_type(getattr(entry_point, "return_type", None)):
                continue
            struct_name = self.unique_stage_output_struct_name(
                "VertexOutput", used_struct_names
            )
            used_struct_names.add(struct_name)
            field_name = "position"
            struct_node = StructNode(
                struct_name,
                [
                    StructMemberNode(
                        field_name,
                        output_variable.var_type,
                        attributes=[AttributeNode("gl_Position")],
                    )
                ],
            )
            lowerings[id(stage_node)] = {
                "source_name": output_variable.name,
                "field_name": field_name,
                "local_name": self.unique_stage_output_local_name(entry_point),
                "struct_name": struct_name,
                "struct": struct_node,
            }
        return lowerings

    def unique_stage_output_struct_name(self, preferred, used_names):
        if preferred not in used_names:
            return preferred
        base = "SPIRVVertexOutput"
        if base not in used_names:
            return base
        index = 2
        while f"{base}{index}" in used_names:
            index += 1
        return f"{base}{index}"

    def unique_stage_output_local_name(self, entry_point):
        used_names = {
            getattr(parameter, "name", "")
            for parameter in getattr(entry_point, "parameters", []) or []
            if getattr(parameter, "name", "")
        }
        body = getattr(entry_point, "body", None)
        if body is not None and hasattr(body, "walk"):
            used_names.update(
                getattr(node, "name", "")
                for node in body.walk()
                if isinstance(node, VariableNode) and getattr(node, "name", "")
            )
        if "output" not in used_names:
            return "output"
        base = "spirvPositionOutput"
        if base not in used_names:
            return base
        index = 2
        while f"{base}{index}" in used_names:
            index += 1
        return f"{base}{index}"

    def stage_interface_member_locations(self, ast, target_stage, structs):
        structs_by_name = {
            getattr(struct, "name", ""): struct
            for struct in structs
            if getattr(struct, "name", "")
        }
        interface_struct_names = self.stage_interface_struct_names(
            ast, target_stage, structs_by_name
        )
        locations = {}
        for struct in structs:
            struct_name = getattr(struct, "name", "")
            if struct_name not in interface_struct_names:
                continue

            used_locations = set()
            unbound_members = []
            for member in getattr(struct, "members", []) or []:
                binding_kind, location = self.member_stage_binding(member)
                if binding_kind == "location":
                    if location is not None:
                        used_locations.add(location)
                    continue
                if binding_kind == "builtin":
                    continue
                unbound_members.append(member)

            next_location = 0
            for member in unbound_members:
                while next_location in used_locations:
                    next_location += 1
                locations[(struct_name, member.name)] = next_location
                used_locations.add(next_location)
                next_location += 1
        return locations

    def stage_interface_struct_names(self, ast, target_stage, structs_by_name):
        names = set()
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            if stage_name not in {"vertex", "fragment"}:
                continue
            entry_point = self.stage_output_lowered_entry_point(
                stage_node, getattr(stage_node, "entry_point", None)
            )
            if entry_point is None:
                continue

            for parameter in getattr(entry_point, "parameters", []) or []:
                if self.is_stage_resource_parameter(parameter):
                    continue
                struct_name = self.stage_interface_struct_type_name(
                    getattr(parameter, "param_type", None), structs_by_name
                )
                if struct_name:
                    names.add(struct_name)

            struct_name = self.stage_interface_struct_type_name(
                getattr(entry_point, "return_type", None), structs_by_name
            )
            if struct_name:
                names.add(struct_name)
        return names

    def stage_interface_struct_type_name(self, type_node, structs_by_name):
        struct_name = self.struct_type_name(type_node)
        if struct_name in structs_by_name:
            return struct_name
        return None

    def member_stage_binding(self, member):
        for attr in getattr(member, "attributes", []) or []:
            semantic = str(getattr(attr, "name", attr))
            key = self.semantic_key(semantic)
            if key == "builtin":
                return "builtin", None
            if key == "location":
                return "location", self.attribute_integer_argument(attr)
            if self.BUILTIN_SEMANTICS.get(key):
                return "builtin", None
            location = self.semantic_location(semantic, direction="generic")
            if location is not None:
                return "location", location
        return None, None

    def attribute_integer_argument(self, attr):
        arguments = getattr(attr, "arguments", []) or []
        if len(arguments) != 1:
            return None
        return self.integer_literal_value(arguments[0])

    def integer_literal_value(self, expr):
        if isinstance(expr, LiteralNode):
            value = str(expr.value)
            match = re.fullmatch(r"([+-]?(?:0[xX][0-9a-fA-F]+|\d+))[uUlL]*", value)
            if match:
                try:
                    return int(match.group(1), 0)
                except ValueError:
                    return None
        if (
            isinstance(expr, UnaryOpNode)
            and expr.operator == "+"
            and not expr.is_postfix
        ):
            return self.integer_literal_value(expr.operand)
        return None

    def vertex_position_output_variable(self, stage_node):
        if normalize_stage_name(getattr(stage_node, "stage", None)) != "vertex":
            return None
        for variable in getattr(stage_node, "local_variables", []) or []:
            if self.is_vertex_position_output_variable(variable):
                return variable
        return None

    def is_vertex_position_output_variable(self, variable):
        attribute_keys = {
            self.semantic_key(str(getattr(attribute, "name", attribute)))
            for attribute in getattr(variable, "attributes", []) or []
        }
        if "output" not in attribute_keys:
            return False
        return any(
            self.BUILTIN_SEMANTICS.get(attribute_key) == "position"
            for attribute_key in attribute_keys
        )

    def is_void_type(self, vtype):
        return self.semantic_key(str(getattr(vtype, "name", vtype))) == "void"

    def stage_output_lowered_entry_point(self, stage_node, entry_point):
        lowering = self._stage_output_lowerings.get(id(stage_node))
        if lowering is None:
            return entry_point
        lowered = copy.copy(entry_point)
        lowered.return_type = NamedType(lowering["struct_name"])
        lowered.body = self.rewrite_stage_output_body(entry_point.body, lowering)
        return lowered

    def rewrite_stage_output_body(self, body, lowering):
        rewritten = []
        saw_return = False
        for statement in getattr(body, "statements", []) or []:
            rewritten_statement = self.rewrite_stage_output_statement(
                statement, lowering
            )
            if isinstance(rewritten_statement, ReturnNode):
                saw_return = True
            rewritten.append(rewritten_statement)
        if not saw_return:
            rewritten.append(ReturnNode(IdentifierNode(lowering["local_name"])))
        output_declaration = VariableNode(
            lowering["local_name"], NamedType(lowering["struct_name"])
        )
        return BlockNode([output_declaration, *rewritten])

    def rewrite_stage_output_statement(self, statement, lowering):
        if isinstance(statement, ExpressionStatementNode):
            expression = statement.expression
            if isinstance(expression, AssignmentNode):
                rewritten = copy.copy(statement)
                rewritten.expression = self.rewrite_stage_output_assignment(
                    expression, lowering
                )
                return rewritten
            return statement
        if isinstance(statement, AssignmentNode):
            return self.rewrite_stage_output_assignment(statement, lowering)
        if isinstance(statement, ReturnNode) and statement.value is None:
            return ReturnNode(IdentifierNode(lowering["local_name"]))
        if isinstance(statement, BlockNode):
            return BlockNode(
                [
                    self.rewrite_stage_output_statement(child, lowering)
                    for child in getattr(statement, "statements", []) or []
                ]
            )
        if isinstance(statement, IfNode):
            rewritten = copy.copy(statement)
            rewritten.then_branch = self.rewrite_stage_output_statement(
                statement.then_branch, lowering
            )
            if statement.else_branch is not None:
                rewritten.else_branch = self.rewrite_stage_output_statement(
                    statement.else_branch, lowering
                )
            rewritten.if_body = rewritten.then_branch
            rewritten.else_body = rewritten.else_branch
            return rewritten
        if isinstance(statement, (ForNode, WhileNode, DoWhileNode, LoopNode)):
            rewritten = copy.copy(statement)
            rewritten.body = self.rewrite_stage_output_statement(
                statement.body, lowering
            )
            return rewritten
        return statement

    def rewrite_stage_output_assignment(self, assignment, lowering):
        if not self.is_stage_output_assignment_target(assignment.target, lowering):
            return assignment
        rewritten = copy.copy(assignment)
        rewritten.target = MemberAccessNode(
            IdentifierNode(lowering["local_name"]), lowering["field_name"]
        )
        rewritten.left = rewritten.target
        return rewritten

    def is_stage_output_assignment_target(self, target, lowering):
        return (
            isinstance(target, IdentifierNode)
            and target.name == lowering["source_name"]
        )

    def generate_workgroup_size_attribute(self, stage_node):
        return (
            "@workgroup_size("
            + ", ".join(self.stage_workgroup_size_values(stage_node))
            + ")"
        )

    def stage_workgroup_size_values(self, stage_node):
        config = getattr(stage_node, "execution_config", {}) or {}
        values = config.get("numthreads") or [
            config.get("local_size_x", "1"),
            config.get("local_size_y", "1"),
            config.get("local_size_z", "1"),
        ]
        if isinstance(values, str):
            values = [part.strip() for part in values.split(",")]
        values = list(values)[:3]
        while len(values) < 3:
            values.append("1")
        return tuple(str(value) for value in values)

    def generate_function(self, func):
        self.validate_helper_function_references(func)
        previous_function_return_type = self._current_function_return_type
        self._current_function_return_type = getattr(func, "return_type", None)
        self.push_identifier_scope(
            getattr(param, "name", "") for param in getattr(func, "parameters", [])
        )
        self.register_parameter_value_types(func)
        self.push_pointer_identifier_scope(self.buffer_pointer_parameter_names(func))
        try:
            signature = self.generate_function_signature(func)
            body = self.generate_block(func.body, indent=0)
        finally:
            self.pop_pointer_identifier_scope()
            self.pop_identifier_scope()
            self._current_function_return_type = previous_function_return_type
        return f"{signature} {body}"

    def generate_function_signature(
        self,
        func,
        name=None,
        return_attributes=(),
        leading_parameters=(),
        skip_parameter_ids=(),
        parameter_attributes=False,
    ):
        function_name = name or self.function_declaration_identifier_name(func)
        skip_parameter_ids = set(skip_parameter_ids or ())
        parameters = ", ".join(
            list(leading_parameters)
            + [
                self.generate_parameter(
                    param, include_attributes=parameter_attributes
                )
                for param in func.parameters
                if id(param) not in skip_parameter_ids
            ]
        )
        return_type = self.type_name_string(func.return_type)
        if return_type == "void":
            return f"fn {function_name}({parameters})"
        return_prefix = " ".join(return_attributes)
        if return_prefix:
            return f"fn {function_name}({parameters}) -> {return_prefix} {return_type}"
        return f"fn {function_name}({parameters}) -> {return_type}"

    def generate_parameter(self, node, include_attributes=True):
        attributes = (
            self.wgsl_attributes(node.attributes, direction="in")
            if include_attributes
            else ""
        )
        prefix = f"{attributes} " if attributes else ""
        parameter_name = self.identifier_name(node.name)
        if self.is_glsl_buffer_block_parameter(node):
            raise ValueError(
                "WGSL target does not support GLSL buffer block parameters yet; "
                "use module-scope storage bindings"
            )
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.param_type
        )
        if unsupported_storage_type is not None:
            raise ValueError(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        if self.structured_buffer_element_type(node.param_type) is not None:
            raise ValueError(
                "WGSL target does not support StructuredBuffer parameters yet; "
                "declare them as module-scope storage resources"
            )
        sampled_texture_type = self.sampled_texture_type(node.param_type)
        if sampled_texture_type is not None:
            return (
                f"{prefix}{parameter_name}: {sampled_texture_type}, "
                f"{self.texture_sampler_name(parameter_name)}: "
                f"{self.companion_sampler_type(node.param_type)}"
            )
        if self.is_sampler_type(node.param_type):
            return f"{prefix}{parameter_name}: {self.sampler_type(node.param_type)}"
        if self.is_buffer_pointer_type(node.param_type, node.qualifiers):
            return (
                f"{prefix}{parameter_name}: "
                f"{self.buffer_pointer_parameter_type(node.param_type)}"
            )
        parameter = (
            f"{prefix}{parameter_name}: {self.type_name_string(node.param_type)}"
        )
        resource_parameters = self.resource_member_parameter_declarations(
            node.name, node.param_type
        )
        if resource_parameters:
            parameter += ", " + ", ".join(resource_parameters)
        return parameter

    def stage_implicit_builtin_parameters(self, function, stage_name):
        referenced = self.stage_direct_builtin_references(function)
        workgroup_size_references = self.direct_workgroup_size_references(function)
        if not referenced and not workgroup_size_references:
            return ()

        if workgroup_size_references and stage_name != "compute":
            raise ValueError(
                "WGSL target does not support gl_WorkGroupSize outside compute stages"
            )

        existing = self.existing_parameter_builtin_names(function)

        unsupported = sorted(
            original
            for original, builtin in referenced.items()
            if builtin not in self.INPUT_BUILTIN_TYPE_MAP
        )
        if unsupported:
            raise ValueError(
                "WGSL target does not support implicit builtin identifier(s): "
                + ", ".join(unsupported)
                + "; model them as entry-point parameters or return values"
            )

        stage_allowed = self.STAGE_INPUT_BUILTINS.get(stage_name, set())
        invalid_for_stage = sorted(
            original
            for original, builtin in referenced.items()
            if builtin in self.INPUT_BUILTIN_TYPE_MAP and builtin not in stage_allowed
        )
        if invalid_for_stage:
            raise ValueError(
                "WGSL target does not support implicit builtin identifier(s) "
                f"in {stage_name} stage: "
                + ", ".join(invalid_for_stage)
                + "; model them as entry-point parameters or return values"
            )

        parameter_names = {
            getattr(param, "name", "") for param in getattr(function, "parameters", [])
        }
        colliding_builtins = sorted(
            builtin
            for builtin in set(referenced.values())
            if builtin in parameter_names and builtin not in existing
        )
        if colliding_builtins:
            builtin = colliding_builtins[0]
            raise ValueError(
                "WGSL target cannot inject builtin "
                f"{builtin} because an entry parameter already uses that name "
                f"without @builtin({builtin})"
            )

        parameters = []
        for builtin, builtin_type in self.INPUT_BUILTIN_TYPE_MAP.items():
            if builtin not in referenced.values() or builtin in existing:
                continue
            parameters.append(f"@builtin({builtin}) {builtin}: {builtin_type}")
        return tuple(parameters)

    def validate_helper_function_references(self, function):
        builtin_references = self.stage_direct_builtin_references(function)
        if builtin_references:
            raise ValueError(
                "WGSL target does not support direct builtin identifier(s) "
                f"in helper function {function.name}: "
                + ", ".join(sorted(builtin_references))
                + "; pass builtin values through entry-point parameters instead"
            )

        workgroup_size_references = self.direct_workgroup_size_references(function)
        if workgroup_size_references:
            raise ValueError(
                "WGSL target does not support gl_WorkGroupSize inside helper "
                f"function {function.name}; keep it directly in compute entry points"
            )

        barrier_calls = self.function_barrier_call_names(function)
        if barrier_calls:
            raise ValueError(
                "WGSL target does not support barrier() inside helper function "
                f"{function.name} yet; keep barriers directly in compute entry points"
            )

    def direct_workgroup_size_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        references = []
        for node in body.walk():
            if (
                isinstance(node, IdentifierNode)
                and node.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES
            ):
                references.append(node.name)
        return tuple(references)

    def function_barrier_call_names(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        calls = []
        for node in body.walk():
            if not isinstance(node, FunctionCallNode):
                continue
            function_name = self.expression_name(node.function)
            if self.semantic_key(function_name) in self.BARRIER_FUNCTION_NAMES:
                calls.append(function_name)
        return tuple(calls)

    def stage_direct_builtin_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return {}
        referenced = {}
        for node in body.walk():
            if not isinstance(node, IdentifierNode):
                continue
            builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(node.name)
            if builtin:
                referenced[node.name] = builtin
        return referenced

    def existing_parameter_builtin_names(self, function):
        builtin_names = set()
        for param in function.parameters:
            for attr in getattr(param, "attributes", []) or []:
                key = self.semantic_key(str(getattr(attr, "name", attr)))
                builtin = self.BUILTIN_SEMANTICS.get(key)
                if builtin:
                    builtin_names.add(builtin)
        return builtin_names

    def generate_struct(self, node):
        if getattr(node, "generic_params", None):
            raise ValueError("WGSL target does not support generic structs yet")
        if node.name in self._glsl_buffer_block_struct_names:
            self.validate_glsl_buffer_block_struct(node)
        lines = [f"struct {self.type_identifier_name(node.name)} {{"]
        for member in node.members:
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                if node.name in self._glsl_buffer_block_struct_names:
                    raise ValueError(
                        "WGSL target does not support resource member "
                        f"{node.name}.{member.name} of type {resource_type_name} "
                        "inside GLSL buffer blocks; declare textures, samplers, "
                        "and storage resources as module-scope bindings"
                    )
                if self.supported_struct_resource_member(member):
                    continue
                raise ValueError(
                    "WGSL target does not support resource member "
                    f"{node.name}.{member.name} of type {resource_type_name}; "
                    "declare textures, samplers, and storage resources as "
                    "module-scope bindings instead of user-struct fields"
                )
            attributes = self.wgsl_attributes(member.attributes, direction="generic")
            implicit_location = self._stage_interface_member_locations.get(
                (node.name, member.name)
            )
            if implicit_location is not None:
                attributes = " ".join(
                    part
                    for part in (f"@location({implicit_location})", attributes)
                    if part
                )
            member_name = self.struct_member_identifier_name(node.name, member.name)
            member_type = member.member_type
            if node.name in self._uniform_buffer_struct_names:
                if self.uniform_scalar_array_wrapper_name(member_type) is not None:
                    attributes = " ".join(
                        attribute
                        for attribute in (attributes, "@align(16)")
                        if attribute
                    )
                member_type_name = self.uniform_buffer_member_type_name(member_type)
            else:
                member_type_name = self.type_name_string(member_type)
            prefix = f"{attributes} " if attributes else ""
            lines.append(
                f"    {prefix}{member_name}: {member_type_name},"
            )
        lines.append("};")
        return "\n".join(lines)

    def generate_cbuffer(self, node):
        lines = [f"struct {self.type_identifier_name(node.name)} {{"]
        for member in getattr(node, "members", []) or []:
            self.validate_cbuffer_member(node, member)
            member_name = self.struct_member_identifier_name(node.name, member.name)
            prefix = (
                "@align(16) "
                if self.uniform_scalar_array_wrapper_name(member.member_type) is not None
                else ""
            )
            lines.append(
                f"    {prefix}{member_name}: "
                f"{self.uniform_buffer_member_type_name(member.member_type)},"
            )
        lines.append("};")

        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        instance_name = self.cbuffer_instance_name(node)
        lines.append(
            f"{attributes}\nvar<uniform> {instance_name}: "
            f"{self.type_identifier_name(node.name)};"
        )
        return "\n".join(lines)

    def generate_constant(self, node):
        value = self.generate_expression(node.value)
        return (
            f"const {self.module_identifier_name(node.name)}: "
            f"{self.type_name_string(node.const_type)} = {value};"
        )

    def generate_global_variable(self, node):
        sampled_texture_type = self.sampled_texture_type(node.var_type)
        if sampled_texture_type is not None:
            return self.generate_sampled_texture_global_variable(
                node, sampled_texture_type
            )
        if self.is_sampler_type(node.var_type):
            return self.generate_sampler_global_variable(node)
        storage_texture_type = self.storage_texture_type(node.var_type, node)
        if storage_texture_type is not None:
            return self.generate_storage_texture_global_variable(
                node, storage_texture_type
            )
        if self.is_buffer_pointer_type(node.var_type, node.qualifiers):
            return self.generate_buffer_pointer_global_variable(node)
        if self.is_glsl_buffer_block_variable(node):
            return self.generate_glsl_buffer_block_global_variable(node)

        qualifier_names = {str(qualifier).lower() for qualifier in node.qualifiers}
        address_space = "private"
        access = ""
        attributes = ""
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.var_type
        )
        if unsupported_storage_type is not None:
            raise ValueError(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        storage_buffer_access = self.structured_buffer_access(node.var_type)
        if storage_buffer_access:
            address_space = "storage"
            access = f", {storage_buffer_access}"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "uniform" in qualifier_names:
            address_space = "uniform"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "buffer" in qualifier_names or "storage" in qualifier_names:
            address_space = "storage"
            access = ", read_write"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "workgroup" in qualifier_names or "shared" in qualifier_names:
            address_space = "workgroup"

        initializer = ""
        if node.initial_value is not None and address_space == "private":
            initializer = f" = {self.generate_expression(node.initial_value)}"
        prefix = f"{attributes}\n" if attributes else ""
        variable_name = self.module_identifier_name(node.name)
        declaration = (
            f"{prefix}var<{address_space}{access}> {variable_name}: "
            f"{self.type_name_string(node.var_type, allow_storage_resources=True)}"
            f"{initializer};"
        )
        resource_declarations = self.generate_resource_member_global_variables(
            node.name, node.var_type
        )
        if resource_declarations:
            declaration += "\n" + "\n".join(resource_declarations)
        return declaration

    def generate_stage_resource_parameter(self, node):
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.param_type
        )
        if unsupported_storage_type is not None:
            raise ValueError(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        storage_buffer_access = self.structured_buffer_access(node.param_type)
        if storage_buffer_access:
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
            return (
                f"{attributes}\nvar<storage, {storage_buffer_access}> "
                f"{self.module_identifier_name(node.name)}: "
                f"{self.type_name_string(node.param_type, allow_storage_resources=True)};"
            )
        uniform_type = self.stage_uniform_parameter_type(node)
        if uniform_type is not None:
            self.validate_uniform_binding_type(uniform_type)
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
            return (
                f"{attributes}\nvar<uniform> {self.module_identifier_name(node.name)}: "
                f"{self.type_name_string(uniform_type)};"
            )
        raise ValueError(
            "WGSL target cannot lower stage resource parameter "
            f"{node.name} of type {node.param_type}"
        )

    def generate_sampled_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for sampled texture "
                f"resource {node.name}"
            )
        texture_attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        sampler_attributes = self.sampler_binding_attributes_for_texture(
            texture_attributes,
            node=node,
        )
        texture_name = self.module_identifier_name(node.name)
        sampler_name = self.texture_sampler_name(texture_name)
        return (
            f"{texture_attributes}\nvar {texture_name}: {texture_type};\n"
            f"{sampler_attributes}\nvar {sampler_name}: "
            f"{self.companion_sampler_type(node.var_type)};"
        )

    def generate_sampler_global_variable(self, node):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for sampler resource "
                f"{node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return (
            f"{attributes}\nvar {self.module_identifier_name(node.name)}: "
            f"{self.sampler_type(node.var_type)};"
        )

    def generate_storage_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for storage texture "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        self._module_storage_texture_access_modes[node.name] = (
            self.storage_texture_access(node)
        )
        return (
            f"{attributes}\nvar {self.module_identifier_name(node.name)}: "
            f"{texture_type};"
        )

    def generate_resource_member_global_variables(self, root_name, root_type):
        declarations = []
        for info in self.resource_paths_for_type(root_type):
            binding_name = self.resource_member_binding_name(root_name, info["path"])
            info = {**info, "binding_name": binding_name, "root_name": root_name}
            self._module_resource_bindings[(root_name, tuple(info["path"]))] = info
            declarations.append(self.generate_resource_member_binding(info))
        return declarations

    def generate_resource_member_binding(self, info):
        member = info["member"]
        binding_name = info["binding_name"]
        sampled_texture_type = self.sampled_texture_type(info["member_type"])
        if sampled_texture_type is not None:
            texture_attributes = (
                self.explicit_binding_attributes(member)
                or self.next_binding_attributes()
            )
            sampler_attributes = self.sampler_binding_attributes_for_texture(
                texture_attributes,
                node=member,
            )
            sampler_name = self.texture_sampler_name(binding_name)
            return (
                f"{texture_attributes}\nvar {binding_name}: {sampled_texture_type};\n"
                f"{sampler_attributes}\nvar {sampler_name}: "
                f"{self.companion_sampler_type(info['member_type'])};"
            )
        if self.is_sampler_type(info["member_type"]):
            attributes = (
                self.explicit_binding_attributes(member)
                or self.next_binding_attributes()
            )
            return (
                f"{attributes}\nvar {binding_name}: "
                f"{self.sampler_type(info['member_type'])};"
            )
        raise ValueError(
            "WGSL target does not support resource member "
            f"{info['owner_struct']}.{member.name} of type {info['resource_type_name']}; "
            "declare textures, samplers, and storage resources as module-scope "
            "bindings instead of user-struct fields"
        )

    def generate_buffer_pointer_global_variable(self, node):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for buffer pointer "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return (
            f"{attributes}\nvar<storage, read_write> "
            f"{self.module_identifier_name(node.name)}: "
            f"{self.buffer_pointer_storage_type(node.var_type)};"
        )

    def generate_glsl_buffer_block_global_variable(self, node):
        self.validate_glsl_buffer_block_variable(node)
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for GLSL buffer block "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        access = self.glsl_buffer_block_access(node)
        type_name = self.glsl_buffer_block_struct_name(node.var_type)
        return (
            f"{attributes}\nvar<storage, {access}> "
            f"{self.module_identifier_name(node.name)}: "
            f"{self.type_identifier_name(type_name)};"
        )

    def generate_statement(self, stmt, indent=0):
        pad = "    " * indent
        if isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)
        if isinstance(stmt, VariableNode):
            mutable_keyword = "var" if stmt.is_mutable else "let"
            var_type = self.local_variable_declaration_type(stmt)
            initializer = ""
            if stmt.initial_value is not None:
                initializer = (
                    " = "
                    + self.generate_expression_for_target(
                        stmt.initial_value, var_type
                    )
                )
            variable_name = self.declare_local_identifier(stmt.name)
            line = (
                f"{pad}{mutable_keyword} {variable_name}: "
                f"{self.type_name_string(var_type)}{initializer};"
            )
            self.register_local_identifier(stmt.name)
            self.register_value_type(stmt.name, var_type)
            self.register_resource_aliases(
                stmt.name, var_type, initializer=stmt.initial_value
            )
            return line
        if isinstance(stmt, ExpressionStatementNode):
            return f"{pad}{self.generate_expression(stmt.expression)};"
        if isinstance(stmt, AssignmentNode):
            return f"{pad}{self.generate_assignment(stmt)};"
        if isinstance(stmt, ReturnNode):
            if stmt.value is None:
                return f"{pad}return;"
            value = self.generate_expression_for_target(
                stmt.value, self._current_function_return_type
            )
            return f"{pad}return {value};"
        if isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        if isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        if isinstance(stmt, WhileNode):
            return f"{pad}while ({self.generate_expression(stmt.condition)}) {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, LoopNode):
            return f"{pad}loop {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, BreakNode):
            return f"{pad}break;"
        if isinstance(stmt, ContinueNode):
            return f"{pad}continue;"
        if isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        if isinstance(stmt, DoWhileNode):
            raise ValueError("WGSL target does not support do-while statements")
        if isinstance(stmt, ForInNode):
            raise ValueError("WGSL target does not support for-in statements")
        if isinstance(stmt, MatchNode):
            raise ValueError("WGSL target does not support match statements")
        raise ValueError(
            f"WGSL target does not support statement {type(stmt).__name__}"
        )

    def generate_block(self, block, indent=0):
        if block is None:
            return "{}"
        statements = getattr(block, "statements", [])
        if not statements:
            return "{}"
        pad = "    " * indent
        lines = [f"{pad}{{"]
        self.push_identifier_scope()
        try:
            for stmt in statements:
                lines.append(self.generate_statement(stmt, indent + 1))
        finally:
            self.pop_identifier_scope()
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_if(self, node, indent):
        pad = "    " * indent
        code = (
            f"{pad}if ({self.generate_expression(node.condition)}) "
            f"{self.generate_block(node.then_branch, indent)}"
        )
        if node.else_branch is not None:
            code += f" else {self.generate_block(node.else_branch, indent)}"
        return code

    def generate_for(self, node, indent):
        pad = "    " * indent
        self.push_identifier_scope()
        try:
            init = self.generate_for_initializer(node.init)
            condition = (
                self.generate_expression(node.condition) if node.condition else ""
            )
            update = self.generate_expression(node.update) if node.update else ""
            return (
                f"{pad}for ({init}; {condition}; {update}) "
                f"{self.generate_block(node.body, indent)}"
            )
        finally:
            self.pop_identifier_scope()

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, VariableNode):
            var_type = self.local_variable_declaration_type(init)
            initializer = ""
            if init.initial_value is not None:
                initializer = (
                    " = "
                    + self.generate_expression_for_target(
                        init.initial_value, var_type
                    )
                )
            variable_name = self.declare_local_identifier(init.name)
            line = (
                f"var {variable_name}: {self.type_name_string(var_type)}"
                f"{initializer}"
            )
            self.register_local_identifier(init.name)
            self.register_value_type(init.name, var_type)
            self.register_resource_aliases(
                init.name, var_type, initializer=init.initial_value
            )
            return line
        if isinstance(init, AssignmentNode):
            return self.generate_assignment(init)
        if isinstance(init, ExpressionStatementNode):
            return self.generate_expression(init.expression)
        return self.generate_expression(init)

    def generate_switch(self, node, indent):
        pad = "    " * indent
        lines = [f"{pad}switch ({self.generate_expression(node.expression)}) {{"]
        for case in node.cases:
            label = (
                "default"
                if case.value is None
                else f"case {self.generate_expression(case.value)}"
            )
            lines.append(f"{pad}    {label}: {{")
            for stmt in case.statements:
                lines.append(self.generate_statement(stmt, indent + 2))
            lines.append(f"{pad}    }}")
        if node.default_case is not None:
            lines.append(
                f"{pad}    default: {self.generate_block(node.default_case, indent + 1)}"
            )
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_assignment(self, node):
        self.validate_storage_assignment_target(node.target)
        target_type = self.expression_type(node.target)
        return (
            f"{self.generate_expression(node.target)} {node.operator} "
            f"{self.generate_expression_for_target(node.value, target_type)}"
        )

    def generate_expression_for_target(self, expr, target_type):
        previous_expected_type = self._current_expression_expected_type
        self._current_expression_expected_type = target_type
        try:
            if isinstance(expr, FunctionCallNode):
                rendered = self.generate_function_call(expr, expected_type=target_type)
            else:
                rendered = self.generate_expression(expr)
            source_type = self.expression_type(expr)
            if isinstance(expr, FunctionCallNode):
                resolved_function = self.resolve_function_overload(
                    self.expression_name(expr.function),
                    expr.arguments,
                    expected_type=target_type,
                )
                if resolved_function is not None:
                    source_type = getattr(resolved_function, "return_type", None)
        finally:
            self._current_expression_expected_type = previous_expected_type
        narrowed = self.vector_narrowing_expression(rendered, target_type, source_type)
        if narrowed is not None:
            return narrowed
        target_scalar = self.integer_scalar_type(target_type)
        source_scalar = self.integer_scalar_type(source_type)
        if target_scalar == "i32" and source_scalar == "u32":
            return f"i32({rendered})"
        if target_scalar == "u32" and source_scalar == "i32":
            return f"u32({rendered})"
        return rendered

    def local_variable_declaration_type(self, node):
        declared_type = getattr(node, "var_type", None)
        if not self.is_inferred_local_declaration_type(declared_type):
            return declared_type
        initializer = getattr(node, "initial_value", None)
        inferred_type = (
            self.expression_type(initializer) if initializer is not None else None
        )
        if inferred_type is not None:
            return inferred_type
        raise ValueError(
            "WGSL target cannot infer a concrete type for local variable "
            f"{getattr(node, 'name', '<unnamed>')} declared as void"
        )

    def is_inferred_local_declaration_type(self, vtype):
        return self.is_void_local_declaration_type(vtype) or self.is_auto_type(vtype)

    def is_void_local_declaration_type(self, vtype):
        if vtype is None:
            return True
        try:
            return self.type_name_string(vtype) == "void"
        except ValueError:
            return False

    def is_auto_type(self, vtype):
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name).lower() == "auto"
        if isinstance(vtype, str):
            return vtype.strip().lower() == "auto"
        return False

    def generate_vector_narrowing_conversion(self, target_type, expr):
        return self.vector_narrowing_expression(
            self.generate_expression(expr), target_type, self.expression_type(expr)
        )

    def vector_narrowing_expression(self, rendered, target_type, source_type):
        target_shape = self.vector_shape(target_type)
        source_shape = self.vector_shape(source_type)
        if target_shape is None or source_shape is None:
            return None
        target_element, target_size = target_shape
        source_element, source_size = source_shape
        if source_size <= target_size:
            return None
        components = "xyzw"[:target_size]
        narrowed = f"{rendered}.{components}"
        if target_element == source_element:
            return narrowed
        return f"{self.type_name_string(target_type)}({narrowed})"

    def generate_expression(self, expr):
        if expr is None:
            return ""
        if isinstance(expr, LiteralNode):
            return self.generate_literal(expr)
        if isinstance(expr, IdentifierNode):
            if self.is_builtin_option_none_expression(expr):
                none_value = self.generate_builtin_option_none_value()
                if none_value is not None:
                    return none_value
            if expr.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES:
                return self.generate_workgroup_size_literal()
            mapped_builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(expr.name)
            if mapped_builtin:
                return mapped_builtin
            if not self.is_local_identifier(expr.name):
                cbuffer_access = self._cbuffer_member_accesses.get(expr.name)
                if cbuffer_access:
                    return cbuffer_access
            return self.identifier_name(expr.name)
        if isinstance(expr, BinaryOpNode):
            option_none_comparison = self.generate_builtin_option_none_comparison(
                expr.left, expr.operator, expr.right
            )
            if option_none_comparison is not None:
                return option_none_comparison
            return (
                f"({self.generate_expression(expr.left)} {expr.operator} "
                f"{self.generate_expression(expr.right)})"
            )
        if isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            if expr.is_postfix:
                if expr.operator == "++":
                    return f"{operand} += 1"
                if expr.operator == "--":
                    return f"{operand} -= 1"
                return f"{operand}{expr.operator}"
            if expr.operator == "++":
                return f"{operand} += 1"
            if expr.operator == "--":
                return f"{operand} -= 1"
            return f"{expr.operator}{operand}"
        if isinstance(expr, TernaryOpNode):
            return (
                f"select({self.generate_expression(expr.false_expr)}, "
                f"{self.generate_expression(expr.true_expr)}, "
                f"{self.generate_expression(expr.condition)})"
            )
        if isinstance(expr, FunctionCallNode):
            return self.generate_function_call(expr)
        if isinstance(expr, ConstructorNode):
            return self.generate_constructor(expr)
        if isinstance(expr, MemberAccessNode):
            resource_binding = self.resource_member_binding_for_access(expr)
            if resource_binding is not None:
                return resource_binding["binding_name"]
            member_name = self.member_access_identifier_name(expr)
            if (
                isinstance(expr.object_expr, IdentifierNode)
                and self.is_pointer_identifier(expr.object_expr.name)
            ):
                pointer_name = self.identifier_name(expr.object_expr.name)
                return f"(*{pointer_name}).{member_name}"
            return f"{self.generate_expression(expr.object_expr)}." f"{member_name}"
        if isinstance(expr, SwizzleNode):
            return f"{self.generate_expression(expr.vector_expr)}." f"{expr.components}"
        if isinstance(expr, ArrayAccessNode):
            if isinstance(
                expr.array_expr, IdentifierNode
            ) and self.is_pointer_identifier(expr.array_expr.name):
                pointer_name = self.identifier_name(expr.array_expr.name)
                return (
                    f"(*{pointer_name})"
                    f"[{self.generate_expression(expr.index_expr)}]"
                )
            access = (
                f"{self.generate_expression(expr.array_expr)}"
                f"[{self.generate_expression(expr.index_expr)}]"
            )
            if self.is_uniform_scalar_array_access(expr.array_expr):
                return f"{access}.value"
            return access
        if isinstance(expr, ArrayLiteralNode):
            return (
                "array("
                + ", ".join(
                    self.generate_expression(element) for element in expr.elements
                )
                + ")"
            )
        if isinstance(expr, CastNode):
            narrowed = self.generate_vector_narrowing_conversion(
                expr.target_type, expr.expression
            )
            if narrowed is not None:
                return narrowed
            return (
                f"{self.type_name_string(expr.target_type)}"
                f"({self.generate_expression(expr.expression)})"
            )
        if isinstance(expr, RangeNode):
            raise ValueError("WGSL target does not support range expressions")
        if isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        raise ValueError(
            f"WGSL target does not support expression {type(expr).__name__}"
        )

    def generate_literal(self, node):
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        text = str(value)
        literal_type = self.type_name_string(getattr(node, "literal_type", ""))
        if literal_type == "f32" and re.fullmatch(r"[-+]?\d+", text):
            return f"{text}.0"
        return text

    def generate_workgroup_size_literal(self):
        values = self._current_workgroup_size or ("1", "1", "1")
        return (
            "vec3<u32>(" + ", ".join(self.u32_literal(value) for value in values) + ")"
        )

    def u32_literal(self, value):
        text = str(value)
        if re.fullmatch(r"\d+", text):
            return f"{text}u"
        return text

    def generate_function_call(self, node, expected_type=None):
        function_name = self.expression_name(node.function)
        normalized_name = self.semantic_key(function_name)
        option_call = self.generate_builtin_option_call(node, function_name)
        if option_call is not None:
            return option_call
        if normalized_name in self.STRUCTURED_BUFFER_FREE_HELPERS:
            return self.generate_structured_buffer_free_helper_call(
                node, normalized_name
            )
        if normalized_name in self.UNSUPPORTED_STORAGE_BUFFER_FREE_HELPERS:
            raise ValueError(
                "WGSL target does not support storage buffer helper "
                f"{function_name} yet"
            )
        if isinstance(node.function, MemberAccessNode):
            member_name = self.semantic_key(node.function.member)
            if member_name in self.STRUCTURED_BUFFER_MEMBER_HELPERS:
                return self.generate_structured_buffer_member_helper_call(
                    node, member_name
                )
            if member_name in self.UNSUPPORTED_STRUCTURED_BUFFER_MEMBER_HELPERS:
                receiver_type = self.expression_type(node.function.object_expr)
                if self.storage_buffer_like_type_name(receiver_type) is not None:
                    raise ValueError(
                        "WGSL target does not support storage buffer member helper "
                        f"{node.function.member} yet"
                    )
        if normalized_name in self.TEXTURE_FUNCTION_NAMES:
            return self.generate_texture_function_call(node, function_name)
        if normalized_name == "imageload":
            return self.generate_image_load_call(node, function_name)
        if normalized_name == "imagestore":
            return self.generate_image_store_call(node, function_name)
        if normalized_name in self.BARRIER_FUNCTION_NAMES:
            return self.generate_barrier_call(node, function_name)
        derivative_name = self.DERIVATIVE_FUNCTION_NAME_MAP.get(normalized_name)
        if derivative_name is not None:
            return self.generate_derivative_call(node, derivative_name, function_name)
        if function_name == "mod":
            return self.generate_mod_call(node)

        if self.is_type_constructor_name(function_name):
            if len(node.arguments) == 1:
                narrowed = self.generate_vector_narrowing_conversion(
                    function_name, node.arguments[0]
                )
                if narrowed is not None:
                    return narrowed
            args = self.generate_call_arguments(function_name, node.arguments)
            return f"{self.type_name_string(function_name)}({args})"
        resolved_function = None
        if isinstance(node.function, IdentifierNode):
            resolved_function = self.resolve_function_overload(
                function_name, node.arguments, expected_type=expected_type
            )
        args = self.generate_call_arguments(
            function_name, node.arguments, function=resolved_function
        )
        mapped_name = self.FUNCTION_NAME_MAP.get(function_name, function_name)
        if mapped_name == function_name and isinstance(node.function, IdentifierNode):
            if resolved_function is not None:
                mapped_name = self.function_declaration_identifier_name(
                    resolved_function
                )
            else:
                mapped_name = self.function_identifier_name(function_name)
        elif isinstance(node.function, MemberAccessNode):
            mapped_name = self.generate_expression(node.function)
        return f"{mapped_name}({args})"

    def generate_derivative_call(self, node, mapped_name, function_name):
        if len(node.arguments) != 1:
            raise ValueError(
                "WGSL target supports derivative intrinsic "
                f"{function_name}() with exactly 1 argument; got "
                f"{len(node.arguments)}"
            )
        return f"{mapped_name}({self.generate_expression(node.arguments[0])})"

    def generate_structured_buffer_free_helper_call(self, node, helper_name):
        if helper_name == "buffer_load":
            if len(node.arguments) != 2:
                raise ValueError(
                    "WGSL target supports buffer_load() with exactly 2 arguments; "
                    f"got {len(node.arguments)}"
                )
            resource, index = node.arguments
            self.require_structured_buffer_resource(resource, "buffer_load")
            return self.structured_buffer_index_expression(resource, index)

        if len(node.arguments) != 3:
            raise ValueError(
                "WGSL target supports buffer_store() with exactly 3 arguments; "
                f"got {len(node.arguments)}"
            )
        resource, index, value = node.arguments
        self.require_writable_structured_buffer_resource(resource, "buffer_store")
        return (
            f"{self.structured_buffer_index_expression(resource, index)} = "
            f"{self.generate_expression(value)}"
        )

    def generate_structured_buffer_member_helper_call(self, node, helper_name):
        receiver = node.function.object_expr
        if helper_name == "load":
            if len(node.arguments) != 1:
                raise ValueError(
                    "WGSL target supports StructuredBuffer.Load() with exactly "
                    f"1 argument; got {len(node.arguments)}"
                )
            self.require_structured_buffer_resource(receiver, "Load")
            return self.structured_buffer_index_expression(receiver, node.arguments[0])

        if len(node.arguments) != 2:
            raise ValueError(
                "WGSL target supports RWStructuredBuffer.Store() with exactly "
                f"2 arguments; got {len(node.arguments)}"
            )
        self.require_writable_structured_buffer_resource(receiver, "Store")
        return (
            f"{self.structured_buffer_index_expression(receiver, node.arguments[0])} = "
            f"{self.generate_expression(node.arguments[1])}"
        )

    def structured_buffer_index_expression(self, resource, index):
        if isinstance(resource, IdentifierNode) and self.is_pointer_identifier(
            resource.name
        ):
            resource_name = self.identifier_name(resource.name)
            return f"(*{resource_name})[{self.generate_expression(index)}]"
        return (
            f"{self.generate_expression(resource)}[{self.generate_expression(index)}]"
        )

    def require_structured_buffer_resource(self, resource, helper_name):
        resource_type = self.expression_type(resource)
        if self.structured_buffer_element_type(resource_type) is None and not (
            isinstance(resource, IdentifierNode)
            and self.is_pointer_identifier(resource.name)
        ):
            raise ValueError(
                "WGSL target requires "
                f"{helper_name}() to use a StructuredBuffer or RWStructuredBuffer "
                "resource"
            )

    def require_writable_structured_buffer_resource(self, resource, helper_name):
        self.require_structured_buffer_resource(resource, helper_name)
        resource_type = self.expression_type(resource)
        access = self.structured_buffer_access(resource_type)
        if access == "read":
            raise ValueError(
                "WGSL target cannot store through read-only StructuredBuffer "
                f"resource in {helper_name}()"
            )

    def generate_mod_call(self, node):
        if len(node.arguments) != 2:
            raise ValueError(
                "WGSL target supports mod() calls with exactly 2 arguments; got "
                f"{len(node.arguments)}"
            )
        left = self.generate_expression(node.arguments[0])
        right = self.generate_expression(node.arguments[1])
        return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"

    def generate_call_arguments(self, function_name, arguments, function=None):
        metadata_key = (
            self.function_signature_key(function) if function is not None else function_name
        )
        texture_parameter_indices = self._function_texture_parameters.get(
            metadata_key, self._function_texture_parameters.get(function_name, ())
        )
        pointer_parameter_indices = self._function_pointer_parameters.get(
            metadata_key, self._function_pointer_parameters.get(function_name, ())
        )
        resource_member_parameters = self._function_resource_member_parameters.get(
            metadata_key,
            self._function_resource_member_parameters.get(function_name, {}),
        )
        if (
            not texture_parameter_indices
            and not pointer_parameter_indices
            and not resource_member_parameters
        ):
            parameter_types = [
                getattr(parameter, "param_type", None)
                for parameter in getattr(function, "parameters", []) or []
            ]
            return ", ".join(
                self.generate_expression_for_target(arg, parameter_types[index])
                if index < len(parameter_types)
                else self.generate_expression(arg)
                for index, arg in enumerate(arguments)
            )

        texture_parameter_indices = set(texture_parameter_indices)
        pointer_parameter_indices = set(pointer_parameter_indices)
        parameter_types = [
            getattr(parameter, "param_type", None)
            for parameter in getattr(function, "parameters", []) or []
        ]
        rendered = []
        for index, arg in enumerate(arguments):
            if index in pointer_parameter_indices:
                rendered.append(self.pointer_argument_expression(arg))
            else:
                rendered.append(
                    self.generate_expression_for_target(arg, parameter_types[index])
                    if index < len(parameter_types)
                    else self.generate_expression(arg)
                )
            if index in texture_parameter_indices:
                rendered.append(self.texture_sampler_expression(arg))
            if index in resource_member_parameters:
                rendered.extend(
                    self.resource_member_call_arguments(
                        arg, resource_member_parameters[index]
                    )
                )
        return ", ".join(rendered)

    def generate_texture_function_call(self, node, function_name):
        normalized_name = self.semantic_key(function_name)
        args = list(node.arguments)
        if normalized_name == "texture":
            return self.generate_texture_sample_call(function_name, args)
        if normalized_name == "texturelod":
            return self.generate_texture_sample_level_call(function_name, args)
        if normalized_name == "texturelodoffset":
            return self.generate_texture_sample_level_offset_call(function_name, args)
        if normalized_name == "texturegrad":
            return self.generate_texture_sample_grad_call(function_name, args)
        if normalized_name == "texturegradoffset":
            return self.generate_texture_sample_grad_offset_call(function_name, args)
        if normalized_name == "textureoffset":
            return self.generate_texture_sample_offset_call(function_name, args)
        if normalized_name == "texturecompare":
            return self.generate_texture_sample_compare_call(function_name, args)
        if normalized_name == "texturecompareoffset":
            return self.generate_texture_sample_compare_offset_call(function_name, args)
        if normalized_name == "texturecomparelod":
            return self.generate_texture_sample_compare_level_call(function_name, args)
        if normalized_name == "texturecomparelodoffset":
            return self.generate_texture_sample_compare_level_offset_call(
                function_name, args
            )
        if normalized_name == "texturesize":
            return self.generate_texture_dimensions_call(args)
        if normalized_name == "texelfetch":
            return self.generate_texel_fetch_call(function_name, args)
        raise ValueError(
            "WGSL target does not support CrossGL texture function "
            f"{function_name} yet"
        )

    def generate_texture_call_args(self, args, *, function_name, implicit, explicit):
        if len(args) == implicit:
            texture = args[0]
            return [
                self.generate_expression(texture),
                self.texture_sampler_expression(texture),
                *(self.generate_expression(arg) for arg in args[1:]),
            ]
        if len(args) == explicit:
            return [self.generate_expression(arg) for arg in args]
        raise ValueError(
            f"WGSL target supports {function_name}() calls with {implicit} or "
            f"{explicit} argument(s); got {len(args)}"
        )

    def generate_texture_builtin_call(
        self, builtin_name, function_name, args, *, implicit, explicit
    ):
        return (
            f"{builtin_name}("
            + ", ".join(
                self.generate_texture_call_args(
                    args,
                    function_name=function_name,
                    implicit=implicit,
                    explicit=explicit,
                )
            )
            + ")"
        )

    def generate_texture_sample_call(self, function_name, args):
        if len(args) == 2:
            texture, coords = args
            return (
                f"textureSample({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.generate_expression(coords)})"
            )
        if len(args) == 3:
            texture, sampler, coords = args
            return (
                f"textureSample({self.generate_expression(texture)}, "
                f"{self.generate_expression(sampler)}, "
                f"{self.generate_expression(coords)})"
            )
        raise ValueError(
            "WGSL target supports texture() calls with texture/coords or "
            "texture/sampler/coords arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_sample_level_call(self, function_name, args):
        if len(args) == 3:
            texture, coords, level = args
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.generate_expression(coords)}, "
                f"{self.generate_expression(level)})"
            )
        if len(args) == 4:
            texture, sampler, coords, level = args
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.generate_expression(sampler)}, "
                f"{self.generate_expression(coords)}, "
                f"{self.generate_expression(level)})"
            )
        raise ValueError(
            "WGSL target supports textureLod() calls with texture/coords/lod or "
            "texture/sampler/coords/lod arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_sample_level_offset_call(self, function_name, args):
        return self.generate_texture_builtin_call(
            "textureSampleLevel", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_grad_call(self, function_name, args):
        return self.generate_texture_builtin_call(
            "textureSampleGrad", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_grad_offset_call(self, function_name, args):
        return self.generate_texture_builtin_call(
            "textureSampleGrad", function_name, args, implicit=5, explicit=6
        )

    def generate_texture_sample_offset_call(self, function_name, args):
        if len(args) == 3:
            return self.generate_texture_builtin_call(
                "textureSample", function_name, args, implicit=3, explicit=4
            )
        if len(args) == 4:
            if self.is_sampler_type(self.expression_type(args[1])):
                return self.generate_texture_builtin_call(
                    "textureSample", function_name, args, implicit=3, explicit=4
                )
            texture, coords, offset, bias = (
                self.generate_expression(arg) for arg in args
            )
            return (
                "textureSampleBias("
                f"{texture}, {self.texture_sampler_expression(args[0])}, "
                f"{coords}, {bias}, {offset})"
            )
        if len(args) == 5:
            call_args = self.generate_texture_call_args(
                args, function_name=function_name, implicit=4, explicit=5
            )
            texture, sampler, coords, offset, bias = call_args
            return (
                "textureSampleBias("
                f"{texture}, {sampler}, {coords}, {bias}, {offset})"
            )
        raise ValueError(
            "WGSL target supports textureOffset() calls with texture/coords/offset, "
            "texture/sampler/coords/offset, or texture/sampler/coords/offset/bias "
            f"arguments; got {len(args)} argument(s)"
        )

    def generate_texture_sample_compare_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=3)
        return self.generate_texture_builtin_call(
            "textureSampleCompare", function_name, args, implicit=3, explicit=4
        )

    def generate_texture_sample_compare_offset_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=4)
        return self.generate_texture_builtin_call(
            "textureSampleCompare", function_name, args, implicit=4, explicit=5
        )

    def generate_texture_sample_compare_level_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=4)
        self.require_zero_compare_level_operand(args, function_name, implicit=4)
        call_args = self.generate_texture_call_args(
            args, function_name=function_name, implicit=4, explicit=5
        )
        call_args.pop()
        return "textureSampleCompareLevel(" + ", ".join(call_args) + ")"

    def generate_texture_sample_compare_level_offset_call(self, function_name, args):
        self.require_depth_texture_operand(args[0], function_name)
        self.require_comparison_sampler_operand(args, function_name, implicit=5)
        self.require_zero_compare_level_operand(args, function_name, implicit=5)
        call_args = self.generate_texture_call_args(
            args, function_name=function_name, implicit=5, explicit=6
        )
        offset = call_args.pop()
        call_args.pop()
        call_args.append(offset)
        return "textureSampleCompareLevel(" + ", ".join(call_args) + ")"

    def generate_texture_dimensions_call(self, args):
        if len(args) not in {1, 2}:
            raise ValueError(
                "WGSL target supports textureSize() calls with texture or "
                f"texture/lod arguments; got {len(args)} argument(s)"
            )
        return (
            "textureDimensions("
            + ", ".join(self.generate_expression(arg) for arg in args)
            + ")"
        )

    def generate_texel_fetch_call(self, function_name, args):
        if len(args) != 3:
            raise ValueError(
                "WGSL target supports texelFetch() calls with exactly 3 "
                f"arguments; got {len(args)} for {function_name}"
            )
        texture, coords, level = args
        return (
            f"textureLoad({self.generate_expression(texture)}, "
            f"{self.generate_expression(coords)}, "
            f"{self.generate_expression(level)})"
        )

    def generate_image_load_call(self, node, function_name):
        if len(node.arguments) != 2:
            raise ValueError(
                "WGSL target supports imageLoad() calls with exactly 2 "
                f"arguments; got {len(node.arguments)} for {function_name}"
            )
        image, coords = node.arguments
        self.require_readable_storage_texture_resource(image, function_name)
        return (
            f"textureLoad({self.generate_expression(image)}, "
            f"{self.generate_expression(coords)})"
        )

    def generate_image_store_call(self, node, function_name):
        if len(node.arguments) != 3:
            raise ValueError(
                "WGSL target supports imageStore() calls with exactly 3 "
                f"arguments; got {len(node.arguments)} for {function_name}"
            )
        image, coords, value = node.arguments
        self.require_writable_storage_texture_resource(image, function_name)
        return (
            f"textureStore({self.generate_expression(image)}, "
            f"{self.generate_expression(coords)}, "
            f"{self.generate_expression(value)})"
        )

    def texture_sampler_expression(self, texture_expr):
        resource_binding = self.resource_member_binding_for_access(texture_expr)
        if resource_binding is not None:
            return self.texture_sampler_name(resource_binding["binding_name"])
        if isinstance(texture_expr, IdentifierNode):
            return self.texture_sampler_name(self.identifier_name(texture_expr.name))
        raise ValueError(
            "WGSL target cannot infer a companion sampler for texture expression "
            f"{self.generate_expression(texture_expr)}; pass an explicit sampler"
        )

    def require_depth_texture_operand(self, texture_expr, function_name):
        texture_type = self.expression_type(texture_expr)
        if not self.is_depth_texture_type(texture_type):
            raise ValueError(
                f"WGSL target requires {function_name}() to use a shadow/depth "
                "texture resource"
            )

    def require_comparison_sampler_operand(self, args, function_name, *, implicit):
        if len(args) == implicit:
            return
        if len(args) != implicit + 1:
            return
        sampler_type = self.expression_type(args[1])
        if not self.is_comparison_sampler_type(sampler_type):
            raise ValueError(
                f"WGSL target requires {function_name}() explicit sampler operand "
                "to use samplerComparisonState"
            )

    def require_zero_compare_level_operand(self, args, function_name, *, implicit):
        if self.semantic_key(function_name).endswith("offset"):
            level_index = len(args) - 2
        else:
            level_index = len(args) - 1
        if len(args) <= level_index:
            return
        level_expr = args[level_index]
        if isinstance(level_expr, LiteralNode) and level_expr.value in {0, 0.0}:
            return
        raise ValueError(
            f"WGSL target only lowers {function_name}() when the explicit LOD "
            "operand is literal 0 because textureSampleCompareLevel samples "
            "mip level 0"
        )

    def pointer_argument_expression(self, pointer_expr):
        if isinstance(pointer_expr, IdentifierNode) and self.is_pointer_identifier(
            pointer_expr.name
        ):
            return self.identifier_name(pointer_expr.name)
        return f"&{self.generate_expression(pointer_expr)}"

    def generate_barrier_call(self, node, function_name):
        if node.arguments:
            raise ValueError(
                f"WGSL target does not support arguments for barrier function {function_name}"
            )
        if self._current_stage_name != "compute":
            raise ValueError(
                "WGSL target only supports barrier() inside compute stages"
            )
        return "workgroupBarrier()"

    def generate_constructor(self, node):
        if len(node.arguments) == 1:
            narrowed = self.generate_vector_narrowing_conversion(
                node.constructor_type, node.arguments[0]
            )
            if narrowed is not None:
                return narrowed
        args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
        return f"{self.type_name_string(node.constructor_type)}({args})"

    def type_name_string(self, vtype, allow_storage_resources=False):
        if vtype is None:
            return "void"
        if isinstance(vtype, PrimitiveType):
            return self.PRIMITIVE_TYPE_MAP.get(vtype.name.lower(), vtype.name)
        if isinstance(vtype, VectorType):
            return f"vec{vtype.size}<{self.type_name_string(vtype.element_type)}>"
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            self.validate_not_resource_array(vtype)
            element = self.type_name_string(vtype.element_type)
            if vtype.size is None:
                return f"array<{element}>"
            size = (
                self.generate_expression(vtype.size)
                if hasattr(vtype.size, "__class__")
                and not isinstance(vtype.size, (str, int))
                else str(vtype.size)
            )
            return f"array<{element}, {size}>"
        if isinstance(vtype, NamedType):
            option_type = self.builtin_option_specialized_type_name(vtype)
            if option_type is not None:
                return option_type
            storage_element = self.structured_buffer_element_type(vtype)
            if storage_element is not None:
                if not allow_storage_resources:
                    raise ValueError(
                        "WGSL target only supports StructuredBuffer resources as "
                        "module-scope storage bindings"
                    )
                return f"array<{self.type_name_string(storage_element)}>"
            generic_vector = self.generic_vector_type_name(vtype)
            if generic_vector is not None:
                return generic_vector
            if vtype.generic_args:
                raise ValueError(
                    "WGSL target does not support generic named type "
                    f"{self.raw_type_name_string(vtype)} yet"
                )
            return self.map_type_name(vtype.name)
        if isinstance(vtype, GenericType):
            raise ValueError("WGSL target does not support generic types yet")
        if isinstance(vtype, PointerType):
            raise ValueError(
                "WGSL target only supports pointer types for buffer/storage "
                "resources and helper parameters"
            )
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        if isinstance(vtype, str):
            option_type = self.builtin_option_specialized_type_name(vtype)
            if option_type is not None:
                return option_type
            return self.map_type_name(vtype)
        return str(vtype)

    def uniform_buffer_member_type_name(self, vtype):
        wrapper_name = self.uniform_scalar_array_wrapper_name(vtype)
        if wrapper_name is None:
            return self.type_name_string(vtype)
        return self.array_type_name(wrapper_name, vtype)

    def array_type_name(self, element_type_name, vtype):
        if not isinstance(vtype, ArrayType):
            raise ValueError("WGSL target expected an array type")
        if vtype.size is None:
            return f"array<{element_type_name}>"
        size = (
            self.generate_expression(vtype.size)
            if hasattr(vtype.size, "__class__")
            and not isinstance(vtype.size, (str, int))
            else str(vtype.size)
        )
        return f"array<{element_type_name}, {size}>"

    def uniform_scalar_array_wrapper_name(self, vtype):
        scalar_type = self.uniform_scalar_array_element_type_name(vtype)
        if scalar_type is None:
            return None
        return self._uniform_scalar_array_wrappers.get(scalar_type)

    def uniform_scalar_array_element_type_name(self, vtype):
        if not isinstance(vtype, ArrayType) or vtype.size is None:
            return None
        scalar_type = self.scalar_type_name(vtype.element_type)
        if scalar_type in {"f32", "i32", "u32"}:
            return scalar_type
        return None

    def generate_uniform_scalar_array_wrappers(self):
        sections = []
        for scalar_type, wrapper_name in sorted(
            self._uniform_scalar_array_wrappers.items()
        ):
            sections.append(
                f"struct {wrapper_name} {{\n"
                f"    @size(16) value: {scalar_type},\n"
                "};"
            )
        return "\n\n".join(sections)

    def collect_builtin_option_specializations(self, root):
        specializations = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (int, float, bool)):
                return
            if isinstance(value, str):
                payload_type = self.builtin_option_payload_type(value)
                if payload_type is not None:
                    specializations[value] = {
                        "type_name": value,
                        "struct_name": generic_enum_specialization_name(value),
                        "payload_type": payload_type,
                    }
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            payload_type = self.builtin_option_payload_type(value)
            if payload_type is not None:
                type_text = self.raw_type_name_string(value)
                specializations[type_text] = {
                    "type_name": type_text,
                    "struct_name": generic_enum_specialization_name(type_text),
                    "payload_type": payload_type,
                }

            if not hasattr(value, "__dict__"):
                return
            for child in vars(value).values():
                visit(child)

        visit(root)
        return dict(sorted(specializations.items()))

    def builtin_option_specialization_member_types(self, specializations):
        member_types = {}
        for type_text, specialization in (specializations or {}).items():
            fields = {
                "variant": PrimitiveType("int"),
                "Some_0": specialization["payload_type"],
            }
            member_types[type_text] = dict(fields)
            member_types[specialization["struct_name"]] = dict(fields)
        return member_types

    def generate_builtin_option_declarations(self):
        if not self._builtin_option_specializations:
            return ""

        sections = ["const Option_Some: i32 = 0;\nconst Option_None: i32 = 1;"]
        for specialization in self._builtin_option_specializations.values():
            struct_name = specialization["struct_name"]
            payload_type = self.type_name_string(specialization["payload_type"])
            sections.append(
                f"struct {struct_name} {{\n"
                "    variant: i32,\n"
                f"    Some_0: {payload_type},\n"
                "};"
            )
            sections.append(
                f"fn {struct_name}_Some_make(payload0: {payload_type}) -> {struct_name} {{\n"
                f"    var result: {struct_name};\n"
                "    result.variant = Option_Some;\n"
                "    result.Some_0 = payload0;\n"
                "    return result;\n"
                "}"
            )
            sections.append(
                f"fn {struct_name}_None_make() -> {struct_name} {{\n"
                f"    var result: {struct_name};\n"
                "    result.variant = Option_None;\n"
                "    result.Some_0 = "
                f"{self.default_value_expression_for_type(specialization['payload_type'])};\n"
                "    return result;\n"
                "}"
            )
        return "\n\n".join(sections)

    def default_value_expression_for_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name == "bool":
            return "false"
        if type_name == "f32":
            return "0.0"
        if type_name in {"i32", "u32"}:
            return f"{type_name}(0)"
        return f"{type_name}()"

    def raw_type_name_string(self, vtype):
        if vtype is None:
            return None
        if isinstance(vtype, PrimitiveType):
            return vtype.name
        if isinstance(vtype, VectorType):
            return (
                f"vec{vtype.size}<"
                f"{self.raw_type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.raw_type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            element = self.raw_type_name_string(vtype.element_type)
            if vtype.size is None:
                return f"{element}[]"
            return f"{element}[{vtype.size}]"
        if isinstance(vtype, NamedType):
            if not vtype.generic_args:
                return str(vtype.name)
            generic_args = ", ".join(
                self.raw_type_name_string(arg) for arg in vtype.generic_args
            )
            return f"{vtype.name}<{generic_args}>"
        if isinstance(vtype, GenericType):
            return str(vtype.name)
        if isinstance(vtype, str):
            return vtype
        if isinstance(vtype, ReferenceType):
            return self.raw_type_name_string(vtype.referenced_type)
        if isinstance(vtype, PointerType):
            return (
                f"{self.raw_type_name_string(vtype.pointee_type)}*"
            )
        return str(vtype)

    def builtin_option_payload_type(self, vtype):
        if isinstance(vtype, NamedType):
            base_name = str(vtype.name).rsplit("::", 1)[-1]
            if base_name == "Option" and len(vtype.generic_args) == 1:
                return vtype.generic_args[0]
            return None
        if isinstance(vtype, str):
            base_name, generic_args = generic_type_parts(vtype)
            if base_name.rsplit("::", 1)[-1] == "Option" and len(generic_args) == 1:
                return generic_args[0]
        return None

    def builtin_option_specialization(self, vtype):
        if self.builtin_option_payload_type(vtype) is None:
            return None
        return self._builtin_option_specializations.get(self.raw_type_name_string(vtype))

    def builtin_option_specialized_type_name(self, vtype):
        specialization = self.builtin_option_specialization(vtype)
        if specialization is None:
            return None
        return specialization["struct_name"]

    def expected_builtin_option_specialization(self):
        return self.builtin_option_specialization(
            self._current_expression_expected_type
        ) or self.builtin_option_specialization(self._current_function_return_type)

    def is_builtin_option_none_expression(self, expr):
        return (
            isinstance(expr, IdentifierNode)
            and str(expr.name).rsplit("::", 1)[-1] == "None"
        )

    def generate_builtin_option_none_value(self):
        specialization = self.expected_builtin_option_specialization()
        if specialization is None:
            return None
        return f"{specialization['struct_name']}_None_make()"

    def generate_builtin_option_call(self, node, function_name):
        base_name = str(function_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            specialization = self.expected_builtin_option_specialization()
            if specialization is None:
                return None
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                raise ValueError(
                    f"WGSL target expects Some() to receive 1 argument, got {len(args)}"
                )
            payload = self.generate_expression_for_target(
                args[0], specialization["payload_type"]
            )
            return f"{specialization['struct_name']}_Some_make({payload})"

        if base_name in {"is_Some", "is_None", "unwrap_Some"}:
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                raise ValueError(
                    "WGSL target expects "
                    f"{function_name}() to receive 1 argument, got {len(args)}"
                )
            subject = self.generate_expression(args[0])
            if base_name == "is_Some":
                return f"({subject}.variant == Option_Some)"
            if base_name == "is_None":
                return f"({subject}.variant == Option_None)"
            return f"{subject}.Some_0"

        return None

    def builtin_option_call_type(self, node, function_name):
        base_name = str(function_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            specialization = self.expected_builtin_option_specialization()
            return None if specialization is None else specialization["type_name"]
        if base_name in {"is_Some", "is_None"}:
            return PrimitiveType("bool")
        if base_name == "unwrap_Some":
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                return None
            option_type = self.expression_type(args[0])
            return self.builtin_option_payload_type(option_type)
        return None

    def generate_builtin_option_none_comparison(self, left, operator, right):
        if operator not in {"==", "!="}:
            return None
        left_is_none = self.is_builtin_option_none_expression(left)
        right_is_none = self.is_builtin_option_none_expression(right)
        if left_is_none == right_is_none:
            return None
        subject = right if left_is_none else left
        subject_expr = self.generate_expression(subject)
        comparison = "==" if operator == "==" else "!="
        return f"({subject_expr}.variant {comparison} Option_None)"

    def map_type_name(self, type_name):
        normalized = type_name.strip()
        lower = normalized.lower()
        if lower in self.PRIMITIVE_TYPE_MAP:
            return self.PRIMITIVE_TYPE_MAP[lower]
        if self.is_resource_type_name(lower):
            raise ValueError(
                "WGSL target does not support CrossGL resource type "
                f"{normalized} yet; split texture/sampler/storage bindings are required"
            )

        wgsl_vector_match = re.fullmatch(r"vec([234])<\s*([^>]+)\s*>", normalized, re.I)
        if wgsl_vector_match:
            size = wgsl_vector_match.group(1)
            element = wgsl_vector_match.group(2)
            return f"vec{size}<{self.map_type_name(element)}>"

        float16_vector_match = self.FLOAT16_VECTOR_TYPE_RE.match(lower)
        if float16_vector_match:
            return f"vec{float16_vector_match.group(1)}<f32>"

        vector_match = self.VECTOR_TYPE_RE.match(lower)
        if vector_match:
            size = vector_match.group(1)
            element = "f32"
            if lower.startswith(("int", "ivec")):
                element = "i32"
            elif lower.startswith(("uint", "uvec")):
                element = "u32"
            elif lower.startswith(("bool", "bvec")):
                element = "bool"
            return f"vec{size}<{element}>"

        matrix_match = self.MATRIX_TYPE_RE.match(lower)
        if matrix_match:
            columns = matrix_match.group(1)
            rows = matrix_match.group(2) or columns
            return f"mat{columns}x{rows}<f32>"

        return self.type_identifier_name(normalized)

    def generic_vector_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        if str(vtype.name).lower() not in {"vec", "vector"}:
            return None
        if len(vtype.generic_args) < 2:
            return None
        size = self.literal_integer_value(vtype.generic_args[1])
        if size not in {2, 3, 4}:
            return None
        return f"vec{size}<{self.type_name_string(vtype.generic_args[0])}>"

    def literal_integer_value(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, LiteralNode):
            return self.literal_integer_value(value.value)
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                return None
        return None

    def is_resource_type_name(self, lower_type_name):
        return lower_type_name in self.RESOURCE_TYPE_NAMES

    def structured_buffer_element_type(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.STRUCTURED_BUFFER_TYPE_NAMES:
            return None
        if len(vtype.generic_args) != 1:
            raise ValueError(
                "WGSL target requires StructuredBuffer resources to declare one "
                "element type"
            )
        return vtype.generic_args[0]

    def structured_buffer_access(self, vtype):
        if self.structured_buffer_element_type(vtype) is None:
            return None
        base_name = str(vtype.name).lower()
        if base_name in self.WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES:
            return "read_write"
        return "read"

    def constant_buffer_element_type(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.CONSTANT_BUFFER_TYPE_NAMES:
            return None
        if len(vtype.generic_args) != 1:
            raise ValueError(
                "WGSL target requires ConstantBuffer resources to declare one "
                "element type"
            )
        return vtype.generic_args[0]

    def stage_uniform_parameter_type(self, parameter):
        constant_buffer_type = self.constant_buffer_element_type(parameter.param_type)
        if constant_buffer_type is not None:
            return constant_buffer_type

        if not self.has_binding_attribute(parameter):
            return None

        qualifier_names = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "constant" not in qualifier_names:
            return None

        param_type = getattr(parameter, "param_type", None)
        if isinstance(param_type, ReferenceType):
            return param_type.referenced_type
        if isinstance(param_type, PointerType):
            pointee_type = param_type.pointee_type
            if self.struct_type_name(pointee_type) in self._structs_by_name:
                return pointee_type
            return None
        return param_type

    def stage_resource_module_type(self, parameter):
        uniform_type = self.stage_uniform_parameter_type(parameter)
        if uniform_type is not None:
            return uniform_type
        return getattr(parameter, "param_type", None)

    def validate_uniform_binding_type(self, uniform_type):
        struct_name = self.struct_type_name(self.array_element_type(uniform_type))
        struct = self._structs_by_name.get(struct_name)
        if struct is None:
            return
        for member in getattr(struct, "members", []) or []:
            self.validate_cbuffer_member(struct, member)

    def unsupported_storage_buffer_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES:
            return None
        return str(vtype.name)

    def storage_buffer_like_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if (
            base_name in self.STRUCTURED_BUFFER_TYPE_NAMES
            or base_name in self.UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES
        ):
            return str(vtype.name)
        return None

    def validate_not_resource_array(self, vtype):
        resource_type = self.resource_array_element_type_name(vtype)
        if resource_type is None:
            return
        raise ValueError(
            "WGSL target does not support resource arrays of "
            f"{resource_type}; WebGPU/WGSL requires texture, sampler, image, "
            "and storage-buffer resources to be declared as individual "
            "module-scope bindings"
        )

    def resource_array_element_type_name(self, vtype):
        if not isinstance(vtype, ArrayType):
            return None
        element_type = self.array_element_type(vtype)
        resource_type = self.resource_type_name(element_type)
        if resource_type is not None and self.is_resource_type_name(resource_type):
            return self.type_display_name(element_type)
        return self.storage_buffer_like_type_name(element_type)

    def type_display_name(self, vtype):
        if isinstance(vtype, NamedType):
            return str(vtype.name)
        if isinstance(vtype, str):
            return vtype.strip()
        return str(vtype)

    def is_glsl_buffer_block_variable(self, node):
        if self.is_buffer_pointer_type(
            getattr(node, "var_type", None), getattr(node, "qualifiers", [])
        ):
            return False
        if self.glsl_buffer_block_attribute(node) is not None:
            return True
        qualifier_names = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "buffer" not in qualifier_names:
            return False
        struct_name = self.glsl_buffer_block_struct_name(
            getattr(node, "var_type", None)
        )
        return struct_name in self._structs_by_name

    def is_glsl_buffer_block_parameter(self, node):
        return self.glsl_buffer_block_attribute(node) is not None

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            if key == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        if attr is None:
            qualifier_names = {
                self.semantic_key(str(qualifier))
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if "buffer" in qualifier_names:
                return "std430"
            return None
        arguments = getattr(attr, "arguments", []) or []
        if not arguments:
            return None
        return self.semantic_key(self.generate_attribute_argument(arguments[0]))

    def glsl_buffer_block_struct_name(self, vtype):
        return self.struct_type_name(self.array_element_type(vtype))

    def glsl_buffer_block_access(self, node):
        qualifiers = {
            self.semantic_key(str(qualifier))
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            self.semantic_key(str(getattr(attr, "name", attr)))
            for attr in getattr(node, "attributes", []) or []
        }
        names = qualifiers | attributes
        if "readonly" in names:
            return "read"
        if "writeonly" in names:
            return "write"
        return "read_write"

    def validate_glsl_buffer_block_layout(self, node, resource_name):
        layout = self.glsl_buffer_block_layout(node)
        if layout not in self.SUPPORTED_GLSL_BUFFER_BLOCK_LAYOUTS:
            layout_name = layout or "unspecified"
            raise ValueError(
                "WGSL target only supports std430 GLSL buffer block layout "
                f"for {resource_name}; got {layout_name}"
            )

    def validate_glsl_buffer_block_variable(self, node):
        self.validate_glsl_buffer_block_layout(node, node.name)
        if isinstance(getattr(node, "var_type", None), ArrayType):
            raise ValueError(
                "WGSL target does not support GLSL buffer block arrays yet; "
                "declare each block as a separate storage binding"
            )
        struct_name = self.glsl_buffer_block_struct_name(node.var_type)
        if not struct_name:
            raise ValueError(
                "WGSL target requires GLSL buffer block resource "
                f"{node.name} to use a named struct type"
            )

    def validate_glsl_buffer_block_struct(self, node):
        members = list(getattr(node, "members", []) or [])
        for index, member in enumerate(members):
            member_type = getattr(member, "member_type", None)
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                continue
            if self.structured_buffer_element_type(self.array_element_type(member_type)):
                raise ValueError(
                    "WGSL target does not support storage-buffer resource member "
                    f"{node.name}.{member.name} inside GLSL buffer blocks"
                )
            if (
                self.unsized_array_type(member_type) is not None
                and index != len(members) - 1
            ):
                raise ValueError(
                    "WGSL target requires runtime-sized array member "
                    f"{node.name}.{member.name} to be the final GLSL buffer block member"
                )

    def collect_glsl_buffer_block_struct_names(self, global_variables, functions):
        struct_names = set()
        for variable in global_variables:
            if not self.is_glsl_buffer_block_variable(variable):
                continue
            struct_name = self.glsl_buffer_block_struct_name(variable.var_type)
            if struct_name:
                struct_names.add(struct_name)
        for function in functions:
            for parameter in getattr(function, "parameters", []) or []:
                if not self.is_glsl_buffer_block_parameter(parameter):
                    continue
                struct_name = self.glsl_buffer_block_struct_name(parameter.param_type)
                if struct_name:
                    struct_names.add(struct_name)
        return struct_names

    def module_storage_access_modes(self, global_variables, stage_resource_parameters=()):
        modes = {}
        for variable in global_variables:
            if self.is_glsl_buffer_block_variable(variable):
                modes[variable.name] = self.glsl_buffer_block_access(variable)
                continue
            access = self.structured_buffer_access(getattr(variable, "var_type", None))
            if access:
                modes[variable.name] = access
        for parameter in stage_resource_parameters:
            access = self.structured_buffer_access(
                getattr(parameter, "param_type", None)
            )
            if access:
                modes[parameter.name] = access
        return modes

    def inferred_storage_texture_access_modes(
        self, global_variables, stage_resource_parameters, functions
    ):
        storage_texture_names = {
            getattr(node, "name", "")
            for node in list(global_variables) + list(stage_resource_parameters)
            if self.STORAGE_TEXTURE_DIMENSION_MAP.get(
                self.resource_type_name(self.declared_resource_type(node)) or ""
            )
            is not None
        }
        storage_texture_names.discard("")
        if not storage_texture_names:
            return {}

        modes = {}
        for function in functions:
            body = getattr(function, "body", None)
            if body is None or not hasattr(body, "walk"):
                continue
            for node in body.walk():
                if not isinstance(node, FunctionCallNode):
                    continue
                access = self.storage_image_operation_access_requirement(
                    self.expression_name(node.function)
                )
                if access is None or not getattr(node, "arguments", None):
                    continue
                access_path = self.expression_access_path(node.arguments[0])
                if access_path is None:
                    continue
                root_name, path = access_path
                if path or root_name not in storage_texture_names:
                    continue
                modes[root_name] = merge_image_access_requirement(
                    modes.get(root_name), access
                )
        return modes

    def storage_image_operation_access_requirement(self, function_name):
        normalized_name = self.semantic_key(function_name)
        if normalized_name == "imageload":
            return "read"
        if normalized_name == "imagestore":
            return "write"
        if normalized_name.startswith("imageatomic"):
            return "read_write"
        return None

    def is_buffer_pointer_type(self, vtype, qualifiers=()):
        if not isinstance(vtype, PointerType):
            return False
        qualifier_names = {str(qualifier).lower() for qualifier in qualifiers or []}
        return bool(qualifier_names.intersection({"buffer", "storage"}))

    def buffer_pointer_element_type(self, vtype):
        if not isinstance(vtype, PointerType):
            raise ValueError("WGSL target expected a buffer pointer type")
        return vtype.pointee_type

    def buffer_pointer_storage_type(self, vtype):
        element_type = self.type_name_string(self.buffer_pointer_element_type(vtype))
        return f"array<{element_type}>"

    def buffer_pointer_parameter_type(self, vtype):
        return f"ptr<storage, {self.buffer_pointer_storage_type(vtype)}, read_write>"

    def validate_cbuffer_member(self, cbuffer, member):
        member_type = getattr(member, "member_type", None)
        element_type = self.array_element_type(member_type)
        resource_type_name = self.resource_type_name(element_type)
        if resource_type_name is not None and not self.is_resource_type_name(
            resource_type_name
        ):
            resource_type_name = None
        if resource_type_name:
            display_type_name = (
                str(element_type.name)
                if isinstance(element_type, NamedType)
                else str(element_type)
            )
            raise ValueError(
                "WGSL target does not support resource member "
                f"{cbuffer.name}.{member.name} of type {display_type_name} "
                "inside uniform buffers; declare resources as module-scope "
                "bindings instead"
            )
        if self.structured_buffer_element_type(element_type) is not None:
            raise ValueError(
                "WGSL target does not support storage-buffer resource member "
                f"{cbuffer.name}.{member.name} inside uniform buffers; "
                "declare StructuredBuffer resources as module-scope bindings"
            )
        if self.unsized_array_type(member_type) is not None:
            raise ValueError(
                "WGSL target does not support runtime-sized array member "
                f"{cbuffer.name}.{member.name} inside uniform buffers"
            )

    def uniform_buffer_struct_names(
        self, cbuffers, global_variables, stage_resource_parameters
    ):
        names = {
            getattr(cbuffer, "name", "")
            for cbuffer in cbuffers
            if getattr(cbuffer, "name", "")
        }
        for variable in global_variables:
            qualifier_names = {
                str(qualifier).lower()
                for qualifier in getattr(variable, "qualifiers", []) or []
            }
            if "uniform" not in qualifier_names:
                continue
            struct_name = self.struct_type_name(
                self.array_element_type(getattr(variable, "var_type", None))
            )
            if struct_name:
                names.add(struct_name)
        for parameter in stage_resource_parameters:
            uniform_type = self.stage_uniform_parameter_type(parameter)
            struct_name = self.struct_type_name(self.array_element_type(uniform_type))
            if struct_name:
                names.add(struct_name)
        names.discard("")
        return names

    def uniform_scalar_array_wrapper_names(self, cbuffers, structs):
        scalar_types = set()
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                scalar_type = self.uniform_scalar_array_element_type_name(
                    getattr(member, "member_type", None)
                )
                if scalar_type is not None:
                    scalar_types.add(scalar_type)
        for struct in structs:
            if getattr(struct, "name", "") not in self._uniform_buffer_struct_names:
                continue
            for member in getattr(struct, "members", []) or []:
                scalar_type = self.uniform_scalar_array_element_type_name(
                    getattr(member, "member_type", None)
                )
                if scalar_type is not None:
                    scalar_types.add(scalar_type)

        used_names = set(self._type_identifier_names.values())
        used_names.update(self._function_identifier_names.values())
        used_names.update(self._module_identifier_names.values())
        wrappers = {}
        suffixes = {"f32": "F32", "i32": "I32", "u32": "U32"}
        for scalar_type in sorted(scalar_types):
            base_name = f"UniformArrayElement{suffixes[scalar_type]}"
            wrapper_name = self.unique_wgsl_identifier(base_name, used_names)
            wrappers[scalar_type] = wrapper_name
            used_names.add(wrapper_name)
        return wrappers

    def is_uniform_scalar_array_access(self, array_expr):
        if isinstance(array_expr, IdentifierNode):
            if self.is_local_identifier(array_expr.name):
                return False
            member_type = self._cbuffer_member_types.get(array_expr.name)
            return self.uniform_scalar_array_wrapper_name(member_type) is not None
        if isinstance(array_expr, MemberAccessNode):
            object_type = self.array_element_type(
                self.expression_type(array_expr.object_expr)
            )
            struct_name = self.struct_type_name(object_type)
            if struct_name not in self._uniform_buffer_struct_names:
                return False
            member_type = self._struct_member_types.get((struct_name, array_expr.member))
            return self.uniform_scalar_array_wrapper_name(member_type) is not None
        return False

    def unsized_array_type(self, vtype):
        while isinstance(vtype, ArrayType):
            if vtype.size is None:
                return vtype
            vtype = vtype.element_type
        return None

    def sampled_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        if type_name is None:
            return None
        return self.SAMPLED_TEXTURE_TYPE_MAP.get(type_name)

    def storage_texture_type(self, vtype, node):
        type_name = self.resource_type_name(vtype)
        dimension = self.STORAGE_TEXTURE_DIMENSION_MAP.get(type_name or "")
        if dimension is None:
            return None
        texture_format = self.storage_texture_format(node)
        access = self.storage_texture_access(node)
        return f"texture_storage_{dimension}<{texture_format}, {access}>"

    def storage_texture_format(self, node):
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            texture_format = self.STORAGE_TEXTURE_FORMAT_MAP.get(key)
            if texture_format is not None:
                return texture_format
        name = getattr(node, "name", "storage texture")
        raise ValueError(
            "WGSL target requires storage image resource "
            f"{name} to declare a representable image format such as "
            "layout(rgba8) or @rgba8"
        )

    def storage_texture_access(self, node):
        explicit_access = explicit_image_access(node, self.generate_attribute_argument)
        if explicit_access is not None:
            return explicit_access
        names = self.resource_qualifier_attribute_names(node)
        if "constant" in names:
            return "read"
        if names & {"rw", "coherent"}:
            return "read_write"
        name = getattr(node, "name", None)
        if name:
            inferred_access = self._inferred_storage_texture_access_modes.get(name)
            if inferred_access is not None:
                return inferred_access
        diagnostic_name = name or "storage texture"
        raise ValueError(
            "WGSL target requires storage image resource "
            f"{diagnostic_name} to declare an access mode "
            "(@readonly, @writeonly, or @readwrite) or use it in "
            "imageLoad/imageStore so access can be inferred"
        )

    def resource_qualifier_attribute_names(self, node):
        names = {
            self.semantic_key(str(qualifier))
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        names.update(
            self.semantic_key(str(getattr(attr, "name", attr)))
            for attr in getattr(node, "attributes", []) or []
        )
        return names

    def require_writable_storage_texture_resource(self, resource, function_name):
        resource_type = self.expression_type(resource)
        if self.STORAGE_TEXTURE_DIMENSION_MAP.get(
            self.resource_type_name(resource_type) or ""
        ) is None:
            raise ValueError(
                f"WGSL target requires {function_name}() to use a storage image "
                "resource"
            )
        access = self.storage_texture_access_for_expression(resource)
        if access == "read":
            raise ValueError(
                "WGSL target cannot store through read-only storage image "
                f"resource in {function_name}()"
            )

    def require_readable_storage_texture_resource(self, resource, function_name):
        resource_type = self.expression_type(resource)
        if self.STORAGE_TEXTURE_DIMENSION_MAP.get(
            self.resource_type_name(resource_type) or ""
        ) is None:
            raise ValueError(
                f"WGSL target requires {function_name}() to use a storage image "
                "resource"
            )
        access = self.storage_texture_access_for_expression(resource)
        if access == "write":
            raise ValueError(
                "WGSL target cannot load from write-only storage image "
                f"resource in {function_name}()"
            )

    def storage_texture_access_for_expression(self, expr):
        access_path = self.expression_access_path(expr)
        if access_path is None:
            return None
        root_name, path = access_path
        if path:
            return None
        return self._module_storage_texture_access_modes.get(root_name)

    def is_depth_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        return type_name in {
            "sampler2darrayshadow",
            "sampler2dshadow",
            "samplercubearrayshadow",
            "samplercubeshadow",
        }

    def is_sampler_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        return (
            type_name in self.SAMPLER_TYPE_NAMES
            or type_name in self.COMPARISON_SAMPLER_TYPE_NAMES
        )

    def is_comparison_sampler_type(self, vtype):
        return self.resource_type_name(vtype) in self.COMPARISON_SAMPLER_TYPE_NAMES

    def sampler_type(self, vtype):
        if self.is_comparison_sampler_type(vtype):
            return "sampler_comparison"
        return "sampler"

    def companion_sampler_type(self, texture_type):
        if self.is_depth_texture_type(texture_type):
            return "sampler_comparison"
        return "sampler"

    def resource_type_name(self, vtype):
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name).lower()
        if isinstance(vtype, str):
            return vtype.strip().lower()
        return None

    def struct_member_resource_type_name(self, member):
        member_type = getattr(member, "member_type", None)
        type_name = self.resource_type_name(member_type)
        if type_name is None or not self.is_resource_type_name(type_name):
            return None
        if isinstance(member_type, NamedType):
            return str(member_type.name)
        return str(member_type)

    def supported_struct_resource_member(self, member):
        return self.sampled_texture_type(
            getattr(member, "member_type", None)
        ) is not None or self.is_sampler_type(getattr(member, "member_type", None))

    def collect_identifier_metadata(
        self,
        structs,
        cbuffers,
        constants,
        global_variables,
        stage_resource_parameters,
        helper_functions,
    ):
        type_names = [
            getattr(node, "name", "") for node in list(structs) + list(cbuffers)
        ]
        module_names = [
            getattr(node, "name", "")
            for node in list(constants)
            + list(global_variables)
            + list(stage_resource_parameters)
        ]
        function_groups = self.function_groups_by_name(helper_functions)
        self._function_overloads_by_name = function_groups
        function_names = []
        function_name_keys = {}
        used_function_names = {name for name in type_names + module_names if name}
        for function_name, functions in function_groups.items():
            if len(functions) == 1:
                emitted_name = self.unique_function_module_name(
                    function_name, used_function_names
                )
                function_names.append(emitted_name)
                function_name_keys[self.function_signature_key(functions[0])] = (
                    emitted_name
                )
                continue
            for function in functions:
                overload_name = (
                    f"{function_name}_{self.function_overload_suffix(function)}"
                )
                emitted_name = self.unique_function_module_name(
                    overload_name, used_function_names
                )
                function_names.append(emitted_name)
                function_name_keys[self.function_signature_key(function)] = (
                    emitted_name
                )
        module_scope_names = self.wgsl_identifier_map(
            type_names + function_names + module_names
        )
        self._type_identifier_names = {
            name: module_scope_names[name] for name in type_names if name
        }
        self._function_identifier_names = {
            name: module_scope_names[name] for name in function_names if name
        }
        self._function_signature_identifier_names = {
            signature_key: module_scope_names[function_name]
            for signature_key, function_name in function_name_keys.items()
            if function_name
        }
        self._module_identifier_names = {
            name: module_scope_names[name] for name in module_names if name
        }
        self._struct_member_identifier_names = {}
        for struct in structs:
            self.register_struct_member_identifier_metadata(struct)

    def collect_cbuffer_member_identifier_metadata(self, cbuffers):
        for cbuffer in cbuffers:
            self.register_struct_member_identifier_metadata(cbuffer)

    def register_struct_member_identifier_metadata(self, struct):
        struct_name = getattr(struct, "name", "")
        member_names = [
            getattr(member, "name", "")
            for member in getattr(struct, "members", []) or []
            if getattr(member, "name", "")
        ]
        member_map = self.wgsl_identifier_map(member_names)
        for source_name, emitted_name in member_map.items():
            self._struct_member_identifier_names[(struct_name, source_name)] = (
                emitted_name
            )

    def collect_struct_type_metadata(self, structs):
        self._structs_by_name = {struct.name: struct for struct in structs}
        self._struct_member_types = {}
        self._struct_resource_paths = {}
        for struct in structs:
            for member in getattr(struct, "members", []) or []:
                if not hasattr(member, "member_type"):
                    continue
                self._struct_member_types[(struct.name, member.name)] = (
                    member.member_type
                )
        for struct in structs:
            self._struct_resource_paths[struct.name] = tuple(
                self.resource_paths_for_type(NamedType(struct.name))
            )

    def resource_paths_for_type(self, vtype, prefix=(), seen=()):
        vtype = self.array_element_type(vtype)
        struct_name = self.struct_type_name(vtype)
        if not struct_name or struct_name in seen:
            return ()
        struct = self._structs_by_name.get(struct_name)
        if struct is None:
            return ()
        if getattr(struct, "generic_params", None):
            return ()

        paths = []
        for member in getattr(struct, "members", []) or []:
            if not hasattr(member, "member_type"):
                continue
            member_type = getattr(member, "member_type", None)
            member_path = tuple(prefix) + (member.name,)
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                if not self.supported_struct_resource_member(member):
                    continue
                info = {
                    "owner_struct": struct.name,
                    "member": member,
                    "member_name": member.name,
                    "member_type": member_type,
                    "path": member_path,
                    "resource_type_name": resource_type_name,
                }
                paths.append(info)
                continue
            paths.extend(
                self.resource_paths_for_type(
                    member_type, prefix=member_path, seen=tuple(seen) + (struct_name,)
                )
            )
        return tuple(paths)

    def struct_type_name(self, vtype):
        option_type = self.builtin_option_specialized_type_name(vtype)
        if option_type is not None:
            return option_type
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name)
        if isinstance(vtype, str):
            return vtype
        return None

    def array_element_type(self, vtype):
        while isinstance(vtype, ArrayType):
            vtype = vtype.element_type
        return vtype

    def resource_member_binding_name(self, root_name, path):
        return "_".join((root_name, *path))

    def resource_member_binding_for_access(self, expr):
        access_path = self.expression_access_path(expr)
        if access_path is None:
            return None
        root_name, path = access_path
        if not path:
            return None

        alias = self.resource_alias_binding(root_name, path)
        if alias is not None:
            return alias

        binding = self._module_resource_bindings.get((root_name, tuple(path)))
        if binding is not None:
            return binding
        return None

    def expression_access_path(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name, ()
        if isinstance(expr, MemberAccessNode):
            parent = self.expression_access_path(expr.object_expr)
            if parent is None:
                return None
            root_name, path = parent
            return root_name, tuple(path) + (expr.member,)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_access_path(expr.array_expr)
        return None

    def expression_type(self, expr):
        if isinstance(expr, IdentifierNode):
            if expr.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES:
                return "vec3<u32>"
            mapped_builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(expr.name)
            if mapped_builtin:
                return self.INPUT_BUILTIN_TYPE_MAP.get(mapped_builtin)
            value_type = self.value_type(expr.name)
            if value_type is not None:
                return value_type
            module_type = self._module_variable_types.get(expr.name)
            if module_type is not None:
                return module_type
            if not self.is_local_identifier(expr.name):
                return self._cbuffer_member_types.get(expr.name)
            return None
        if isinstance(expr, MemberAccessNode):
            object_type = self.array_element_type(
                self.expression_type(expr.object_expr)
            )
            component_type = self.vector_component_type(object_type)
            if component_type is not None and self.is_vector_member(expr.member):
                if len(expr.member) == 1:
                    return component_type
                return f"vec{len(expr.member)}<{component_type}>"
            struct_name = self.struct_type_name(object_type)
            if not struct_name:
                return None
            return self._struct_member_types.get((struct_name, expr.member))
        if isinstance(expr, BinaryOpNode):
            return self.binary_expression_type(expr)
        if isinstance(expr, FunctionCallNode):
            function_name = self.expression_name(expr.function)
            option_call_type = self.builtin_option_call_type(expr, function_name)
            if option_call_type is not None:
                return option_call_type
            if self.is_type_constructor_name(function_name):
                return self.type_name_string(function_name)
            resolved_function = self.resolve_function_overload(
                function_name, expr.arguments
            )
            if resolved_function is not None:
                return self._function_return_types_by_signature.get(
                    self.function_signature_key(resolved_function),
                    getattr(resolved_function, "return_type", None),
                )
            return self._function_return_types.get(function_name)
        if isinstance(expr, ConstructorNode):
            return expr.constructor_type
        if isinstance(expr, CastNode):
            return expr.target_type
        if isinstance(expr, LiteralNode):
            return getattr(expr, "literal_type", None)
        if isinstance(expr, SwizzleNode):
            vector_type = self.expression_type(expr.vector_expr)
            component_type = self.vector_component_type(vector_type)
            if component_type is None:
                return None
            if len(expr.components) == 1:
                return component_type
            return f"vec{len(expr.components)}<{component_type}>"
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_type(expr.array_expr)
            if isinstance(array_type, ArrayType):
                return array_type.element_type
            storage_element = self.structured_buffer_element_type(array_type)
            if storage_element is not None:
                return storage_element
            if isinstance(array_type, PointerType):
                return array_type.pointee_type
            return self.array_element_type(array_type)
        return None

    def binary_expression_type(self, expr):
        if expr.operator not in {"+", "-", "*", "/", "%"}:
            return None
        left_type = self.expression_type(expr.left)
        right_type = self.expression_type(expr.right)
        left_vector = self.vector_shape(left_type)
        right_vector = self.vector_shape(right_type)
        left_matrix = self.matrix_shape(left_type)
        right_matrix = self.matrix_shape(right_type)
        left_scalar = self.scalar_type_name(left_type)
        right_scalar = self.scalar_type_name(right_type)
        if expr.operator == "*":
            if left_matrix is not None and right_vector is not None:
                matrix_element, columns, rows = left_matrix
                vector_element, vector_size = right_vector
                if matrix_element == vector_element and columns == vector_size:
                    return f"vec{rows}<{matrix_element}>"
            if left_vector is not None and right_matrix is not None:
                vector_element, vector_size = left_vector
                matrix_element, columns, rows = right_matrix
                if matrix_element == vector_element and rows == vector_size:
                    return f"vec{columns}<{matrix_element}>"
        if left_vector is not None and right_vector is not None:
            if left_vector == right_vector:
                return left_type
            return None
        if left_vector is not None and right_scalar is not None:
            return left_type
        if right_vector is not None and left_scalar is not None:
            return right_type
        if left_scalar is not None and left_scalar == right_scalar:
            return left_scalar
        if left_type == right_type:
            return left_type
        return None

    def scalar_type_name(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name in {"bool", "f32", "i32", "u32"}:
            return type_name
        return None

    def vector_shape(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name is None:
            return None
        match = re.fullmatch(r"vec([234])<([^>]+)>", type_name)
        if match:
            return match.group(2), int(match.group(1))
        return None

    def matrix_shape(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name is None:
            return None
        match = re.fullmatch(r"mat([234])x([234])<([^>]+)>", type_name)
        if match:
            return match.group(3), int(match.group(1)), int(match.group(2))
        return None

    def integer_scalar_type(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name in {"i32", "u32"}:
            return type_name
        return None

    def vector_component_type(self, vtype):
        if isinstance(vtype, VectorType):
            return self.type_name_string(vtype.element_type)
        if isinstance(vtype, str):
            match = re.fullmatch(r"vec[234]<([^>]+)>", self.map_type_name(vtype))
            if match:
                return match.group(1)
        return None

    def is_vector_member(self, member):
        return bool(member) and all(component in "xyzwrgba" for component in member)

    def value_type(self, name):
        for scope in reversed(self._value_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_value_type(self, name, vtype):
        if self._value_type_scopes and name:
            self._value_type_scopes[-1][name] = vtype

    def register_parameter_value_types(self, function):
        for parameter in getattr(function, "parameters", []) or []:
            value_type = (
                self.stage_resource_module_type(parameter)
                if self.is_stage_resource_parameter(parameter)
                else parameter.param_type
            )
            self.register_value_type(parameter.name, value_type)
            self.register_parameter_resource_aliases(parameter)

    def register_parameter_resource_aliases(self, parameter):
        for info in self.resource_paths_for_type(parameter.param_type):
            self.register_resource_alias(
                parameter.name,
                info["path"],
                {
                    **info,
                    "binding_name": self.resource_member_binding_name(
                        parameter.name, info["path"]
                    ),
                },
            )

    def register_resource_aliases(self, name, vtype, initializer=None):
        if initializer is None:
            return
        initializer_access = self.expression_access_path(initializer)
        if initializer_access is None:
            return
        _initializer_root, initializer_path = initializer_access
        for info in self.resource_paths_for_type(vtype):
            binding = self.resource_binding_for_access_path(
                initializer_access[0], tuple(initializer_path) + tuple(info["path"])
            )
            if binding is not None:
                self.register_resource_alias(name, info["path"], binding)

    def register_resource_alias(self, root_name, path, binding):
        if self._resource_alias_scopes and root_name:
            self._resource_alias_scopes[-1].setdefault(root_name, {})[
                tuple(path)
            ] = binding

    def resource_alias_binding(self, root_name, path):
        for scope in reversed(self._resource_alias_scopes):
            binding = scope.get(root_name, {}).get(tuple(path))
            if binding is not None:
                return binding
        return None

    def resource_binding_for_access_path(self, root_name, path):
        alias = self.resource_alias_binding(root_name, path)
        if alias is not None:
            return alias
        return self._module_resource_bindings.get((root_name, tuple(path)))

    def storage_access_mode(self, name):
        return self._module_storage_access_modes.get(name)

    def validate_storage_assignment_target(self, target):
        access_path = self.expression_access_path(target)
        if access_path is None:
            return
        root_name, _path = access_path
        if self.storage_access_mode(root_name) == "read":
            raise ValueError(
                "WGSL target cannot write read-only GLSL buffer block resource "
                f"{root_name}"
            )

    def resource_member_parameter_declarations(self, root_name, root_type):
        declarations = []
        for info in self.resource_paths_for_type(root_type):
            binding_name = self.resource_member_binding_name(root_name, info["path"])
            sampled_texture_type = self.sampled_texture_type(info["member_type"])
            if sampled_texture_type is not None:
                declarations.append(f"{binding_name}: {sampled_texture_type}")
                declarations.append(
                    f"{self.texture_sampler_name(binding_name)}: "
                    f"{self.companion_sampler_type(info['member_type'])}"
                )
            elif self.is_sampler_type(info["member_type"]):
                declarations.append(
                    f"{binding_name}: {self.sampler_type(info['member_type'])}"
                )
        return declarations

    def resource_member_call_arguments(self, arg, resource_paths):
        arguments = []
        access = self.expression_access_path(arg)
        if access is None:
            raise ValueError(
                "WGSL target cannot forward resource-bearing struct expression "
                f"{self.generate_expression(arg)}; pass a named struct value"
            )
        root_name, root_path = access
        for info in resource_paths:
            binding = self.resource_binding_for_access_path(
                root_name, tuple(root_path) + tuple(info["path"])
            )
            if binding is None:
                raise ValueError(
                    "WGSL target cannot forward resource member "
                    f"{'.'.join((root_name, *root_path, *info['path']))}; "
                    "no module binding or resource alias is available"
                )
            arguments.append(binding["binding_name"])
            if self.sampled_texture_type(info["member_type"]) is not None:
                arguments.append(self.texture_sampler_name(binding["binding_name"]))
        return arguments

    def texture_sampler_name(self, texture_name):
        return f"{texture_name}_sampler"

    def module_identifier_name(self, name):
        return self._module_identifier_names.get(name, self.safe_wgsl_identifier(name))

    def function_declaration_identifier_name(self, function):
        signature_key = self.function_signature_key(function)
        if signature_key in self._function_signature_identifier_names:
            return self._function_signature_identifier_names[signature_key]
        return self.function_identifier_name(getattr(function, "name", ""))

    def function_identifier_name(self, name):
        if name in self._function_identifier_names:
            return self._function_identifier_names[name]
        overloads = self._function_overloads_by_name.get(name, ())
        if overloads:
            return self.function_declaration_identifier_name(overloads[0])
        return self.safe_wgsl_identifier(name)

    def function_groups_by_name(self, functions):
        groups = {}
        for function in functions:
            function_name = getattr(function, "name", "")
            if not function_name:
                continue
            groups.setdefault(function_name, []).append(function)
        return groups

    def unique_function_module_name(self, base_name, used_names):
        candidate = self.safe_wgsl_identifier(base_name)
        if candidate in used_names or self.requires_wgsl_identifier_escape(candidate):
            candidate = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(candidate), used_names
            )
        used_names.add(candidate)
        return candidate

    def function_signature_key(self, function):
        return (
            getattr(function, "name", ""),
            tuple(
                self.function_type_signature(getattr(parameter, "param_type", None))
                for parameter in getattr(function, "parameters", []) or []
            ),
        )

    def function_overload_suffix(self, function):
        parts = [
            self.function_type_suffix(getattr(parameter, "param_type", None))
            for parameter in getattr(function, "parameters", []) or []
        ]
        return "_".join(part for part in parts if part) or "void"

    def function_type_suffix(self, vtype):
        signature = self.function_type_signature(vtype)
        suffix = re.sub(r"[^0-9A-Za-z]+", "_", signature).strip("_")
        return suffix or "value"

    def function_type_signature(self, vtype):
        if vtype is None:
            return "void"
        try:
            mapped_type = self.type_name_string(vtype, allow_storage_resources=True)
        except ValueError:
            mapped_type = None
        if mapped_type is not None:
            return re.sub(r"[^0-9A-Za-z]+", "_", mapped_type).strip("_") or "value"
        if isinstance(vtype, PrimitiveType):
            return self.PRIMITIVE_TYPE_MAP.get(vtype.name.lower(), vtype.name.lower())
        if isinstance(vtype, VectorType):
            element = self.function_type_signature(vtype.element_type)
            return f"vec{vtype.size}_{element}"
        if isinstance(vtype, MatrixType):
            element = self.function_type_signature(vtype.element_type)
            return f"mat{vtype.cols}x{vtype.rows}_{element}"
        if isinstance(vtype, ArrayType):
            size = "unsized" if vtype.size is None else str(vtype.size)
            return f"array_{size}_{self.function_type_signature(vtype.element_type)}"
        if isinstance(vtype, ReferenceType):
            return self.function_type_signature(vtype.referenced_type)
        if isinstance(vtype, PointerType):
            return f"ptr_{self.function_type_signature(vtype.pointee_type)}"
        if isinstance(vtype, NamedType):
            base = self.semantic_key(str(vtype.name)) or str(vtype.name)
            if not vtype.generic_args:
                return base
            args = "_".join(
                self.function_type_signature(arg) for arg in vtype.generic_args
            )
            return f"{base}_{args}"
        if isinstance(vtype, GenericType):
            return self.semantic_key(str(vtype.name)) or str(vtype.name)
        if isinstance(vtype, str):
            return re.sub(r"[^0-9A-Za-z]+", "_", vtype).strip("_") or "value"
        return re.sub(r"[^0-9A-Za-z]+", "_", str(vtype)).strip("_") or "value"

    def resolve_function_overload(self, function_name, arguments, expected_type=None):
        candidates = [
            function
            for function in self._function_overloads_by_name.get(function_name, ())
            if len(getattr(function, "parameters", []) or []) == len(arguments)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        argument_types = [self.expression_type(argument) for argument in arguments]
        scored = []
        for function in candidates:
            parameters = getattr(function, "parameters", []) or []
            score = 0
            compatible = True
            known_argument_count = 0
            for argument_type, parameter in zip(argument_types, parameters):
                if argument_type is None:
                    continue
                known_argument_count += 1
                if not self.function_types_compatible(
                    argument_type, getattr(parameter, "param_type", None)
                ):
                    compatible = False
                    break
                score += 4
            if not compatible:
                continue
            if expected_type is not None:
                if self.function_types_compatible(
                    getattr(function, "return_type", None), expected_type
                ):
                    score += 2
                elif known_argument_count == 0:
                    continue
            scored.append((score, function))
        if not scored:
            return candidates[0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def function_types_compatible(self, actual_type, expected_type):
        if actual_type is None or expected_type is None:
            return False
        return self.function_type_match_name(actual_type) == self.function_type_match_name(
            expected_type
        )

    def function_type_match_name(self, vtype):
        try:
            return self.type_name_string(vtype, allow_storage_resources=True)
        except ValueError:
            return self.function_type_signature(vtype)

    def type_identifier_name(self, name):
        return self._type_identifier_names.get(name, self.safe_wgsl_identifier(name))

    def struct_member_identifier_name(self, struct_name, member_name):
        return self._struct_member_identifier_names.get(
            (struct_name, member_name), self.safe_wgsl_identifier(member_name)
        )

    def member_access_identifier_name(self, expr):
        object_type = self.array_element_type(self.expression_type(expr.object_expr))
        if self.vector_component_type(object_type) is not None and self.is_vector_member(
            expr.member
        ):
            return expr.member
        struct_name = self.struct_type_name(object_type)
        if not struct_name:
            return self.safe_wgsl_identifier(expr.member)
        return self.struct_member_identifier_name(struct_name, expr.member)

    def identifier_name(self, name):
        for aliases in reversed(self._identifier_alias_scopes):
            if name in aliases:
                return aliases[name]
        if name in self._module_identifier_names:
            return self._module_identifier_names[name]
        return self.safe_wgsl_identifier(name)

    def declare_local_identifier(self, name):
        if not self._identifier_alias_scopes:
            return self.safe_wgsl_identifier(name)
        aliases = self._identifier_alias_scopes[-1]
        if name in aliases:
            return aliases[name]
        used_names = set(self._identifier_scopes[-1])
        used_names.update(aliases.values())
        emitted_name = self.safe_wgsl_identifier(name)
        if emitted_name in used_names or self.requires_wgsl_identifier_escape(name):
            emitted_name = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(name), used_names
            )
        aliases[name] = emitted_name
        self._identifier_scopes[-1].add(name)
        return emitted_name

    def wgsl_identifier_map(self, names):
        unique_names = []
        seen = set()
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            unique_names.append(name)

        used_names = {
            name
            for name in unique_names
            if not self.requires_wgsl_identifier_escape(name)
        }
        identifier_map = {}
        for name in unique_names:
            if not self.requires_wgsl_identifier_escape(name):
                identifier_map[name] = name
                continue
            emitted_name = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(name), used_names
            )
            identifier_map[name] = emitted_name
            used_names.add(emitted_name)
        return identifier_map

    def safe_wgsl_identifier(self, name):
        if self.requires_wgsl_identifier_escape(name):
            return self.escaped_wgsl_identifier_base(name)
        return name

    def escaped_wgsl_identifier_base(self, name):
        name = str(name)
        if name == "_":
            return "identifier_"
        if name.startswith("__"):
            return f"{name.lstrip('_') or 'identifier'}_"
        return f"{name}_"

    def unique_wgsl_identifier(self, base_name, used_names):
        candidate = base_name
        suffix = 2
        while candidate in used_names or self.requires_wgsl_identifier_escape(
            candidate
        ):
            separator = "" if base_name.endswith("_") else "_"
            candidate = f"{base_name}{separator}{suffix}"
            suffix += 1
        return candidate

    def requires_wgsl_identifier_escape(self, name):
        name = str(name)
        return (
            name in self.WGSL_RESERVED_IDENTIFIERS
            or name == "_"
            or name.startswith("__")
        )

    def is_type_constructor_name(self, name):
        lower = str(name).lower()
        return (
            lower in self.PRIMITIVE_TYPE_MAP
            or self.TYPE_CONSTRUCTOR_RE.match(lower)
            or re.fullmatch(r"vec[234]<[^>]+>", lower)
        )

    def expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, MemberAccessNode):
            return f"{self.expression_name(expr.object_expr)}.{expr.member}"
        return self.generate_expression(expr)

    def stage_return_attributes(self, stage_name, function):
        attributes = getattr(function, "attributes", []) or []
        if not attributes and stage_name == "vertex":
            return ()
        return_attributes = self.wgsl_attributes(attributes, direction="out")
        return (return_attributes,) if return_attributes else ()

    def wgsl_attributes(self, attributes, direction="generic"):
        rendered = []
        for attr in attributes or []:
            semantic = str(getattr(attr, "name", attr))
            explicit_attribute = self.explicit_wgsl_attribute(attr, semantic)
            if explicit_attribute:
                rendered.append(explicit_attribute)
                continue

            key = self.semantic_key(semantic)
            builtin = self.BUILTIN_SEMANTICS.get(key)
            if builtin:
                rendered.append(f"@builtin({builtin})")
                continue

            location = self.semantic_location(semantic, direction)
            if location is not None:
                rendered.append(f"@location({location})")

        return " ".join(rendered)

    def explicit_wgsl_attribute(self, attr, name):
        key = self.semantic_key(name)
        if key not in {
            "builtin",
            "invariant",
            "interpolate",
            "location",
        }:
            return None

        arguments = getattr(attr, "arguments", []) or []
        if not arguments:
            if key in {"builtin", "interpolate", "location"}:
                return None
            return f"@{key}"

        rendered_args = ", ".join(
            self.generate_attribute_argument(arg) for arg in arguments
        )
        return f"@{key}({rendered_args})"

    def generate_attribute_argument(self, argument):
        if isinstance(argument, IdentifierNode):
            return argument.name
        return self.generate_expression(argument)

    def semantic_key(self, semantic):
        return re.sub(r"[^a-z0-9_]+", "", semantic.lower())

    def semantic_location(self, semantic, direction):
        key = self.semantic_key(semantic)
        if key in {"gl_fragcolor", "sv_target", "color"}:
            return 0
        semantic_bases = {
            "position": 0,
            "normal": 1,
            "texcoord": 2,
            "uv": 2,
            "tangent": 6,
            "bitangent": 7,
        }
        for base, offset in semantic_bases.items():
            if key == base:
                return offset
            if key.startswith(base):
                suffix = key[len(base) :]
                if suffix.isdigit():
                    return offset + int(suffix)
        numeric_suffix = re.search(r"(\d+)$", key)
        if numeric_suffix:
            return int(numeric_suffix.group(1))
        return None

    def next_binding_attributes(self, group="0"):
        binding = self.next_available_binding(group, self._global_binding_index)
        self._global_binding_index = binding + 1
        self.mark_allocated_binding(group, binding)
        return f"@group({group}) @binding({binding})"

    def binding_group_from_attributes(self, attributes):
        match = re.search(r"@group\(([^)]+)\)", attributes)
        return match.group(1) if match else "0"

    def sampler_binding_attributes_for_texture(self, texture_attributes, *, node=None):
        group = self.binding_group_from_attributes(texture_attributes)
        if node is not None:
            reserved_sampler = self._reserved_sampler_bindings.get(id(node))
            if reserved_sampler is not None:
                group, binding = reserved_sampler
                self.mark_allocated_binding(group, binding)
                self._global_binding_index = max(
                    self._global_binding_index, binding + 1
                )
                return f"@group({group}) @binding({binding})"
        match = re.search(r"@binding\(([^)]+)\)", texture_attributes)
        if not match:
            return self.next_binding_attributes(group=group)
        try:
            start = int(str(match.group(1)), 0) + 1
        except ValueError:
            return self.next_binding_attributes(group=group)
        binding = self.next_available_binding(group, start)
        self.mark_allocated_binding(group, binding)
        self._global_binding_index = max(self._global_binding_index, binding + 1)
        return f"@group({group}) @binding({binding})"

    def next_available_binding(self, group, start):
        binding = start
        reserved = self._reserved_bindings_by_group.get(str(group), set())
        allocated = self._allocated_bindings_by_group.get(str(group), set())
        while binding in reserved or binding in allocated:
            binding += 1
        return binding

    def mark_allocated_binding(self, group, binding):
        self._allocated_bindings_by_group.setdefault(str(group), set()).add(binding)

    def reserve_binding(self, group, binding):
        self._reserved_bindings_by_group.setdefault(str(group), set()).add(binding)

    def resource_binding_diagnostic_name(self, node):
        name = getattr(node, "name", None)
        if name:
            return name
        return getattr(
            node, "member_type", getattr(node, "var_type", type(node).__name__)
        )

    def numeric_binding_components(self, node):
        group, binding = self.resolved_explicit_binding_components(node)
        try:
            binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return None
        return str(group), binding

    def reserve_explicit_binding_for_node(self, node):
        group, binding, register_class = self.explicit_binding_info(node)
        try:
            numeric_binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return None
        group = str(group)
        if register_class is not None:
            source_key = (group, register_class, numeric_binding)
            owner = self._hlsl_register_source_owners.get(source_key)
            if owner is not None:
                raise ValueError(
                    "WGSL target resource binding collision: "
                    f"{self.resource_binding_diagnostic_name(node)} and "
                    f"{self.resource_binding_diagnostic_name(owner)} both declare "
                    f"@group({group}) @binding({numeric_binding})"
                )
            self._hlsl_register_source_owners[source_key] = node
            owner = self._explicit_binding_owners.setdefault(group, {}).get(
                numeric_binding
            )
            if owner is None:
                target_binding = numeric_binding
            else:
                owner_class = self._explicit_binding_register_classes.get(
                    (group, numeric_binding)
                )
                if owner_class is None:
                    raise ValueError(
                        "WGSL target resource binding collision: "
                        f"{self.resource_binding_diagnostic_name(node)} and "
                        f"{self.resource_binding_diagnostic_name(owner)} "
                        f"both declare @group({group}) @binding({numeric_binding})"
                    )
                target_binding = self.next_available_binding(
                    group, numeric_binding + 1
                )
            self._hlsl_register_binding_allocations[source_key] = target_binding
            self._explicit_binding_owners[group][target_binding] = node
            self._explicit_binding_register_classes[(group, target_binding)] = (
                register_class
            )
            self.reserve_binding(group, target_binding)
            return group, target_binding

        owner = self._explicit_binding_owners.setdefault(group, {}).get(numeric_binding)
        if owner is not None:
            raise ValueError(
                "WGSL target resource binding collision: "
                f"{self.resource_binding_diagnostic_name(node)} and "
                f"{self.resource_binding_diagnostic_name(owner)} both declare "
                f"@group({group}) @binding({numeric_binding})"
            )
        self._explicit_binding_owners[group][numeric_binding] = node
        self._explicit_binding_register_classes[(group, numeric_binding)] = None
        self.reserve_binding(group, numeric_binding)
        return group, numeric_binding

    def reserve_explicit_sampler_for_texture_node(self, node):
        components = self.numeric_binding_components(node)
        if components is None:
            return
        group, texture_binding = components
        sampler_binding = self.next_available_binding(group, texture_binding + 1)
        self.reserve_binding(group, sampler_binding)
        self._reserved_sampler_bindings[id(node)] = (group, sampler_binding)

    def declared_resource_type(self, node):
        return getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "member_type", None)),
        )

    def explicit_resource_binding_nodes(
        self, cbuffers, global_variables, stage_resource_parameters=()
    ):
        for node in cbuffers:
            yield node
        for node in global_variables:
            yield node
            for info in self.resource_paths_for_type(self.declared_resource_type(node)):
                yield info["member"]
        for node in stage_resource_parameters:
            yield node
            for info in self.resource_paths_for_type(self.declared_resource_type(node)):
                yield info["member"]

    def reserve_explicit_resource_bindings(
        self, cbuffers, global_variables, stage_resource_parameters=()
    ):
        resource_nodes = list(
            self.explicit_resource_binding_nodes(
                cbuffers, global_variables, stage_resource_parameters
            )
        )
        for node in resource_nodes:
            self.reserve_explicit_binding_for_node(node)
        for node in resource_nodes:
            resource_type = self.declared_resource_type(node)
            if self.sampled_texture_type(resource_type) is not None:
                self.reserve_explicit_sampler_for_texture_node(node)

    def explicit_binding_components(self, node):
        group, binding, _register_class = self.explicit_binding_info(node)
        return group, binding

    def resolved_explicit_binding_components(self, node):
        group, binding, register_class = self.explicit_binding_info(node)
        if binding is None or register_class is None:
            return group, binding
        try:
            numeric_binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return group, binding
        source_key = (str(group), register_class, numeric_binding)
        target_binding = self._hlsl_register_binding_allocations.get(source_key)
        if target_binding is None:
            return group, binding
        return group, target_binding

    def explicit_binding_info(self, node):
        group = "0"
        binding = None
        register_class = None
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            arguments = getattr(attr, "arguments", []) or []
            if key in {"group", "set", "space"} and arguments:
                group = self.generate_attribute_argument(arguments[0])
            elif key in {"binding", "buffer"} and arguments:
                binding = self.generate_attribute_argument(arguments[0])
                register_class = None
            elif key == "register" and arguments:
                binding, group, register_class = self.register_attribute_components(
                    arguments, group
                )
        return group, binding, register_class

    def explicit_binding_attributes(self, node):
        group, binding = self.resolved_explicit_binding_components(node)
        if binding is None:
            return ""
        try:
            numeric_binding = int(str(binding), 0)
            next_binding = numeric_binding + 1
            self.mark_allocated_binding(group, numeric_binding)
        except ValueError:
            next_binding = self._global_binding_index
        self._global_binding_index = max(self._global_binding_index, next_binding)
        return f"@group({group}) @binding({binding})"

    def register_attribute_binding(self, arguments, default_group):
        binding, group, _register_class = self.register_attribute_components(
            arguments, default_group
        )
        return binding, group

    def register_attribute_components(self, arguments, default_group):
        binding_text = self.generate_attribute_argument(arguments[0])
        group = default_group
        register_class = None
        binding_match = re.fullmatch(r"\s*([A-Za-z]+)?\s*(\d+)\s*", str(binding_text))
        if binding_match:
            register_class = (
                binding_match.group(1).lower() if binding_match.group(1) else None
            )
            binding = binding_match.group(2)
        else:
            fallback_match = re.search(r"\d+", str(binding_text))
            binding = fallback_match.group(0) if fallback_match else str(binding_text)
        if len(arguments) > 1:
            space_text = self.generate_attribute_argument(arguments[1])
            group_match = re.search(r"\d+", str(space_text))
            group = group_match.group(0) if group_match else str(space_text)
        return binding, group, register_class

    def cbuffer_instance_name(self, node):
        return self.safe_wgsl_identifier(f"_{self.type_identifier_name(node.name)}")

    def cbuffer_member_accesses(self, cbuffers):
        accesses = {}
        for cbuffer in cbuffers:
            instance_name = self.cbuffer_instance_name(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", "")
                if not member_name:
                    continue
                if member_name in accesses:
                    raise ValueError(
                        "WGSL target cannot flatten duplicate cbuffer member "
                        f"name: {member_name}"
                    )
                accesses[member_name] = (
                    f"{instance_name}."
                    f"{self.struct_member_identifier_name(cbuffer.name, member_name)}"
                )
        return accesses

    def cbuffer_member_types(self, cbuffers):
        member_types = {}
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", "")
                if member_name:
                    member_types[member_name] = getattr(member, "member_type", None)
        return member_types

    def function_texture_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        texture_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.sampled_texture_type(parameter.param_type) is not None
            ]
            if indices:
                for key in self.function_metadata_keys(function):
                    texture_parameters[key] = tuple(indices)
        return texture_parameters

    def function_resource_member_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        resource_parameters = {}
        for function in functions:
            parameter_paths = {}
            for index, parameter in enumerate(
                getattr(function, "parameters", []) or []
            ):
                paths = self.resource_paths_for_type(parameter.param_type)
                if paths:
                    parameter_paths[index] = tuple(paths)
            if parameter_paths:
                for key in self.function_metadata_keys(function):
                    resource_parameters[key] = parameter_paths
        return resource_parameters

    def function_buffer_pointer_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        pointer_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.is_buffer_pointer_type(
                    parameter.param_type, getattr(parameter, "qualifiers", [])
                )
            ]
            if indices:
                for key in self.function_metadata_keys(function):
                    pointer_parameters[key] = tuple(indices)
        return pointer_parameters

    def function_metadata_keys(self, function):
        function_name = getattr(function, "name", "")
        keys = [self.function_signature_key(function)]
        if len(self._function_overloads_by_name.get(function_name, ())) <= 1:
            keys.append(function_name)
        return tuple(key for key in keys if key)

    def buffer_pointer_parameter_names(self, function):
        return tuple(
            getattr(parameter, "name", "")
            for parameter in getattr(function, "parameters", []) or []
            if self.is_buffer_pointer_type(
                parameter.param_type, getattr(parameter, "qualifiers", [])
            )
        )

    def has_binding_attribute(self, node):
        _group, binding = self.explicit_binding_components(node)
        return binding is not None

    def is_stage_resource_parameter(self, parameter):
        param_type = getattr(parameter, "param_type", None)
        if self.structured_buffer_element_type(param_type) is not None:
            return True
        if self.stage_uniform_parameter_type(parameter) is not None:
            return True
        return False

    def push_identifier_scope(self, names=()):
        names = [name for name in names if name]
        self._identifier_scopes.append(set(names))
        self._identifier_alias_scopes.append(self.wgsl_identifier_map(names))
        self._value_type_scopes.append({})
        self._resource_alias_scopes.append({})

    def pop_identifier_scope(self):
        self._identifier_scopes.pop()
        self._identifier_alias_scopes.pop()
        self._value_type_scopes.pop()
        self._resource_alias_scopes.pop()

    def register_local_identifier(self, name):
        if not self._identifier_scopes:
            return
        if name:
            self._identifier_scopes[-1].add(name)

    def is_local_identifier(self, name):
        return any(name in scope for scope in reversed(self._identifier_scopes))

    def push_pointer_identifier_scope(self, names=()):
        self._pointer_identifier_scopes.append({name for name in names if name})

    def pop_pointer_identifier_scope(self):
        self._pointer_identifier_scopes.pop()

    def is_pointer_identifier(self, name):
        return any(name in scope for scope in reversed(self._pointer_identifier_scopes))

    def _collect_structs(self, ast, target_stage):
        structs = list(getattr(ast, "structs", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            structs.extend(getattr(stage_node, "local_structs", []) or [])
        return self._dedupe_by_name(structs)

    def _collect_cbuffers(self, ast, target_stage):
        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            cbuffers.extend(getattr(stage_node, "local_cbuffers", []) or [])
        return self._dedupe_by_name(cbuffers)

    def _collect_global_variables(self, ast, target_stage):
        variables = list(getattr(ast, "global_variables", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            lowered_output_name = (
                self._stage_output_lowerings.get(id(stage_node), {}).get("source_name")
            )
            variables.extend(
                variable
                for variable in getattr(stage_node, "local_variables", []) or []
                if getattr(variable, "name", None) != lowered_output_name
            )
        return self._dedupe_by_name(variables)

    def _collect_stage_resource_parameters(self, ast, target_stage):
        parameters = []
        for stage_node in self._stage_nodes(ast, target_stage):
            entry_point = getattr(stage_node, "entry_point", None)
            if entry_point is None:
                continue
            parameters.extend(
                parameter
                for parameter in getattr(entry_point, "parameters", []) or []
                if self.is_stage_resource_parameter(parameter)
            )
        return self._dedupe_by_name(parameters)

    def _helper_functions(self, ast, target_stage):
        stage_entries = {
            id(stage_node.entry_point)
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        }
        helpers = []
        for func in getattr(ast, "functions", []) or []:
            if id(func) not in stage_entries and not self._function_stage_name(func):
                helpers.append(func)
        for stage_node in self._stage_nodes(ast, target_stage):
            helpers.extend(getattr(stage_node, "local_functions", []) or [])
        return self._dedupe_functions(helpers)

    def _stage_nodes(self, ast, target_stage):
        nodes = []
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(stage_node)
        if nodes:
            return nodes

        for func in getattr(ast, "functions", []) or []:
            stage_name = self._function_stage_name(func)
            if not stage_name:
                continue
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(
                _FunctionStageNode(
                    stage_name,
                    func,
                    execution_config=self._function_execution_config(func),
                )
            )
        return nodes

    def _function_stage_name(self, func):
        qualifiers = getattr(func, "qualifiers", []) or []
        for qualifier in qualifiers:
            stage_name = normalize_stage_name(qualifier)
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", ""))
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        return None

    def _function_execution_config(self, func):
        config = {}
        for attr in getattr(func, "attributes", []) or []:
            key = str(getattr(attr, "name", "")).lower()
            if key not in {"numthreads", "workgroup_size"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 3:
                continue
            config["numthreads"] = [
                self.generate_attribute_argument(argument) for argument in arguments
            ]
        return config

    def _dedupe_by_name(self, nodes):
        seen = set()
        deduped = []
        for node in nodes:
            name = getattr(node, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(node)
        return deduped

    def _dedupe_functions(self, funcs):
        seen = set()
        deduped = []
        for func in funcs:
            key = self.function_signature_key(func)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(func)
        return deduped


class _FunctionStageNode:
    def __init__(self, stage, entry_point, execution_config=None):
        self.stage = stage
        self.entry_point = entry_point
        self.execution_config = execution_config or {}
        self.layout_qualifiers = []
        self.local_structs = []
        self.local_functions = []
