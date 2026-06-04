"""Reverse code generator that emits CrossGL from GLSL AST nodes."""

import re

from .OpenglAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    LayoutNode,
    MemberAccessNode,
    NumberNode,
    PostfixOpNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)

try:
    from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
except ImportError:
    CROSSGL_KEYWORDS = {}

CROSSGL_RESERVED_IDENTIFIERS = set(CROSSGL_KEYWORDS) | {"true", "false"}
SHORT_INTEGER_LITERAL_SUFFIX_RE = re.compile(
    r"^(?P<body>(?:0[xX][0-9a-fA-F]+|\d+))(?P<suffix>[uU][sS]|[sS])$"
)
DOUBLE_FLOAT_LITERAL_SUFFIX_RE = re.compile(
    r"^(?P<body>"
    r"0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?\d+"
    r"|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?"
    r"|\d+[eE][+-]?\d+"
    r")[lL][fF]$"
)


class GLSLToCrossGLConverter:
    """Serialize OpenGL backend AST nodes back into CrossGL source."""

    BUFFER_BLOCK_LAYOUT_QUALIFIERS = {"std140", "std430", "scalar"}
    RAY_STORAGE_QUALIFIERS = {
        "raypayloadext": "rayPayloadEXT",
        "raypayloadinext": "rayPayloadInEXT",
        "hitattributeext": "hitAttributeEXT",
        "callabledataext": "callableDataEXT",
        "callabledatainext": "callableDataInEXT",
    }
    STORAGE_QUALIFIER_ATTRIBUTES = {
        **RAY_STORAGE_QUALIFIERS,
        "taskpayloadsharedext": "taskPayloadSharedEXT",
    }
    INTERFACE_QUALIFIER_NAMES = {
        "in": "in",
        "out": "out",
        "inout": "inout",
        "patch": "patch",
        "flat": "flat",
        "smooth": "smooth",
        "noperspective": "noperspective",
        "centroid": "centroid",
        "sample": "sample",
        "perprimitive": "perprimitive",
        "perprimitiveext": "perprimitive",
        "perprimitivenv": "perprimitive",
        "pervertex": "pervertex",
        "pervertexext": "pervertex",
        "pervertexnv": "pervertex",
        "perview": "perview",
        "perviewext": "perview",
        "perviewnv": "perview",
    }
    VARIABLE_QUALIFIER_ATTRIBUTES = {
        "invariant": "invariant",
        "precise": "precise",
        "nonuniformext": "nonuniformEXT",
        "lowp": "lowp",
        "mediump": "mediump",
        "highp": "highp",
    }
    LAYOUT_ATTRIBUTE_NAMES = (
        "constant_id",
        "location",
        "component",
        "index",
        "input_attachment_index",
        "stream",
        "xfb_buffer",
        "xfb_offset",
        "xfb_stride",
    )
    BLEND_SUPPORT_LAYOUT_ATTRIBUTE_NAMES = (
        "blend_support_multiply",
        "blend_support_screen",
        "blend_support_overlay",
        "blend_support_darken",
        "blend_support_lighten",
        "blend_support_colordodge",
        "blend_support_colorburn",
        "blend_support_hardlight",
        "blend_support_softlight",
        "blend_support_difference",
        "blend_support_exclusion",
        "blend_support_hsl_hue",
        "blend_support_hsl_saturation",
        "blend_support_hsl_color",
        "blend_support_hsl_luminosity",
        "blend_support_all_equations",
    )
    BARE_LAYOUT_ATTRIBUTE_NAMES = (
        "depth_any",
        "depth_greater",
        "depth_less",
        "depth_unchanged",
        "origin_upper_left",
        "pixel_center_integer",
        *BLEND_SUPPORT_LAYOUT_ATTRIBUTE_NAMES,
    )
    NON_STRUCT_STAGE_TYPES = {
        "compute",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "ray_generation",
        "ray_intersection",
        "ray_any_hit",
        "ray_closest_hit",
        "ray_miss",
        "ray_callable",
    }
    RAY_QUERY_SIMPLE_METHODS = {
        "rayQueryInitializeEXT": "Initialize",
        "rayQueryProceedEXT": "Proceed",
        "rayQueryTerminateEXT": "Abort",
        "rayQueryGenerateIntersectionEXT": "GenerateIntersection",
        "rayQueryConfirmIntersectionEXT": "ConfirmIntersection",
        "rayQueryGetRayTMinEXT": "RayTMin",
        "rayQueryGetRayFlagsEXT": "RayFlags",
        "rayQueryGetWorldRayOriginEXT": "WorldRayOrigin",
        "rayQueryGetWorldRayDirectionEXT": "WorldRayDirection",
        "rayQueryGetIntersectionCandidateAABBOpaqueEXT": "CandidateAABBOpaque",
    }
    RAY_QUERY_COMMITTED_METHODS = {
        "rayQueryGetIntersectionTypeEXT": ("CandidateType", "CommittedType"),
        "rayQueryGetIntersectionPrimitiveIndexEXT": (
            "CandidatePrimitiveIndex",
            "CommittedPrimitiveIndex",
        ),
        "rayQueryGetIntersectionInstanceIdEXT": (
            "CandidateInstanceID",
            "CommittedInstanceID",
        ),
        "rayQueryGetIntersectionGeometryIndexEXT": (
            "CandidateGeometryIndex",
            "CommittedGeometryIndex",
        ),
        "rayQueryGetIntersectionInstanceCustomIndexEXT": (
            "CandidateInstanceCustomIndex",
            "CommittedInstanceCustomIndex",
        ),
        "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT": (
            "CandidateInstanceShaderBindingTableRecordOffset",
            "CommittedInstanceShaderBindingTableRecordOffset",
        ),
        "rayQueryGetIntersectionObjectRayOriginEXT": (
            "CandidateObjectRayOrigin",
            "CommittedObjectRayOrigin",
        ),
        "rayQueryGetIntersectionObjectRayDirectionEXT": (
            "CandidateObjectRayDirection",
            "CommittedObjectRayDirection",
        ),
        "rayQueryGetIntersectionTEXT": ("CandidateRayT", "CommittedRayT"),
        "rayQueryGetIntersectionBarycentricsEXT": (
            "CandidateTriangleBarycentrics",
            "CommittedTriangleBarycentrics",
        ),
        "rayQueryGetIntersectionFrontFaceEXT": (
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
        ),
        "rayQueryGetIntersectionTriangleVertexPositionsEXT": (
            "CandidateTriangleVertexPositions",
            "CommittedTriangleVertexPositions",
        ),
        "rayQueryGetIntersectionObjectToWorldEXT": (
            "CandidateObjectToWorld",
            "CommittedObjectToWorld",
        ),
        "rayQueryGetIntersectionWorldToObjectEXT": (
            "CandidateWorldToObject",
            "CommittedWorldToObject",
        ),
    }
    RAY_QUERY_TRANSFORM_FUNCTIONS = {
        "rayQueryGetIntersectionObjectToWorldEXT",
        "rayQueryGetIntersectionWorldToObjectEXT",
    }
    VERTEX_BUILTIN_OUTPUT_TYPES = {
        "gl_Position": "vec4",
        "gl_PointSize": "float",
    }
    DEFAULT_SHADER_TYPE = "vertex"

    def __init__(self, shader_type="vertex"):
        self.shader_type = shader_type
        self.indent_level = 0
        self.indent_str = "    "

        self.function_map = {
            "dot": "dot",
            "normalize": "normalize",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "pow": "pow",
            "exp": "exp",
            "log": "log",
            "sqrt": "sqrt",
            "inversesqrt": "inverseSqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "fract": "fract",
            "mod": "mod",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "mix",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "reflect": "reflect",
            "refract": "refract",
            "traceRayEXT": "TraceRay",
            "reportIntersectionEXT": "ReportHit",
            "executeCallableEXT": "CallShader",
            "terminateRayEXT": "AcceptHitAndEndSearch",
            "ignoreIntersectionEXT": "IgnoreHit",
            "SetMeshOutputsEXT": "SetMeshOutputCounts",
            "EmitMeshTasksEXT": "DispatchMesh",
            "interpolateAtCentroid": "interpolate_at_centroid",
            "interpolateAtSample": "interpolate_at_sample",
            "interpolateAtOffset": "interpolate_at_offset",
            "dFdx": "ddx",
            "dFdy": "ddy",
            "fwidth": "fwidth",
            "dFdxFine": "ddx_fine",
            "dFdxCoarse": "ddx_coarse",
            "dFdyFine": "ddy_fine",
            "dFdyCoarse": "ddy_coarse",
            "fwidthFine": "fwidth_fine",
            "fwidthCoarse": "fwidth_coarse",
        }
        self.texture_function_operations = {
            "texture": "sample",
            "texture1D": "sample",
            "texture2D": "sample",
            "texture3D": "sample",
            "textureCube": "sample",
            "textureLod": "sample_lod",
            "textureLodOffset": "sample_lod",
            "textureGrad": "sample_grad",
            "textureGradOffset": "sample_grad",
            "textureProj": "sample_projected",
            "texture1DProj": "sample_projected",
            "texture2DProj": "sample_projected",
            "texture3DProj": "sample_projected",
            "textureProjLod": "sample_projected",
            "textureProjLodOffset": "sample_projected",
            "textureProjGrad": "sample_projected",
            "textureProjGradOffset": "sample_projected",
            "textureOffset": "sample_offset",
            "textureGather": "gather",
            "textureGatherOffset": "gather",
            "textureGatherOffsets": "gather",
            "textureGatherCompare": "gather_compare",
            "textureGatherCompareOffset": "gather_compare",
            "textureGatherCompareOffsets": "gather_compare",
            "textureCompare": "compare",
            "textureCompareOffset": "compare",
            "textureCompareLod": "compare",
            "textureCompareLodOffset": "compare",
            "textureCompareGrad": "compare",
            "textureCompareGradOffset": "compare",
            "textureCompareProj": "compare",
            "textureCompareProjOffset": "compare",
            "textureCompareProjLod": "compare",
            "textureCompareProjLodOffset": "compare",
            "textureCompareProjGrad": "compare",
            "textureCompareProjGradOffset": "compare",
            "texelFetch": "fetch",
            "texelFetchOffset": "fetch",
            "textureSize": "query_size",
            "textureQueryLevels": "query_levels",
            "textureQueryLod": "query_lod",
            "textureSamples": "query_samples",
        }
        self.legacy_texture_function_names = {
            "texture1D": "texture",
            "texture2D": "texture",
            "texture3D": "texture",
            "textureCube": "texture",
            "texture1DProj": "textureProj",
            "texture2DProj": "textureProj",
            "texture3DProj": "textureProj",
        }
        self.image_function_operations = {
            "imageLoad": "load",
            "imageStore": "store",
            "imageSize": "query_size",
            "imageSamples": "query_samples",
            "imageAtomicAdd": "atomic",
            "imageAtomicMin": "atomic",
            "imageAtomicMax": "atomic",
            "imageAtomicAnd": "atomic",
            "imageAtomicOr": "atomic",
            "imageAtomicXor": "atomic",
            "imageAtomicExchange": "atomic",
            "imageAtomicCompSwap": "atomic",
        }

        # Mapping of GLSL types to CrossGL types
        self.type_map = {
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "bvec2": "bvec2",
            "bvec3": "bvec3",
            "bvec4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "texture1D": "texture1D",
            "texture2D": "texture2D",
            "texture3D": "texture3D",
            "textureCube": "textureCube",
            "texture1DArray": "texture1DArray",
            "texture2DArray": "texture2DArray",
            "textureCubeArray": "textureCubeArray",
            "sampler": "sampler",
            "sampler1D": "sampler1D",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCube": "samplerCube",
            "sampler1DArray": "sampler1DArray",
            "sampler2DArray": "sampler2DArray",
            "samplerCubeArray": "samplerCubeArray",
            "sampler2DShadow": "sampler2DShadow",
            "sampler1DShadow": "sampler1DShadow",
            "sampler1DArrayShadow": "sampler1DArrayShadow",
            "sampler2DArrayShadow": "sampler2DArrayShadow",
            "samplerCubeShadow": "samplerCubeShadow",
            "samplerCubeArrayShadow": "samplerCubeArrayShadow",
            "sampler2DRect": "sampler2DRect",
            "sampler2DRectShadow": "sampler2DRectShadow",
            "samplerBuffer": "samplerBuffer",
            "sampler2DMS": "sampler2DMS",
            "sampler2DMSArray": "sampler2DMSArray",
            "isampler1D": "isampler1D",
            "isampler2D": "isampler2D",
            "isampler3D": "isampler3D",
            "isamplerCube": "isamplerCube",
            "isampler1DArray": "isampler1DArray",
            "isampler2DArray": "isampler2DArray",
            "isamplerCubeArray": "isamplerCubeArray",
            "isampler2DRect": "isampler2DRect",
            "isamplerBuffer": "isamplerBuffer",
            "isampler2DMS": "isampler2DMS",
            "isampler2DMSArray": "isampler2DMSArray",
            "usampler1D": "usampler1D",
            "usampler2D": "usampler2D",
            "usampler3D": "usampler3D",
            "usamplerCube": "usamplerCube",
            "usampler1DArray": "usampler1DArray",
            "usampler2DArray": "usampler2DArray",
            "usamplerCubeArray": "usamplerCubeArray",
            "usampler2DRect": "usampler2DRect",
            "usamplerBuffer": "usamplerBuffer",
            "usampler2DMS": "usampler2DMS",
            "usampler2DMSArray": "usampler2DMSArray",
            "image1D": "image1D",
            "image2D": "image2D",
            "image3D": "image3D",
            "imageCube": "imageCube",
            "image1DArray": "image1DArray",
            "image2DArray": "image2DArray",
            "imageCubeArray": "imageCubeArray",
            "image2DRect": "image2DRect",
            "imageBuffer": "imageBuffer",
            "image2DMS": "image2DMS",
            "image2DMSArray": "image2DMSArray",
            "subpassInput": "subpassInput",
            "subpassInputMS": "subpassInputMS",
            "isubpassInput": "isubpassInput",
            "isubpassInputMS": "isubpassInputMS",
            "usubpassInput": "usubpassInput",
            "usubpassInputMS": "usubpassInputMS",
            "iimage1D": "iimage1D",
            "iimage2D": "iimage2D",
            "iimage3D": "iimage3D",
            "iimageCube": "iimageCube",
            "iimage1DArray": "iimage1DArray",
            "iimage2DArray": "iimage2DArray",
            "iimageCubeArray": "iimageCubeArray",
            "iimage2DRect": "iimage2DRect",
            "iimageBuffer": "iimageBuffer",
            "iimage2DMS": "iimage2DMS",
            "iimage2DMSArray": "iimage2DMSArray",
            "uimage1D": "uimage1D",
            "uimage2D": "uimage2D",
            "uimage3D": "uimage3D",
            "uimageCube": "uimageCube",
            "uimage1DArray": "uimage1DArray",
            "uimage2DArray": "uimage2DArray",
            "uimageCubeArray": "uimageCubeArray",
            "uimage2DRect": "uimage2DRect",
            "uimageBuffer": "uimageBuffer",
            "uimage2DMS": "uimage2DMS",
            "uimage2DMSArray": "uimage2DMSArray",
            "accelerationStructureEXT": "accelerationStructureEXT",
            "rayQueryEXT": "rayQueryEXT",
            "void": "void",
        }

        # Map of GLSL operators to CrossGL operators
        self.operator_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "GREATER_THAN": ">",
            "LESS_THAN": "<",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "LOGICAL_AND": "&&",
            "LOGICAL_XOR": "!=",
            "LOGICAL_OR": "||",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "MOD": "%",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "BITWISE_XOR": "^",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_AND": "&=",
            "ASSIGN_OR": "|=",
            "ASSIGN_XOR": "^=",
            "^^": "!=",
        }

        # Shader-specific info
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []
        self.structs_by_name = {}
        self.structured_buffer_names = set()
        self.structured_buffer_instance_members = {}
        self.converted_ssbo_struct_names = set()
        self.interface_block_struct_names = set()
        self.flattened_uniform_block_instances = {}
        self.task_payload_shared_names = set()
        self.variable_type_scopes = []
        self.function_name_renames = {}

    def indent(self):
        return self.indent_str * self.indent_level

    def increase_indent(self):
        self.indent_level += 1

    def decrease_indent(self):
        self.indent_level -= 1

    def stage_struct_name(self):
        return "".join(part.capitalize() for part in self.shader_type.split("_"))

    def _qualifier_set(self, var):
        qualifiers = getattr(var, "qualifiers", None) or []
        return {str(q).lower() for q in qualifiers}

    def _is_input_var(self, var):
        io_type = str(getattr(var, "io_type", "") or "").upper()
        qualifiers = self._qualifier_set(var)
        return io_type == "IN" or "in" in qualifiers or "inout" in qualifiers

    def _is_output_var(self, var):
        io_type = str(getattr(var, "io_type", "") or "").upper()
        qualifiers = self._qualifier_set(var)
        return io_type == "OUT" or "out" in qualifiers or "inout" in qualifiers

    def _is_resource_type(self, type_name):
        if not type_name:
            return False
        name = str(type_name)
        return name == "accelerationStructureEXT" or name.startswith(
            (
                "__sampler",
                "texture",
                "subpassInput",
                "isubpassInput",
                "usubpassInput",
                "sampler",
                "isampler",
                "usampler",
                "image",
                "iimage",
                "uimage",
            )
        )

    def resource_function_descriptor(self, name):
        if name in self.texture_function_operations:
            return {
                "name": name,
                "function": self.legacy_texture_function_names.get(name, name),
                "resource": "texture",
                "operation": self.texture_function_operations[name],
            }
        if name in self.image_function_operations:
            return {
                "name": name,
                "function": name,
                "resource": "image",
                "operation": self.image_function_operations[name],
            }
        return None

    def push_variable_type_scope(self):
        self.variable_type_scopes.append({})

    def pop_variable_type_scope(self):
        self.variable_type_scopes.pop()

    def register_variable_type(self, var):
        if not self.variable_type_scopes:
            return
        name = getattr(var, "name", None)
        vtype = getattr(var, "vtype", None)
        if name and vtype:
            self.variable_type_scopes[-1][name] = vtype

    def register_parameter_type(self, param):
        if isinstance(param, VariableNode):
            self.register_variable_type(param)
            return
        if isinstance(param, tuple) and len(param) == 2:
            param_type, param_name = param
            if self.variable_type_scopes:
                self.variable_type_scopes[-1][param_name] = param_type

    def lookup_variable_type(self, name):
        for scope in reversed(self.variable_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def expression_resource_type(self, expr):
        if isinstance(expr, VariableNode):
            return self.lookup_variable_type(expr.name)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_resource_type(expr.array)
        return None

    def expression_is_shadow_sampler(self, expr):
        resource_type = self.expression_resource_type(expr)
        return bool(resource_type and str(resource_type).endswith("Shadow"))

    def shadow_gather_import_name(self, name, args):
        if not args or not self.expression_is_shadow_sampler(args[0]):
            return name
        if name == "textureGather" and len(args) >= 3:
            return "textureGatherCompare"
        if name == "textureGatherOffset" and len(args) >= 4:
            return "textureGatherCompareOffset"
        if name == "textureGatherOffsets" and len(args) >= 4:
            return "textureGatherCompareOffsets"
        return name

    def sanitize_crossgl_identifier(self, name):
        if name in CROSSGL_RESERVED_IDENTIFIERS:
            return f"{name}_"
        return name

    def collect_function_name_renames(self, functions):
        renames = {}
        used_names = {getattr(function, "name", "") for function in functions}
        for function in functions:
            name = getattr(function, "name", None)
            if not name or name == "main":
                continue
            safe_name = self.sanitize_crossgl_identifier(name)
            while safe_name in used_names and safe_name != name:
                safe_name += "_"
            if safe_name != name:
                renames[name] = safe_name
                used_names.add(safe_name)
        return renames

    def format_function_name(self, name):
        return self.function_name_renames.get(name, name)

    def _is_image_resource_type(self, type_name):
        if not type_name:
            return False
        return str(type_name).startswith(("image", "iimage", "uimage"))

    def _is_buffer_qualified(self, var):
        return "buffer" in self._qualifier_set(var)

    def ssbo_binding_attribute_suffix(self, var):
        layout = getattr(var, "layout", None) or {}
        binding = layout.get("binding")
        return (
            f" @binding({self.layout_value_to_string(binding)})"
            if binding is not None
            else ""
        )

    def ssbo_block_attribute_suffix(self, var):
        layout_names = self.ssbo_block_layout_names(var)
        attributes = [f"@glsl_buffer_block({', '.join(layout_names)})"]
        if self.is_shader_record_buffer_block(var):
            self.validate_shader_record_layout(var)

        binding = self.ssbo_binding(var)
        if binding is not None:
            attributes.append(f"@binding({self.layout_value_to_string(binding)})")

        qualifiers = self._qualifier_set(var)
        for qualifier in ("coherent", "volatile", "restrict", "readonly", "writeonly"):
            if qualifier in qualifiers:
                attributes.append(f"@{qualifier}")

        return " " + " ".join(attributes)

    def ssbo_binding(self, var):
        layout = getattr(var, "layout", None) or {}
        return layout.get("binding")

    def ssbo_block_layout_names(self, var):
        layout = getattr(var, "layout", None) or {}
        layout_names = []
        for name, value in layout.items():
            normalized = str(name).lower()
            if value is not None:
                continue
            if normalized in self.BUFFER_BLOCK_LAYOUT_QUALIFIERS:
                layout_names.append(normalized)
            elif normalized == "shaderrecordext":
                layout_names.append("shaderRecordEXT")
        return layout_names or ["std430"]

    def is_shader_record_buffer_block(self, var):
        return any(
            str(layout_name).lower() == "shaderrecordext"
            for layout_name in self.ssbo_block_layout_names(var)
        )

    def validate_shader_record_layout(self, var):
        if self.ssbo_binding(var) is not None:
            raise ValueError(
                "GLSL shaderRecordEXT buffer blocks cannot declare binding layout "
                "qualifiers"
            )

    def ssbo_element_member(self, var):
        if not self._is_buffer_qualified(var):
            return None
        if self.is_shader_record_buffer_block(var):
            return None

        struct = self.structs_by_name.get(getattr(var, "vtype", None))
        if struct is not None:
            members = getattr(struct, "members", None) or getattr(struct, "fields", [])
            if len(members) != 1:
                return None
            member = members[0]
            if not getattr(member, "is_array", False):
                return None
            return member

        if getattr(var, "is_array", False):
            return var

        return None

    def ssbo_block_declaration(self, var):
        if not self._is_buffer_qualified(var):
            return None
        if self.ssbo_element_member(var) is not None:
            return None
        struct = self.structs_by_name.get(getattr(var, "vtype", None))
        if struct is None:
            return None
        return self.generate_variable_declaration(
            var
        ) + self.ssbo_block_attribute_suffix(var)

    def structured_buffer_type(self, var, member):
        base = (
            "StructuredBuffer"
            if "readonly" in self._qualifier_set(var)
            else "RWStructuredBuffer"
        )
        return f"{base}<{self.convert_type(member.vtype)}>"

    def prepare_structured_buffers(self, node):
        self.structs_by_name = {struct.name: struct for struct in node.structs}
        self.interface_block_struct_names = {
            struct.name
            for struct in node.structs
            if self.is_graphics_interface_block_struct(struct)
        }
        self.structured_buffer_names = set()
        self.structured_buffer_instance_members = {}
        self.converted_ssbo_struct_names = set()
        self.variable_type_scopes = [{}]
        self.flattened_uniform_block_instances = {}

        for var in getattr(node, "uniforms", []) or []:
            self.register_variable_type(var)
        for var in getattr(node, "global_variables", []) or []:
            self.register_variable_type(var)
        for var in getattr(node, "io_variables", []) or []:
            self.register_variable_type(var)
        for var in getattr(node, "global_variables", []) or []:
            member = self.ssbo_element_member(var)
            if member is None:
                continue

            self.structured_buffer_names.add(var.name)
            if getattr(var, "vtype", None) in self.structs_by_name:
                self.converted_ssbo_struct_names.add(var.vtype)
                self.structured_buffer_instance_members[(var.name, member.name)] = True
            else:
                block_name = getattr(var, "interface_block", None)
                if block_name:
                    self.converted_ssbo_struct_names.add(block_name)

    def structured_buffer_declaration(self, var):
        member = self.ssbo_element_member(var)
        if member is None:
            return None
        array_suffix = self.array_suffix(var)
        return (
            f"{self.structured_buffer_type(var, member)} {var.name}{array_suffix}"
            f"{self.ssbo_binding_attribute_suffix(var)}"
        )

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

    def image_resource_attribute_suffix(self, var):
        var_type = getattr(var, "vtype", None)
        storage_attributes = self.storage_qualifier_attributes(var)
        if not self._is_resource_type(var_type) and not storage_attributes:
            return ""

        attributes = []
        layout = getattr(var, "layout", None) or {}
        binding = layout.get("binding")
        if binding is not None:
            attributes.append(f"@binding({self.layout_value_to_string(binding)})")
        input_attachment_index = layout.get("input_attachment_index")
        if input_attachment_index is not None:
            attributes.append(
                "@input_attachment_index("
                f"{self.layout_value_to_string(input_attachment_index)})"
            )

        if self._is_image_resource_type(var_type):
            supported_formats = self.supported_image_formats()
            for key in layout:
                format_name = str(key).lower()
                if format_name in supported_formats:
                    attributes.append(f"@{format_name}")
                    break

        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        for qualifier in ("coherent", "volatile", "restrict", "readonly", "writeonly"):
            if qualifier in qualifiers:
                attributes.append(f"@{qualifier}")

        attributes.extend(storage_attributes)

        return f" {' '.join(attributes)}" if attributes else ""

    def layout_attribute_suffix(self, layout):
        layout = layout or {}
        attributes = []
        for name in self.LAYOUT_ATTRIBUTE_NAMES:
            value = layout.get(name)
            if value is not None:
                attributes.append(f"@{name}({self.layout_value_to_string(value)})")
        for name in self.BARE_LAYOUT_ATTRIBUTE_NAMES:
            if name in layout and layout.get(name) is None:
                attributes.append(f"@{name}")
        return f" {' '.join(attributes)}" if attributes else ""

    def variable_layout_attribute_suffix(self, var):
        return self.layout_attribute_suffix(getattr(var, "layout", None))

    def interface_member_layout_attribute_suffix(self, var):
        if hasattr(var, "interface_member_layout"):
            return self.layout_attribute_suffix(
                getattr(var, "interface_member_layout", None)
            )
        return self.variable_layout_attribute_suffix(var)

    def storage_qualifier_attributes(self, var):
        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        return [
            f"@{attribute}"
            for qualifier, attribute in self.STORAGE_QUALIFIER_ATTRIBUTES.items()
            if qualifier in qualifiers
        ]

    def is_task_payload_shared_variable(self, var):
        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        return "taskpayloadsharedext" in qualifiers

    def is_buffer_reference_forward_declaration(self, var):
        if not isinstance(var, VariableNode):
            return False
        if getattr(var, "vtype", ""):
            return False
        layout = {str(key).lower() for key in getattr(var, "layout", None) or {}}
        return "buffer" in self._qualifier_set(var) and "buffer_reference" in layout

    def variable_qualifier_attribute_suffix(self, var):
        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        attributes = [
            f"@{attribute}"
            for qualifier, attribute in self.VARIABLE_QUALIFIER_ATTRIBUTES.items()
            if qualifier in qualifiers
        ]
        return f" {' '.join(attributes)}" if attributes else ""

    def is_qualifier_only_builtin_declaration(self, var):
        if not isinstance(var, VariableNode):
            return False
        if getattr(var, "vtype", ""):
            return False
        name = getattr(var, "name", "")
        if name not in self.VERTEX_BUILTIN_OUTPUT_TYPES:
            return False
        return bool(getattr(var, "qualifiers", None) or getattr(var, "layout", None))

    def qualifier_only_builtin_qualifiers(self, node):
        qualifiers_by_name = {}
        for var in getattr(node, "global_variables", []) or []:
            if not self.is_qualifier_only_builtin_declaration(var):
                continue
            builtin_qualifiers = qualifiers_by_name.setdefault(var.name, [])
            for qualifier in getattr(var, "qualifiers", []) or []:
                if qualifier not in builtin_qualifiers:
                    builtin_qualifiers.append(qualifier)
        return qualifiers_by_name

    def apply_builtin_redeclaration_qualifiers(self, vars_, qualifiers_by_name):
        for var in vars_:
            qualifiers = qualifiers_by_name.get(getattr(var, "name", None))
            if not qualifiers:
                continue
            var_qualifiers = list(getattr(var, "qualifiers", None) or [])
            for qualifier in qualifiers:
                if qualifier not in var_qualifiers:
                    var_qualifiers.append(qualifier)
            var.qualifiers = var_qualifiers

    def builtin_output_qualifiers(self, name, qualifiers_by_name):
        qualifiers = ["out"]
        for qualifier in qualifiers_by_name.get(name, []):
            if qualifier not in qualifiers:
                qualifiers.append(qualifier)
        return qualifiers

    def interface_qualifier_attribute_suffix(self, var):
        block_qualifiers = {"in", "out", "inout"}
        qualifiers = [str(q).lower() for q in getattr(var, "qualifiers", []) or []]
        attributes = []
        for qualifier in qualifiers:
            if qualifier in block_qualifiers:
                continue
            mapped = self.INTERFACE_QUALIFIER_NAMES.get(qualifier)
            if mapped is not None and mapped not in attributes:
                attributes.append(mapped)
        return (
            f" {' '.join(f'@{attribute}' for attribute in attributes)}"
            if attributes
            else ""
        )

    def stage_struct_member_attribute_suffix(self, var):
        return self.variable_layout_attribute_suffix(
            var
        ) + self.variable_qualifier_attribute_suffix(var)

    def fragment_return_attribute_suffix(self, var):
        return (
            self.variable_layout_attribute_suffix(var)
            + self.interface_qualifier_attribute_suffix(var)
            + self.variable_qualifier_attribute_suffix(var)
        )

    def fragment_uses_direct_output_declarations(
        self, fragment_writes_depth=False, fragment_writes_sample_mask=False
    ):
        return self.shader_type == "fragment" and (
            len(self.outputs) > 1
            or fragment_writes_depth
            or fragment_writes_sample_mask
        )

    def fragment_main_writes_name(self, node, name):
        if self.shader_type != "fragment":
            return False
        for function in getattr(node, "functions", []) or []:
            if getattr(function, "name", None) != "main":
                continue
            return self.statements_write_name(getattr(function, "body", []) or [], name)
        return False

    def statements_write_name(self, statements, name):
        return any(
            self.statement_writes_name(statement, name) for statement in statements
        )

    def statement_writes_name(self, statement, name):
        if isinstance(statement, AssignmentNode):
            return self.expression_base_name(getattr(statement, "left", None)) == name

        for child_name in (
            "then_branch",
            "if_body",
            "else_branch",
            "else_body",
            "else_if_chain",
            "else_if_bodies",
            "body",
            "statements",
            "cases",
            "default_case",
            "default",
        ):
            child = getattr(statement, child_name, None)
            if child is None:
                continue
            if self.child_writes_name(child, name):
                return True
        return False

    def child_writes_name(self, child, name):
        if isinstance(child, list):
            return any(self.child_writes_name(item, name) for item in child)
        if isinstance(child, tuple):
            return any(self.child_writes_name(item, name) for item in child)
        return self.statement_writes_name(child, name)

    def generate_stage_struct_member(self, var):
        var_type = self.convert_type(var.vtype)
        var_name = var.name
        qualifier_prefix = self.interface_member_qualifier_prefix(var)
        if qualifier_prefix:
            qualifier_prefix += " "
        semantic = ""
        if getattr(var, "semantic", None):
            semantic = f" @ {var.semantic}"
        array_suffix = self.array_suffix(var)
        attributes = self.stage_struct_member_attribute_suffix(var)
        return f"{qualifier_prefix}{var_type} {var_name}{array_suffix}{attributes}{semantic};\n"

    def interface_member_qualifier_prefix(self, var):
        block_qualifiers = {"in", "out", "inout", "patch"}
        qualifiers = [str(q).lower() for q in getattr(var, "qualifiers", []) or []]
        emitted = []
        for qualifier in qualifiers:
            if qualifier in block_qualifiers:
                continue
            mapped = self.INTERFACE_QUALIFIER_NAMES.get(qualifier)
            if mapped is not None and mapped not in emitted:
                emitted.append(mapped)
        return " ".join(emitted)

    def interface_qualifier_prefix(self, var):
        qualifiers = [str(q).lower() for q in getattr(var, "qualifiers", []) or []]
        emitted = []
        for qualifier in qualifiers:
            mapped = self.INTERFACE_QUALIFIER_NAMES.get(qualifier)
            if mapped is not None and mapped not in emitted:
                emitted.append(mapped)
        return " ".join(emitted)

    def layout_value_to_string(self, value):
        if isinstance(value, str):
            return value
        return self.generate_expression(value)

    def format_layout(self, layout_entry):
        layout = (
            layout_entry.get("layout", {}) if isinstance(layout_entry, dict) else {}
        )
        qualifiers = (
            layout_entry.get("qualifiers", []) if isinstance(layout_entry, dict) else []
        )
        parts = []
        for key, value in layout.items():
            if value is None:
                parts.append(str(key))
            else:
                parts.append(f"{key} = {self.layout_value_to_string(value)}")
        layout_str = f"layout({', '.join(parts)})" if parts else "layout()"
        if qualifiers:
            layout_str += " " + " ".join(qualifiers)
        declaration_type = (
            layout_entry.get("type") if isinstance(layout_entry, dict) else None
        )
        if declaration_type:
            layout_str += f" {declaration_type}"
        return layout_str.strip()

    def is_graphics_interface_block_struct(self, node):
        if not getattr(node, "interface_block", False):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "interface_qualifiers", []) or []
        }
        return bool(qualifiers & {"in", "out", "inout", "patch"})

    def is_graphics_interface_block_variable(self, var):
        block_name = getattr(var, "interface_block", None)
        return bool(block_name and block_name in self.interface_block_struct_names)

    def has_push_constant_layout(self, node):
        layout = getattr(node, "layout", None) or getattr(
            node, "interface_layout", None
        )
        return any(str(key).lower() == "push_constant" for key in layout or {})

    def is_push_constant_uniform(self, var):
        if self.has_push_constant_layout(var):
            return True

        block_name = getattr(var, "interface_block", None)
        if block_name is None:
            block_name = getattr(var, "vtype", None)
        block_struct = self.structs_by_name.get(block_name)
        return bool(block_struct and self.has_push_constant_layout(block_struct))

    def is_push_constant_interface_block_struct(self, struct):
        return bool(
            getattr(struct, "interface_block", False)
            and self.has_push_constant_layout(struct)
        )

    def push_constant_block_name(self, var):
        return getattr(var, "interface_block", None) or getattr(var, "vtype", None)

    def push_constant_block_fields(self, block_name, uniforms):
        block_struct = self.structs_by_name.get(block_name)
        if block_struct is not None:
            return getattr(block_struct, "members", None) or getattr(
                block_struct, "fields", []
            )
        return uniforms

    def record_flattened_push_constant_instance(self, block_name, uniforms, fields):
        field_names = {getattr(field, "name", None) for field in fields}
        field_names.discard(None)
        if not field_names:
            return

        block_struct = self.structs_by_name.get(block_name)
        instance_names = set()
        if block_struct is not None:
            instance_name = getattr(block_struct, "interface_instance_name", None)
            if instance_name:
                instance_names.add(str(instance_name).split("[", 1)[0])

        for uniform in uniforms:
            if getattr(uniform, "vtype", None) == block_name:
                instance_names.add(str(uniform.name).split("[", 1)[0])

        for instance_name in instance_names:
            if instance_name:
                self.flattened_uniform_block_instances[instance_name] = field_names

    def generate_push_constant_block(self, block_name, uniforms):
        fields = self.push_constant_block_fields(block_name, uniforms)
        self.record_flattened_push_constant_instance(block_name, uniforms, fields)

        result = f"cbuffer {block_name} @push_constant {{\n"
        self.increase_indent()
        for field in fields:
            var_type = self.convert_type(getattr(field, "vtype", ""))
            var_name = getattr(field, "name", "")
            array_suffix = self.array_suffix(field)
            result += self.indent() + f"{var_type} {var_name}{array_suffix};\n"
        self.decrease_indent()
        result += self.indent_str + "};\n"
        return result

    def interface_block_attribute_prefix(self, node):
        if not self.is_graphics_interface_block_struct(node):
            return ""

        attributes = []
        qualifiers = [
            str(qualifier)
            for qualifier in getattr(node, "interface_qualifiers", []) or []
        ]
        if qualifiers:
            attributes.append(f"@glsl_interface_block({', '.join(qualifiers)})")

        layout = getattr(node, "interface_layout", None) or {}
        for key in self.LAYOUT_ATTRIBUTE_NAMES:
            value = layout.get(key)
            if value is not None:
                attributes.append(f"@{key}({self.layout_value_to_string(value)})")

        instance_name = getattr(node, "interface_instance_name", None)
        if instance_name:
            attributes.append(f"@glsl_interface_instance({instance_name})")
            if getattr(node, "interface_instance_is_array", False):
                array_sizes = getattr(node, "interface_array_sizes", None) or []
                if not array_sizes:
                    array_size = getattr(node, "interface_array_size", None)
                    array_sizes = [array_size] if array_size is not None else []
                if not array_sizes or any(size is None for size in array_sizes):
                    attributes.append("@glsl_interface_array")
                else:
                    sizes = ", ".join(
                        self.generate_expression(size) for size in array_sizes
                    )
                    attributes.append(f"@glsl_interface_array({sizes})")

        return f"{' '.join(attributes)} " if attributes else ""

    def generate(self, ast):
        if ast is None:
            return "// Empty shader"

        if not isinstance(ast, ShaderNode):
            return f"// Unexpected AST node type: {type(ast)}"

        previous_shader_type = self.shader_type
        ast_shader_type = getattr(ast, "shader_type", None)
        if self.shader_type is None:
            self.shader_type = str(
                getattr(ast_shader_type, "value", ast_shader_type)
                or self.DEFAULT_SHADER_TYPE
            )
        try:
            return self.generate_shader(ast)
        finally:
            self.shader_type = previous_shader_type

    def generate_shader(self, node):
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []
        self.task_payload_shared_names = {
            var.name
            for var in getattr(node, "global_variables", []) or []
            if self.is_task_payload_shared_variable(var)
        }
        self.function_name_renames = self.collect_function_name_renames(
            getattr(node, "functions", []) or []
        )
        self.prepare_structured_buffers(node)

        for var in node.io_variables:
            if isinstance(var, (LayoutNode, VariableNode)):
                if self._is_input_var(var):
                    self.inputs.append(var)
                if self._is_output_var(var):
                    self.outputs.append(var)

        builtin_redeclaration_qualifiers = self.qualifier_only_builtin_qualifiers(node)
        self.apply_builtin_redeclaration_qualifiers(
            [*self.inputs, *self.outputs], builtin_redeclaration_qualifiers
        )

        fragment_writes_depth = self.fragment_main_writes_name(node, "gl_FragDepth")
        fragment_writes_sample_mask = self.fragment_main_writes_name(
            node, "gl_SampleMask"
        )

        # Ensure vertex stages include gl_Position
        if self.shader_type == "vertex":
            output_names = {
                var.name for var in self.outputs if isinstance(var, VariableNode)
            }
            if "gl_Position" not in output_names:
                builtin = VariableNode(
                    "vec4",
                    "gl_Position",
                    qualifiers=self.builtin_output_qualifiers(
                        "gl_Position", builtin_redeclaration_qualifiers
                    ),
                    semantic="gl_Position",
                )
                self.outputs.append(builtin)
                output_names.add("gl_Position")

            for name, vtype in self.VERTEX_BUILTIN_OUTPUT_TYPES.items():
                if (
                    name == "gl_Position"
                    or name not in builtin_redeclaration_qualifiers
                ):
                    continue
                if name in output_names:
                    continue
                self.outputs.append(
                    VariableNode(
                        vtype,
                        name,
                        qualifiers=self.builtin_output_qualifiers(
                            name, builtin_redeclaration_qualifiers
                        ),
                        semantic=name,
                    )
                )
                output_names.add(name)

        # Ensure fragment outputs include gl_FragColor if no outputs declared
        if (
            self.shader_type == "fragment"
            and not self.outputs
            and not fragment_writes_depth
            and not fragment_writes_sample_mask
        ):
            builtin = VariableNode(
                "vec4", "gl_FragColor", qualifiers=["out"], semantic="gl_FragColor"
            )
            self.outputs.append(builtin)
        fragment_uses_direct_outputs = self.fragment_uses_direct_output_declarations(
            fragment_writes_depth, fragment_writes_sample_mask
        )

        for uniform in node.uniforms:
            self.uniform_vars.append(uniform)

        result = ""
        preprocessor = getattr(node, "preprocessor", []) or []
        if preprocessor:
            for line in preprocessor:
                result += f"{line}\n"
            result += "\n"
        result += "shader main {\n"

        # Generate struct definitions
        for struct in node.structs:
            if struct.name in self.converted_ssbo_struct_names:
                continue
            if self.is_push_constant_interface_block_struct(struct):
                continue
            result += self.indent_str + self.generate_struct(struct) + "\n\n"

        # Generate input struct if needed
        if self.inputs and self.shader_type in (
            "vertex",
            "fragment",
        ):
            result += self.indent_str + f"struct {self.stage_struct_name()}Input {{\n"
            self.increase_indent()
            for input_var in self.inputs:
                result += self.indent() + self.generate_stage_struct_member(input_var)
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate output struct for vertex-like stages
        if self.outputs and self.shader_type == "vertex":
            result += self.indent_str + f"struct {self.stage_struct_name()}Output {{\n"
            self.increase_indent()
            for output_var in self.outputs:
                result += self.indent() + self.generate_stage_struct_member(output_var)
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate uniforms: split resource uniforms from constant data
        if self.uniform_vars:
            resource_uniforms = [
                u for u in self.uniform_vars if self._is_resource_type(u.vtype)
            ]
            data_uniforms = [
                u for u in self.uniform_vars if not self._is_resource_type(u.vtype)
            ]
            push_constant_blocks = {}
            ordinary_data_uniforms = []
            for uniform in data_uniforms:
                if self.is_push_constant_uniform(uniform):
                    block_name = self.push_constant_block_name(uniform)
                    push_constant_blocks.setdefault(block_name, []).append(uniform)
                else:
                    ordinary_data_uniforms.append(uniform)

            for uniform in resource_uniforms:
                var_type = self.convert_type(uniform.vtype)
                var_name = uniform.name
                attributes = self.image_resource_attribute_suffix(uniform)
                array_suffix = self.array_suffix(uniform)
                result += (
                    self.indent_str
                    + f"{var_type} {var_name}{array_suffix}{attributes};\n"
                )

            for block_name, uniforms in push_constant_blocks.items():
                result += self.indent_str + self.generate_push_constant_block(
                    block_name, uniforms
                )

            if ordinary_data_uniforms:
                result += self.indent_str + "cbuffer Uniforms {\n"
                self.increase_indent()
                for uniform in ordinary_data_uniforms:
                    var_type = self.convert_type(uniform.vtype)
                    var_name = uniform.name
                    array_suffix = self.array_suffix(uniform)
                    result += self.indent() + f"{var_type} {var_name}{array_suffix};\n"
                self.decrease_indent()
                result += self.indent_str + "};\n"

            result += "\n"

        # Generate global constants
        for const_var in getattr(node, "constant", []) or []:
            result += (
                self.indent_str
                + self.generate_variable_declaration(const_var, array_on_type=True)
                + ";\n"
            )
        if getattr(node, "constant", []):
            result += "\n"

        # Generate global variables
        for global_var in getattr(node, "global_variables", []) or []:
            if self.is_qualifier_only_builtin_declaration(global_var):
                continue
            if self.is_buffer_reference_forward_declaration(global_var):
                continue
            structured_buffer_decl = self.structured_buffer_declaration(global_var)
            if structured_buffer_decl is not None:
                result += self.indent_str + structured_buffer_decl + ";\n"
                continue
            ssbo_block_decl = self.ssbo_block_declaration(global_var)
            if ssbo_block_decl is not None:
                result += self.indent_str + ssbo_block_decl + ";\n"
                continue
            result += (
                self.indent_str + self.generate_variable_declaration(global_var) + ";\n"
            )
        if getattr(node, "global_variables", []):
            result += "\n"

        if fragment_uses_direct_outputs and self.outputs:
            for output_var in self.outputs:
                result += (
                    self.indent_str
                    + self.generate_variable_declaration(output_var)
                    + ";\n"
                )
            result += "\n"

        if self.shader_type in self.NON_STRUCT_STAGE_TYPES and (
            self.inputs or self.outputs
        ):
            for interface_var in [*self.inputs, *self.outputs]:
                if self.is_graphics_interface_block_variable(interface_var):
                    continue
                result += (
                    self.indent_str
                    + self.generate_variable_declaration(interface_var)
                    + ";\n"
                )
            result += "\n"

        # Generate shader function
        result += self.indent_str + f"{self.shader_type} {{\n"

        layouts = getattr(node, "layouts", []) or []
        if layouts:
            self.increase_indent()
            for layout in layouts:
                result += self.indent() + f"{self.format_layout(layout)};\n"
            result += "\n"
            self.decrease_indent()

        main_function = None
        other_functions = []

        for function in node.functions:
            if function.name == "main":
                main_function = function
            else:
                other_functions.append(function)

        # Generate auxiliary functions first
        for function in other_functions:
            self.increase_indent()
            result += self.indent() + self.generate_function(function) + "\n\n"
            self.decrease_indent()

        # Generate the main function if it exists
        if main_function:
            self.increase_indent()
            input_parameter = (
                f"{self.stage_struct_name()}Input input" if self.inputs else ""
            )

            # Determine function signature based on shader type
            if self.shader_type == "vertex":
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({input_parameter})"
                )
            elif self.shader_type == "fragment":
                if fragment_uses_direct_outputs:
                    result += self.indent() + f"void main({input_parameter})"
                else:
                    output_type = "vec4"
                    output_name = "gl_FragColor"
                    output_attributes = ""
                    if self.outputs:
                        output_var = self.outputs[0]
                        output_type = self.convert_type(output_var.vtype)
                        output_name = output_var.name
                        output_attributes = self.fragment_return_attribute_suffix(
                            output_var
                        )
                    result += (
                        self.indent()
                        + f"{output_type} main({input_parameter})"
                        + f"{output_attributes} @ {output_name}"
                    )
            elif self.shader_type in self.NON_STRUCT_STAGE_TYPES:
                result += self.indent() + "void main()"
            else:
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({input_parameter})"
                )

            result += " {\n"

            self.increase_indent()

            # For vertex shaders, create the output struct
            if self.shader_type == "vertex":
                result += self.indent() + f"{self.stage_struct_name()}Output output;\n"

            # For fragment shaders, declare a local output if assignments are used
            if (
                self.shader_type == "fragment"
                and self.outputs
                and not fragment_uses_direct_outputs
            ):
                output_type = self.convert_type(self.outputs[0].vtype)
                output_name = self.outputs[0].name
                result += self.indent() + f"{output_type} {output_name};\n"

            # Generate statements for the main function
            for statement in self.generate_statement_sequence(main_function.body):
                result += self.indent() + statement + "\n"

            if self.shader_type in ("vertex",) and not any(
                isinstance(stmt, ReturnNode) for stmt in main_function.body
            ):
                result += self.indent() + "return output;\n"

            if (
                self.shader_type == "fragment"
                and not fragment_uses_direct_outputs
                and not any(isinstance(stmt, ReturnNode) for stmt in main_function.body)
            ):
                output_name = self.outputs[0].name if self.outputs else "gl_FragColor"
                result += self.indent() + f"return {output_name};\n"

            self.decrease_indent()
            result += self.indent() + "}\n"
            self.decrease_indent()

        result += self.indent_str + "}\n"

        result += "}\n"

        return result

    def generate_struct(self, node):
        result = f"{self.interface_block_attribute_prefix(node)}struct {node.name} {{\n"

        self.increase_indent()
        members = getattr(node, "members", None) or getattr(node, "fields", [])
        for field in members:
            qualifier_prefix = ""
            if isinstance(field, dict):
                var_type = self.convert_type(field.get("type"))
                var_name = field.get("name")
                semantic = ""
                array_suffix = ""
            else:
                var_type = self.convert_type(getattr(field, "vtype", ""))
                var_name = getattr(field, "name", "")
                semantic = ""
                if getattr(field, "semantic", None):
                    semantic = f" @ {field.semantic}"
                array_suffix = self.array_suffix(field)
                qualifier_prefix = ""
                if self.is_graphics_interface_block_struct(node):
                    qualifier_prefix = self.interface_member_qualifier_prefix(field)
                    if qualifier_prefix:
                        qualifier_prefix += " "
                    semantic += self.interface_member_layout_attribute_suffix(field)
                    semantic += self.variable_qualifier_attribute_suffix(field)
            result += (
                self.indent()
                + f"{qualifier_prefix}{var_type} {var_name}{array_suffix}{semantic};\n"
            )
        self.decrease_indent()

        result += self.indent() + "};"
        return result

    def array_suffix(self, node):
        array_sizes = getattr(node, "array_sizes", None)
        if array_sizes:
            return "".join(
                f"[{self.generate_expression(size)}]" if size is not None else "[]"
                for size in array_sizes
            )
        if getattr(node, "array_size", None) is not None:
            return f"[{self.generate_expression(node.array_size)}]"
        if getattr(node, "is_array", False):
            return "[]"
        return ""

    def generate_function_parameter(self, param):
        if isinstance(param, tuple):  # (type, name)
            param_type, param_name = param
            return f"{self.convert_type(param_type)} {param_name}"
        if isinstance(param, VariableNode):
            declaration = self.generate_variable_declaration(
                param,
                array_before_attributes=True,
                memory_qualifier_prefix=False,
            )
            default_value = getattr(param, "default_value", None)
            if default_value is not None:
                declaration += f" = {self.generate_expression(default_value)}"
            return declaration
        return None

    def generate_function(self, node):
        """Render one GLSL function node as a CrossGL function block."""
        self.push_variable_type_scope()
        for param in node.params:
            self.register_parameter_type(param)

        try:
            return_type = self.convert_type(node.return_type)
            name = self.format_function_name(node.name)

            params = []
            for param in node.params:
                param_decl = self.generate_function_parameter(param)
                if param_decl is not None:
                    params.append(param_decl)

            params_str = ", ".join(params)

            result = f"{return_type} {name}({params_str}) {{\n"

            self.increase_indent()
            for statement in self.generate_statement_sequence(node.body):
                result += self.indent() + statement + "\n"
            self.decrease_indent()

            result += self.indent() + "}"
            return result
        finally:
            self.pop_variable_type_scope()

    def generate_statement_sequence(self, statements):
        generated = []
        index = 0
        while index < len(statements):
            folded = self.generate_task_payload_dispatch_statement(statements, index)
            if folded is not None:
                generated.append(folded)
                index += 2
                continue
            generated.append(self.generate_statement(statements[index]))
            index += 1
        return generated

    def generate_task_payload_dispatch_statement(self, statements, index):
        if self.shader_type not in {"task", "amplification", "object"}:
            return None
        if index + 1 >= len(statements):
            return None

        assignment = statements[index]
        dispatch = statements[index + 1]
        payload_expr = self.task_payload_assignment_expression(assignment)
        if payload_expr is None:
            return None
        if not self.is_emit_mesh_tasks_call(dispatch):
            return None

        dispatch_args = [
            self.generate_expression(arg) for arg in getattr(dispatch, "args", [])[:3]
        ]
        payload = self.generate_expression(payload_expr)
        return f"DispatchMesh({', '.join(dispatch_args + [payload])});"

    def task_payload_assignment_expression(self, statement):
        if not isinstance(statement, AssignmentNode):
            return None
        if getattr(statement, "operator", "=") != "=":
            return None
        if not self.task_payload_shared_names:
            return None

        target = self.expression_base_name(getattr(statement, "left", None))
        if target not in self.task_payload_shared_names:
            return None
        return getattr(statement, "right", None)

    def is_emit_mesh_tasks_call(self, statement):
        if not isinstance(statement, FunctionCallNode):
            return False
        if len(getattr(statement, "args", []) or []) != 3:
            return False
        return self.function_call_name(statement) == "EmitMeshTasksEXT"

    def function_call_name(self, node):
        name = getattr(node, "name", None)
        if isinstance(name, VariableNode):
            return name.name
        if isinstance(name, MemberAccessNode):
            return self.generate_member_access(name)
        return name

    def generate_statement(self, node):
        """Render a GLSL statement node as CrossGL source."""
        if isinstance(node, AssignmentNode):
            return self.generate_assignment(node) + ";"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, DoWhileNode):
            return self.generate_do_while(node)
        elif isinstance(node, ReturnNode):
            return self.generate_return(node) + ";"
        elif isinstance(node, VariableNode):
            ray_control = self.ray_control_statement(node)
            if ray_control is not None:
                return ray_control + ";"
            self.register_variable_type(node)
            return self.generate_variable_declaration(node) + ";"
        elif isinstance(node, StructNode):
            return self.generate_struct(node)
        elif isinstance(node, FunctionCallNode):
            return self.generate_function_call(node) + ";"
        elif isinstance(node, SwitchNode):
            return self.generate_switch_statement(node)
        elif isinstance(node, BlockNode):
            return self.generate_block(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        elif isinstance(node, DiscardNode):
            return "discard;"
        elif isinstance(node, PostfixOpNode):
            return self.generate_expression(node) + ";"
        else:
            return self.generate_expression(node) + ";"

    def generate_assignment(self, node):
        if hasattr(node, "left") and hasattr(node, "right"):
            lhs = node.left
            rhs = node.right
            op = getattr(node, "operator", "=")

            if isinstance(lhs, VariableNode) and lhs.vtype:
                var_type = self.convert_type(lhs.vtype)
                var_name = lhs.name
                value = self.generate_expression(rhs)
                return f"{var_type} {var_name} {op} {value}"

            structured_length = self.structured_buffer_length_call(rhs)
            if structured_length is not None and op == "=":
                target = self.generate_expression(lhs)
                return f"buffer_dimensions({structured_length}, {target})"

            structured_access = self.structured_buffer_access_parts(lhs)
            if structured_access is not None:
                buffer_expr, index_expr = structured_access
                value = self.generate_expression(rhs)
                if op != "=":
                    compound_ops = {
                        "+=": "+",
                        "-=": "-",
                        "*=": "*",
                        "/=": "/",
                        "%=": "%",
                        "&=": "&",
                        "|=": "|",
                        "^=": "^",
                        "<<=": "<<",
                        ">>=": ">>",
                    }
                    binary_op = compound_ops.get(op)
                    if binary_op is None:
                        return None
                    current = f"buffer_load({buffer_expr}, {index_expr})"
                    value = f"{current} {binary_op} {value}"
                return f"buffer_store({buffer_expr}, {index_expr}, {value})"

            left_expr = self.generate_expression(lhs)
            right_expr = self.generate_expression(rhs)
            return f"{left_expr} {op} {right_expr}"

        return self.generate_expression(node)

    def is_condition_declaration(self, node):
        return (
            isinstance(node, VariableNode)
            and bool(getattr(node, "vtype", None))
            and getattr(node, "value", None) is not None
        )

    def generate_condition_expression(self, node):
        if self.is_condition_declaration(node):
            return node.name
        return self.generate_expression(node)

    def ray_control_statement(self, node):
        if getattr(node, "vtype", None):
            return None

        name = getattr(node, "name", None)
        mapped_name = self.function_map.get(name)
        if mapped_name not in {"AcceptHitAndEndSearch", "IgnoreHit"}:
            return None

        return f"{mapped_name}()"

    def generate_if(self, node):
        condition_node = getattr(node, "condition", None)
        if condition_node is None:
            condition_node = getattr(node, "if_condition", None)
        wraps_condition_declaration = self.is_condition_declaration(condition_node)
        result = ""
        if wraps_condition_declaration:
            self.register_variable_type(condition_node)
            result += "{\n"
            self.increase_indent()
            result += (
                self.indent()
                + self.generate_variable_declaration(condition_node)
                + ";\n"
            )

        condition = self.generate_condition_expression(condition_node)
        if_prefix = self.indent() if wraps_condition_declaration else ""
        result += if_prefix + f"if ({condition}) {{\n"

        # Generate if body
        self.increase_indent()
        for statement in getattr(node, "if_body", []) or []:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()

        result += self.indent() + "}"

        else_if_chain = []
        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            else_if_chain = list(zip(node.else_if_conditions, node.else_if_bodies))
        elif hasattr(node, "else_if_chain"):
            else_if_chain = node.else_if_chain

        for elif_condition, elif_body in else_if_chain:
            elif_cond = self.generate_expression(elif_condition)
            result += f" else if ({elif_cond}) {{\n"

            self.increase_indent()
            for statement in elif_body:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

            result += self.indent() + "}"

        if node.else_body:
            result += " else {\n"

            self.increase_indent()
            for statement in node.else_body:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

            result += self.indent() + "}"

        if wraps_condition_declaration:
            result += "\n"
            self.decrease_indent()
            result += self.indent() + "}"

        return result

    def generate_for(self, node):
        init = self.generate_for_clause(node.init)
        condition_node = node.condition
        wraps_condition_declaration = self.is_condition_declaration(condition_node)
        condition = (
            ""
            if wraps_condition_declaration
            else self.generate_expression(condition_node) if condition_node else ""
        )
        update_node = getattr(node, "update", None) or getattr(node, "iteration", None)
        if isinstance(update_node, list):
            iteration = ", ".join(
                self.generate_statement(update_part).rstrip(";")
                for update_part in update_node
            )
        else:
            iteration = (
                self.generate_statement(update_node).rstrip(";") if update_node else ""
            )

        result = f"for ({init}; {condition}; {iteration}) {{\n"
        self.increase_indent()
        if wraps_condition_declaration:
            self.register_variable_type(condition_node)
            result += (
                self.indent()
                + self.generate_variable_declaration(condition_node)
                + ";\n"
            )
            result += self.indent() + f"if (!{condition_node.name}) {{\n"
            self.increase_indent()
            result += self.indent() + "break;\n"
            self.decrease_indent()
            result += self.indent() + "}\n"
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_for_clause(self, node):
        if node is None:
            return ""
        if isinstance(node, list):
            return self.generate_for_declaration_list(node)
        return self.generate_statement(node).rstrip(";")

    def generate_for_declaration_list(self, declarations):
        declarations = [decl for decl in declarations if decl is not None]
        if not declarations:
            return ""
        if not all(isinstance(decl, VariableNode) for decl in declarations):
            return ", ".join(self.generate_for_clause(decl) for decl in declarations)

        for decl in declarations:
            self.register_variable_type(decl)

        first = declarations[0]
        first_type = self.variable_declaration_type(first)
        first_qualifiers = list(getattr(first, "qualifiers", None) or [])
        if all(
            self.variable_declaration_type(decl) == first_type
            and list(getattr(decl, "qualifiers", None) or []) == first_qualifiers
            for decl in declarations
        ):
            prefix = self.for_declaration_prefix(first, first_type)
            declarators = []
            for decl in declarations:
                declarator = f"{decl.name}{self.array_suffix(decl)}"
                if getattr(decl, "value", None) is not None:
                    declarator += f" = {self.generate_expression(decl.value)}"
                declarators.append(declarator)
            return prefix + ", ".join(declarators)

        return ", ".join(
            self.generate_variable_declaration(decl) for decl in declarations
        )

    def for_declaration_prefix(self, node, var_type):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        prefix_parts = []
        if "shared" in qualifiers:
            prefix_parts.append("shared")
        if getattr(node, "is_const", False) or "const" in qualifiers:
            prefix_parts.append("const")
        interface_prefix = self.interface_qualifier_prefix(node)
        if interface_prefix:
            prefix_parts.append(interface_prefix)
        return f"{' '.join(prefix_parts + [var_type])} "

    def generate_while(self, node):
        if self.is_condition_declaration(node.condition):
            self.register_variable_type(node.condition)
            condition_name = node.condition.name
            result = "while (true) {\n"
            self.increase_indent()
            result += (
                self.indent()
                + self.generate_variable_declaration(node.condition)
                + ";\n"
            )
            result += self.indent() + f"if (!{condition_name}) {{\n"
            self.increase_indent()
            result += self.indent() + "break;\n"
            self.decrease_indent()
            result += self.indent() + "}\n"
            for statement in node.body:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()
            result += self.indent() + "}"
            return result

        condition = self.generate_expression(node.condition)
        result = f"while ({condition}) {{\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.generate_expression(node.condition)
        result = "do {\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + f"}} while ({condition});"
        return result

    def generate_block(self, node):
        result = "{\n"
        self.increase_indent()
        for statement in node.statements:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_return(self, node):
        if node.value is None:
            return "return"
        return f"return {self.generate_expression(node.value)}"

    def generate_expression(self, node):
        """Render an OpenGL backend expression node as CrossGL syntax."""
        if node is None:
            return ""

        if isinstance(node, str):
            return node
        elif isinstance(node, NumberNode):
            return self.normalize_number_literal(node.value)
        elif isinstance(node, (int, float)):
            return str(node)
        elif isinstance(node, VariableNode):
            if self.shader_type in (
                "vertex",
                "fragment",
            ) and any(var.name == node.name for var in self.inputs):
                return f"input.{node.name}"
            if self.shader_type in ("vertex",) and any(
                var.name == node.name for var in self.outputs
            ):
                return f"output.{node.name}"
            return node.name
        elif isinstance(node, BinaryOpNode):
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            operator = self.operator_map.get(node.op, node.op)
            if operator == ",":
                return f"({left}, {right})"
            return f"({left} {operator} {right})"
        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand)
            operator = self.operator_map.get(node.op, node.op)
            return f"({operator}{operand})"
        elif isinstance(node, PostfixOpNode):
            operand = self.generate_expression(node.operand)
            return f"({operand}{node.op})"
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        elif isinstance(node, FunctionCallNode):
            return self.generate_function_call(node)
        elif isinstance(node, MemberAccessNode):
            return self.generate_member_access(node)
        elif isinstance(node, ArrayAccessNode):
            return self.generate_array_access(node)
        elif isinstance(node, TernaryOpNode):
            condition = self.generate_expression(node.condition)
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, VectorConstructorNode):
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{self.convert_type(node.type_name)}({args})"
        elif isinstance(node, InitializerListNode):
            elements = ", ".join(
                self.generate_expression(element) for element in node.elements
            )
            return f"{{ {elements} }}"
        else:
            return str(node)

    def normalize_number_literal(self, value):
        text = str(value)
        double_float_match = DOUBLE_FLOAT_LITERAL_SUFFIX_RE.match(text)
        if double_float_match:
            return double_float_match.group("body")
        match = SHORT_INTEGER_LITERAL_SUFFIX_RE.match(text)
        if not match:
            return text
        suffix = match.group("suffix")
        if suffix[0].lower() == "u":
            return f"{match.group('body')}u"
        return match.group("body")

    def generate_function_call(self, node):
        structured_length = self.structured_buffer_length_call(node)
        if structured_length is not None:
            return f"buffer_dimensions({structured_length})"

        name = node.name
        if isinstance(name, MemberAccessNode):
            name = self.generate_member_access(name)
        elif isinstance(name, VariableNode):
            name = name.name

        ray_query_call = self.generate_ray_query_method_call(name, node.args)
        if ray_query_call is not None:
            return ray_query_call

        name = self.shadow_gather_import_name(name, node.args)

        if name in [
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
        ]:
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{self.convert_type(name)}({args})"

        if name in self.structs_by_name:
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{name}{{{args}}}"

        descriptor = self.resource_function_descriptor(name)
        mapped_name = (
            descriptor["function"]
            if descriptor is not None
            else self.function_map.get(name, self.format_function_name(name))
        )

        args = ", ".join(self.generate_expression(arg) for arg in node.args)

        return f"{mapped_name}({args})"

    def generate_ray_query_method_call(self, name, args):
        if name in self.RAY_QUERY_SIMPLE_METHODS and args:
            query = self.generate_expression(args[0])
            method = self.RAY_QUERY_SIMPLE_METHODS[name]
            method_args = ", ".join(self.generate_expression(arg) for arg in args[1:])
            return f"{query}.{method}({method_args})"

        if name not in self.RAY_QUERY_COMMITTED_METHODS or len(args) < 2:
            return None

        committed = self.ray_query_committed_argument(args[1])
        if committed is None:
            return None

        query = self.generate_expression(args[0])
        candidate_method, committed_method = self.RAY_QUERY_COMMITTED_METHODS[name]
        method = committed_method if committed else candidate_method
        method_args = ", ".join(self.generate_expression(arg) for arg in args[2:])
        return f"{query}.{method}({method_args})"

    def ray_query_committed_argument(self, arg):
        if isinstance(arg, str):
            value = arg.lower()
        elif isinstance(arg, VariableNode):
            value = str(arg.name).lower()
        elif isinstance(arg, NumberNode):
            value = str(arg.value).lower()
        else:
            value = str(arg).lower()

        if value == "true":
            return True
        if value == "false":
            return False
        return None

    def generate_member_access(self, node):
        flattened_member = self.flattened_uniform_block_member(node)
        if flattened_member is not None:
            return flattened_member

        object_name = ""
        if isinstance(node.object, VariableNode):
            if self.shader_type in (
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and any(var.name == node.object.name for var in self.inputs):
                object_name = f"input.{node.object.name}"
            elif self.shader_type in ("vertex",) and any(
                var.name == node.object.name for var in self.outputs
            ):
                object_name = f"output.{node.object.name}"
            else:
                object_name = node.object.name
        else:
            object_name = self.generate_expression(node.object)

        return f"{object_name}.{node.member}"

    def flattened_uniform_block_member(self, node):
        instance_name = self.expression_base_name(getattr(node, "object", None))
        if instance_name is None:
            return None
        fields = self.flattened_uniform_block_instances.get(instance_name)
        if fields is None or node.member not in fields:
            return None
        return node.member

    def generate_array_access(self, node):
        structured_access = self.structured_buffer_access_parts(node)
        if structured_access is not None:
            buffer_expr, index_expr = structured_access
            return f"buffer_load({buffer_expr}, {index_expr})"

        array = self.generate_expression(node.array)
        index = self.generate_expression(node.index)
        return f"{array}[{index}]"

    def structured_buffer_access_parts(self, node):
        if not isinstance(node, ArrayAccessNode):
            return None

        index = self.generate_expression(node.index)
        array_expr = node.array

        if (
            isinstance(array_expr, VariableNode)
            and array_expr.name in self.structured_buffer_names
        ):
            return array_expr.name, index

        if isinstance(array_expr, MemberAccessNode):
            base_name = self.expression_base_name(array_expr.object)
            if (
                base_name
                and (base_name, array_expr.member)
                in self.structured_buffer_instance_members
            ):
                return (
                    self.generate_buffer_receiver_expression(array_expr.object),
                    index,
                )

        return None

    def structured_buffer_length_call(self, node):
        if not isinstance(node, FunctionCallNode) or getattr(node, "args", None):
            return None
        if not isinstance(node.name, MemberAccessNode) or node.name.member != "length":
            return None

        target = node.name.object
        if (
            isinstance(target, VariableNode)
            and target.name in self.structured_buffer_names
        ):
            return target.name
        if isinstance(target, MemberAccessNode):
            base_name = self.expression_base_name(target.object)
            if (
                base_name
                and (base_name, target.member)
                in self.structured_buffer_instance_members
            ):
                return self.generate_buffer_receiver_expression(target.object)
        return None

    def generate_buffer_receiver_expression(self, node):
        if isinstance(node, ArrayAccessNode):
            array = self.generate_buffer_receiver_expression(node.array)
            index = self.generate_expression(node.index)
            return f"{array}[{index}]"
        return self.generate_expression(node)

    def expression_base_name(self, node):
        if isinstance(node, str):
            return node
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, ArrayAccessNode):
            return self.expression_base_name(node.array)
        if isinstance(node, MemberAccessNode):
            return self.expression_base_name(node.object)
        return None

    def convert_type(self, type_name):
        return self.type_map.get(type_name, type_name)

    def variable_declaration_type(self, node):
        var_type = self.convert_type(node.vtype)
        if var_type == "mat4x3" and self.is_ray_query_transform_call(
            getattr(node, "value", None)
        ):
            return "mat3x4"
        return var_type

    def is_ray_query_transform_call(self, node):
        if not isinstance(node, FunctionCallNode):
            return False
        name = node.name
        if isinstance(name, VariableNode):
            name = name.name
        return name in self.RAY_QUERY_TRANSFORM_FUNCTIONS

    def generate_variable_declaration(
        self,
        node,
        array_before_attributes=False,
        array_on_type=False,
        memory_qualifier_prefix=True,
    ):
        var_type = self.variable_declaration_type(node)
        var_name = node.name
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        prefix_parts = []
        if "shared" in qualifiers:
            prefix_parts.append("shared")
        if getattr(node, "is_const", False) or "const" in qualifiers:
            prefix_parts.append("const")
        if (
            memory_qualifier_prefix
            and not self._is_resource_type(var_type)
            and "buffer" not in qualifiers
        ):
            for qualifier in (
                "coherent",
                "volatile",
                "restrict",
                "readonly",
                "writeonly",
            ):
                if qualifier in qualifiers:
                    prefix_parts.append(qualifier)
        interface_prefix = self.interface_qualifier_prefix(node)
        if interface_prefix:
            prefix_parts.append(interface_prefix)
        prefix = f"{' '.join(prefix_parts)} " if prefix_parts else ""
        array_suffix = self.array_suffix(node)
        if array_on_type and array_suffix:
            var_type = f"{var_type}{array_suffix}"
            array_suffix = ""
        attributes = (
            self.variable_layout_attribute_suffix(node)
            + self.image_resource_attribute_suffix(node)
            + self.variable_qualifier_attribute_suffix(node)
        )
        declarator = (
            f"{var_name}{array_suffix}{attributes}"
            if array_before_attributes
            else f"{var_name}{attributes}{array_suffix}"
        )

        if getattr(node, "value", None) is not None:
            value = self.generate_expression(node.value)
            return f"{prefix}{var_type} {declarator} = {value}"

        return f"{prefix}{var_type} {declarator}"

    def generate_switch_statement(self, node):
        expression = self.generate_expression(node.expression)
        result = f"switch ({expression}) {{\n"

        for case in node.cases:
            case_value = self.generate_expression(case.value)
            result += self.indent() + f"case {case_value}:\n"

            self.increase_indent()
            for statement in case.statements:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        if node.default:
            result += self.indent() + "default:\n"

            self.increase_indent()
            for statement in node.default:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        result += self.indent() + "}"
        return result
