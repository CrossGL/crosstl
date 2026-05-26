"""Reverse code generator that emits CrossGL from GLSL AST nodes."""

from .OpenglAst import (
    ShaderNode,
    VariableNode,
    AssignmentNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    WhileNode,
    DoWhileNode,
    LayoutNode,
    VectorConstructorNode,
    MemberAccessNode,
    TernaryOpNode,
    ArrayAccessNode,
    SwitchNode,
    BlockNode,
    NumberNode,
    PostfixOpNode,
    BreakNode,
    ContinueNode,
    DiscardNode,
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
        "pervertex": "pervertex",
        "perview": "perview",
    }
    VARIABLE_QUALIFIER_ATTRIBUTES = {
        "invariant": "invariant",
        "precise": "precise",
        "lowp": "lowp",
        "mediump": "mediump",
        "highp": "highp",
    }
    LAYOUT_ATTRIBUTE_NAMES = (
        "location",
        "component",
        "index",
        "stream",
        "xfb_buffer",
        "xfb_offset",
        "xfb_stride",
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

    def __init__(self, shader_type="vertex"):
        """Initialize GLSL-to-CrossGL mappings for a shader stage."""
        self.shader_type = shader_type
        self.indent_level = 0
        self.indent_str = "    "

        # Mapping of GLSL built-in functions to CrossGL equivalents
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
        }
        self.texture_function_operations = {
            "texture": "sample",
            "texture2D": "sample",
            "textureCube": "sample",
            "textureLod": "sample_lod",
            "textureLodOffset": "sample_lod",
            "textureGrad": "sample_grad",
            "textureGradOffset": "sample_grad",
            "textureProj": "sample_projected",
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
            "texture2D": "texture",
            "textureCube": "texture",
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
            ("sampler", "isampler", "usampler", "image", "iimage", "uimage")
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

    def _is_image_resource_type(self, type_name):
        if not type_name:
            return False
        return str(type_name).startswith(("image", "iimage", "uimage"))

    def _is_buffer_qualified(self, var):
        return "buffer" in self._qualifier_set(var)

    def ssbo_binding_attribute_suffix(self, var):
        layout = getattr(var, "layout", None) or {}
        binding = layout.get("binding")
        return f" @binding({binding})" if binding is not None else ""

    def ssbo_block_attribute_suffix(self, var):
        layout_names = self.ssbo_block_layout_names(var)
        attributes = [f"@glsl_buffer_block({', '.join(layout_names)})"]
        if self.is_shader_record_buffer_block(var):
            self.validate_shader_record_layout(var)

        binding = self.ssbo_binding(var)
        if binding is not None:
            attributes.append(f"@binding({binding})")

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
            attributes.append(f"@binding({binding})")

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

    def variable_layout_attribute_suffix(self, var):
        layout = getattr(var, "layout", None) or {}
        attributes = []
        for name in self.LAYOUT_ATTRIBUTE_NAMES:
            value = layout.get(name)
            if value is not None:
                attributes.append(f"@{name}({value})")
        return f" {' '.join(attributes)}" if attributes else ""

    def storage_qualifier_attributes(self, var):
        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        return [
            f"@{attribute}"
            for qualifier, attribute in self.STORAGE_QUALIFIER_ATTRIBUTES.items()
            if qualifier in qualifiers
        ]

    def variable_qualifier_attribute_suffix(self, var):
        qualifiers = {str(q).lower() for q in getattr(var, "qualifiers", []) or []}
        attributes = [
            f"@{attribute}"
            for qualifier, attribute in self.VARIABLE_QUALIFIER_ATTRIBUTES.items()
            if qualifier in qualifiers
        ]
        return f" {' '.join(attributes)}" if attributes else ""

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

    def fragment_uses_direct_output_declarations(self):
        return self.shader_type == "fragment" and len(self.outputs) > 1

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
                parts.append(f"{key} = {value}")
        layout_str = f"layout({', '.join(parts)})" if parts else "layout()"
        if qualifiers:
            layout_str += " " + " ".join(qualifiers)
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
                attributes.append(f"@{key}({value})")

        instance_name = getattr(node, "interface_instance_name", None)
        if instance_name:
            attributes.append(f"@glsl_interface_instance({instance_name})")
            if getattr(node, "interface_instance_is_array", False):
                array_size = getattr(node, "interface_array_size", None)
                if array_size is None:
                    attributes.append("@glsl_interface_array")
                else:
                    attributes.append(
                        f"@glsl_interface_array({self.generate_expression(array_size)})"
                    )

        return f"{' '.join(attributes)} " if attributes else ""

    def generate(self, ast):
        """Generate a complete CrossGL shader from a parsed GLSL AST."""
        if ast is None:
            return "// Empty shader"

        if not isinstance(ast, ShaderNode):
            return f"// Unexpected AST node type: {type(ast)}"

        return self.generate_shader(ast)

    def generate_shader(self, node):
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []
        self.prepare_structured_buffers(node)

        for var in node.io_variables:
            if isinstance(var, (LayoutNode, VariableNode)):
                if self._is_input_var(var):
                    self.inputs.append(var)
                if self._is_output_var(var):
                    self.outputs.append(var)

        # Ensure vertex stages include gl_Position
        if self.shader_type == "vertex":
            has_position = any(
                isinstance(var, VariableNode) and var.name == "gl_Position"
                for var in self.outputs
            )
            if not has_position:
                builtin = VariableNode(
                    "vec4", "gl_Position", qualifiers=["out"], semantic="gl_Position"
                )
                self.outputs.append(builtin)

        # Ensure fragment outputs include gl_FragColor if no outputs declared
        if self.shader_type == "fragment" and not self.outputs:
            builtin = VariableNode(
                "vec4", "gl_FragColor", qualifiers=["out"], semantic="gl_FragColor"
            )
            self.outputs.append(builtin)

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

            for uniform in resource_uniforms:
                var_type = self.convert_type(uniform.vtype)
                var_name = uniform.name
                attributes = self.image_resource_attribute_suffix(uniform)
                array_suffix = ""
                if getattr(uniform, "array_size", None) is not None:
                    array_suffix = f"[{self.generate_expression(uniform.array_size)}]"
                result += (
                    self.indent_str
                    + f"{var_type} {var_name}{array_suffix}{attributes};\n"
                )

            if data_uniforms:
                result += self.indent_str + "cbuffer Uniforms {\n"
                self.increase_indent()
                for uniform in data_uniforms:
                    var_type = self.convert_type(uniform.vtype)
                    var_name = uniform.name
                    array_suffix = ""
                    if getattr(uniform, "array_size", None) is not None:
                        array_suffix = (
                            f"[{self.generate_expression(uniform.array_size)}]"
                        )
                    result += self.indent() + f"{var_type} {var_name}{array_suffix};\n"
                self.decrease_indent()
                result += self.indent_str + "};\n"

            result += "\n"

        # Generate global constants
        for const_var in getattr(node, "constant", []) or []:
            result += (
                self.indent_str + self.generate_variable_declaration(const_var) + ";\n"
            )
        if getattr(node, "constant", []):
            result += "\n"

        # Generate global variables
        for global_var in getattr(node, "global_variables", []) or []:
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

        if self.fragment_uses_direct_output_declarations():
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

            # Determine function signature based on shader type
            if self.shader_type == "vertex":
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({self.stage_struct_name()}Input input)"
                )
            elif self.shader_type == "fragment":
                if self.fragment_uses_direct_output_declarations():
                    result += (
                        self.indent()
                        + f"void main({self.stage_struct_name()}Input input)"
                    )
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
                        + f"{output_type} main({self.stage_struct_name()}Input input)"
                        + f"{output_attributes} @ {output_name}"
                    )
            elif self.shader_type in self.NON_STRUCT_STAGE_TYPES:
                result += self.indent() + "void main()"
            else:
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({self.stage_struct_name()}Input input)"
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
                and not self.fragment_uses_direct_output_declarations()
            ):
                output_type = self.convert_type(self.outputs[0].vtype)
                output_name = self.outputs[0].name
                result += self.indent() + f"{output_type} {output_name};\n"

            # Generate statements for the main function
            for statement in main_function.body:
                result += self.indent() + self.generate_statement(statement) + "\n"

            # Add implicit return for stages with output struct if not present
            if self.shader_type in ("vertex",) and not any(
                isinstance(stmt, ReturnNode) for stmt in main_function.body
            ):
                result += self.indent() + "return output;\n"

            # Add implicit return for fragment shaders if not present
            if (
                self.shader_type == "fragment"
                and not self.fragment_uses_direct_output_declarations()
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
                    semantic += self.variable_qualifier_attribute_suffix(field)
            result += (
                self.indent()
                + f"{qualifier_prefix}{var_type} {var_name}{array_suffix}{semantic};\n"
            )
        self.decrease_indent()

        result += self.indent() + "};"
        return result

    def array_suffix(self, node):
        if getattr(node, "array_size", None) is not None:
            return f"[{self.generate_expression(node.array_size)}]"
        if getattr(node, "is_array", False):
            return "[]"
        return ""

    def generate_function(self, node):
        """Render one GLSL function node as a CrossGL function block."""
        return_type = self.convert_type(node.return_type)
        name = node.name

        params = []
        for param in node.params:
            if isinstance(param, tuple):  # (type, name)
                param_type, param_name = param
                params.append(f"{self.convert_type(param_type)} {param_name}")
            elif isinstance(param, VariableNode):
                params.append(f"{self.convert_type(param.vtype)} {param.name}")

        params_str = ", ".join(params)

        result = f"{return_type} {name}({params_str}) {{\n"

        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()

        result += self.indent() + "}"
        return result

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
            return self.generate_variable_declaration(node) + ";"
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
        condition = self.generate_expression(condition_node)
        result = f"if ({condition}) {{\n"

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

        return result

    def generate_for(self, node):
        init = self.generate_statement(node.init).rstrip(";") if node.init else ""
        condition = self.generate_expression(node.condition) if node.condition else ""
        update_node = getattr(node, "update", None) or getattr(node, "iteration", None)
        iteration = (
            self.generate_statement(update_node).rstrip(";") if update_node else ""
        )

        result = f"for ({init}; {condition}; {iteration}) {{\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_while(self, node):
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
        result = "while (true) {\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        result += self.indent() + f"if (!({condition})) {{\n"
        self.increase_indent()
        result += self.indent() + "break;\n"
        self.decrease_indent()
        result += self.indent() + "}\n"
        self.decrease_indent()
        result += self.indent() + "}"
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
            return str(node.value)
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
        else:
            return str(node)

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

        descriptor = self.resource_function_descriptor(name)
        mapped_name = (
            descriptor["function"]
            if descriptor is not None
            else self.function_map.get(name, name)
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

    def generate_array_access(self, node):
        """Generate CrossGL code for an array access expression

        Args:
            node: ArrayAccessNode representing a GLSL array access

        Returns:
            str: The CrossGL array access expression
        """
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
        """Convert a GLSL type to its CrossGL equivalent

        Args:
            type_name: The GLSL type name

        Returns:
            str: The equivalent CrossGL type name
        """
        return self.type_map.get(type_name, type_name)

    def generate_variable_declaration(self, node):
        """Generate CrossGL code for a variable declaration

        Args:
            node: VariableNode representing a GLSL variable declaration

        Returns:
            str: The CrossGL variable declaration
        """
        var_type = self.convert_type(node.vtype)
        var_name = node.name
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        prefix_parts = []
        if getattr(node, "is_const", False) or "const" in qualifiers:
            prefix_parts.append("const")
        interface_prefix = self.interface_qualifier_prefix(node)
        if interface_prefix:
            prefix_parts.append(interface_prefix)
        prefix = f"{' '.join(prefix_parts)} " if prefix_parts else ""
        array_suffix = self.array_suffix(node)
        attributes = (
            self.variable_layout_attribute_suffix(node)
            + self.image_resource_attribute_suffix(node)
            + self.variable_qualifier_attribute_suffix(node)
        )

        if getattr(node, "value", None) is not None:
            value = self.generate_expression(node.value)
            return f"{prefix}{var_type} {var_name}{attributes}{array_suffix} = {value}"

        return f"{prefix}{var_type} {var_name}{attributes}{array_suffix}"

    def generate_switch_statement(self, node):
        """Generate CrossGL code for a switch statement

        Args:
            node: SwitchNode representing a GLSL switch statement

        Returns:
            str: The CrossGL switch statement
        """
        expression = self.generate_expression(node.expression)
        result = f"switch ({expression}) {{\n"

        # Generate case statements
        for case in node.cases:
            case_value = self.generate_expression(case.value)
            result += self.indent() + f"case {case_value}:\n"

            self.increase_indent()
            for statement in case.statements:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        # Generate default case if present
        if node.default:
            result += self.indent() + "default:\n"

            self.increase_indent()
            for statement in node.default:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        result += self.indent() + "}"
        return result
