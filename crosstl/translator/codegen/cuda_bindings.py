"""Built-in function conversion and resource bindings of the CUDA generator."""

from ..refusals import CudaRefusal
from .array_utils import split_array_type_suffix
from .c_narrow_integer_vectors import generic_narrow_vector_spelling
from .c_wide_integer_vectors import wide_integer_vector_constructor
from .cuda_refusals import CudaLoweringRefused


class CudaBindingMixin:
    """Binding members of ``CudaCodeGen``."""

    def cuda_euclidean_norm_function_name(self, func_name, raw_args):
        """Name CrossGL ``norm``/``rnorm`` of an array for its element type.

        CrossGL spells both precisions without the ``f`` suffix. CUDA's ``norm``/``rnorm``
        take ``const double*`` and ``normf``/``rnormf`` take ``const float*``, so an array
        of any other element type, or of one the generator cannot resolve, is refused.
        """
        if (
            func_name not in {"norm", "rnorm"}
            or len(raw_args or []) != 2
            or self.is_user_defined_function(func_name)
        ):
            return None

        array_expr = raw_args[1]
        lvalue = self.strip_address_of_expression(array_expr)
        if lvalue is not array_expr:
            # ``&x`` points at ``x`` itself.
            element_type = self.type_name_string(self.expression_result_type(lvalue))
        else:
            array_type = self.type_name_string(self.expression_result_type(array_expr))
            base_type, array_suffix = split_array_type_suffix(array_type or "")
            pointer = self.cuda_indirect_type_info(array_type)
            if array_suffix.count("[") == 1:
                # A one-dimensional array decays to a pointer to its first element.
                element_type = base_type
            elif pointer is not None and pointer["kind"] == "pointer":
                element_type = pointer["pointee_type"]
            else:
                element_type = None

        component = self.scalar_component_type(element_type) if element_type else None
        if component == "float":
            return f"{func_name}f"
        if component == "double":
            return func_name
        elements = f"{element_type} elements" if element_type else "unresolved elements"
        raise CudaLoweringRefused(
            [
                f"Euclidean norm: {func_name} of {elements}: CUDA has normf and "
                "rnormf for float arrays and norm and rnorm for double arrays"
            ]
        )

    def convert_builtin_function(self, func_name):
        """Convert CrossGL built-in functions to CUDA equivalents"""
        func_name = generic_narrow_vector_spelling(func_name) or func_name
        wide_vector_constructor = wide_integer_vector_constructor(func_name)
        if wide_vector_constructor is not None:
            return wide_vector_constructor
        function_mapping = {
            # Math functions
            "sqrt": "sqrtf",
            "pow": "powf",
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "log": "logf",
            "log2": "log2f",
            "exp": "expf",
            "exp2": "exp2f",
            "inversesqrt": "rsqrtf",
            "inverseSqrt": "rsqrtf",
            "rsqrt": "rsqrtf",
            "fma": "fmaf",
            "mad": "fmaf",
            "abs": "fabsf",
            "round": "roundf",
            "roundEven": "rintf",
            "trunc": "truncf",
            "mod": "fmodf",
            "mix": "lerp",
            "min": "fminf",
            "max": "fmaxf",
            "floor": "floorf",
            "ceil": "ceilf",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "f16vec3": "cgl_make_half3",
            "float16vec3": "cgl_make_half3",
            "half3": "cgl_make_half3",
            "vec3<f16>": "cgl_make_half3",
            "vec3<float16>": "cgl_make_half3",
            "vec3<half>": "cgl_make_half3",
            "f16vec4": "cgl_make_half4",
            "float16vec4": "cgl_make_half4",
            "half4": "cgl_make_half4",
            "vec4<f16>": "cgl_make_half4",
            "vec4<float16>": "cgl_make_half4",
            "vec4<half>": "cgl_make_half4",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "vec2<i8>": "make_char2",
            "vec3<i8>": "make_char3",
            "vec4<i8>": "make_char4",
            "vec2<u8>": "make_uchar2",
            "vec3<u8>": "make_uchar3",
            "vec4<u8>": "make_uchar4",
            "vec2<i16>": "make_short2",
            "vec3<i16>": "make_short3",
            "vec4<i16>": "make_short4",
            "vec2<u16>": "make_ushort2",
            "vec3<u16>": "make_ushort3",
            "vec4<u16>": "make_ushort4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            "atomicCompSwap": "atomicCAS",
            "atomicInc": "atomicInc",
            "atomicDec": "atomicDec",
            # Synchronization
            "barrier": "__syncthreads",
            "groupMemoryBarrier": "__threadfence_block",
            "memoryBarrier": "__threadfence",
            "memoryBarrierShared": "__threadfence_block",
            "memoryBarrierBuffer": "__threadfence",
            "memoryBarrierImage": "__threadfence",
            "allMemoryBarrier": "__threadfence",
            "deviceMemoryBarrier": "__threadfence",
            "workgroupBarrier": "__syncthreads",
            "workgroupExecutionBarrier": "__syncthreads",
            # Ray tracing placeholders
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
            "TraceRay": "cgl_trace_ray",
            "CallShader": "cgl_call_shader",
            "ReportHit": "cgl_report_hit",
            "IgnoreHit": "cgl_ignore_hit",
            "AcceptHitAndEndSearch": "cgl_accept_hit_and_end_search",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        if func_name in function_mapping:
            return function_mapping[func_name]
        # A program function or a CUDA library function, spelled as it is; a built-in function of another
        # language left in the output is refused by render_checked_program.
        return func_name

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
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

    def resource_register_space_value(self, value):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_cuda_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if (
                attr_name in {"binding", "buffer", "sampler", "texture", "uav"}
                or attr_name == "register"
            ):
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_cuda_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if attr_name in {"set", "group"} and arguments:
                set_index = self.resource_binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.resource_register_space_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.resource_register_space_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def cuda_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_value_to_string(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def cuda_explicit_resource_binding_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if (
                attr_name in {"binding", "buffer", "sampler", "texture", "uav"}
                or attr_name == "register"
            ):
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return attr_name
        return None

    def cuda_resource_register_class(self, node):
        """Return the HLSL register class letter from an explicit ``register`` attribute."""
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            raw_value = self.attribute_value_to_string(arguments[0]) or ""
            raw_value = str(raw_value).strip().lower()
            if (
                raw_value
                and raw_value[0] in ("b", "s", "t", "u")
                and raw_value[1:].isdigit()
            ):
                return raw_value[0]
        return None

    def cuda_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name, 1
        if "[" not in type_name or "]" not in type_name:
            return type_name, 1
        base_type = type_name.split("[", 1)[0].strip()
        suffix = type_name[type_name.find("[") :]
        count = 1
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                break
            size_text = suffix[1:close_bracket].strip()
            if size_text.isdigit():
                count *= int(size_text)
            suffix = suffix[close_bracket + 1 :]
        return base_type, count

    def cuda_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_node(node):
            return "glsl_buffer_block"

        base_type, _count = self.cuda_resource_base_type_and_count(type_name)
        if not base_type:
            return None
        generic_parts = self.generic_type_parts(base_type)
        base_name = generic_parts[0] if generic_parts is not None else base_type
        base_name = base_name.rsplit("::", 1)[-1]

        if (
            self.structured_buffer_type_parts(base_type) is not None
            or self.byte_address_buffer_base_type(base_type) is not None
        ):
            return "buffer"

        canonical = self.resource_base_type(base_name) or base_name
        canonical = canonical.rsplit("::", 1)[-1]
        mapped_type = self.convert_crossgl_type_to_cuda(canonical)
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        if mapped_base == "CglRayTracingAccelerationStructure":
            return "acceleration_structure"
        if canonical in {"sampler", "SamplerState"}:
            return "sampler"
        if canonical.startswith("sampler"):
            return "texture"
        if canonical.startswith(("image", "iimage", "uimage")):
            return "image"
        return None

    def cuda_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def cuda_resource_binding_validation_count(self, node, resource_kind, count):
        if (
            resource_kind in {"image", "sampler", "texture"}
            and count != 1
            and self.cuda_explicit_resource_binding_attribute(node) == "binding"
        ):
            return 1
        return count

    def cuda_resource_binding_family_keys(self, base_namespace, set_index):
        """Yield the unscoped key and all class-scoped keys for a base namespace."""
        yield (base_namespace, set_index)
        for cls in ("b", "s", "t", "u"):
            yield (f"{base_namespace}:{cls}", set_index)

    def cuda_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.cuda_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def cuda_resource_binding_family_conflicts(
        self, base_namespace, set_index, binding, count
    ):
        for key in self.cuda_resource_binding_family_keys(base_namespace, set_index):
            if self.cuda_resource_binding_range_conflicts(key, binding, count):
                return True
        return False

    def next_available_cuda_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.cuda_resource_binding_cursors.get(key, 0)
        while self.cuda_resource_binding_family_conflicts(
            namespace, set_index, binding, count
        ):
            binding += 1
        self.cuda_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_cuda_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        base_namespace = namespace.split(":", 1)[0] if ":" in namespace else namespace

        # Register-class resources only conflict with the same class, but still
        # conflict with classless resources in the unscoped namespace. Classless
        # resources conflict with every class-scoped range in the same family.
        if ":" in namespace:
            keys_to_check = [(namespace, set_index), (base_namespace, set_index)]
        else:
            keys_to_check = self.cuda_resource_binding_family_keys(
                base_namespace, set_index
            )

        for check_key in keys_to_check:
            for used_start, used_end, used_name in self.cuda_used_resource_bindings.get(
                check_key, []
            ):
                if binding <= used_end and used_start <= end:
                    if used_start == binding and used_end == end and used_name == name:
                        return
                    raise CudaRefusal(
                        "Conflicting CUDA resource binding for "
                        f"'{name}': {base_namespace} set {set_index} binding "
                        f"{binding}-{end} overlaps '{used_name}' binding "
                        f"{used_start}-{used_end}"
                    )

        ranges = self.cuda_used_resource_bindings.setdefault(key, [])
        ranges.append((binding, end, name))
        self.cuda_resource_binding_cursors[key] = max(
            self.cuda_resource_binding_cursors.get(key, 0), end + 1
        )

    def cuda_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.cuda_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return ""

        _base_type, count = self.cuda_resource_base_type_and_count(type_name)
        validation_count = self.cuda_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.cuda_resource_binding_namespace(resource_kind)
        register_class = self.cuda_resource_register_class(node)
        if register_class is not None:
            namespace = f"{namespace}:{register_class}"
        set_index = self.explicit_cuda_resource_set_index(node)
        binding = self.explicit_cuda_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_cuda_resource_binding(
                namespace, set_index, validation_count
            )
            binding_source = "automatic"
            self.reserve_cuda_resource_binding(
                namespace, set_index, binding, validation_count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_cuda_resource_binding(
                namespace, set_index, binding, validation_count, name
            )

        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
        ]
        if resource_kind == "glsl_buffer_block":
            layout = self.glsl_buffer_block_layout(node)
            if layout:
                parts.append(f"layout={layout}")
            access = self.explicit_resource_access(node)
            if access:
                parts.append(f"access={access}")
        parts.extend(
            [
                f"set={set_index}",
                f"binding={binding}",
                f"binding_source={binding_source}",
            ]
        )
        if count != 1:
            parts.append(f"count={count}")
        register_metadata = self.cuda_resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        return " ".join(parts)

    def reserve_explicit_cuda_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.cuda_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_cuda_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.cuda_resource_base_type_and_count(type_name)
        validation_count = self.cuda_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.cuda_resource_binding_namespace(resource_kind)
        register_class = self.cuda_resource_register_class(node)
        if register_class is not None:
            namespace = f"{namespace}:{register_class}"
        set_index = self.explicit_cuda_resource_set_index(node)
        self.reserve_cuda_resource_binding(
            namespace, set_index, binding, validation_count, name
        )

    def reserve_explicit_cuda_resource_bindings(self, ast):
        for node in getattr(ast, "global_variables", []) or []:
            type_name = self.get_variable_node_type(node) or "float"
            self.reserve_explicit_cuda_resource_binding(node, type_name)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.reserve_explicit_cuda_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        stages = getattr(ast, "stages", {}) or {}
        for stage in stages.values():
            for node in getattr(stage, "local_variables", []) or []:
                type_name = self.get_variable_node_type(node) or "float"
                self.reserve_explicit_cuda_resource_binding(node, type_name)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_cuda_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None
