"""CrossGL-to-CUDA code generator."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ConstantNode,
    ContinueNode,
    EnumNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .array_utils import parse_array_type
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .resource_arrays import format_array_declarator
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .stage_utils import normalize_stage_name, stage_matches
from .vector_arithmetic import VectorArithmeticMixin

CUDA_WAVE_OP_ARITIES = {
    "WaveGetLaneCount": 0,
    "WaveGetLaneIndex": 0,
    "WaveIsFirstLane": 0,
    "WaveActiveSum": 1,
    "WaveActiveProduct": 1,
    "WaveActiveBitAnd": 1,
    "WaveActiveBitOr": 1,
    "WaveActiveBitXor": 1,
    "WaveActiveMin": 1,
    "WaveActiveMax": 1,
    "WaveActiveAllTrue": 1,
    "WaveActiveAnyTrue": 1,
    "WaveActiveAllEqual": 1,
    "WaveActiveBallot": 1,
    "WaveActiveCountBits": 1,
    "WaveReadLaneAt": 2,
    "WaveReadLaneFirst": 1,
    "WavePrefixSum": 1,
    "WavePrefixProduct": 1,
    "WavePrefixCountBits": 1,
    "QuadReadAcrossX": 1,
    "QuadReadAcrossY": 1,
    "QuadReadAcrossDiagonal": 1,
    "QuadReadLaneAt": 2,
    "WaveMatch": 1,
    "WaveMultiPrefixSum": 2,
    "WaveMultiPrefixCountBits": 2,
    "WaveMultiPrefixProduct": 2,
    "WaveMultiPrefixBitAnd": 2,
    "WaveMultiPrefixBitOr": 2,
    "WaveMultiPrefixBitXor": 2,
}

CUDA_WAVE_PREDICATE_ARGUMENT_OPS = {
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

CUDA_WAVE_UINT_RESULT_OPS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

CUDA_WAVE_BOOL_RESULT_OPS = {
    "WaveIsFirstLane",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
}

CUDA_WAVE_UVEC4_RESULT_OPS = {
    "WaveActiveBallot",
    "WaveMatch",
}


class CudaCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit CUDA source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "CUDA"
    query_return_index_binary_ops = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"}
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "allMemoryBarrier",
        "deviceMemoryBarrier",
        "workgroupBarrier",
    }
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
        "RWTexture1D": "image1D",
        "RWTexture1DArray": "image1DArray",
        "RWTexture2D": "image2D",
        "RWTexture2DArray": "image2DArray",
        "RWTexture3D": "image3D",
        "RWTextureCube": "imageCube",
        "RWTextureCubeArray": "imageCubeArray",
        "RWTexture2DMS": "image2DMS",
        "RWTexture2DMSArray": "image2DMSArray",
    }

    def __init__(self):
        """Initialize CUDA type maps and per-generation visitor state."""
        self.indent_level = 0
        self.output = []
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.enum_variant_constants = {}
        self.cuda_resource_binding_cursors = {}
        self.cuda_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.struct_member_image_accesses = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.query_metadata_aliases = {}
        self.query_return_sources = {}
        self.query_local_resource_names_by_function = {}
        self.query_metadata_snapshot_locals_by_function = {}
        self.struct_query_metadata_members = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.current_stage_name = None
        self.cuda_function_capture_params = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False
        self.builtin_map = {
            "gl_LocalInvocationID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize": "make_uint3(blockDim.x, blockDim.y, blockDim.z)",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups": "make_uint3(gridDim.x, gridDim.y, gridDim.z)",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "gl_LocalInvocationIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GroupThreadID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GroupThreadID.x": "threadIdx.x",
            "SV_GroupThreadID.y": "threadIdx.y",
            "SV_GroupThreadID.z": "threadIdx.z",
            "SV_GROUPTHREADID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GROUPTHREADID.x": "threadIdx.x",
            "SV_GROUPTHREADID.y": "threadIdx.y",
            "SV_GROUPTHREADID.z": "threadIdx.z",
            "SV_GroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GroupID.x": "blockIdx.x",
            "SV_GroupID.y": "blockIdx.y",
            "SV_GroupID.z": "blockIdx.z",
            "SV_GROUPID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GROUPID.x": "blockIdx.x",
            "SV_GROUPID.y": "blockIdx.y",
            "SV_GROUPID.z": "blockIdx.z",
            "SV_DispatchThreadID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DispatchThreadID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DispatchThreadID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DispatchThreadID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_DISPATCHTHREADID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DISPATCHTHREADID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DISPATCHTHREADID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DISPATCHTHREADID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_GroupIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GROUPINDEX": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
        }

    def generate(self, ast_node):
        """Generate complete CUDA source for a CrossGL AST."""
        self.output = []
        self.indent_level = 0
        self.validate_supported_stage_types(ast_node)
        self.reject_unsupported_generic_functions(ast_node)
        self.variable_types = {}
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.enum_variant_constants = self.collect_cuda_enum_variant_constants(ast_node)
        self.cuda_resource_binding_cursors = {}
        self.cuda_used_resource_bindings = {}
        (
            self.struct_member_types,
            self.struct_member_image_accesses,
        ) = self.collect_struct_member_metadata(ast_node)
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(ast_node)
        self.helper_functions = {}
        self.query_metadata_aliases = {}
        self.query_return_sources = self.collect_simple_query_return_sources(ast_node)
        self.query_local_resource_names_by_function = {}
        self.query_metadata_snapshot_locals_by_function = {}
        self.struct_query_metadata_members = self.collect_struct_query_metadata_members(
            ast_node
        )
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False
        self.current_stage_name = None
        self.cuda_function_capture_params = {}
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(ast_node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(ast_node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(ast_node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.reserve_explicit_cuda_resource_bindings(ast_node)
        self.visit(ast_node)
        self.insert_helper_functions()
        return "\n".join(self.output)

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions before emitting non-compilable CUDA code."""
        functions = list(getattr(ast_node, "functions", []) or [])
        for stage in (getattr(ast_node, "stages", {}) or {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
            functions.extend(getattr(stage, "local_functions", []) or [])

        for func in functions:
            generic_params = getattr(func, "generic_params", []) or []
            if not generic_params:
                continue
            names = [
                getattr(param, "name", str(param))
                for param in generic_params
                if getattr(param, "name", str(param))
            ]
            suffix = f" ({', '.join(names)})" if names else ""
            raise ValueError(
                f"CUDA codegen does not support generic functions{suffix}; "
                "specialize the function before CUDA generation"
            )

    def unsupported_stage_types(self):
        return set()

    def cuda_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def cuda_ray_stage_metadata_name(self, stage_name):
        return {
            "ray_any_hit": "any_hit",
            "ray_callable": "callable",
            "ray_closest_hit": "closest_hit",
            "ray_generation": "ray_generation",
            "ray_intersection": "intersection",
            "ray_miss": "miss",
        }.get(stage_name, stage_name)

    def has_geometry_stage(self, ast_node, target_stage=None):
        if not stage_matches(target_stage, "geometry"):
            return False

        return self.has_stage(ast_node, "geometry")

    def has_tessellation_stage(self, ast_node, target_stage=None):
        if not (
            stage_matches(target_stage, "tessellation_control")
            or stage_matches(target_stage, "tessellation_evaluation")
        ):
            return False

        return self.has_stage(ast_node, "tessellation_control") or self.has_stage(
            ast_node, "tessellation_evaluation"
        )

    def cuda_mesh_task_stage_names(self):
        return {"mesh", "task", "object", "amplification"}

    def has_mesh_task_stage(self, ast_node, target_stage=None):
        stage_names = self.cuda_mesh_task_stage_names()
        if target_stage is not None and not any(
            stage_matches(target_stage, stage_name) for stage_name in stage_names
        ):
            return False

        return any(self.has_stage(ast_node, stage_name) for stage_name in stage_names)

    def has_stage(self, ast_node, stage_name):
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == stage_name:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == stage_name
                for qualifier in qualifiers
            ):
                return True

        return False

    def collect_cuda_enum_variant_constants(self, root):
        """Collect plain enum path names that CUDA can lower to C++ enumerators."""
        constants = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
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

            if isinstance(value, EnumNode):
                if any(
                    self.cuda_enum_variant_has_payload(variant)
                    for variant in getattr(value, "variants", []) or []
                ):
                    return
                for variant in getattr(value, "variants", []) or []:
                    constants[f"{value.name}::{variant.name}"] = variant.name
                return

            if not hasattr(value, "__dict__"):
                return
            for child in vars(value).values():
                visit(child)

        visit(root)
        return constants

    def cuda_enum_variant_has_payload(self, variant):
        return bool(getattr(variant, "data", None) or getattr(variant, "fields", None))

    def generate_cuda_geometry_stream_helpers(self):
        return (
            "template <typename T>\n"
            "struct CglCudaPointStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglCudaLineStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglCudaTriangleStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
        )

    def generate_cuda_tessellation_patch_helpers(self):
        return (
            "template <typename T, int N>\n"
            "struct CglCudaInputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n\n"
            "template <typename T, int N>\n"
            "struct CglCudaOutputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n\n"
        )

    def generate_cuda_mesh_task_helpers(self):
        return (
            "__device__ inline void cgl_cuda_set_mesh_output_counts(\n"
            "    unsigned int vertex_count, unsigned int primitive_count)\n"
            "{\n"
            "    (void)vertex_count;\n"
            "    (void)primitive_count;\n"
            "}\n\n"
            "__device__ inline void cgl_cuda_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "}\n\n"
            "template <typename Payload>\n"
            "__device__ inline void cgl_cuda_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z,\n"
            "    const Payload& payload)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "    (void)payload;\n"
            "}\n\n"
        )

    def generate_cuda_matrix_type_helpers(self):
        components = ("x", "y", "z", "w")

        def matrix_type(scalar, columns, rows):
            return f"{scalar}{columns}x{rows}"

        def vector_type(scalar, size):
            return f"{scalar}{size}"

        def vector_constructor(scalar, size, args):
            return f"make_{scalar}{size}({', '.join(args)})"

        lines = ["// CrossGL matrix value helpers"]
        for scalar in ("float", "double"):
            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    count = columns * rows
                    params = ", ".join(f"{scalar} v{i}" for i in range(count))
                    assignments = " ".join(f"m[{i}] = v{i};" for i in range(count))
                    zero = f"{scalar}(0)"
                    add_args = ", ".join(f"m[{i}] + rhs.m[{i}]" for i in range(count))
                    sub_args = ", ".join(f"m[{i}] - rhs.m[{i}]" for i in range(count))
                    neg_args = ", ".join(f"-m[{i}]" for i in range(count))
                    mul_args = ", ".join(f"m[{i}] * rhs" for i in range(count))
                    div_args = ", ".join(f"m[{i}] / rhs" for i in range(count))
                    column_params = ", ".join(
                        f"{vector_type(scalar, rows)} c{column}"
                        for column in range(columns)
                    )
                    column_assignments = " ".join(
                        (f"m[{column * rows + row}] = " f"c{column}.{components[row]};")
                        for column in range(columns)
                        for row in range(rows)
                    )
                    one = f"{scalar}(1)"
                    lines.extend(
                        [
                            f"struct {type_name} {{",
                            f"    {scalar} m[{count}];",
                            f"    static const int CGL_COLUMNS = {columns};",
                            f"    static const int CGL_ROWS = {rows};",
                            f"    __host__ __device__ {type_name}() {{ }}",
                            f"    __host__ __device__ explicit {type_name}({scalar} diagonal) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                        ]
                    )
                    for index in range(min(columns, rows)):
                        lines.append(f"        m[{index * rows + index}] = diagonal;")
                    lines.extend(
                        [
                            "    }",
                            f"    __host__ __device__ {type_name}({column_params}) {{ {column_assignments} }}",
                            (
                                "    template <typename Matrix, "
                                "typename = decltype(Matrix::CGL_COLUMNS), "
                                "typename = decltype(Matrix::CGL_ROWS)>"
                            ),
                            f"    __host__ __device__ explicit {type_name}(const Matrix& source) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                            f"        for (int column = 0; column < {columns}; ++column) {{",
                            f"            for (int row = 0; row < {rows}; ++row) {{",
                            "                if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {",
                            f"                    m[column * {rows} + row] = source.m[column * Matrix::CGL_ROWS + row];",
                            "                } else if (column == row) {",
                            f"                    m[column * {rows} + row] = {one};",
                            "                }",
                            "            }",
                            "        }",
                            "    }",
                            f"    __host__ __device__ {type_name}({params}) {{ {assignments} }}",
                            f"    __host__ __device__ {scalar}& operator[](int index) {{ return m[index]; }}",
                            f"    __host__ __device__ const {scalar}& operator[](int index) const {{ return m[index]; }}",
                            f"    __host__ __device__ {type_name} operator+(const {type_name}& rhs) const {{",
                            f"        return {type_name}({add_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-(const {type_name}& rhs) const {{",
                            f"        return {type_name}({sub_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-() const {{",
                            f"        return {type_name}({neg_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator*({scalar} rhs) const {{",
                            f"        return {type_name}({mul_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator/({scalar} rhs) const {{",
                            f"        return {type_name}({div_args});",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator+=(const {type_name}& rhs) {{",
                            "        *this = *this + rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator-=(const {type_name}& rhs) {{",
                            "        *this = *this - rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator*=({scalar} rhs) {{",
                            "        *this = *this * rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator/=({scalar} rhs) {{",
                            "        *this = *this / rhs;",
                            "        return *this;",
                            "    }",
                            "};",
                            "",
                            f"__host__ __device__ inline {type_name} operator*({scalar} lhs, const {type_name}& rhs) {{",
                            "    return rhs * lhs;",
                            "}",
                            "",
                        ]
                    )

            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    result_type = matrix_type(scalar, rows, columns)
                    transpose_values = [
                        f"value.m[{row * rows + column}]"
                        for column in range(rows)
                        for row in range(columns)
                    ]
                    lines.extend(
                        [
                            f"__host__ __device__ inline {result_type} transpose(const {type_name}& value) {{",
                            f"    return {result_type}({', '.join(transpose_values)});",
                            "}",
                            "",
                        ]
                    )

            for size in range(2, 5):
                type_name = matrix_type(scalar, size, size)
                zero = f"{scalar}(0)"
                one = f"{scalar}(1)"
                lines.extend(
                    [
                        f"__host__ __device__ inline {type_name} inverse(const {type_name}& value) {{",
                        f"    {type_name} a(value);",
                        f"    {type_name} result({one});",
                        f"    for (int column = 0; column < {size}; ++column) {{",
                        "        int pivot_row = column;",
                        f"        {scalar} pivot_value = a.m[column * {size} + column];",
                        (
                            f"        {scalar} pivot_abs = pivot_value < {zero} ? "
                            "-pivot_value : pivot_value;"
                        ),
                        f"        for (int row = column + 1; row < {size}; ++row) {{",
                        f"            {scalar} candidate_value = a.m[column * {size} + row];",
                        (
                            f"            {scalar} candidate_abs = "
                            f"candidate_value < {zero} ? "
                            "-candidate_value : candidate_value;"
                        ),
                        "            if (candidate_abs > pivot_abs) {",
                        "                pivot_abs = candidate_abs;",
                        "                pivot_row = row;",
                        "            }",
                        "        }",
                        f"        if (pivot_abs == {zero}) {{",
                        "            return result;",
                        "        }",
                        "        if (pivot_row != column) {",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                {scalar} tmp = a.m[c * {size} + column];",
                        f"                a.m[c * {size} + column] = a.m[c * {size} + pivot_row];",
                        f"                a.m[c * {size} + pivot_row] = tmp;",
                        f"                tmp = result.m[c * {size} + column];",
                        f"                result.m[c * {size} + column] = result.m[c * {size} + pivot_row];",
                        f"                result.m[c * {size} + pivot_row] = tmp;",
                        "            }",
                        "        }",
                        f"        {scalar} pivot = a.m[column * {size} + column];",
                        f"        for (int c = 0; c < {size}; ++c) {{",
                        f"            a.m[c * {size} + column] /= pivot;",
                        f"            result.m[c * {size} + column] /= pivot;",
                        "        }",
                        f"        for (int row = 0; row < {size}; ++row) {{",
                        "            if (row == column) {",
                        "                continue;",
                        "            }",
                        f"            {scalar} factor = a.m[column * {size} + row];",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                a.m[c * {size} + row] -= factor * a.m[c * {size} + column];",
                        f"                result.m[c * {size} + row] -= factor * result.m[c * {size} + column];",
                        "            }",
                        "        }",
                        "    }",
                        "    return result;",
                        "}",
                        "",
                    ]
                )

            for left_columns in range(2, 5):
                for left_rows in range(2, 5):
                    left_type = matrix_type(scalar, left_columns, left_rows)
                    in_vector = vector_type(scalar, left_columns)
                    out_vector = vector_type(scalar, left_rows)
                    values = []
                    for row in range(left_rows):
                        terms = [
                            f"lhs.m[{column * left_rows + row}] * rhs.{components[column]}"
                            for column in range(left_columns)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {left_type}& lhs, const {in_vector}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_rows, values)};",
                            "}",
                            "",
                        ]
                    )

                    in_vector = vector_type(scalar, left_rows)
                    out_vector = vector_type(scalar, left_columns)
                    values = []
                    for column in range(left_columns):
                        terms = [
                            f"lhs.{components[row]} * rhs.m[{column * left_rows + row}]"
                            for row in range(left_rows)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {in_vector}& lhs, const {left_type}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_columns, values)};",
                            "}",
                            "",
                        ]
                    )

                    for right_columns in range(2, 5):
                        right_type = matrix_type(scalar, right_columns, left_columns)
                        result_type = matrix_type(scalar, right_columns, left_rows)
                        values = []
                        for column in range(right_columns):
                            for row in range(left_rows):
                                terms = [
                                    (
                                        f"lhs.m[{shared * left_rows + row}] * "
                                        f"rhs.m[{column * left_columns + shared}]"
                                    )
                                    for shared in range(left_columns)
                                ]
                                values.append(" + ".join(terms))
                        lines.extend(
                            [
                                f"__host__ __device__ inline {result_type} operator*(const {left_type}& lhs, const {right_type}& rhs) {{",
                                f"    return {result_type}({', '.join(values)});",
                                "}",
                                "",
                            ]
                        )
        return "\n".join(lines)

    def validate_supported_stage_types(self, ast_node, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast_node, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise ValueError(
                f"CUDA output does not support stage type(s): {stage_list}"
            )

    def visit(self, node):
        """Dispatch an AST node to its CUDA visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback visitor for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            builtin_alias = self.cuda_stage_builtin_alias_expression(node)
            if builtin_alias is not None:
                return builtin_alias
            ray_builtin = self.cuda_ray_builtin_expression(node)
            if ray_builtin is not None and node not in self.variable_types:
                return ray_builtin
            return self.builtin_map.get(node, node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        """Append a line of CUDA output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def emit_generated_code(self, code):
        """Append pre-indented generated code to the output stream."""
        for line in code.rstrip("\n").splitlines():
            self.output.append(line.rstrip())

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        if isinstance(node, AssignmentNode):
            self.emit_assignment_statement(node)
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        """Normalize block-like statement containers to a list."""
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def collect_struct_member_metadata(self, root):
        """Collect struct member types and storage-image access before emission."""
        member_types_by_struct = {}
        member_accesses_by_struct = {}
        for struct in getattr(root, "structs", []) or []:
            struct_name = getattr(struct, "name", None)
            if not struct_name:
                continue
            member_types = {}
            member_accesses = {}
            for member in getattr(struct, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if hasattr(member, "member_type"):
                    member_type = member.member_type
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_type = self.resource_type_with_access(member_type, member)
                member_types[member_name] = member_type
                member_access = self.explicit_resource_access(member)
                if member_access is not None and (
                    self.is_storage_image_type(member_type)
                    or self.is_buffer_resource_type(member_type)
                ):
                    member_accesses[member_name] = member_access
            member_types_by_struct[struct_name] = member_types
            member_accesses_by_struct[struct_name] = member_accesses
        return member_types_by_struct, member_accesses_by_struct

    def collect_struct_query_metadata_members(self, root):
        """Collect struct resource members that need embedded query sidecars."""
        has_resource_query = False
        for node in self.query_walk_nodes(root):
            if not isinstance(node, FunctionCallNode):
                continue
            if self.raw_function_call_name(node) in self.query_function_names:
                has_resource_query = True
                break
        if not has_resource_query:
            return {}

        metadata_members = {}
        for struct_name, member_types in self.struct_member_types.items():
            for member_name, member_type in member_types.items():
                if not self.is_queryable_resource_type(member_type):
                    continue
                metadata_members.setdefault(struct_name, set()).add(member_name)
        return metadata_members

    def statement_body_terminates(self, body):
        """Return true when the body already exits the active control flow."""
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def visit_ShaderNode(self, node):
        """Render a full shader/program AST as a CUDA translation unit."""
        self.emit("#include <cuda_runtime.h>")
        self.emit("#include <device_launch_parameters.h>")
        self.emit("")
        self.emit_generated_code(self.generate_cuda_matrix_type_helpers())
        self.emit("")
        if self.has_geometry_stage(node):
            self.emit_generated_code(self.generate_cuda_geometry_stream_helpers())
        if self.has_tessellation_stage(node):
            self.emit_generated_code(self.generate_cuda_tessellation_patch_helpers())
        if self.has_mesh_task_stage(node):
            self.emit_generated_code(self.generate_cuda_mesh_task_helpers())

        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)
            self.emit("")

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit_cbuffer(cbuffer)
            self.emit("")

        constants = getattr(node, "constants", [])
        for constant in constants:
            self.visit(constant)
            self.emit("")

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)
            self.emit("")

        emitted_stage_local_variables = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in global_vars
            if getattr(var, "name", None)
        }
        for stage in (getattr(node, "stages", {}) or {}).values():
            for var in getattr(stage, "local_variables", []) or []:
                if self.is_cuda_shared_variable(var):
                    continue
                var_name = getattr(var, "name", None)
                var_type = self.type_name_string(self.get_variable_node_type(var))
                if not var_name:
                    continue
                previous_type = emitted_stage_local_variables.get(var_name)
                if previous_type is not None:
                    if previous_type != var_type:
                        raise ValueError(
                            "Conflicting CUDA stage-local declaration for "
                            f"'{var_name}': {previous_type} differs from {var_type}"
                        )
                    self.register_variable_type(var_name, var_type, var)
                    continue
                self.visit(var)
                self.emit("")
                emitted_stage_local_variables[var_name] = var_type

        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)
            self.emit("")

        # Handle legacy shader structure
        if hasattr(node, "stages") and node.stages:
            emitted_local_functions = set()
            stage_entry_name_counts = self.stage_entry_name_counts(node.stages)
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    local_captures = self.collect_stage_local_function_captures(stage)
                    self.cuda_function_capture_params.update(local_captures)
                    saved_stage_name = self.current_stage_name
                    saved_override = getattr(
                        self, "current_stage_entry_function_name", None
                    )
                    saved_stage_local_variables = getattr(
                        self, "current_stage_local_variables", []
                    )
                    try:
                        self.current_stage_name = saved_stage_name
                        for func in getattr(stage, "local_functions", []):
                            if id(func) in emitted_local_functions:
                                continue
                            self.visit(func)
                            self.emit("")
                            emitted_local_functions.add(id(func))

                        self.current_stage_name = stage_name
                        self.current_stage_entry_function_name = (
                            self.stage_entry_function_name(
                                stage_name, stage.entry_point, stage_entry_name_counts
                            )
                        )
                        self.current_stage_local_variables = (
                            getattr(stage, "local_variables", []) or []
                        )
                        self.visit(stage.entry_point)
                    finally:
                        self.current_stage_entry_function_name = saved_override
                        self.current_stage_name = saved_stage_name
                        self.current_stage_local_variables = saved_stage_local_variables
                        for captured_name in local_captures:
                            self.cuda_function_capture_params.pop(captured_name, None)
                    self.emit("")

    def stage_entry_name_counts(self, stages):
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if normalize_stage_name(stage_name) == "compute" and name == "main":
            return "compute_main"
        if stage_name in self.cuda_ray_stage_names() and name == "main":
            return f"{stage_name}_{name}"
        if (
            stage_name
            in {
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            }
            | self.cuda_mesh_task_stage_names()
            and name == "main"
        ):
            return f"{stage_name}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name

    def collect_stage_local_function_captures(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        entry_params = list(
            getattr(entry_point, "parameters", getattr(entry_point, "params", []))
        )
        entry_params_by_name = {
            getattr(param, "name", None): param for param in entry_params
        }
        entry_params_by_name = {
            name: param for name, param in entry_params_by_name.items() if name
        }
        if not entry_params_by_name:
            return {}

        captures = {}
        for function in getattr(stage, "local_functions", []) or []:
            if getattr(function, "body", None) is None:
                continue
            used_names = self.collect_function_free_identifier_names(function)
            captured_params = [
                entry_params_by_name[name]
                for name in entry_params_by_name
                if name in used_names
            ]
            if captured_params:
                captures[getattr(function, "name", None)] = captured_params
        return {name: params for name, params in captures.items() if name}

    def is_cuda_shared_variable(self, node):
        """Return whether a stage-local variable maps to CUDA shared memory."""
        return any(
            "shared" in qualifier_name or "workgroup" in qualifier_name
            for qualifier_name in (
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            )
        )

    def collect_function_free_identifier_names(self, function):
        params = getattr(function, "parameters", getattr(function, "params", []))
        bound_names = {getattr(param, "name", None) for param in params}
        bound_names = {name for name in bound_names if name}
        body = getattr(function, "body", None)
        local_names = {
            getattr(node, "name", None)
            for node in self.query_walk_nodes(body)
            if isinstance(node, VariableNode)
        }
        bound_names.update(name for name in local_names if name)
        return {
            node.name
            for node in self.query_walk_nodes(body)
            if isinstance(node, IdentifierNode) and node.name not in bound_names
        }

    def visit_FunctionNode(self, node):
        """Render a CrossGL function or compute entry point as CUDA code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        saved_query_metadata_aliases = self.query_metadata_aliases
        saved_current_function_name = self.current_function_name
        saved_current_function_return_type = self.current_function_return_type
        saved_stage_builtin_aliases = self.stage_builtin_aliases
        saved_current_function_is_kernel_entry = self.current_function_is_kernel_entry
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_structured_buffer_length_parameters = {}
        self.query_metadata_aliases = {}
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False
        qualifiers = []
        stage_qualifiers = self.function_stage_qualifier_names(node)

        if stage_qualifiers:
            for qualifier_name in stage_qualifiers:
                if qualifier_name == "compute":
                    qualifiers.append("__global__")
                elif qualifier_name in ["vertex", "fragment"]:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        stage_name = self.function_stage_name(node)
        is_kernel_entry = "__global__" in qualifiers
        self.current_function_is_kernel_entry = is_kernel_entry

        if hasattr(node, "return_type"):
            self.current_function_return_type = node.return_type
            return_type = self.convert_crossgl_type_to_cuda(node.return_type)
        else:
            self.current_function_return_type = "void"
            return_type = "void"

        return_semantic = self.semantic_from_node(node)
        self.validate_cuda_return_semantic(
            stage_name, self.current_function_return_type, return_semantic
        )
        self.validate_cuda_struct_return_semantics(
            stage_name, self.current_function_return_type
        )
        if is_kernel_entry:
            self.current_function_return_type = "void"
            return_type = "void"

        qualifier_str = " ".join(qualifiers)

        params = []
        param_list = list(getattr(node, "parameters", getattr(node, "params", [])))
        param_list.extend(self.cuda_function_capture_params.get(node.name, []))
        self.validate_cuda_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_cuda_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_cuda_tessellation_stage(node, stage_name)
        if stage_name in self.cuda_mesh_task_stage_names():
            self.validate_cuda_mesh_task_stage(node, stage_name)

        for param in param_list:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "void"

            declaration_type = self.resource_type_with_access(param_type, param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            builtin_role = self.cuda_compute_builtin_parameter_role(stage_name, param)
            if param_name and builtin_role is not None:
                self.register_variable_type(param_name, declaration_type, param)
                self.stage_builtin_aliases[param_name] = builtin_role
                continue

            geometry_stream_declaration = self.format_cuda_geometry_stream_parameter(
                declaration_type,
                param_name,
            )
            if geometry_stream_declaration is not None:
                self.register_variable_type(
                    param_name,
                    self.cuda_geometry_stream_mapped_type(declaration_type),
                    param,
                )
                params.append(geometry_stream_declaration)
            else:
                self.register_variable_type(param_name, declaration_type, param)
                params.append(
                    self.format_typed_declarator(declaration_type, param_name)
                )
            metadata_param = self.query_metadata_parameter(param_name, declaration_type)
            if metadata_param:
                params.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, declaration_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                params.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, declaration_type
            )
            if counter_param:
                params.append(counter_param)

        param_str = ", ".join(params)
        if stage_name in self.cuda_ray_stage_names():
            self.emit(
                "// CrossGL ray stage: "
                f"{self.cuda_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.emit_generated_code(
                self.generate_cuda_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.emit_generated_code(
                self.generate_cuda_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.cuda_mesh_task_stage_names():
            self.emit_generated_code(
                self.generate_cuda_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.emit(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    self.emit(
                        f"// CrossGL parameter semantic: {param.name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )
        function_name = getattr(self, "current_stage_entry_function_name", None)
        if not function_name:
            function_name = node.name
        if is_kernel_entry and function_name == "main":
            function_name = "compute_main"
        signature_prefix = qualifier_str
        if is_kernel_entry:
            signature_prefix = f'extern "C" {qualifier_str}'
        body = getattr(node, "body", None)
        if stage_name is None and body is None:
            self.emit(f"{signature_prefix} {return_type} {function_name}({param_str});")
            self.variable_types = saved_variable_types
            self.image_resource_accesses = saved_image_resource_accesses
            self.buffer_resource_accesses = saved_buffer_resource_accesses
            self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
            self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
            self.query_metadata_aliases = saved_query_metadata_aliases
            self.current_function_name = saved_current_function_name
            self.current_function_return_type = saved_current_function_return_type
            self.stage_builtin_aliases = saved_stage_builtin_aliases
            self.current_function_is_kernel_entry = (
                saved_current_function_is_kernel_entry
            )
            self.current_structured_buffer_length_parameters = (
                saved_structured_buffer_length_parameters
            )
            return
        self.emit(f"{signature_prefix} {return_type} {function_name}({param_str}) {{")

        self.indent_level += 1

        for local_var in getattr(self, "current_stage_local_variables", []) or []:
            if self.is_cuda_shared_variable(local_var):
                self.visit(local_var)

        self.emit_body(body)

        self.indent_level -= 1
        self.emit("}")
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.query_metadata_aliases = saved_query_metadata_aliases
        self.current_function_name = saved_current_function_name
        self.current_function_return_type = saved_current_function_return_type
        self.stage_builtin_aliases = saved_stage_builtin_aliases
        self.current_function_is_kernel_entry = saved_current_function_is_kernel_entry
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_semantics = {}
        member_image_accesses = {}
        query_metadata_members = self.struct_query_metadata_members.get(
            node.name, set()
        )
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            else:
                member_type = "float"

            member_type = self.resource_type_with_access(member_type, member)
            member_types[member.name] = member_type
            member_access = self.explicit_resource_access(member)
            if member_access is not None and (
                self.is_storage_image_type(member_type)
                or self.is_buffer_resource_type(member_type)
            ):
                member_image_accesses[member.name] = member_access
            semantic = self.semantic_from_node(member)
            if semantic:
                member_semantics[member.name] = semantic
                self.validate_cuda_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )
            semantic_comment = (
                f" // CrossGL semantic: {self.map_semantic(semantic)}"
                if semantic
                else ""
            )
            self.emit(
                f"{self.format_typed_declarator(member_type, member.name)};"
                f"{semantic_comment}"
            )
            if member.name in query_metadata_members:
                self.resource_query_info_required = True
                metadata_declarator = self.format_typed_declarator(
                    self.query_metadata_type(member_type),
                    self.query_metadata_name(member.name),
                    dynamic_array_as_pointer=False,
                )
                self.emit(f"{metadata_declarator};")

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.struct_member_image_accesses[node.name] = member_image_accesses
        self.indent_level -= 1
        self.emit("};")

    def visit_EnumNode(self, node):
        self.emit(f"enum {node.name} {{")
        self.indent_level += 1

        variants = getattr(node, "variants", [])
        for index, variant in enumerate(variants):
            suffix = "," if index < len(variants) - 1 else ""
            value = getattr(variant, "value", None)
            if value is not None:
                self.emit(f"{variant.name} = {self.visit(value)}{suffix}")
            else:
                self.emit(f"{variant.name}{suffix}")

        self.indent_level -= 1
        self.emit("};")

    def visit_ConstantNode(self, node: ConstantNode):
        const_type = self.type_name_string(getattr(node, "const_type", "auto"))
        value = getattr(node, "value", None)
        self.register_variable_type(node.name, const_type, node, value)
        declaration_type = self.glsl_buffer_block_declaration_type(const_type, node)
        declaration = self.format_typed_declarator(declaration_type, node.name)
        if value is not None:
            declaration += f" = {self.visit(value)}"
        self.emit(f"const {declaration};")

    def format_variable_declaration(self, node):
        var_type = None
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))

        if hasattr(node, "var_type"):
            var_type = node.var_type
        elif hasattr(node, "vtype"):
            var_type = node.vtype

        if not var_type and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(
                node.name,
                inferred_type or "auto",
                source_node=initial_value,
            )
            if self.is_query_metadata_snapshot_local(
                node.name,
                inferred_type or "auto",
            ):
                self.query_metadata_aliases[node.name] = self.query_metadata_name(
                    node.name
                )
            else:
                self.register_query_metadata_alias(
                    node.name,
                    inferred_type or "auto",
                    initial_value,
                )
            return f"auto {node.name} = {self.visit(initial_value)}"

        if var_type:
            var_type = self.resource_type_with_access(var_type, node)
            self.register_variable_type(node.name, var_type, node, initial_value)
            if self.is_query_metadata_snapshot_local(node.name, var_type):
                self.query_metadata_aliases[node.name] = self.query_metadata_name(
                    node.name
                )
            else:
                self.register_query_metadata_alias(node.name, var_type, initial_value)
            # Check for special memory qualifiers
            qualifiers = []
            if hasattr(node, "qualifiers"):
                for qualifier in node.qualifiers:
                    qualifier = str(qualifier).lower()
                    if qualifier == "const":
                        qualifiers.append("const")
                    elif qualifier == "static":
                        qualifiers.append("static")
                    elif "workgroup" in qualifier or "shared" in qualifier:
                        qualifiers.append("__shared__")
                    elif "uniform" in qualifier:
                        qualifiers.append("__constant__")

            qualifier_str = " ".join(qualifiers)
            if qualifier_str:
                qualifier_str += " "

            declaration_type = self.glsl_buffer_block_declaration_type(var_type, node)
            declaration = (
                f"{qualifier_str}"
                f"{self.format_typed_declarator(declaration_type, node.name)}"
            )
            if initial_value is not None:
                declaration += f" = {self.visit(initial_value)}"
            return declaration

        return node.name

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        type_name = self.type_name_string(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.convert_crossgl_type_to_cuda(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.convert_crossgl_type_to_cuda(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if self.is_semantic_name(attr_name):
                return attr_name
        return None

    def is_semantic_name(self, name):
        if name is None:
            return False
        name = str(name)
        upper_name = name.upper()
        if name.startswith("gl_"):
            return True
        if upper_name in {
            "BINORMAL",
            "CALLABLE_DATA",
            "CALLABLEDATAEXT",
            "CALLABLEDATAINEXT",
            "COLOR",
            "HIT_ATTRIBUTE",
            "HITATTRIBUTEEXT",
            "NORMAL",
            "PAYLOAD",
            "POSITION",
            "RAYPAYLOADEXT",
            "RAYPAYLOADINEXT",
            "SV_DEPTH",
            "SV_DISPATCHRAYSDIMENSIONS",
            "SV_DISPATCHRAYSINDEX",
            "SV_DISPATCHTHREADID",
            "SV_DRAWID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_POSITION",
            "SV_PRIMITIVEID",
            "SV_SAMPLEINDEX",
            "SV_STARTINSTANCELOCATION",
            "SV_STARTVERTEXLOCATION",
            "SV_TARGET",
            "SV_VERTEXID",
            "TANGENT",
            "TEXCOORD",
        }:
            return True
        for prefix in ("COLOR", "SV_TARGET", "TEXCOORD"):
            if upper_name.startswith(prefix) and upper_name[len(prefix) :].isdigit():
                return True
        return False

    def map_semantic(self, semantic):
        if semantic is None:
            return ""
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth(any)"
        if lower_name == "gl_fragcolor":
            return "target(0)"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "SV_TARGET":
            return "target(0)"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "TEXCOORD":
            return "texcoord"
        if (
            upper_name.startswith("TEXCOORD")
            and upper_name[len("TEXCOORD") :].isdigit()
        ):
            return f"texcoord({upper_name[len('TEXCOORD'):]})"
        if upper_name == "COLOR":
            return "color"
        if upper_name.startswith("COLOR") and upper_name[len("COLOR") :].isdigit():
            return f"color({upper_name[len('COLOR'):]})"
        generic_semantics = {
            "BINORMAL": "binormal",
            "NORMAL": "normal",
            "POSITION": "position",
            "TANGENT": "tangent",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_BaseVertex": "start_vertex_location",
            "gl_BaseInstance": "start_instance_location",
            "gl_DrawID": "draw_id",
            "gl_WorkGroupID": "workgroup_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "payload": "ray_payload",
            "rayPayloadEXT": "ray_payload",
            "rayPayloadInEXT": "ray_payload",
            "hit_attribute": "hit_attribute",
            "hitAttributeEXT": "hit_attribute",
            "callable_data": "callable_data",
            "callableDataEXT": "callable_data",
            "callableDataInEXT": "callable_data",
            "gl_LaunchIDEXT": "launch_id",
            "gl_LaunchSizeEXT": "launch_size",
            "gl_HitTEXT": "hit_t",
            "gl_HitKindEXT": "hit_kind",
            "SV_DispatchRaysIndex": "launch_id",
            "SV_DispatchRaysDimensions": "launch_size",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DrawID": "draw_id",
            "SV_GroupID": "workgroup_id",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_InstanceID": "instance_id",
            "SV_IsFrontFace": "front_facing",
            "SV_PrimitiveID": "primitive_id",
            "SV_SampleIndex": "sample_index",
            "SV_StartInstanceLocation": "start_instance_location",
            "SV_StartVertexLocation": "start_vertex_location",
            "SV_VertexID": "vertex_id",
        }
        return generic_semantics.get(semantic_name, semantic_name)

    def cuda_semantic_output_kind(self, semantic):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name in {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_vertexid",
            "gl_workgroupid",
        } or upper_name in {
            "SV_DISPATCHTHREADID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
            return "input_only"
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def is_cuda_float4_type(self, type_name):
        return self.map_type(type_name) == "float4"

    def is_cuda_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "float"

    def validate_cuda_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.cuda_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return
        if kind in {"position", "color"}:
            if self.is_cuda_float4_type(type_name):
                return
            raise ValueError(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "expected vec4-compatible type"
            )
        if kind == "depth" and not self.is_cuda_float_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "expected float type"
            )

    def validate_cuda_output_semantic_stage(self, stage_name, semantic, context):
        kind = self.cuda_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise ValueError(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if stage_name is None:
            return
        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if stage_name not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise ValueError(
                f"Unsupported {semantic} {context} for CUDA {stage_name} stage; "
                f"valid stage is {allowed}"
            )

    def function_stage_name(self, node):
        if self.current_stage_name:
            return self.current_stage_name
        for qualifier_name in self.function_stage_qualifier_names(node):
            return qualifier_name
        return None

    def function_stage_qualifier_names(self, node):
        """Return normalized stage qualifiers from legacy qualifiers and attributes."""
        supported_stage_names = {
            "compute",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
        } | self.cuda_mesh_task_stage_names()
        qualifiers = list(getattr(node, "qualifiers", []) or [])
        qualifier = getattr(node, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(attr_name)

        return [
            qualifier_name
            for qualifier_name in (
                normalize_stage_name(qualifier) for qualifier in qualifiers
            )
            if qualifier_name in supported_stage_names
        ]

    def cuda_stage_attribute_arguments(self, func, attribute_name):
        requested = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == requested:
                return self.attribute_arguments(attr)
        return []

    def cuda_geometry_maxvertexcount(self, func):
        arguments = self.cuda_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise ValueError("CUDA geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise ValueError(
                "CUDA geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                f"CUDA geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def cuda_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.cuda_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"CUDA {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def cuda_tessellation_output_control_points(self, func):
        arguments = self.cuda_stage_attribute_arguments(func, "outputcontrolpoints")
        if not arguments:
            raise ValueError(
                "CUDA tessellation_control stage requires outputcontrolpoints "
                "attribute"
            )
        if len(arguments) != 1:
            raise ValueError(
                "CUDA tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                "CUDA tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        return value_text

    def canonical_cuda_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def cuda_tessellation_domain(self, func, stage_name, required=False):
        value = self.cuda_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise ValueError(f"CUDA {stage_name} stage requires domain attribute")
            return None
        canonical = self.canonical_cuda_tessellation_domain(value)
        if canonical is None:
            raise ValueError(
                f"CUDA {stage_name} stage domain '{value}' must be tri, quad, "
                "or isoline"
            )
        return canonical

    def cuda_tessellation_partitioning(self, func, stage_name):
        value = self.cuda_stage_attribute_value(func, "partitioning", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if normalized not in valid_partitioning:
            valid = ", ".join(sorted(valid_partitioning))
            raise ValueError(
                f"CUDA {stage_name} stage partitioning '{value}' must be one of: "
                f"{valid}"
            )
        return normalized

    def validate_cuda_tessellation_stage(self, func, stage_name):
        if stage_name == "tessellation_control":
            self.cuda_tessellation_output_control_points(func)
            self.cuda_tessellation_domain(func, stage_name)
        elif stage_name == "tessellation_evaluation":
            self.cuda_tessellation_domain(func, stage_name, required=True)
        self.cuda_tessellation_partitioning(func, stage_name)

    def generate_cuda_tessellation_stage_comments(self, func, stage_name):
        lines = []
        if stage_name == "tessellation_control":
            output_points = self.cuda_tessellation_output_control_points(func)
            lines.append(
                "// CrossGL tessellation control stage: "
                f"outputcontrolpoints={output_points}\n"
            )
        else:
            domain = self.cuda_tessellation_domain(func, stage_name, required=True)
            lines.append(
                "// CrossGL tessellation evaluation stage: " f"domain={domain}\n"
            )

        domain = self.cuda_tessellation_domain(
            func, stage_name, required=stage_name == "tessellation_evaluation"
        )
        if stage_name == "tessellation_control" and domain is not None:
            lines.append(f"// CrossGL tessellation domain: {domain}\n")

        partitioning = self.cuda_tessellation_partitioning(func, stage_name)
        if partitioning is not None:
            lines.append(f"// CrossGL tessellation partitioning: {partitioning}\n")

        output_topology = self.cuda_stage_attribute_value(
            func, "outputtopology", stage_name
        )
        if output_topology is not None:
            lines.append(
                "// CrossGL tessellation output topology: " f"{output_topology}\n"
            )

        patch_constant_function = self.cuda_stage_attribute_value(
            func, "patchconstantfunc", stage_name
        )
        if patch_constant_function is not None:
            lines.append(
                "// CrossGL tessellation patch constant function: "
                f"{patch_constant_function}\n"
            )

        return "".join(lines)

    def cuda_mesh_task_stage_label(self, stage_name):
        return {
            "amplification": "amplification/task",
            "object": "object/task",
        }.get(stage_name, stage_name)

    def cuda_mesh_task_positive_attribute(self, func, attribute_name, stage_name):
        value = self.cuda_stage_attribute_value(func, attribute_name, stage_name)
        if value is None:
            return None
        value_int = self.literal_int_value(
            self.cuda_stage_attribute_arguments(func, attribute_name)[0]
        )
        if value_int is not None and value_int <= 0:
            raise ValueError(
                f"CUDA {stage_name} stage {attribute_name} ({value_int}) "
                "must be positive"
            )
        return value

    def cuda_mesh_task_numthreads(self, func, stage_name):
        for attribute_name in ("numthreads", "local_size", "workgroup_size"):
            arguments = self.cuda_stage_attribute_arguments(func, attribute_name)
            if not arguments:
                continue
            if len(arguments) != 3:
                raise ValueError(
                    f"CUDA {stage_name} stage {attribute_name} requires exactly "
                    "three arguments"
                )
            values = []
            for axis, argument in zip("xyz", arguments):
                value = self.literal_int_value(argument)
                if value is not None and value <= 0:
                    raise ValueError(
                        f"CUDA {stage_name} stage {attribute_name} {axis} "
                        f"dimension ({value}) must be positive"
                    )
                values.append(self.attribute_value_to_string(argument))
            return ", ".join(values)
        return None

    def cuda_mesh_task_output_topology(self, func, stage_name):
        value = self.cuda_stage_attribute_value(func, "outputtopology", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        topology_aliases = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        topology = topology_aliases.get(normalized)
        if topology is None:
            raise ValueError(
                f"CUDA {stage_name} stage outputtopology '{value}' must be "
                "point, line, or triangle"
            )
        return topology

    def validate_cuda_mesh_task_stage(self, func, stage_name):
        self.cuda_mesh_task_numthreads(func, stage_name)
        if stage_name == "mesh":
            self.cuda_mesh_task_output_topology(func, stage_name)
            self.cuda_mesh_task_positive_attribute(func, "max_vertices", stage_name)
            self.cuda_mesh_task_positive_attribute(func, "max_primitives", stage_name)

    def generate_cuda_mesh_task_stage_comments(self, func, stage_name):
        label = self.cuda_mesh_task_stage_label(stage_name)
        lines = [f"// CrossGL {label} stage\n"]

        numthreads = self.cuda_mesh_task_numthreads(func, stage_name)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {numthreads}\n")

        if stage_name == "mesh":
            topology = self.cuda_mesh_task_output_topology(func, stage_name)
            if topology is not None:
                lines.append(f"// CrossGL mesh output topology: {topology}\n")
            for attribute_name in ("max_vertices", "max_primitives"):
                value = self.cuda_mesh_task_positive_attribute(
                    func, attribute_name, stage_name
                )
                if value is not None:
                    lines.append(f"// CrossGL mesh {attribute_name}: {value}\n")

        return "".join(lines)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def cuda_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def cuda_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.cuda_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def cuda_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def cuda_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            if isinstance(size, int):
                return size
            return self.literal_int_value(size)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        return array_size

    def validate_cuda_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.cuda_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.cuda_parameter_is_array(parameter):
                raise ValueError(
                    "CUDA geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.cuda_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
                    "CUDA geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_cuda_geometry_stage(self, func, parameters):
        self.cuda_geometry_maxvertexcount(func)

        if not any(
            self.cuda_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise ValueError(
                "CUDA geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.cuda_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise ValueError(
                "CUDA geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_cuda_geometry_input_primitive_arity(parameters)

    def generate_cuda_geometry_stage_comments(self, func, parameters):
        maxvertexcount = self.cuda_geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.cuda_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.cuda_geometry_stream_info(
                self.parameter_raw_type(parameter)
            )
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"// CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                "// CrossGL geometry input primitive: "
                f"{', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                "// CrossGL geometry output stream: "
                f"{', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def validate_cuda_return_semantic(self, stage_name, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "void":
            raise ValueError(
                f"Unsupported {semantic} return semantic for CUDA codegen; "
                "void return type"
            )
        self.validate_cuda_output_semantic_stage(
            stage_name, semantic, "return semantic"
        )
        self.validate_cuda_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_cuda_struct_return_semantics(self, stage_name, return_type):
        if stage_name is None:
            return
        base_type = self.type_name_string(return_type)
        if not base_type:
            return
        base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_cuda_output_semantic_stage(stage_name, semantic, context)
            self.validate_cuda_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def cuda_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def cuda_stage_parameter_semantic_rules(self):
        ray_stages = self.cuda_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "uint", {"vertex"}),
            "gl_instanceid": ("instance_id", "uint", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "int", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "uint", {"vertex"}),
            "gl_drawid": ("draw_id", "uint", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "uint", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "uint", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "int", {"vertex"}),
            "SV_STARTINSTANCELOCATION": ("start_instance_location", "uint", {"vertex"}),
            "SV_DRAWID": ("draw_id", "uint", {"vertex"}),
            "gl_position": ("position", "float4", {"fragment"}),
            "gl_fragcoord": ("position", "float4", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "float2", {"fragment"}),
            "gl_primitiveid": (
                "primitive_id",
                "uint",
                {"fragment"} | ray_hit_stages,
            ),
            "gl_sampleid": ("sample_index", "uint", {"fragment"}),
            "SV_POSITION": ("position", "float4", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": (
                "primitive_id",
                "uint",
                {"fragment"} | ray_hit_stages,
            ),
            "SV_SAMPLEINDEX": ("sample_index", "uint", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "uint3", {"compute"}),
            "gl_localinvocationid": ("local_invocation_id", "uint3", {"compute"}),
            "gl_globalinvocationid": ("global_invocation_id", "uint3", {"compute"}),
            "gl_localinvocationindex": ("local_invocation_index", "uint", {"compute"}),
            "SV_GROUPID": ("workgroup_id", "uint3", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "uint3", {"compute"}),
            "SV_DISPATCHTHREADID": ("global_invocation_id", "uint3", {"compute"}),
            "SV_GROUPINDEX": ("local_invocation_index", "uint", {"compute"}),
            "gl_launchidext": ("launch_id", "uint3", ray_stages),
            "gl_launchsizeext": ("launch_size", "uint3", ray_stages),
            "gl_hittext": ("hit_t", "float", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "uint", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "uint3", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": ("launch_size", "uint3", ray_stages),
        }

    def cuda_compute_builtin_parameter_role(self, stage_name, param):
        """Return the compute built-in role for a semantic kernel parameter."""
        if stage_name != "compute":
            return None

        semantic = self.semantic_from_node(param)
        if semantic is None:
            return None

        rule = self.cuda_stage_parameter_semantic_rules().get(
            self.cuda_stage_parameter_semantic_key(semantic)
        )
        if rule is None:
            return None

        role, _expected_type, allowed_stages = rule
        if "compute" not in allowed_stages:
            return None
        if role in {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "workgroup_id",
        }:
            return role
        return None

    def cuda_compute_builtin_expression(self, role, component=None):
        """Return the CUDA expression for a CrossGL compute built-in role."""
        local_index = (
            "(threadIdx.z * blockDim.y * blockDim.x + "
            "threadIdx.y * blockDim.x + threadIdx.x)"
        )
        component_expressions = {
            "global_invocation_id": {
                "x": "(blockIdx.x * blockDim.x + threadIdx.x)",
                "y": "(blockIdx.y * blockDim.y + threadIdx.y)",
                "z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            },
            "local_invocation_id": {
                "x": "threadIdx.x",
                "y": "threadIdx.y",
                "z": "threadIdx.z",
            },
            "workgroup_id": {
                "x": "blockIdx.x",
                "y": "blockIdx.y",
                "z": "blockIdx.z",
            },
            "workgroup_size": {
                "x": "blockDim.x",
                "y": "blockDim.y",
                "z": "blockDim.z",
            },
            "num_workgroups": {
                "x": "gridDim.x",
                "y": "gridDim.y",
                "z": "gridDim.z",
            },
        }

        if role == "local_invocation_index":
            return local_index if component is None else None

        role_components = component_expressions.get(role)
        if role_components is None:
            return None
        if component is not None:
            if component in role_components:
                return role_components[component]
            swizzle_components = tuple(str(component))
            if all(part in role_components for part in swizzle_components):
                constructor = {
                    2: "make_uint2",
                    3: "make_uint3",
                    4: "make_uint4",
                }.get(len(swizzle_components))
                if constructor is not None:
                    values = ", ".join(
                        role_components[part] for part in swizzle_components
                    )
                    return f"{constructor}({values})"
            return None
        return "make_uint3({x}, {y}, {z})".format(**role_components)

    def cuda_compute_builtin_role_for_name(self, name):
        """Return the compute built-in role represented by a source identifier."""
        return {
            "gl_GlobalInvocationID": "global_invocation_id",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DISPATCHTHREADID": "global_invocation_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_GROUPTHREADID": "local_invocation_id",
            "gl_WorkGroupID": "workgroup_id",
            "SV_GroupID": "workgroup_id",
            "SV_GROUPID": "workgroup_id",
            "gl_WorkGroupSize": "workgroup_size",
            "gl_NumWorkGroups": "num_workgroups",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GROUPINDEX": "local_invocation_index",
        }.get(name)

    def cuda_compute_builtin_type(self, name):
        """Return the CUDA value type for a compute built-in identifier."""
        role = self.cuda_compute_builtin_role_for_name(name)
        if role is None:
            return None
        if role == "local_invocation_index":
            return "uint"
        return "uint3"

    def cuda_compute_builtin_member_type(self, name, member):
        """Return the type of a direct compute built-in member/swizzle."""
        root_type = self.cuda_compute_builtin_type(name)
        vector_info = self.vector_type_info(root_type)
        if vector_info is None:
            return None
        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        components = [component_aliases.get(component) for component in str(member)]
        if not components or any(component is None for component in components):
            return None
        if any(component not in vector_info["components"] for component in components):
            return None
        if len(components) == 1:
            return vector_info["component_type"]
        return self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )

    def cuda_stage_builtin_alias_expression(self, name, component=None):
        role = self.stage_builtin_aliases.get(name)
        if role is None:
            return None
        return self.cuda_compute_builtin_expression(role, component)

    def validate_cuda_stage_parameter_semantic_type(
        self, param, semantic, expected_type
    ):
        param_type = self.resource_type_with_access(
            self.get_parameter_type(param), param
        )
        actual_type = self.convert_crossgl_type_to_cuda(param_type)
        if actual_type == expected_type:
            return
        raise ValueError(
            f"Unsupported {semantic} stage parameter semantic for CUDA codegen; "
            f"expected {expected_type} type"
        )

    def validate_cuda_stage_parameter_semantics(self, stage_name, parameters):
        if stage_name is None:
            return

        rules = self.cuda_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for param in parameters or []:
            semantic = self.semantic_from_node(param)
            if semantic is None:
                continue

            semantic_key = self.cuda_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if stage_name not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise ValueError(
                        f"Unsupported {semantic} stage parameter semantic for CUDA "
                        f"{stage_name} stage; valid stage is {allowed}"
                    )
                param_name = getattr(param, "name", getattr(param, "param_name", None))
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise ValueError(
                        f"Duplicate CUDA stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{param_name}'"
                    )
                seen_system_semantics[mapped_semantic] = param_name
                self.validate_cuda_stage_parameter_semantic_type(
                    param, semantic, expected_type
                )
                continue

            kind = self.cuda_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise ValueError(
                    f"Unsupported {semantic} stage parameter semantic for CUDA "
                    f"{stage_name} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def visit_VariableNode(self, node):
        var_type = self.get_variable_node_type(node)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if isinstance(initial_value, MatchNode):
            self.emit_match_expression_variable(node, initial_value, var_type)
            return None

        metadata_comment = self.cuda_resource_metadata_comment(node, var_type)
        if metadata_comment:
            self.emit(metadata_comment)
        declaration = self.format_variable_declaration(node)
        if declaration != node.name:
            self.emit(f"{declaration};")
            initial_value = getattr(
                node,
                "initial_value",
                getattr(node, "value", None),
            )
            effective_var_type = var_type
            if effective_var_type is None and initial_value is not None:
                effective_var_type = self.expression_result_type(initial_value)
            metadata_declaration = self.query_metadata_snapshot_declaration(
                node.name,
                effective_var_type,
                initial_value,
            )
            if metadata_declaration is None:
                metadata_declaration = self.query_metadata_declaration(
                    node.name,
                    effective_var_type,
                )
            if metadata_declaration:
                self.emit(f"{metadata_declaration};")
            length_declaration = self.structured_buffer_length_declaration(
                node.name, var_type
            )
            if length_declaration:
                self.emit(f"{length_declaration};")
            counter_declaration = self.structured_buffer_counter_declaration(
                node.name, var_type
            )
            if counter_declaration:
                self.emit(f"{counter_declaration};")
            return None

        return node.name

    def emit_match_expression_variable(self, node, match_node, var_type):
        if var_type is None:
            try:
                var_type = infer_match_expression_result_type(self, match_node)
            except ValueError:
                var_type = "int"
        var_type = var_type or "auto"
        self.register_variable_type(node.name, var_type, node)

        self.emit(f"{self.format_typed_declarator(var_type, node.name)};")
        try:
            assignment = generate_match_expression_assignment(
                self,
                match_node,
                node.name,
                var_type,
                self.indent_level,
                "CUDA",
            )
        except ValueError as error:
            assignment = self.generate_match_expression_diagnostic_assignment(
                node.name,
                var_type,
                self.cuda_match_error_reason(error),
                self.indent_level,
                "CUDA",
            )
        self.emit_generated_code(assignment)

    def visit_ExpressionStatementNode(self, node):
        if isinstance(node.expression, AssignmentNode):
            self.emit_assignment_statement(node.expression)
            return None

        expr = self.visit(node.expression)
        if expr and expr.strip():
            self.emit(f"{expr};")

    def emit_assignment_statement(self, node):
        assignment = self.format_assignment_expression(node)
        if assignment and assignment.strip():
            self.emit(f"{assignment};")

        metadata_assignment = self.format_query_metadata_assignment(node)
        if metadata_assignment:
            self.emit(f"{metadata_assignment};")

    def format_for_clause_expression(self, node):
        """Return a for-clause expression, preserving CUDA metadata sidecars."""
        if node is None:
            return ""
        if isinstance(node, VariableNode):
            return self.format_variable_declaration(node)

        expr_node = getattr(node, "expression", node)
        if isinstance(expr_node, AssignmentNode):
            assignment = self.format_assignment_expression(expr_node)
            metadata_assignment = self.format_query_metadata_assignment(expr_node)
            parts = [
                part
                for part in (assignment, metadata_assignment)
                if part and part.strip()
            ]
            return ", ".join(parts)

        return self.visit(expr_node) or ""

    def visit_IdentifierNode(self, node):
        name = getattr(node, "name", str(node))
        builtin_alias = self.cuda_stage_builtin_alias_expression(name)
        if builtin_alias is not None:
            return builtin_alias
        ray_builtin = self.cuda_ray_builtin_expression(name)
        if ray_builtin is not None and name not in self.variable_types:
            return ray_builtin
        enum_constant = self.enum_variant_constants.get(name)
        if enum_constant is not None:
            return enum_constant
        return self.builtin_map.get(name, name)

    def cuda_ray_builtin_expression(self, name):
        helper_name = {
            "gl_LaunchIDEXT": "cgl_ray_launch_id",
            "gl_LaunchSizeEXT": "cgl_ray_launch_size",
            "gl_HitTEXT": "cgl_ray_hit_t",
            "gl_HitKindEXT": "cgl_ray_hit_kind",
            "gl_WorldRayOriginEXT": "cgl_ray_world_origin",
            "gl_WorldRayDirectionEXT": "cgl_ray_world_direction",
            "gl_ObjectRayOriginEXT": "cgl_ray_object_origin",
            "gl_ObjectRayDirectionEXT": "cgl_ray_object_direction",
            "gl_RayTminEXT": "cgl_ray_t_min",
            "gl_IncomingRayFlagsEXT": "cgl_ray_incoming_flags",
            "gl_InstanceCustomIndexEXT": "cgl_ray_instance_custom_index",
            "gl_GeometryIndexEXT": "cgl_ray_geometry_index",
        }.get(str(name))
        if helper_name is None:
            return None
        self.require_cuda_ray_runtime_helpers()
        return f"{helper_name}()"

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def visit_LiteralNode(self, node):
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_AssignmentNode(self, node):
        return self.format_assignment_expression(node)

    def format_assignment_expression(self, node):
        operator = getattr(node, "operator", "=")
        diagnostic = self.glsl_buffer_block_write_diagnostic(node.target, "assignment")
        if diagnostic is not None:
            return diagnostic
        diagnostic = self.structured_buffer_element_write_diagnostic(
            node.target, "assignment", operator
        )
        if diagnostic is not None:
            return diagnostic
        swizzle_assignment = self.format_swizzle_assignment_expression(
            node.target,
            node.value,
            operator,
        )
        if swizzle_assignment is not None:
            return swizzle_assignment
        self.assignment_lhs_depth += 1
        try:
            target = self.visit(node.target)
        finally:
            self.assignment_lhs_depth -= 1
        value = self.visit(node.value)
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_value = self.lower_vector_binary_operation(
                node.target,
                target,
                node.value,
                value,
                compound_binary_ops[operator],
            )
            if lowered_value is not None:
                return f"{target} = {lowered_value}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.target,
                    target,
                    node.value,
                    value,
                )
                if modulo is not None:
                    return f"{target} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_value = self.lower_vector_bitwise_operation(
                node.target,
                target,
                node.value,
                value,
                compound_bitwise_ops[operator],
            )
            if lowered_value is not None:
                return f"{target} = {lowered_value}"
        return f"{target} {operator} {value}"

    def format_swizzle_assignment_expression(self, target_node, value_node, operator):
        if not isinstance(target_node, MemberAccessNode):
            return None
        object_node = getattr(
            target_node, "object_expr", getattr(target_node, "object", None)
        )
        components = self.member_swizzle_components(target_node)
        if object_node is None or components is None or len(components) <= 1:
            return None

        object_info = self.vector_type_info(self.expression_result_type(object_node))
        if object_info is None:
            return None
        result_type = self.vector_type_for_components(
            object_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        object_expr = self.visit(object_node)
        value_expr = self.visit(value_node)
        value_info = self.vector_type_info(self.expression_result_type(value_node))
        temp_name = self.next_cuda_temp_variable("swizzle_value")
        if value_info is not None:
            if len(value_info["components"]) != len(components):
                return None
            temp_type = result_info["type"]
            value_parts = [f"{temp_name}.{part}" for part in result_info["components"]]
        else:
            scalar_type = self.scalar_component_type(
                self.expression_result_type(value_node)
            )
            if scalar_type is None:
                scalar_type = self.vector_scalar_parameter_type(object_info)
            temp_type = self.convert_crossgl_type_to_cuda(scalar_type)
            value_parts = None

        statements = [f"{temp_type} {temp_name} = {value_expr}"]
        if value_parts is None:
            value_parts = [temp_name] * len(components)

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
        binary_op = compound_ops.get(operator)
        for component, value_part in zip(components, value_parts):
            target = f"{object_expr}.{component}"
            if operator == "=":
                statements.append(f"{target} = {value_part}")
            elif binary_op is not None:
                statements.append(f"{target} = ({target} {binary_op} {value_part})")
            else:
                return None
        return "; ".join(statements)

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        logical = self.lower_vector_logical_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if logical is not None:
            return logical
        bitwise = self.lower_vector_bitwise_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if bitwise is not None:
            return bitwise
        comparison = self.lower_vector_comparison_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if comparison is not None:
            return comparison
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        if operator == "%":
            modulo = self.lower_scalar_modulo_operation(
                node.left,
                left,
                node.right,
                right,
            )
            if modulo is not None:
                return modulo
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        if operator == "not":
            operator = "!"
        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return f"{operand}{operator}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, operator)
        if lowered is not None:
            return lowered
        return f"{operator}{operand}"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        if hasattr(node, "function"):
            func_name = self.visit(function_expr)
        else:
            func_name = getattr(node, "name", "unknown")

        raw_args = []
        if hasattr(node, "arguments"):
            raw_args = node.arguments
        elif hasattr(node, "args"):
            raw_args = node.args

        args = [self.visit(arg) for arg in raw_args]

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)
        if func_name in CUDA_WAVE_OP_ARITIES:
            return self.generate_wave_operation(func_name, raw_args, args)
        if func_name in {"SetMeshOutputCounts", "DispatchMesh"}:
            return self.generate_mesh_task_call_expression(func_name, raw_args, args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, raw_args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                function_expr, func_name, raw_args, args
            )
            if buffer_call is not None:
                return buffer_call

            structured_atomic_call = self.generate_structured_buffer_atomic_call(
                func_name, raw_args, args
            )
            if structured_atomic_call is not None:
                return structured_atomic_call

            plain_atomic_call = self.generate_plain_atomic_call(
                func_name, raw_args, args
            )
            if plain_atomic_call is not None:
                return plain_atomic_call

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

        args = self.cuda_user_function_call_arguments(func_name, raw_args, args)
        if is_user_function:
            return f"{func_name}({', '.join(args)})"

        if func_name in self.synchronization_builtins and raw_args:
            raise ValueError(
                f"CUDA synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        if func_name == "abs" and len(args) == 1:
            abs_call = self.generate_abs_call(raw_args, args)
            if abs_call is not None:
                return abs_call

        if func_name == "sign" and len(args) == 1:
            sign_call = self.generate_sign_call(raw_args, args)
            if sign_call is not None:
                return sign_call

        if func_name == "mod" and len(args) == 2:
            mod_call = self.generate_mod_call(raw_args, args)
            if mod_call is not None:
                return mod_call

        if func_name in {"fract", "frac"} and len(args) == 1:
            fract_call = self.generate_fract_call(raw_args, args)
            if fract_call is not None:
                return fract_call

        if func_name == "clamp" and len(args) == 3:
            return self.generate_clamp_call(raw_args, args)

        if func_name in {"min", "max"} and len(args) == 2:
            min_max_call = self.generate_min_max_call(func_name, raw_args, args)
            if min_max_call is not None:
                return min_max_call

        if func_name == "atan2" and len(args) == 2:
            atan2_call = self.generate_atan2_call(raw_args, args)
            if atan2_call is not None:
                return atan2_call

        if func_name == "mix" and len(args) == 3:
            mix_call = self.generate_mix_call(raw_args, args)
            if mix_call is not None:
                return mix_call

        if func_name == "saturate" and len(args) == 1:
            saturate_call = self.generate_saturate_call(raw_args, args)
            if saturate_call is not None:
                return saturate_call

        if func_name == "smoothstep" and len(args) == 3:
            smoothstep_call = self.generate_smoothstep_call(raw_args, args)
            if smoothstep_call is not None:
                return smoothstep_call

        if func_name in {"dot", "cross", "length", "normalize"}:
            geometric_call = self.generate_vector_geometric_call(
                func_name,
                raw_args,
                args,
            )
            if geometric_call is not None:
                return geometric_call

        vector_math_call = self.generate_vector_scalar_math_call(
            func_name,
            raw_args,
            args,
        )
        if vector_math_call is not None:
            return vector_math_call

        vector_info = self.vector_type_info(func_name)
        if vector_info:
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, raw_args, args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, raw_args, args
            )
            if constructor_call is not None:
                return constructor_call
            args = self.generate_vector_constructor_args(vector_info, raw_args, args)

        if func_name in self.struct_member_types:
            args = self.struct_constructor_arguments(func_name, raw_args, args)

        scalar_math_call = self.generate_scalar_math_call(func_name, raw_args, args)
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)

        func_name = self.convert_builtin_function(func_name)
        return f"{func_name}({args_str})"

    def visit_MeshOpNode(self, node):
        operation = getattr(node, "operation", "")
        raw_args = getattr(node, "arguments", []) or []
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_mesh_task_call_expression(operation, raw_args, args)

    def generate_mesh_task_call_expression(self, operation, raw_args, args):
        stage_name = getattr(self, "current_stage_name", None)
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise ValueError(
                    "CUDA SetMeshOutputCounts is only valid in mesh stages"
                )
            if len(raw_args) != 2:
                raise ValueError(
                    "CUDA SetMeshOutputCounts requires exactly two arguments"
                )
            return f"cgl_cuda_set_mesh_output_counts({', '.join(args)})"

        if operation == "DispatchMesh":
            if stage_name not in {"task", "object", "amplification"}:
                raise ValueError(
                    "CUDA DispatchMesh is only valid in task, object, or "
                    "amplification stages"
                )
            if len(raw_args) not in {3, 4}:
                raise ValueError(
                    "CUDA DispatchMesh requires exactly three arguments, or four "
                    "with a task payload"
                )
            return f"cgl_cuda_dispatch_mesh({', '.join(args)})"

        return None

    def visit_RayTracingOpNode(self, node):
        operation = getattr(node, "operation", "")
        if operation in self.function_return_types:
            args = ", ".join(
                self.visit(arg) for arg in getattr(node, "arguments", []) or []
            )
            return f"{operation}({args})"
        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(node, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, arguments):
        if operation not in {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "IgnoreHit",
            "AcceptHitAndEndSearch",
        }:
            return None

        self.require_cuda_ray_runtime_helpers()
        generated_args = [self.visit(arg) for arg in arguments or []]
        if operation == "TraceRay" and len(generated_args) == 11:
            ray_desc = (
                "CglRayDesc("
                f"{generated_args[6]}, {generated_args[7]}, "
                f"{generated_args[8]}, {generated_args[9]}"
                ")"
            )
            generated_args = generated_args[:6] + [ray_desc, generated_args[10]]
        elif operation == "ReportHit" and len(generated_args) == 2:
            generated_args.append("CglBuiltInTriangleIntersectionAttributes{}")

        helper_name = self.convert_builtin_function(operation)
        return f"{helper_name}({', '.join(generated_args)})"

    def visit_RayQueryOpNode(self, node):
        operation = getattr(node, "operation", "")
        helper_name = self.require_cuda_ray_query_helper(operation)
        args = [self.visit(getattr(node, "query_expr", None))]
        args.extend(self.visit(arg) for arg in getattr(node, "arguments", []) or [])
        return f"{helper_name}({', '.join(args)})"

    def visit_WaveOpNode(self, node):
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_wave_operation(
            getattr(node, "operation", "WaveOp"), raw_args, args
        )

    def generate_wave_operation(self, operation, raw_args, args):
        expected_count = CUDA_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise ValueError(f"Unsupported CUDA wave intrinsic {operation}")
        if len(raw_args) != expected_count:
            raise ValueError(
                f"CUDA wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(raw_args)}"
            )

        if operation == "WaveGetLaneCount":
            return "warpSize"
        if operation == "WaveGetLaneIndex":
            return "(threadIdx.x & (warpSize - 1))"
        if operation == "WaveIsFirstLane":
            return "((threadIdx.x & (warpSize - 1)) == 0)"

        diagnostic = self.cuda_wave_operation_diagnostic(operation, raw_args)
        if diagnostic is not None:
            return diagnostic

        helper_name = self.require_wave_helper(operation, raw_args)
        return f"{helper_name}({', '.join(args)})"

    def require_wave_helper(self, operation, raw_args):
        result_type = self.map_type(self.wave_result_type(operation, raw_args))
        arg_types = [
            self.wave_argument_type(operation, index, arg)
            for index, arg in enumerate(raw_args)
        ]
        helper_name = self.wave_helper_name(operation, result_type, arg_types)
        if helper_name in self.helper_functions:
            return helper_name

        parameter_names = self.wave_helper_parameter_names(operation, len(arg_types))
        params = [
            f"{arg_type} {parameter_name}"
            for parameter_name, arg_type in zip(parameter_names, arg_types)
        ]
        body = self.wave_helper_body(operation, result_type)
        helper = (
            f"__device__ inline {result_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"{body}"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def cuda_wave_operation_diagnostic(self, operation, raw_args):
        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveBallot",
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type is not None and self.map_type(arg_type) != "bool":
                return self.unsupported_cuda_wave_operation(
                    operation, raw_args, "requires bool predicate input"
                )
            if (
                operation.startswith("WaveMultiPrefix")
                and len(raw_args) > 1
                and self.map_type(self.expression_result_type(raw_args[1])) != "uint4"
            ):
                return self.unsupported_cuda_wave_operation(
                    operation, raw_args, "requires uvec4 partition mask"
                )
            return None

        scalar_ops = {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveActiveAllEqual",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "WavePrefixSum",
            "WavePrefixProduct",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
            "WaveMatch",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
        if operation not in scalar_ops or not raw_args:
            return None

        value_type = self.wave_argument_type(operation, 0, raw_args[0])
        if self.vector_type_info(value_type) is not None:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires scalar input"
            )

        kind = self.cuda_wave_scalar_kind(value_type)
        if kind is None:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, f"does not support {value_type or 'unknown'} input"
            )

        if operation in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        } and kind not in {"signed", "unsigned"}:
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires integer scalar input"
            )

        if (
            operation
            in {
                "WaveActiveSum",
                "WaveActiveProduct",
                "WaveActiveMin",
                "WaveActiveMax",
                "WavePrefixSum",
                "WavePrefixProduct",
                "WaveMultiPrefixSum",
                "WaveMultiPrefixProduct",
            }
            and kind == "bool"
        ):
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires numeric scalar input"
            )

        if (
            operation.startswith("WaveMultiPrefix")
            and len(raw_args) > 1
            and self.map_type(self.expression_result_type(raw_args[1])) != "uint4"
        ):
            return self.unsupported_cuda_wave_operation(
                operation, raw_args, "requires uvec4 partition mask"
            )

        return None

    def unsupported_cuda_wave_operation(self, operation, raw_args, reason):
        result_type = self.wave_result_type(operation, raw_args)
        fallback = self.diagnostic_zero_value_for_type(result_type)
        return f"/* unsupported CUDA wave intrinsic {operation}: {reason} */ {fallback}"

    def cuda_wave_scalar_kind(self, type_name):
        mapped_type = self.map_type(type_name)
        if mapped_type in {"bool"}:
            return "bool"
        if mapped_type in {
            "char",
            "short",
            "int",
            "long",
            "long long",
            "i8",
            "i16",
            "i32",
            "i64",
        }:
            return "signed"
        if mapped_type in {
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
            "uint",
            "u8",
            "u16",
            "u32",
            "u64",
        }:
            return "unsigned"
        if mapped_type in {"float", "double", "half"}:
            return "float"
        return None

    def wave_result_type(self, operation, raw_args):
        if operation in CUDA_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in CUDA_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in CUDA_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if raw_args:
            return self.expression_result_type(raw_args[0]) or "uint"
        return "uint"

    def wave_argument_type(self, operation, index, arg):
        if index == 0 and operation in CUDA_WAVE_PREDICATE_ARGUMENT_OPS:
            return self.map_type("bool")
        if index == 1 and operation.startswith("WaveMultiPrefix"):
            return self.map_type("uvec4")
        if index == 1 and operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return self.map_type("uint")
        return self.map_type(self.expression_result_type(arg) or "uint")

    def wave_helper_name(self, operation, result_type, arg_types):
        suffix_parts = [self.wave_type_suffix(result_type)]
        suffix_parts.extend(self.wave_type_suffix(arg_type) for arg_type in arg_types)
        return (
            f"cgl_cuda_{self.wave_operation_suffix(operation)}_{'_'.join(suffix_parts)}"
        )

    def wave_operation_suffix(self, operation):
        name = operation[0].lower() + operation[1:]
        suffix = []
        for char in name:
            if char.isupper():
                suffix.append("_")
                suffix.append(char.lower())
            else:
                suffix.append(char)
        return "".join(suffix)

    def wave_type_suffix(self, type_name):
        suffix = type_name.replace("unsigned int", "uint")
        suffix = suffix.replace(" ", "_").replace("*", "_ptr")
        suffix = suffix.replace("&", "_ref").replace("::", "_")
        return "".join(char if char.isalnum() else "_" for char in suffix).strip("_")

    def wave_helper_parameter_names(self, operation, arg_count):
        if arg_count == 1:
            if operation in CUDA_WAVE_PREDICATE_ARGUMENT_OPS:
                return ["predicate"]
            return ["value"]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return ["value", "lane"]
        if operation == "WaveMultiPrefixCountBits":
            return ["predicate", "mask"]
        return ["value", "mask"]

    def wave_helper_body(self, operation, result_type):
        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            vote = "__all_sync" if operation == "WaveActiveAllTrue" else "__any_sync"
            return (
                "    unsigned int mask = __activemask();\n"
                f"    return ({vote}(mask, predicate) != 0);\n"
            )

        if operation == "WaveActiveAllEqual":
            return (
                "    unsigned int mask = __activemask();\n"
                f"    {result_type} first = __shfl_sync(mask, value, 0);\n"
                "    return (__all_sync(mask, value == first) != 0);\n"
            )

        if operation == "WaveActiveBallot":
            return (
                "    unsigned int bits = __ballot_sync(__activemask(), predicate);\n"
                "    return make_uint4(bits, 0u, 0u, 0u);\n"
            )

        if operation == "WaveActiveCountBits":
            return (
                "    unsigned int bits = __ballot_sync(__activemask(), predicate);\n"
                "    return static_cast<uint>(__popc(bits));\n"
            )

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveActiveMin",
            "WaveActiveMax",
        }:
            return self.wave_active_reduction_helper_body(operation, result_type)

        if operation in {"WavePrefixSum", "WavePrefixProduct"}:
            return self.wave_prefix_reduction_helper_body(operation, result_type)

        if operation == "WavePrefixCountBits":
            return self.wave_prefix_count_bits_helper_body()

        if operation == "WaveReadLaneAt":
            return "    return __shfl_sync(__activemask(), value, static_cast<int>(lane));\n"

        if operation == "WaveReadLaneFirst":
            return (
                "    unsigned int mask = __activemask();\n"
                "    int first_lane = __ffs(mask) - 1;\n"
                "    return __shfl_sync(mask, value, first_lane);\n"
            )

        if operation in {
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            lane_mask = {
                "QuadReadAcrossX": "1",
                "QuadReadAcrossY": "2",
                "QuadReadAcrossDiagonal": "3",
            }[operation]
            return f"    return __shfl_xor_sync(__activemask(), value, {lane_mask});\n"

        if operation == "QuadReadLaneAt":
            return (
                "    uint lane_index = (threadIdx.x & (warpSize - 1));\n"
                "    uint source_lane = (lane_index & ~3u) | (lane & 3u);\n"
                "    return __shfl_sync(\n"
                "        __activemask(), value, static_cast<int>(source_lane));\n"
            )

        if operation == "WaveMatch":
            return self.wave_match_helper_body(result_type)

        if operation in {
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            return self.wave_multi_prefix_helper_body(operation, result_type)

        if operation == "WaveMultiPrefixCountBits":
            return self.wave_multi_prefix_count_bits_helper_body()

        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "    return predicate;\n"
        if result_type == "bool":
            return "    return false;\n"
        return "    return value;\n"

    def wave_active_reduction_helper_body(self, operation, result_type):
        combine = self.wave_combine_expression(operation, "result", "other")
        return (
            "    unsigned int mask = __activemask();\n"
            f"    {result_type} result = value;\n"
            "    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {\n"
            f"        {result_type} other = __shfl_down_sync(mask, result, offset);\n"
            f"        result = {combine};\n"
            "    }\n"
            "    return result;\n"
        )

    def wave_prefix_reduction_helper_body(self, operation, result_type):
        identity = self.wave_identity_expression(operation, result_type)
        combine_running = self.wave_combine_expression(operation, "running", "other")
        combine_result = self.wave_combine_expression(operation, "result", "other")
        return (
            "    unsigned int mask = __activemask();\n"
            "    uint lane_index = (threadIdx.x & (warpSize - 1));\n"
            f"    {result_type} running = value;\n"
            f"    {result_type} result = {identity};\n"
            "    for (int offset = 1; offset < warpSize; offset <<= 1) {\n"
            f"        {result_type} other = __shfl_up_sync(mask, running, offset);\n"
            "        if (lane_index >= static_cast<uint>(offset)) {\n"
            f"            result = {combine_result};\n"
            f"            running = {combine_running};\n"
            "        }\n"
            "    }\n"
            "    return result;\n"
        )

    def wave_prefix_count_bits_helper_body(self):
        return (
            "    unsigned int mask = __activemask();\n"
            "    uint lane_index = (threadIdx.x & (warpSize - 1));\n"
            "    unsigned int bits = __ballot_sync(mask, predicate);\n"
            "    unsigned int lower_mask =\n"
            "        lane_index == 0u ? 0u : ((1u << lane_index) - 1u);\n"
            "    return static_cast<uint>(__popc(bits & lower_mask));\n"
        )

    def wave_match_helper_body(self, _result_type):
        return (
            "    unsigned int active = __activemask();\n"
            "    unsigned int matching = 0u;\n"
            "    for (int candidate = 0; candidate < warpSize; ++candidate) {\n"
            "        auto other = __shfl_sync(active, value, candidate);\n"
            "        if (value == other) {\n"
            "            matching |= (1u << candidate);\n"
            "        }\n"
            "    }\n"
            "    return make_uint4(matching, 0u, 0u, 0u);\n"
        )

    def wave_multi_prefix_helper_body(self, operation, result_type):
        identity = self.wave_identity_expression(operation, result_type)
        combine = self.wave_combine_expression(operation, "result", "other")
        return (
            "    unsigned int active = __activemask();\n"
            "    unsigned int partition = mask.x & active;\n"
            "    uint lane_index = (threadIdx.x & (warpSize - 1));\n"
            f"    {result_type} result = {identity};\n"
            "    for (int candidate = 0; candidate < warpSize; ++candidate) {\n"
            "        auto other = __shfl_sync(active, value, candidate);\n"
            "        if (candidate < static_cast<int>(lane_index) &&\n"
            "            (partition & (1u << candidate)) != 0u) {\n"
            f"            result = {combine};\n"
            "        }\n"
            "    }\n"
            "    return result;\n"
        )

    def wave_multi_prefix_count_bits_helper_body(self):
        return (
            "    unsigned int active = __activemask();\n"
            "    unsigned int partition = mask.x & active;\n"
            "    uint lane_index = (threadIdx.x & (warpSize - 1));\n"
            "    unsigned int bits = __ballot_sync(active, predicate);\n"
            "    unsigned int lower_mask =\n"
            "        lane_index == 0u ? 0u : ((1u << lane_index) - 1u);\n"
            "    return static_cast<uint>(__popc(bits & partition & lower_mask));\n"
        )

    def wave_identity_expression(self, operation, result_type):
        if operation in {"WaveActiveBitAnd", "WaveMultiPrefixBitAnd"}:
            return f"static_cast<{result_type}>(~0ull)"
        if operation in {
            "WaveActiveProduct",
            "WavePrefixProduct",
            "WaveMultiPrefixProduct",
        }:
            return f"static_cast<{result_type}>(1)"
        return f"static_cast<{result_type}>(0)"

    def wave_combine_expression(self, operation, left, right):
        if operation in {"WaveActiveSum", "WavePrefixSum", "WaveMultiPrefixSum"}:
            return f"({left} + {right})"
        if operation in {
            "WaveActiveProduct",
            "WavePrefixProduct",
            "WaveMultiPrefixProduct",
        }:
            return f"({left} * {right})"
        if operation in {"WaveActiveBitAnd", "WaveMultiPrefixBitAnd"}:
            return f"({left} & {right})"
        if operation in {"WaveActiveBitOr", "WaveMultiPrefixBitOr"}:
            return f"({left} | {right})"
        if operation in {"WaveActiveBitXor", "WaveMultiPrefixBitXor"}:
            return f"({left} ^ {right})"
        if operation == "WaveActiveMin":
            return f"(({right}) < ({left}) ? ({right}) : ({left}))"
        if operation == "WaveActiveMax":
            return f"(({right}) > ({left}) ? ({right}) : ({left}))"
        return left

    def struct_constructor_arguments(self, struct_name, raw_args, args):
        """Expand struct constructors with resource-member metadata sidecars."""
        metadata_members = self.struct_query_metadata_members.get(struct_name, set())
        if not metadata_members:
            return args

        expanded_args = []
        member_types = list(self.struct_member_types.get(struct_name, {}).items())
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(raw_args) or index >= len(member_types):
                continue
            member_name, member_type = member_types[index]
            if member_name not in metadata_members:
                continue
            metadata_arg = self.query_metadata_expression(raw_args[index])
            if metadata_arg is None:
                metadata_arg = self.unavailable_query_metadata_argument(
                    member_name,
                    member_type,
                )
            expanded_args.append(metadata_arg)
        return expanded_args

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured-buffer loads and stores to CUDA pointer indexing."""
        byte_address_call = self.generate_byte_address_buffer_call(
            function_expr, func_name, raw_args, args
        )
        if byte_address_call is not None:
            return byte_address_call

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "Append" and args:
                return self.generate_structured_buffer_append(
                    buffer_expr, buffer_type, args[0], operation
                )
            if operation == "Consume":
                return self.generate_structured_buffer_consume(
                    buffer_expr, buffer_type, operation
                )
            if operation == "GetDimensions":
                return self.generate_structured_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )
            access = self.format_structured_buffer_access(buffer_expr, raw_args, args)
            if access is None:
                return None
            if operation == "Load":
                diagnostic = self.structured_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "read",
                    self.diagnostic_zero_value_for_type(
                        self.array_access_element_type(buffer_type)
                    ),
                )
                if diagnostic is not None:
                    return diagnostic
                return access
            if operation == "Store":
                diagnostic = self.structured_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "write",
                    "((void)0)",
                )
                if diagnostic is not None:
                    return diagnostic
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.unsupported_structured_buffer_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "read",
                self.diagnostic_zero_value_for_type(
                    self.array_access_element_type(buffer_type)
                ),
            )
            if diagnostic is not None:
                return diagnostic
            return f"{args[0]}[{args[1]}]"

        if func_name == "buffer_append" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_append(
                raw_args[0], buffer_type, args[1], func_name
            )

        if func_name == "buffer_consume" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_consume(
                raw_args[0], buffer_type, func_name
            )

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "write",
                "((void)0)",
            )
            if diagnostic is not None:
                return diagnostic
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.unsupported_structured_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_structured_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower structured-buffer dimensions through an explicit length sidecar."""
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA structured buffer dimensions "
                "requires explicit length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def generate_structured_buffer_append(
        self, buffer_expr, buffer_type, value, operation
    ):
        """Lower AppendStructuredBuffer writes through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        diagnostic = self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "write",
            "((void)0)",
        )
        if diagnostic is not None:
            return diagnostic
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        fallback = self.diagnostic_zero_value_for_type(
            parts[1] if parts is not None else None
        )
        diagnostic = self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "readwrite",
            fallback,
        )
        if diagnostic is not None:
            return diagnostic
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the CUDA helper for AppendStructuredBuffer operations."""
        helper_name = "cgl_append_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline void "
            f"{helper_name}(T* buffer, uint* counter, const T& value)\n"
            "{\n"
            "    uint index = atomicAdd(counter, 1u);\n"
            "    buffer[index] = value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_structured_buffer_consume_helper(self):
        """Register the CUDA helper for ConsumeStructuredBuffer operations."""
        helper_name = "cgl_consume_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline T "
            f"{helper_name}(const T* buffer, uint* counter)\n"
            "{\n"
            "    uint index = atomicSub(counter, 1u) - 1u;\n"
            "    return buffer[index];\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def structured_buffer_atomic_operations(self):
        """Return generic atomic mappings for RWStructuredBuffer element targets."""
        integer_kinds = {"int", "uint"}
        add_exchange_kinds = {*integer_kinds, "float"}
        return {
            "atomicAdd": ("atomicAdd", 2, add_exchange_kinds, "int/uint/float"),
            "atomicSub": ("atomicSub", 2, integer_kinds, "int/uint"),
            "atomicMin": ("atomicMin", 2, integer_kinds, "int/uint"),
            "atomicMax": ("atomicMax", 2, integer_kinds, "int/uint"),
            "atomicAnd": ("atomicAnd", 2, integer_kinds, "int/uint"),
            "atomicOr": ("atomicOr", 2, integer_kinds, "int/uint"),
            "atomicXor": ("atomicXor", 2, integer_kinds, "int/uint"),
            "atomicInc": ("atomicInc", 2, {"uint"}, "uint"),
            "atomicDec": ("atomicDec", 2, {"uint"}, "uint"),
            "atomicExchange": ("atomicExch", 2, add_exchange_kinds, "int/uint/float"),
            "atomicCompareExchange": ("atomicCAS", 3, integer_kinds, "int/uint"),
            "atomicCompSwap": ("atomicCAS", 3, integer_kinds, "int/uint"),
        }

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to CUDA atomics."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not raw_args:
            return None

        target = self.structured_buffer_atomic_target(raw_args[0])
        if target is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_type = target["target_type"]
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        access_diagnostic = self.structured_buffer_access_diagnostic(
            func_name,
            target["buffer_type"],
            target["target_expr"],
            "readwrite",
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.cuda_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = self.visit(target["target_expr"])
        value_args = ", ".join(args[1:])
        return f"{intrinsic}(&{target_expr}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower CUDA atomics on ordinary scalar lvalues to pointer operands."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        raw_target = raw_args[0] if raw_args else None
        target_expr = (
            self.strip_address_of_expression(raw_target) if raw_target else None
        )
        address_taken = target_expr is not raw_target
        target_type = self.expression_result_type(target_expr) if target_expr else None
        indirect_info = self.cuda_indirect_type_info(target_type)
        pointee_type = (
            indirect_info["pointee_type"] if indirect_info is not None else None
        )
        atomic_type = pointee_type or target_type
        fallback = self.diagnostic_zero_value_for_type(atomic_type)

        if len(args) != required_arg_count:
            return self.unsupported_plain_atomic_call(
                func_name,
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        if target_expr is None or (
            pointee_type is None and not self.is_plain_atomic_lvalue(target_expr)
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                "requires assignable scalar target",
                fallback,
            )

        if (
            address_taken
            and indirect_info is not None
            and indirect_info["kind"] == "pointer"
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {indirect_info['type_label']} requires pointer target, "
                "not address of pointer",
                fallback,
            )

        readonly_reason = self.plain_atomic_readonly_target_reason(
            target_expr, indirect_info
        )
        if readonly_reason is not None:
            return self.unsupported_plain_atomic_call(
                func_name,
                readonly_reason,
                fallback,
            )
        access_diagnostic = self.glsl_buffer_block_read_write_diagnostic(
            target_expr, func_name, fallback
        )
        if access_diagnostic is not None:
            return access_diagnostic

        scalar_kind = self.cuda_atomic_scalar_kind(atomic_type)
        if scalar_kind not in supported_kinds:
            type_label = (
                indirect_info["type_label"]
                if indirect_info is not None
                else self.type_name_string(atomic_type) or "unknown target"
            )
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {type_label} requires supported scalar "
                f"{supported_kinds_label} target",
                fallback,
            )

        target_code = self.visit(target_expr)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            target_pointer = target_code
        else:
            target_pointer = f"&{target_code}"
        value_args = ", ".join(args[1:])
        return f"{intrinsic}({target_pointer}, {value_args})"

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a CUDA atomic."""
        return isinstance(
            expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        )

    def cuda_atomic_pointer_pointee_type(self, type_name):
        """Return a scalar pointee type for pointer-typed CUDA atomic operands."""
        info = self.cuda_indirect_type_info(type_name)
        return info["pointee_type"] if info is not None else None

    def cuda_indirect_type_info(self, type_name):
        """Return pointer/reference metadata for CUDA indirect type spellings."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if type_name.startswith("ptr<") and type_name.endswith(">"):
            pointee = type_name[4:-1].strip()
            return {
                "kind": "pointer",
                "pointee_type": pointee,
                "readonly": False,
                "type_label": f"{pointee}*",
            }

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        for candidate in (type_name, mapped_type):
            info = self.cuda_pointer_or_reference_type_info(candidate)
            if info is not None:
                return info
        return None

    def cuda_pointer_or_reference_type_info(self, type_name):
        """Parse a CUDA pointer/reference spelling into target metadata."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        stripped = type_name.strip()
        for suffix, kind in (("*", "pointer"), ("&", "reference")):
            if not stripped.endswith(suffix):
                continue
            pointee = stripped[: -len(suffix)].strip()
            readonly = pointee.startswith("const ")
            if readonly:
                pointee = pointee[len("const ") :].strip()
            return {
                "kind": kind,
                "pointee_type": pointee,
                "readonly": readonly,
                "type_label": stripped,
            }
        return None

    def plain_atomic_readonly_target_reason(self, target_expr, indirect_info):
        """Return a diagnostic reason for readonly plain atomic targets."""
        if indirect_info is not None and indirect_info["readonly"]:
            if indirect_info["kind"] == "reference":
                return (
                    f"on {indirect_info['type_label']} requires mutable "
                    "reference target"
                )
            return (
                f"on {indirect_info['type_label']} requires writable pointer " "target"
            )

        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            array_type = self.expression_result_type(array_expr)
            array_info = self.cuda_indirect_type_info(array_type)
            if array_info is not None and array_info["readonly"]:
                if array_info["kind"] == "reference":
                    return (
                        f"on {array_info['type_label']} requires mutable "
                        "reference target"
                    )
                return (
                    f"on {array_info['type_label']} requires writable pointer " "target"
                )
        return None

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Return diagnostic code for unsupported plain CUDA atomics."""
        return (
            f"/* unsupported {self.resource_backend_name()} atomic call: "
            f"{operation} {reason} */ {fallback}"
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
        target_expr = self.strip_address_of_expression(target_expr)
        element_access = self.structured_buffer_element_access(target_expr)
        if element_access is None:
            return None

        array_expr = getattr(
            element_access, "array_expr", getattr(element_access, "array", None)
        )
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        target_type = self.expression_result_type(target_expr) or parts[1]
        return {
            "buffer_type": buffer_type,
            "target_expr": target_expr,
            "target_type": target_type,
        }

    def strip_address_of_expression(self, expr):
        """Return the lvalue inside an address-of expression."""
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "&":
                return getattr(expr, "operand", expr)
        return expr

    def structured_buffer_element_access(self, target_expr):
        """Return the structured-buffer element access inside an atomic lvalue."""
        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            buffer_type = self.expression_result_type(array_expr)
            if self.structured_buffer_type_parts(buffer_type) is not None:
                return target_expr

        if isinstance(target_expr, MemberAccessNode):
            object_expr = getattr(
                target_expr, "object_expr", getattr(target_expr, "object", None)
            )
            return self.structured_buffer_element_access(object_expr)

        return None

    def cuda_atomic_scalar_kind(self, type_name):
        """Return the CUDA atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
        if type_name in {"int", "i32"} or mapped_type == "int":
            return "int"
        if type_name in {"float", "f32"} or mapped_type == "float":
            return "float"
        return None

    def unsupported_structured_buffer_atomic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for unsupported structured-buffer atomics."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer atomic: "
            f"{operation} on {buffer_type} {reason} */ {fallback}"
        )

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer methods to typed CUDA byte-pointer helpers."""
        member_call = self.byte_address_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "GetDimensions":
                return self.generate_byte_address_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )

            if operation in self.byte_address_buffer_atomic_operations():
                return self.generate_byte_address_buffer_atomic(
                    buffer_expr, buffer_type, operation, args
                )

            component_count = self.byte_address_buffer_component_count(operation)
            if component_count is None:
                return None

            buffer_name = self.visit(buffer_expr)
            if operation.startswith("Load") and len(args) >= 1:
                diagnostic = self.byte_address_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "read",
                    self.diagnostic_zero_value_for_type(
                        self.byte_address_buffer_value_type(component_count)
                    ),
                )
                if diagnostic is not None:
                    return diagnostic
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                diagnostic = self.byte_address_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "write",
                    "((void)0)",
                )
                if diagnostic is not None:
                    return diagnostic
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.unsupported_byte_address_buffer_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "read",
                "0u",
            )
            if diagnostic is not None:
                return diagnostic
            helper_name = self.require_byte_address_load_helper(1)
            return f"{helper_name}({args[0]}, {args[1]})"

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            return self.generate_byte_address_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "write",
                "((void)0)",
            )
            if diagnostic is not None:
                return diagnostic
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.unsupported_byte_address_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def byte_address_buffer_atomic_operations(self):
        """Return supported RWByteAddressBuffer atomic method mappings."""
        return {
            "InterlockedAdd": ("add", "atomicAdd", 2),
            "InterlockedMin": ("min", "atomicMin", 2),
            "InterlockedMax": ("max", "atomicMax", 2),
            "InterlockedAnd": ("and", "atomicAnd", 2),
            "InterlockedOr": ("or", "atomicOr", 2),
            "InterlockedXor": ("xor", "atomicXor", 2),
            "InterlockedExchange": ("exchange", "atomicExch", 2),
            "InterlockedCompareExchange": ("compare_exchange", "atomicCAS", 3),
        }

    def generate_byte_address_buffer_atomic(
        self, buffer_expr, buffer_type, operation, args
    ):
        """Lower RWByteAddressBuffer Interlocked* methods to CUDA atomics."""
        operation_info = self.byte_address_buffer_atomic_operations().get(operation)
        if operation_info is None:
            return None

        operation_name, intrinsic, required_args = operation_info
        has_out_arg = len(args) == required_args + 1
        fallback = "((void)0)" if has_out_arg else "0u"
        if len(args) < required_args or len(args) > required_args + 1:
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )
        access_diagnostic = self.byte_address_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "readwrite",
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic
        if not self.byte_address_buffer_is_writable(buffer_type):
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )

        helper_name = self.require_byte_address_atomic_helper(operation_name, intrinsic)
        buffer_name = self.visit(buffer_expr)
        helper_args = [buffer_name, *args[:required_args]]
        call = f"{helper_name}({', '.join(helper_args)})"
        if has_out_arg:
            return f"{args[required_args]} = {call}"
        return call

    def byte_address_atomic_helper_name(self, operation):
        """Return a stable CUDA helper name for a byte-address atomic operation."""
        return f"cgl_byte_address_atomic_{operation}_uint"

    def require_byte_address_atomic_helper(self, operation, intrinsic):
        """Register a CUDA byte-address atomic helper and return its name."""
        helper_name = self.byte_address_atomic_helper_name(operation)
        if helper_name in self.helper_functions:
            return helper_name

        pointer_expr = "reinterpret_cast<unsigned int*>(buffer + offset)"
        if operation == "compare_exchange":
            helper = (
                "__device__ inline uint "
                f"{helper_name}(unsigned char* buffer, uint offset, "
                "uint compare_value, uint value)\n"
                "{\n"
                f"    return {intrinsic}({pointer_expr}, compare_value, value);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        helper = (
            "__device__ inline uint "
            f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
            "{\n"
            f"    return {intrinsic}({pointer_expr}, value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_member_call(self, function_expr):
        """Return byte-address buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def byte_address_buffer_component_count(self, operation):
        """Return the uint lane count for Load/Store byte-address operations."""
        operation_counts = {
            "Load": 1,
            "Load2": 2,
            "Load3": 3,
            "Load4": 4,
            "Store": 1,
            "Store2": 2,
            "Store3": 3,
            "Store4": 4,
        }
        return operation_counts.get(operation)

    def byte_address_buffer_value_type(self, component_count):
        """Return the CUDA uint vector type for a byte-address operation."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def byte_address_helper_suffix(self, component_count):
        """Return a stable helper-name suffix for byte-address operations."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def require_byte_address_load_helper(self, component_count):
        """Register a CUDA byte-address load helper and return its name."""
        helper_name = (
            f"cgl_byte_address_load_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        if component_count == 1:
            helper = (
                "__device__ inline uint "
                f"{helper_name}(const unsigned char* buffer, uint offset)\n"
                "{\n"
                "    return *reinterpret_cast<const uint*>(buffer + offset);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_load_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        args = [
            f"{scalar_helper}(buffer, offset + {index * 4}u)"
            for index, _ in enumerate(components)
        ]
        value_type = self.byte_address_buffer_value_type(component_count)
        constructor = self.convert_builtin_function(value_type)
        helper = (
            f"__device__ inline {value_type} "
            f"{helper_name}(const unsigned char* buffer, uint offset)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_byte_address_store_helper(self, component_count):
        """Register a CUDA byte-address store helper and return its name."""
        helper_name = (
            f"cgl_byte_address_store_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        value_type = self.byte_address_buffer_value_type(component_count)
        if component_count == 1:
            helper = (
                "__device__ inline void "
                f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
                "{\n"
                "    *reinterpret_cast<uint*>(buffer + offset) = value;\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_store_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        lines = [
            f"    {scalar_helper}(buffer, offset + {index * 4}u, value.{component});"
            for index, component in enumerate(components)
        ]
        helper = (
            f"__device__ inline void {helper_name}"
            f"(unsigned char* buffer, uint offset, {value_type} value)\n"
            "{\n" + "\n".join(lines) + "\n}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def unsupported_byte_address_buffer_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for unsupported byte-address buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def format_structured_buffer_access(self, buffer_expr, raw_args, args):
        """Format one CUDA pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def unsupported_structured_buffer_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for unsupported structured-buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def generate_lambda_expression(self, args):
        """Render CrossGL's pseudo-lambda as a CUDA device lambda."""
        if not args:
            return "[&] __device__ () {}"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"[&] __device__ ({params}) {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            param_name = self.lambda_fallback_parameter_name(raw)
            return f"auto {param_name}" if param_name else "auto"

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        if raw:
            return f"{{ return {raw}; }}"
        return "{}"

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw:
            return None
        if any(char in raw for char in "{}()"):
            return None
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "auto"
        return self.convert_crossgl_type_to_cuda(type_name)

    def lambda_fallback_parameter_name(self, raw):
        if not raw:
            return ""
        candidate = raw.rsplit(None, 1)[-1]
        if candidate.isidentifier():
            return candidate
        return raw

    def generate_fract_call(self, raw_args, args):
        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_fract_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"
            return None

        scalar_type = self.fract_scalar_type(arg_type)
        if scalar_type is None:
            return None
        helper_name = self.require_scalar_fract_helper(scalar_type)
        return f"{helper_name}({args[0]})"

    def fract_scalar_type(self, type_name):
        if type_name is not None and not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = (
            self.convert_crossgl_type_to_cuda(type_name)
            if type_name is not None
            else None
        )
        if mapped_type == "double" or type_name in {"double", "f64"}:
            return "double"
        if mapped_type == "float" or type_name is None:
            return "float"
        return None

    def require_scalar_fract_helper(self, scalar_type):
        helper_name = f"cgl_fract_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return value - {floor_name}(value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_fract_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_fract"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_helper_name = self.require_scalar_fract_helper(component_type)
        components = [
            f"{scalar_helper_name}(value.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_constructor_args(self, vector_info, raw_args, args):
        """Flatten vector arguments passed to CUDA make_* constructors."""
        if len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                return args * len(vector_info["components"])

        generated_args = []
        for raw_arg, arg_expr in zip(raw_args, args):
            arg_info = self.vector_type_info(self.expression_result_type(raw_arg))
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        arg_info,
                    )
                )

            if len(generated_args) >= len(vector_info["components"]):
                return generated_args[: len(vector_info["components"])]

        return generated_args

    def vector_argument_lane_expressions(self, raw_arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(raw_arg)
        if swizzle_components is not None:
            object_node = getattr(
                raw_arg,
                "object_expr",
                getattr(raw_arg, "object", None),
            )
            role = self.cuda_compute_builtin_role_for_name(
                getattr(object_node, "name", None)
            )
            if role is not None:
                lanes = [
                    self.cuda_compute_builtin_expression(role, component)
                    for component in swizzle_components
                ]
                if all(lane is not None for lane in lanes):
                    return lanes
            object_expr = self.visit(object_node)
            return [f"{object_expr}.{component}" for component in swizzle_components]

        return [f"{arg_expr}.{component}" for component in arg_info["components"]]

    def member_swizzle_components(self, node):
        if not isinstance(node, MemberAccessNode):
            return None

        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        return components

    def generate_clamp_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        if not value_info:
            scalar_type = self.clamp_scalar_type(value_type)
            scalar_call = self.generate_scalar_clamp_single_eval_call(
                scalar_type,
                raw_args,
                args,
            )
            if scalar_call is not None:
                return scalar_call
            return self.format_clamp_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        min_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        max_info = self.vector_type_info(self.expression_result_type(raw_args[2]))
        if min_info and len(min_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        if max_info and len(max_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"

        helper_name = self.require_vector_clamp_helper(
            value_info,
            min_is_vector=min_info is not None,
            max_is_vector=max_info is not None,
        )
        if helper_name is None:
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def generate_saturate_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        component_type = (
            value_info["component_type"]
            if value_info is not None
            else self.scalar_component_type(value_type)
        )
        if component_type not in {"float", "double"}:
            return None
        return self.generate_clamp_call(
            [raw_args[0], None, None],
            [args[0], "0.0", "1.0"],
        )

    def generate_mix_call(self, raw_args, args):
        left_type = self.expression_result_type(raw_args[0])
        right_type = self.expression_result_type(raw_args[1])
        factor_type = self.expression_result_type(raw_args[2])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        factor_info = self.vector_type_info(factor_type)

        if not left_info and not right_info and not factor_info:
            scalar_bool_mix = self.lower_bool_scalar_mix_operation(
                raw_args[0],
                args[0],
                raw_args[1],
                args[1],
                raw_args[2],
                args[2],
            )
            if scalar_bool_mix is not None:
                return scalar_bool_mix
            scalar_mix_call = self.generate_scalar_mix_single_eval_call(
                raw_args,
                args,
            )
            if scalar_mix_call is not None:
                return scalar_mix_call
            return self.format_mix_component(args[0], args[1], args[2])

        if left_info is None or right_info is None:
            return None
        if (
            len(left_info["components"]) != len(right_info["components"])
            or left_info["component_type"] != right_info["component_type"]
        ):
            return None

        bool_mix = self.lower_bool_vector_mix_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            raw_args[2],
            args[2],
            left_info,
            right_info,
            factor_info,
        )
        if bool_mix is not None:
            return bool_mix

        if factor_info is not None and (
            len(factor_info["components"]) != len(left_info["components"])
            or factor_info["component_type"] != left_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_mix_helper(
            left_info,
            factor_is_vector=factor_info is not None,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def require_vector_mix_helper(self, vector_info, factor_is_vector):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        factor_shape = "vector" if factor_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_mix_{factor_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        factor_type = (
            vector_type
            if factor_is_vector
            else self.vector_scalar_parameter_type(vector_info)
        )
        components = []
        for component in vector_info["components"]:
            factor_component = f"a.{component}" if factor_is_vector else "a"
            components.append(
                self.format_mix_component(
                    f"x.{component}",
                    f"y.{component}",
                    factor_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} x, {vector_type} y, {factor_type} a)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_mix_component(self, left, right, factor):
        return f"({left} + (({right} - {left}) * {factor}))"

    def generate_atan2_call(self, raw_args, args):
        y_type = self.expression_result_type(raw_args[0])
        x_type = self.expression_result_type(raw_args[1])
        y_info = self.vector_type_info(y_type)
        x_info = self.vector_type_info(x_type)

        if y_info is None and x_info is None:
            return None
        if y_info is None or x_info is None:
            return None
        if (
            len(y_info["components"]) != len(x_info["components"])
            or y_info["component_type"] != x_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_atan2_helper(y_info)
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def require_vector_atan2_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_atan2"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_func = "atan2" if component_type == "double" else "atan2f"
        components = [
            f"{scalar_func}(y.{component}, x.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} y, {vector_type} x)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def clamp_scalar_type(self, type_name):
        if type_name is None:
            return "float"
        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        if mapped_type in {"float", "double"}:
            return mapped_type
        if mapped_type in {
            "bool",
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "uint",
            "unsigned int",
            "long long",
            "unsigned long long",
        }:
            return mapped_type
        return "float"

    def require_vector_clamp_helper(self, vector_info, min_is_vector, max_is_vector):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        min_shape = "vector" if min_is_vector else "scalar"
        max_shape = "vector" if max_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_clamp"
        if min_shape != "vector" or max_shape != "vector":
            helper_name += f"_{min_shape}_min_{max_shape}_max"
        if helper_name in self.helper_functions:
            return helper_name

        min_type = vector_type if min_is_vector else scalar_type
        max_type = vector_type if max_is_vector else scalar_type
        components = vector_info["components"]
        constructor = vector_info["constructor"]
        args = []
        for component in components:
            value_component = f"value.{component}"
            min_component = f"min_value.{component}" if min_is_vector else "min_value"
            max_component = f"max_value.{component}" if max_is_vector else "max_value"
            args.append(
                self.format_clamp_component(
                    vector_info["component_type"],
                    value_component,
                    min_component,
                    max_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} value, {min_type} min_value, {max_type} max_value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_clamp_component(self, component_type, value, min_value, max_value):
        if component_type == "float":
            return f"fmaxf({min_value}, fminf({max_value}, {value}))"
        if component_type == "double":
            return f"fmax({min_value}, fmin({max_value}, {value}))"
        return (
            f"(({value}) < ({min_value}) ? ({min_value}) : "
            f"(({value}) > ({max_value}) ? ({max_value}) : ({value})))"
        )

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        raw_object_name = getattr(object_node, "name", None)
        if raw_object_name is not None:
            alias_component = self.cuda_stage_builtin_alias_expression(
                raw_object_name, node.member
            )
            if alias_component is not None:
                return alias_component
            direct_builtin = self.cuda_compute_builtin_expression(
                self.cuda_compute_builtin_role_for_name(raw_object_name), node.member
            )
            if direct_builtin is not None:
                return direct_builtin
            raw_member_access = f"{raw_object_name}.{node.member}"
            if raw_member_access in self.builtin_map:
                return self.builtin_map[raw_member_access]

        obj = self.visit(object_node)
        member_access = f"{obj}.{node.member}"
        if member_access in self.builtin_map:
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, obj)
        if swizzle is not None:
            return swizzle

        return member_access

    def visit_PointerAccessNode(self, node):
        """Visit C-style pointer member access."""
        pointer_expr = self.visit(getattr(node, "pointer_expr", None))
        return f"{pointer_expr}->{node.member}"

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        components = self.member_swizzle_components(node)
        if components is None:
            return None

        if len(components) == 1:
            return f"{object_expr}.{components[0]}"

        result_type = self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        swizzle_call = self.generate_vector_swizzle_single_eval_call(
            result_info,
            vector_info,
            object_node,
            object_expr,
            components,
        )
        if swizzle_call is not None:
            return swizzle_call

        args = [f"{object_expr}.{component}" for component in components]
        return f"{result_info['constructor']}({', '.join(args)})"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        if hasattr(node, "array_expr"):
            array = self.visit(node.array_expr)
        else:
            array = self.visit(node.array)

        if hasattr(node, "index_expr"):
            index = self.visit(node.index_expr)
        else:
            index = self.visit(node.index)

        return f"{array}[{index}]"

    def visit_ArrayLiteralNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "then_branch"):
            self.emit_body(node.then_branch)
        elif hasattr(node, "if_body"):
            self.emit_body(node.if_body)

        self.indent_level -= 1

        if hasattr(node, "else_branch") and node.else_branch:
            self.emit("} else {")
            self.indent_level += 1

            self.emit_body(node.else_branch)

            self.indent_level -= 1
        elif hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init_str = ""
        if node.init:
            init_str = self.format_for_clause_expression(node.init)

        condition_str = ""
        if node.condition:
            condition_str = self.visit(node.condition)

        update_str = ""
        if node.update:
            update_str = self.format_for_clause_expression(node.update)

        self.emit(f"for ({init_str}; {condition_str}; {update_str}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_ForInNode(self, node):
        """Lower CrossGL for-in loops to counted CUDA loops."""
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.visit(iterable.start)
            end = self.visit(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.visit(iterable)
            comparator = "<"

        self.emit(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern}) {{"
        )
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_DoWhileNode(self, node):
        """Visit do-while loop."""
        self.emit("do {")

        self.indent_level += 1
        if hasattr(node, "body"):
            self.emit_body(node.body)
        self.indent_level -= 1

        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"}} while ({condition});")

    def visit_LoopNode(self, node):
        """Lower CrossGL infinite loops to CUDA while loops."""
        self.emit("while (true) {")

        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1

        self.emit("}")

    def visit_SwitchNode(self, node):
        """Visit switch statement"""
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1

        self.emit("}")

    def visit_MatchNode(self, node):
        """Lower CrossGL match statements to CUDA switch or if/else code."""
        try:
            if is_switch_lowerable_match(node):
                code = generate_switch_match(self, node, self.indent_level)
            else:
                code = generate_ordered_conditional_match(
                    self, node, self.indent_level, "CUDA"
                )
        except ValueError as error:
            reason = self.cuda_match_error_reason(error)
            code = (
                f"{'    ' * self.indent_level}"
                f"/* unsupported CUDA match statement: {reason} */\n"
            )
        self.emit_generated_code(code)

    def cuda_match_error_reason(self, error):
        reason = str(error)
        prefixes = (
            "Unsupported match expression for CUDA codegen; ",
            "Unsupported match arm for CUDA codegen; ",
        )
        for prefix in prefixes:
            if reason.startswith(prefix):
                return reason[len(prefix) :]
        return reason

    def generate_match_expression_diagnostic_assignment(
        self, target_variable, target_type, reason, indent, _target_name
    ):
        indent_str = "    " * indent
        fallback = self.diagnostic_zero_value_for_type(target_type)
        return (
            f"{indent_str}{target_variable} = /* unsupported CUDA match "
            f"expression: {reason} */ {fallback};\n"
        )

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def statement_body_has_statements(self, body):
        return bool(self.statement_list(body))

    def generate_scoped_statement_body(self, body, indent):
        saved_output = self.output
        saved_indent = self.indent_level
        self.output = []
        self.indent_level = indent
        try:
            self.emit_body(body)
            if not self.output:
                return ""
            return "\n".join(self.output) + "\n"
        finally:
            self.output = saved_output
            self.indent_level = saved_indent

    def generate_expression(self, node):
        if node is None:
            return ""
        return self.visit(node)

    def generate_expression_with_expected(self, node, _expected_type):
        return self.generate_expression(node)

    def visit_CaseNode(self, node):
        """Visit switch case/default label"""
        if getattr(node, "value", None) is None:
            self.emit("default:")
        else:
            value = self.visit(node.value)
            self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "statements", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if self.current_function_is_kernel_entry:
            if node.value and not isinstance(node.value, MatchNode):
                self.visit(node.value)
            self.emit("return;")
            return
        if node.value:
            if isinstance(node.value, MatchNode):
                return self.emit_match_expression_return(node.value)
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def emit_match_expression_return(self, match_node):
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            raise ValueError(
                "Unsupported match expression for CUDA codegen; return context "
                "requires a concrete non-void result type"
            )

        result_name = self.next_cuda_temp_variable("match_value")
        self.emit(f"{self.format_typed_declarator(return_type, result_name)};")
        try:
            assignment = generate_match_expression_assignment(
                self,
                match_node,
                result_name,
                return_type,
                self.indent_level,
                "CUDA",
            )
        except ValueError as error:
            assignment = self.generate_match_expression_diagnostic_assignment(
                result_name,
                return_type,
                self.cuda_match_error_reason(error),
                self.indent_level,
                "CUDA",
            )
        self.emit_generated_code(assignment)
        self.emit(f"return {result_name};")

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_ContinueNode(self, node):
        """Visit continue statement"""
        self.emit("continue;")

    def visit_BlockNode(self, node):
        """Visit block statement"""
        self.emit_body(node.statements)

    def convert_crossgl_type_to_cuda(self, crossgl_type):
        """Convert CrossGL types to CUDA equivalents"""
        crossgl_type = self.type_name_string(crossgl_type)
        if crossgl_type is None:
            return "void"

        geometry_stream_type = self.cuda_geometry_stream_mapped_type(crossgl_type)
        if geometry_stream_type is not None:
            return geometry_stream_type

        tessellation_patch_type = self.cuda_tessellation_patch_mapped_type(crossgl_type)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        indirect_info = self.cuda_pointer_or_reference_type_info(crossgl_type)
        if indirect_info is not None:
            pointee_type = self.convert_crossgl_type_to_cuda(
                indirect_info["pointee_type"]
            )
            if indirect_info["kind"] == "pointer":
                prefix = "const " if indirect_info["readonly"] else ""
                return f"{prefix}{pointee_type}*"
            if indirect_info["readonly"]:
                if pointee_type.endswith("*"):
                    return f"{pointee_type} const&"
                return f"const {pointee_type}&"
            return f"{pointee_type}&"

        structured_buffer_type = self.cuda_structured_buffer_type(crossgl_type)
        if structured_buffer_type is not None:
            return structured_buffer_type

        byte_address_buffer_type = self.cuda_byte_address_buffer_type(crossgl_type)
        if byte_address_buffer_type is not None:
            return byte_address_buffer_type

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "f32": "float",
            "f64": "double",
            "int": "int",
            "float": "float",
            "double": "double",
            # Vector types (with generics)
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            # Vector types (without generics - for compatibility)
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
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
            # Texture/resource types
            "sampler": "cudaTextureObject_t",
            "sampler1D": "cudaTextureObject_t",
            "sampler1DArray": "cudaTextureObject_t",
            "sampler2D": "cudaTextureObject_t",
            "sampler3D": "cudaTextureObject_t",
            "samplerCube": "cudaTextureObject_t",
            "sampler2DArray": "cudaTextureObject_t",
            "sampler2DShadow": "cudaTextureObject_t",
            "sampler2DArrayShadow": "cudaTextureObject_t",
            "samplerCubeShadow": "cudaTextureObject_t",
            "samplerCubeArray": "cudaTextureObject_t",
            "samplerCubeArrayShadow": "cudaTextureObject_t",
            "sampler2DMS": "cudaTextureObject_t",
            "sampler2DMSArray": "cudaTextureObject_t",
            "image1D": "cudaSurfaceObject_t",
            "image1DArray": "cudaSurfaceObject_t",
            "image2D": "cudaSurfaceObject_t",
            "image3D": "cudaSurfaceObject_t",
            "imageCube": "cudaSurfaceObject_t",
            "imageCubeArray": "cudaSurfaceObject_t",
            "image2DArray": "cudaSurfaceObject_t",
            "image2DMS": "cudaSurfaceObject_t",
            "image2DMSArray": "cudaSurfaceObject_t",
            "iimage1D": "cudaSurfaceObject_t",
            "iimage1DArray": "cudaSurfaceObject_t",
            "iimage2D": "cudaSurfaceObject_t",
            "iimage3D": "cudaSurfaceObject_t",
            "iimageCube": "cudaSurfaceObject_t",
            "iimageCubeArray": "cudaSurfaceObject_t",
            "iimage2DArray": "cudaSurfaceObject_t",
            "iimage2DMS": "cudaSurfaceObject_t",
            "iimage2DMSArray": "cudaSurfaceObject_t",
            "uimage1D": "cudaSurfaceObject_t",
            "uimage1DArray": "cudaSurfaceObject_t",
            "uimage2D": "cudaSurfaceObject_t",
            "uimage3D": "cudaSurfaceObject_t",
            "uimageCube": "cudaSurfaceObject_t",
            "uimageCubeArray": "cudaSurfaceObject_t",
            "uimage2DArray": "cudaSurfaceObject_t",
            "uimage2DMS": "cudaSurfaceObject_t",
            "uimage2DMSArray": "cudaSurfaceObject_t",
            "buffer": "CUdeviceptr",
            "accelerationStructureEXT": "CglRayTracingAccelerationStructure",
            "AccelerationStructure": "CglRayTracingAccelerationStructure",
            "acceleration_structure": "CglRayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
        }

        sampled_resource_type = self.canonical_sampled_resource_type(crossgl_type)
        if sampled_resource_type:
            return type_mapping.get(sampled_resource_type, sampled_resource_type)

        storage_resource_type = self.canonical_storage_resource_type(crossgl_type)
        if storage_resource_type:
            return type_mapping.get(storage_resource_type, storage_resource_type)

        ray_resource_type = self.canonical_ray_resource_type(crossgl_type)
        if ray_resource_type:
            return self.cuda_mapped_type_result(type_mapping[ray_resource_type])

        if crossgl_type.startswith("array<") and crossgl_type.endswith(">"):
            # Extract element type and size
            inner = crossgl_type[6:-1]  # Remove "array<" and ">"
            if "," in inner:
                parts = inner.split(",")
                element_type = parts[0].strip()
                size = parts[1].strip()
                cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
                return f"{cuda_element_type}[{size}]"
            else:
                cuda_element_type = self.convert_crossgl_type_to_cuda(inner)
                return f"{cuda_element_type}*"

        if crossgl_type.startswith("ptr<") and crossgl_type.endswith(">"):
            element_type = crossgl_type[4:-1]  # Remove "ptr<" and ">"
            cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
            return f"{cuda_element_type}*"

        return self.cuda_mapped_type_result(
            type_mapping.get(crossgl_type, crossgl_type)
        )

    def cuda_mapped_type_result(self, mapped_type):
        base_type = str(mapped_type).split("[", 1)[0].strip()
        if base_type in {
            "CglRayTracingAccelerationStructure",
            "CglRayDesc",
            "CglRayQuery",
            "CglBuiltInTriangleIntersectionAttributes",
        }:
            self.require_cuda_ray_runtime_helpers()
        return mapped_type

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def get_parameter_type(self, param):
        """Return a parameter type with CUDA resource access metadata applied."""
        param_type = ResourceQueryMixin.get_parameter_type(self, param)
        return self.resource_type_with_access(param_type, param)

    def get_variable_node_type(self, node):
        """Return a variable type with CUDA resource access metadata applied."""
        var_type = ResourceQueryMixin.get_variable_node_type(self, node)
        return self.resource_type_with_access(var_type, node)

    def map_type(self, type_name):
        return self.convert_crossgl_type_to_cuda(type_name)

    def cuda_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.convert_crossgl_type_to_cuda(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, generic_arg = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.convert_crossgl_type_to_cuda(generic_arg)

    def cuda_geometry_stream_mapped_type(self, type_name):
        stream_info = self.cuda_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglCuda{stream_name}<{output_type}>"

    def format_cuda_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.cuda_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"{stream_type}& {name}"

    def cuda_tessellation_patch_type_info(self, type_name):
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None
        base_name, args = parts
        if base_name not in {"InputPatch", "OutputPatch"} or len(args) != 2:
            return None
        element_type, point_count = args
        if not element_type or not point_count:
            return None
        mapped_element_type = self.convert_crossgl_type_to_cuda(element_type)
        return base_name, mapped_element_type, point_count

    def cuda_tessellation_patch_mapped_type(self, type_name):
        patch_info = self.cuda_tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        base_name, element_type, point_count = patch_info
        helper_name = {
            "InputPatch": "CglCudaInputPatch",
            "OutputPatch": "CglCudaOutputPatch",
        }[base_name]
        return f"{helper_name}<{element_type}, {point_count}>"

    def strip_cuda_indirect_type_qualifiers(self, type_name):
        """Return the underlying type after CUDA pointer/reference wrappers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        stripped = type_name.strip()
        changed = True
        while changed and stripped:
            changed = False
            if stripped.endswith("&") or stripped.endswith("*"):
                stripped = stripped[:-1].strip()
                changed = True
            if stripped.startswith("const "):
                stripped = stripped[len("const ") :].strip()
                changed = True
            if stripped.endswith(" const"):
                stripped = stripped[: -len(" const")].strip()
                changed = True
        return stripped

    def collect_structured_buffer_length_requirements(self, root):
        """Collect buffers that need explicit length sidecar parameters."""
        functions = self.query_collect_functions(root)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_length_names = set()
        function_param_length_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_length_names[func_name])
                function_param_length_names[func_name].add(resource_name)
                return len(function_param_length_names[func_name]) != before
            before = len(global_length_names)
            global_length_names.add(resource_name)
            return len(global_length_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                buffer_expr = self.structured_buffer_dimensions_target(call)
                if buffer_expr is None:
                    continue
                mark_resource_name(func_name, self.get_expression_name(buffer_expr))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_length_names.get(
                        callee_name, set()
                    )
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_length_names[caller_name])
                            function_param_length_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_length_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_length_names)
                            global_length_names.add(arg_name)
                            changed = changed or len(global_length_names) != before

        return (
            global_length_names,
            {
                func_name: names
                for func_name, names in function_param_length_names.items()
                if names
            },
        )

    def structured_buffer_dimensions_target(self, call):
        """Return the structured-buffer expression queried by a dimensions call."""
        func_name = self.raw_function_call_name(call)
        raw_args = getattr(call, "arguments", getattr(call, "args", []))
        if func_name == "buffer_dimensions" and raw_args:
            return raw_args[0]

        function_expr = getattr(call, "function", getattr(call, "name", None))
        if isinstance(function_expr, MemberAccessNode):
            member = getattr(function_expr, "member", None)
            if member == "GetDimensions":
                return getattr(
                    function_expr,
                    "object_expr",
                    getattr(function_expr, "object", None),
                )
        return None

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name
            not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def cuda_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to CUDA pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {cuda_element_type}*"
        return f"{cuda_element_type}*"

    def structured_buffer_length_name(self, name):
        """Return the sidecar length parameter/declaration name for a buffer."""
        return f"{name}_length"

    def structured_buffer_requires_length(self, name):
        """Return whether a global structured buffer needs a length sidecar."""
        return bool(name and name in self.structured_buffer_length_names)

    def structured_buffer_parameter_requires_length(
        self, func_name, name, type_name=None
    ):
        """Return whether a function parameter needs a length sidecar."""
        if not name:
            return False
        if name not in self.structured_buffer_length_function_params.get(
            func_name, set()
        ):
            return False
        return type_name is None or self.buffer_type_supports_length(type_name)

    def buffer_type_supports_length(self, type_name):
        """Return whether a resource type can use a CUDA length sidecar."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def structured_buffer_length_declaration(self, name, type_name):
        """Format a global/local sidecar length declaration when required."""
        if not self.structured_buffer_requires_length(name):
            return None
        if not self.buffer_type_supports_length(type_name):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def structured_buffer_length_parameter(self, func_name, name, type_name):
        """Format a sidecar length parameter when required."""
        if not self.structured_buffer_parameter_requires_length(
            func_name, name, type_name
        ):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def format_structured_buffer_length_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        length_name = self.structured_buffer_length_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"const uint* {length_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("const uint*", length_name, array_suffix)

    def structured_buffer_length_data_expression(self, buffer_expr):
        """Return the sidecar length pointer paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_length = self.structured_buffer_length_data_expression(array_expr)
            if base_length is None:
                return None
            return f"{base_length}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None

        length_parameter = self.current_structured_buffer_length_parameters.get(name)
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(name):
            return self.structured_buffer_length_name(name)
        return None

    def structured_buffer_length_expression(self, buffer_expr):
        """Return the scalar length expression for a structured-buffer resource."""
        length_data = self.structured_buffer_length_data_expression(buffer_expr)
        if length_data is None:
            return None
        return f"{length_data}[0]"

    def structured_buffer_counter_name(self, name):
        """Return the sidecar counter parameter/declaration name for a buffer."""
        return f"{name}_counter"

    def structured_buffer_counter_declaration(self, name, type_name):
        """Format a global/local sidecar counter declaration when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def structured_buffer_counter_parameter(self, name, type_name):
        """Format a sidecar counter parameter when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def format_structured_buffer_counter_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        counter_name = self.structured_buffer_counter_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"uint* {counter_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("uint*", counter_name, array_suffix)

    def structured_buffer_counter_expression(self, buffer_expr):
        """Return the sidecar counter expression paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_counter = self.structured_buffer_counter_expression(array_expr)
            if base_counter is None:
                return None
            return f"{base_counter}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None
        return self.structured_buffer_counter_name(name)

    def cuda_user_function_call_arguments(self, func_name, raw_args, args):
        """Expand user calls with CUDA sidecar resource arguments."""
        callee = self.query_functions_by_name.get(func_name)
        if callee is None:
            return args + self.cuda_captured_function_call_args(func_name)

        params = getattr(callee, "parameters", getattr(callee, "params", []))
        query_params = self.query_metadata_function_params.get(func_name, set())
        expanded_args = []
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue

            param = params[index]
            param_name = getattr(param, "name", None)
            if param_name in query_params:
                metadata_arg = self.query_metadata_expression(raw_args[index])
                if metadata_arg is None:
                    metadata_arg = self.unavailable_query_metadata_argument(
                        param_name,
                        self.get_parameter_type(param),
                    )
                expanded_args.append(metadata_arg)

            param_type = self.get_parameter_type(param)
            if self.structured_buffer_parameter_requires_length(
                func_name, param_name, param_type
            ):
                length_arg = self.structured_buffer_length_data_expression(
                    raw_args[index]
                )
                if length_arg:
                    expanded_args.append(length_arg)

            if self.structured_buffer_requires_counter(param_type):
                counter_arg = self.structured_buffer_counter_expression(raw_args[index])
                if counter_arg:
                    expanded_args.append(counter_arg)
        expanded_args.extend(self.cuda_captured_function_call_args(func_name))
        return expanded_args

    def cuda_captured_function_call_args(self, func_name):
        return [
            self.visit(IdentifierNode(param.name))
            for param in self.cuda_function_capture_params.get(func_name, [])
        ]

    def unavailable_query_metadata_argument(self, param_name, param_type):
        """Return a deterministic zero metadata sidecar for untraceable resources."""
        self.resource_query_info_required = True
        type_name = self.query_type_name(param_type)
        resource_type = self.resource_base_type(type_name) or "unknown resource"
        fallback = "nullptr" if "[" in type_name else "CglResourceQueryInfo{}"
        param_label = f" argument {param_name}" if param_name else " argument"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"metadata unavailable for {resource_type}{param_label} */ {fallback}"
        )

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a CUDA fallback expression for unsupported value-producing calls."""
        if type_name is None:
            return "0"
        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "bool":
                zero = "false"
            elif component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            else:
                zero = "0"
            return f"{vector_info['constructor']}({', '.join([zero] * len(vector_info['components']))})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def cuda_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to CUDA byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None

    def array_access_element_type(self, type_name):
        """Return the element type for CUDA arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        indirect_info = self.cuda_indirect_type_info(type_name)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            return indirect_info["pointee_type"]

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def expression_result_type(self, node):
        """Infer expression result types with CUDA structured-buffer operations."""
        if isinstance(node, (IdentifierNode, VariableNode)):
            builtin_type = self.cuda_compute_builtin_type(getattr(node, "name", None))
            if builtin_type is not None:
                return builtin_type
        if isinstance(node, WaveOpNode):
            return self.wave_result_type(
                getattr(node, "operation", ""), getattr(node, "arguments", []) or []
            )
        if isinstance(node, RayTracingOpNode):
            if getattr(node, "operation", "") == "ReportHit":
                return "bool"
            return None
        if isinstance(node, RayQueryOpNode):
            operation = getattr(node, "operation", "")
            if operation == "Proceed":
                return "bool"
            if operation in {"CandidateRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, MemberAccessNode):
            object_node = getattr(
                node,
                "object_expr",
                getattr(node, "object", None),
            )
            builtin_member_type = self.cuda_compute_builtin_member_type(
                getattr(object_node, "name", None),
                getattr(node, "member", ""),
            )
            if builtin_member_type is not None:
                return builtin_member_type
            member_type = self.member_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, PointerAccessNode):
            member_type = self.pointer_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, FunctionCallNode):
            function_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(function_expr, "name", function_expr)
            if func_name in CUDA_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    getattr(node, "arguments", getattr(node, "args", [])) or [],
                )
            if func_name == "ReportHit":
                return "bool"
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        return super().expression_result_type(node)

    def struct_member_lookup_type(self, type_name):
        """Return the struct key after CUDA pointer/reference wrappers."""
        return self.strip_cuda_indirect_type_qualifiers(type_name)

    def member_access_member_type(self, node):
        """Return the member type for ``object.field`` expressions."""
        object_node = getattr(
            node,
            "object_expr",
            getattr(node, "object", None),
        )
        object_type = self.struct_member_lookup_type(
            self.expression_result_type(object_node)
        )
        return self.struct_member_types.get(object_type, {}).get(
            getattr(node, "member", "")
        )

    def pointer_access_member_type(self, node):
        """Return the member type for ``ptr->field`` expressions."""
        pointer_expr = getattr(node, "pointer_expr", None)
        pointer_type = self.expression_result_type(pointer_expr)
        pointer_info = self.cuda_indirect_type_info(pointer_type)
        if pointer_info is None:
            return None

        struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
        return self.struct_member_types.get(struct_type, {}).get(
            getattr(node, "member", "")
        )

    def resource_call_result_type(self, func_name, raw_args):
        """Infer CUDA-specific resource diagnostic result types."""
        result_type = super().resource_call_result_type(func_name, raw_args)
        if result_type is not None:
            return result_type

        if func_name in self.cuda_shadow_scalar_diagnostic_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(resource_type):
                return "float"

        if func_name in self.cuda_sampled_diagnostic_float4_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if resource_type is None:
                return None
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return "float4"

        if func_name in self.cuda_image_atomic_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            return self.cuda_image_atomic_result_type(resource_type)

        return None

    def cuda_shadow_scalar_diagnostic_calls(self):
        """Return shadow compare calls that CUDA lowers to scalar diagnostics."""
        return {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }

    def cuda_sampled_diagnostic_float4_calls(self):
        """Return sampled-resource calls that CUDA lowers to float4 diagnostics."""
        return {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetchOffset",
        }

    def cuda_image_atomic_calls(self):
        """Return image atomic calls that CUDA lowers to scalar diagnostics."""
        return {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def cuda_image_atomic_result_type(self, resource_type):
        """Return the scalar result type for CUDA image atomic diagnostics."""
        base_type = self.resource_base_type(resource_type)
        if not isinstance(base_type, str):
            return None
        if base_type.startswith("uimage"):
            return "uint"
        return "int"

    def buffer_call_result_type(self, node):
        """Infer result type for structured and byte-address buffer read calls."""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        raw_args = getattr(node, "arguments", getattr(node, "args", []))

        byte_member_call = self.byte_address_buffer_member_call(function_expr)
        if byte_member_call is not None:
            _, operation, _ = byte_member_call
            if operation == "GetDimensions":
                return "uint"
            if operation in self.byte_address_buffer_atomic_operations():
                return "uint"
            if operation.startswith("Load") and raw_args:
                component_count = self.byte_address_buffer_component_count(operation)
                if component_count is not None:
                    return self.byte_address_buffer_value_type(component_count)
            return None

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            _, operation, buffer_type = member_call
            if operation == "Load" and raw_args:
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "Consume":
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "GetDimensions" and not raw_args:
                return "uint"
            return None

        func_name = getattr(function_expr, "name", function_expr)
        if func_name == "buffer_load" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is not None:
                return "uint"
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_consume" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_dimensions" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.buffer_type_supports_length(buffer_type):
                return "uint"
        if func_name in self.structured_buffer_atomic_operations() and raw_args:
            target = self.structured_buffer_atomic_target(raw_args[0])
            if target is not None:
                return target["target_type"]
            target_type = self.expression_result_type(raw_args[0])
            if self.cuda_atomic_scalar_kind(target_type) is not None:
                return target_type
        return None

    def canonical_sampled_resource_type(self, type_name):
        """Return the sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return the image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def canonical_ray_resource_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        if base_type in {
            "accelerationStructureEXT",
            "AccelerationStructure",
            "acceleration_structure",
            "RaytracingAccelerationStructure",
            "RayTracingAccelerationStructure",
        }:
            return "RayTracingAccelerationStructure"
        return None

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return (
            self.canonical_sampled_resource_type(base_type)
            or self.canonical_storage_resource_type(base_type)
            or self.canonical_ray_resource_type(base_type)
            or base_type
        )

    def query_type_name(self, type_name):
        """Return a CUDA-aware type string for resource query decisions."""
        return self.type_name_string(type_name)

    def dimension_query_spec(self, type_name):
        """Return CUDA resource query metadata for supported image shapes."""
        type_name = self.resource_base_type(type_name)
        spec = ResourceQueryMixin.dimension_query_spec(self, type_name)
        if spec is not None:
            return spec

        specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def is_queryable_resource_type(self, type_name):
        """Return whether a resource type can use CUDA query metadata."""
        if type_name is None:
            return False
        resource_type = self.resource_base_type(self.query_type_name(type_name))
        return self.dimension_query_spec(resource_type) is not None

    def collect_function_resource_aliases(self, func):
        """Collect local queryable resource declarations and initializers."""
        local_names = set()
        alias_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            var_type = self.get_variable_node_type(node)
            if not name or not self.is_queryable_resource_type(var_type):
                continue
            local_names.add(name)
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            if initial_value is not None:
                alias_sources[name] = initial_value

        return local_names, alias_sources

    def collect_function_resource_assignments(self, func, local_resource_names):
        """Collect simple assignments into local queryable resource variables."""
        assignment_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, AssignmentNode):
                continue
            if getattr(node, "operator", "=") != "=":
                continue
            target = getattr(node, "target", None)
            if not isinstance(target, (IdentifierNode, VariableNode, str)):
                continue
            target_name = self.get_expression_name(target)
            if target_name not in local_resource_names:
                continue
            value = getattr(node, "value", None)
            if value is not None:
                assignment_sources.setdefault(target_name, []).append(value)

        return assignment_sources

    def is_query_return_alias_type(self, type_name):
        """Return whether a local type can participate in resource-return tracing."""
        if self.is_queryable_resource_type(type_name):
            return True

        type_name = self.type_name_string(type_name)
        if not type_name:
            return False

        if self.struct_member_lookup_type(type_name) in self.struct_member_types:
            return True

        indirect_info = self.cuda_indirect_type_info(type_name)
        if indirect_info is None:
            return False
        pointee_type = self.struct_member_lookup_type(indirect_info["pointee_type"])
        return pointee_type in self.struct_member_types

    def collect_query_return_local_alias_sources(self, statements):
        """Collect simple local aliases before a traceable resource return."""
        local_alias_types = {}
        local_alias_sources = {}
        assigned_names = set()

        for statement in statements:
            if isinstance(statement, VariableNode):
                var_type = self.get_variable_node_type(statement)
                if not self.is_query_return_alias_type(var_type):
                    return None
                name = getattr(statement, "name", None)
                if not name or name in local_alias_types:
                    return None

                local_alias_types[name] = var_type
                initial_value = getattr(
                    statement,
                    "initial_value",
                    getattr(statement, "value", None),
                )
                if initial_value is not None:
                    local_alias_sources[name] = initial_value
                    assigned_names.add(name)
                continue

            if isinstance(statement, IfNode):
                branch_alias = self.query_return_if_alias_source(
                    statement,
                    local_alias_types,
                    assigned_names,
                )
                if branch_alias is None:
                    return None
                target_name, value = branch_alias
                local_alias_sources[target_name] = value
                assigned_names.add(target_name)
                continue

            assignment = self.query_return_simple_assignment(statement)
            if assignment is None:
                return None
            target_name, value = assignment
            if target_name not in local_alias_types or target_name in assigned_names:
                return None

            local_alias_sources[target_name] = value
            assigned_names.add(target_name)

        return local_alias_sources

    def query_return_simple_assignment(self, statement):
        """Return simple assignment target/source for query-return aliases."""
        assignment = statement
        if isinstance(statement, ExpressionStatementNode):
            assignment = getattr(statement, "expression", None)
        if not isinstance(assignment, AssignmentNode):
            return None
        if getattr(assignment, "operator", "=") != "=":
            return None

        target = getattr(assignment, "target", None)
        if not isinstance(target, (IdentifierNode, VariableNode, str)):
            return None
        target_name = self.get_expression_name(target)
        value = getattr(assignment, "value", None)
        if not target_name or value is None:
            return None
        return target_name, value

    def query_return_if_alias_source(
        self,
        statement,
        local_alias_types,
        assigned_names,
    ):
        """Collect a deterministic if/else alias assignment chain."""
        branch_assignments = []
        current_if = statement
        else_assignment = None

        while isinstance(current_if, IfNode):
            condition = getattr(
                current_if,
                "condition",
                getattr(current_if, "if_condition", None),
            )
            then_body = getattr(
                current_if,
                "then_branch",
                getattr(current_if, "if_body", None),
            )
            else_body = getattr(
                current_if,
                "else_branch",
                getattr(current_if, "else_body", None),
            )
            if condition is None or then_body is None or else_body is None:
                return None

            then_assignment = self.query_return_body_alias_assignment(then_body)
            if then_assignment is None:
                return None
            branch_assignments.append((condition, then_assignment))

            else_if_conditions = list(
                getattr(current_if, "else_if_conditions", None) or []
            )
            else_if_bodies = list(getattr(current_if, "else_if_bodies", None) or [])
            if len(else_if_conditions) != len(else_if_bodies):
                return None

            for else_if_condition, else_if_body in zip(
                else_if_conditions,
                else_if_bodies,
            ):
                if else_if_condition is None:
                    return None
                else_if_assignment = self.query_return_body_alias_assignment(
                    else_if_body
                )
                if else_if_assignment is None:
                    return None
                branch_assignments.append((else_if_condition, else_if_assignment))

            if isinstance(else_body, IfNode):
                current_if = else_body
                continue

            else_assignment = self.query_return_body_alias_assignment(else_body)
            if else_assignment is None:
                return None
            break

        if else_assignment is None:
            return None

        target_name = branch_assignments[0][1][0]
        if target_name not in local_alias_types or target_name in assigned_names:
            return None
        if any(
            branch_assignment[0] != target_name
            for _, branch_assignment in branch_assignments
        ):
            return None

        else_name, else_value = else_assignment
        if else_name != target_name:
            return None

        value = else_value
        for branch_condition, (_, branch_value) in reversed(branch_assignments):
            value = TernaryOpNode(branch_condition, branch_value, value)
        return target_name, value

    def query_return_body_alias_assignment(self, body):
        """Return a branch body's only alias assignment, if it is simple."""
        statements = self.statement_list(body)
        if len(statements) != 1:
            return None
        return self.query_return_simple_assignment(statements[0])

    def collect_simple_query_return_sources(self, root):
        """Collect direct resource returns that can reuse caller-side metadata."""
        return_sources = {}
        functions = self.query_collect_functions(root)
        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(root, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}
        global_variable_types = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in getattr(root, "global_variables", [])
        }
        global_variable_types = {
            name: type_name
            for name, type_name in global_variable_types.items()
            if name and type_name
        }

        function_infos = []
        for func in functions:
            func_name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if not func_name or not self.is_queryable_resource_type(return_type):
                continue

            statements = self.statement_list(getattr(func, "body", []))
            if not statements or not isinstance(statements[-1], ReturnNode):
                continue

            local_alias_sources = self.collect_query_return_local_alias_sources(
                statements[:-1]
            )
            if local_alias_sources is None:
                continue

            return_expr = getattr(statements[-1], "value", None)
            params = getattr(func, "parameters", getattr(func, "params", []))
            resource_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
                if self.is_queryable_resource_type(self.get_parameter_type(param))
            }
            resource_param_indices = {
                name: index for name, index in resource_param_indices.items() if name
            }
            all_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
            }
            all_param_indices = {
                name: index for name, index in all_param_indices.items() if name
            }
            param_types = {
                getattr(param, "name", None): self.type_name_string(
                    self.get_parameter_type(param)
                )
                for param in params
            }
            param_types = {
                name: type_name
                for name, type_name in param_types.items()
                if name and type_name
            }
            function_infos.append(
                (
                    func_name,
                    return_expr,
                    resource_param_indices,
                    all_param_indices,
                    param_types,
                    local_alias_sources,
                )
            )

        changed = True
        while changed:
            changed = False
            for (
                func_name,
                return_expr,
                resource_param_indices,
                all_param_indices,
                param_types,
                local_alias_sources,
            ) in function_infos:
                if func_name in return_sources:
                    continue
                return_source = self.query_return_source_descriptor(
                    return_expr,
                    resource_param_indices,
                    global_resource_names,
                    all_param_indices,
                    return_sources,
                    local_alias_sources,
                    param_types=param_types,
                    global_variable_types=global_variable_types,
                )
                if return_source is not None:
                    return_sources[func_name] = return_source
                    changed = True

        return return_sources

    def query_return_source_descriptor(
        self,
        return_expr,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources=None,
        local_alias_sources=None,
        local_visited=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Describe a traceable resource-return expression for caller metadata."""
        if known_return_sources is None:
            known_return_sources = {}
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}

        if isinstance(return_expr, TernaryOpNode):
            true_source = self.query_return_source_descriptor(
                return_expr.true_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            false_source = self.query_return_source_descriptor(
                return_expr.false_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": return_expr.condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if isinstance(return_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(return_expr)
            return_source = known_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                return_expr,
                "arguments",
                getattr(return_expr, "args", []),
            )
            return self.inline_query_return_source_descriptor(
                return_source,
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )

        return_base_expr = return_expr
        return_indices = []
        if isinstance(return_expr, ArrayAccessNode):
            return_base_expr, return_indices = self.query_array_access_parts(
                return_expr
            )
        elif isinstance(return_expr, (MemberAccessNode, PointerAccessNode)):
            return_base_expr = return_expr
        elif not isinstance(return_expr, (IdentifierNode, VariableNode, str)):
            return None

        if isinstance(return_base_expr, (MemberAccessNode, PointerAccessNode)):
            return self.query_return_member_source_descriptor(
                return_base_expr,
                return_indices,
                all_param_indices,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )

        return_name = self.get_expression_name(return_base_expr)
        if not return_name:
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None
        if return_name in local_alias_sources:
            if return_name in local_visited:
                return None
            source_descriptor = self.query_return_source_descriptor(
                local_alias_sources[return_name],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited | {return_name},
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if source_descriptor is None:
                return None
            return self.append_query_return_indices(
                source_descriptor,
                return_indices,
                all_param_indices,
            )
        if return_name in resource_param_indices:
            return {
                "kind": "parameter",
                "index": resource_param_indices[return_name],
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        if return_name in global_resource_names:
            return {
                "kind": "global",
                "name": return_name,
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        return None

    def substitute_query_return_local_aliases(
        self,
        expr,
        local_alias_sources,
        local_visited=None,
    ):
        """Replace helper-local aliases in a return-source expression."""
        if local_visited is None:
            local_visited = set()

        expr_name = self.get_expression_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            if expr_name in local_alias_sources:
                if expr_name in local_visited:
                    return None
                return self.substitute_query_return_local_aliases(
                    local_alias_sources[expr_name],
                    local_alias_sources,
                    local_visited | {expr_name},
                )
            return expr

        if isinstance(expr, BinaryOpNode):
            left = self.substitute_query_return_local_aliases(
                expr.left,
                local_alias_sources,
                local_visited,
            )
            right = self.substitute_query_return_local_aliases(
                expr.right,
                local_alias_sources,
                local_visited,
            )
            if left is None or right is None:
                return None
            return BinaryOpNode(left, expr.operator, right)

        if isinstance(expr, UnaryOpNode):
            operand = self.substitute_query_return_local_aliases(
                expr.operand,
                local_alias_sources,
                local_visited,
            )
            if operand is None:
                return None
            return UnaryOpNode(expr.operator, operand, expr.is_postfix)

        if isinstance(expr, TernaryOpNode):
            condition = self.substitute_query_return_local_aliases(
                expr.condition,
                local_alias_sources,
                local_visited,
            )
            true_expr = self.substitute_query_return_local_aliases(
                expr.true_expr,
                local_alias_sources,
                local_visited,
            )
            false_expr = self.substitute_query_return_local_aliases(
                expr.false_expr,
                local_alias_sources,
                local_visited,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return TernaryOpNode(condition, true_expr, false_expr)

        if isinstance(expr, ArrayAccessNode):
            array_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "array_expr", getattr(expr, "array", None)),
                local_alias_sources,
                local_visited,
            )
            index_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "index_expr", getattr(expr, "index", None)),
                local_alias_sources,
                local_visited,
            )
            if array_expr is None or index_expr is None:
                return None
            return ArrayAccessNode(array_expr, index_expr)

        if isinstance(expr, MemberAccessNode):
            object_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "object_expr", getattr(expr, "object", None)),
                local_alias_sources,
                local_visited,
            )
            if object_expr is None:
                return None
            return MemberAccessNode(object_expr, expr.member)

        if isinstance(expr, PointerAccessNode):
            pointer_expr = self.substitute_query_return_local_aliases(
                expr.pointer_expr,
                local_alias_sources,
                local_visited,
            )
            if pointer_expr is None:
                return None
            return PointerAccessNode(pointer_expr, expr.member)

        if isinstance(expr, FunctionCallNode):
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                substituted = self.substitute_query_return_local_aliases(
                    argument,
                    local_alias_sources,
                    local_visited,
                )
                if substituted is None:
                    return None
                arguments.append(substituted)
            return FunctionCallNode(
                expr.function,
                arguments,
                getattr(expr, "generic_args", []),
            )

        return expr

    def query_return_member_source_descriptor(
        self,
        member_expr,
        return_indices,
        all_param_indices,
        param_types,
        global_variable_types,
        local_alias_sources=None,
        local_visited=None,
    ):
        """Describe a storage-image struct member returned by a helper."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()

        if isinstance(member_expr, PointerAccessNode):
            object_node = getattr(member_expr, "pointer_expr", None)
        else:
            object_node = getattr(
                member_expr,
                "object_expr",
                getattr(member_expr, "object", None),
            )
        member = getattr(member_expr, "member", None)
        member_name = getattr(member, "name", member)
        if object_node is None or not isinstance(member_name, str):
            return None

        object_name = self.get_expression_name(object_node)

        object_kind = None
        object_type = None
        descriptor = {
            "kind": "member",
            "member": member_name,
            "indices": return_indices,
            "param_indices": all_param_indices,
            "object_access": (
                "->" if isinstance(member_expr, PointerAccessNode) else "."
            ),
        }
        if object_name in local_alias_sources:
            if object_name in local_visited:
                return None
            object_kind = "expression"
            object_node = local_alias_sources[object_name]
            object_type = self.query_return_expression_type(
                object_node,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited | {object_name},
            )
        elif object_name in param_types:
            object_kind = "parameter"
            object_type = param_types[object_name]
            descriptor["object_index"] = all_param_indices.get(object_name)
        elif object_name in global_variable_types:
            object_kind = "global"
            object_type = global_variable_types[object_name]
            descriptor["object_name"] = object_name
        else:
            object_kind = "expression"
            object_type = self.query_return_expression_type(
                object_node,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
        if isinstance(member_expr, PointerAccessNode):
            pointer_info = self.cuda_indirect_type_info(object_type)
            if pointer_info is None:
                return None
            object_type = pointer_info["pointee_type"]

        object_type = self.struct_member_lookup_type(object_type)
        member_type = self.struct_member_types.get(object_type, {}).get(member_name)
        if not self.is_storage_image_type(member_type):
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None

        descriptor["object_kind"] = object_kind
        descriptor["object_type"] = object_type
        object_expr = self.substitute_query_return_local_aliases(
            object_node,
            local_alias_sources,
        )
        if object_expr is not None:
            descriptor["object_expr"] = object_expr
        return descriptor

    def query_return_expression_type(
        self,
        expr,
        param_types,
        global_variable_types,
        local_alias_sources=None,
        local_visited=None,
    ):
        """Infer expression types for returned-resource tracing without scope."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()

        expr_name = self.get_expression_name(expr)
        if expr_name in local_alias_sources:
            if expr_name in local_visited:
                return None
            return self.query_return_expression_type(
                local_alias_sources[expr_name],
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited | {expr_name},
            )
        if expr_name in param_types:
            return param_types[expr_name]
        if expr_name in global_variable_types:
            return global_variable_types[expr_name]

        if isinstance(expr, TernaryOpNode):
            true_type = self.query_return_expression_type(
                expr.true_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited.copy(),
            )
            false_type = self.query_return_expression_type(
                expr.false_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited.copy(),
            )
            if self.type_name_string(true_type) == self.type_name_string(false_type):
                return true_type
            return None

        if isinstance(expr, PointerAccessNode):
            pointer_expr = getattr(expr, "pointer_expr", None)
            pointer_type = self.query_return_expression_type(
                pointer_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is None:
                return None
            object_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
            return self.struct_member_types.get(object_type, {}).get(
                getattr(expr, "member", "")
            )

        if isinstance(expr, MemberAccessNode):
            object_node = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.query_return_expression_type(
                    object_node,
                    param_types,
                    global_variable_types,
                    local_alias_sources,
                    local_visited,
                )
            )
            return self.struct_member_types.get(object_type, {}).get(
                getattr(expr, "member", "")
            )

        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.query_return_expression_type(
                array_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
            return self.array_access_element_type(array_type)

        return self.expression_result_type(expr)

    def inline_query_return_source_descriptor(
        self,
        return_source,
        raw_args,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources,
        local_alias_sources=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Inline a traceable callee resource return into the caller context."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}
        source_param_indices = return_source.get("param_indices", {})
        kind = return_source["kind"]

        if kind == "global":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return {
                "kind": "global",
                "name": return_source["name"],
                "indices": indices,
                "param_indices": all_param_indices,
            }

        if kind == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            base_source = self.query_return_source_descriptor(
                raw_args[index],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if base_source is None:
                return None
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return self.append_query_return_indices(
                base_source,
                indices,
                all_param_indices,
            )

        if kind == "ternary":
            condition = self.substitute_query_return_expr(
                return_source["condition"],
                source_param_indices,
                raw_args,
            )
            if condition is None:
                return None
            true_source = self.inline_query_return_source_descriptor(
                return_source["true_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            false_source = self.inline_query_return_source_descriptor(
                return_source["false_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if kind == "member":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            inlined = dict(return_source)
            inlined["indices"] = indices
            inlined["param_indices"] = all_param_indices
            return inlined

        return None

    def append_query_return_indices(self, return_source, indices, param_indices):
        """Append array indices to a flattened resource-return descriptor."""
        if not indices:
            return return_source
        if return_source["kind"] not in {"global", "parameter", "member"}:
            return None
        if any(
            not self.is_safe_query_return_index(index, param_indices)
            for index in indices
        ):
            return None
        combined = dict(return_source)
        combined["indices"] = list(return_source.get("indices", [])) + list(indices)
        combined["param_indices"] = param_indices
        return combined

    def substitute_query_return_indices(
        self,
        indices,
        param_indices,
        raw_args,
        target_param_indices,
    ):
        """Substitute callee index parameters with caller-context expressions."""
        substituted_indices = []
        for index_expr in indices:
            substituted = self.substitute_query_return_expr(
                index_expr,
                param_indices,
                raw_args,
            )
            if substituted is None or not self.is_safe_query_return_index(
                substituted,
                target_param_indices,
            ):
                return None
            substituted_indices.append(substituted)
        return substituted_indices

    def substitute_query_return_expr(self, expr, param_indices, raw_args):
        """Replace callee parameters in a return-source expression."""
        expr_name = self.get_expression_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            if expr_name in param_indices:
                param_index = param_indices[expr_name]
                if param_index >= len(raw_args):
                    return None
                return raw_args[param_index]
            return expr

        if isinstance(expr, BinaryOpNode):
            left = self.substitute_query_return_expr(
                expr.left,
                param_indices,
                raw_args,
            )
            right = self.substitute_query_return_expr(
                expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return BinaryOpNode(left, expr.operator, right)

        if isinstance(expr, UnaryOpNode):
            operand = self.substitute_query_return_expr(
                expr.operand,
                param_indices,
                raw_args,
            )
            if operand is None:
                return None
            return UnaryOpNode(expr.operator, operand, expr.is_postfix)

        if isinstance(expr, TernaryOpNode):
            condition = self.substitute_query_return_expr(
                expr.condition,
                param_indices,
                raw_args,
            )
            true_expr = self.substitute_query_return_expr(
                expr.true_expr,
                param_indices,
                raw_args,
            )
            false_expr = self.substitute_query_return_expr(
                expr.false_expr,
                param_indices,
                raw_args,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return TernaryOpNode(condition, true_expr, false_expr)

        if isinstance(expr, ArrayAccessNode):
            array_expr = self.substitute_query_return_expr(
                expr.array_expr,
                param_indices,
                raw_args,
            )
            index_expr = self.substitute_query_return_expr(
                expr.index_expr,
                param_indices,
                raw_args,
            )
            if array_expr is None or index_expr is None:
                return None
            return ArrayAccessNode(array_expr, index_expr)

        if isinstance(expr, MemberAccessNode):
            object_expr = self.substitute_query_return_expr(
                expr.object_expr,
                param_indices,
                raw_args,
            )
            if object_expr is None:
                return None
            return MemberAccessNode(object_expr, expr.member)

        if isinstance(expr, PointerAccessNode):
            pointer_expr = self.substitute_query_return_expr(
                expr.pointer_expr,
                param_indices,
                raw_args,
            )
            if pointer_expr is None:
                return None
            return PointerAccessNode(pointer_expr, expr.member)

        if isinstance(expr, FunctionCallNode):
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                substituted = self.substitute_query_return_expr(
                    argument,
                    param_indices,
                    raw_args,
                )
                if substituted is None:
                    return None
                arguments.append(substituted)
            return FunctionCallNode(
                expr.function,
                arguments,
                getattr(expr, "generic_args", []),
            )

        return expr

    def query_array_access_parts(self, expr):
        """Return the base expression and ordered indices for nested array access."""
        indices = []
        current = expr
        while isinstance(current, ArrayAccessNode):
            index_node = getattr(
                current,
                "index_expr",
                getattr(current, "index", None),
            )
            array_node = getattr(
                current,
                "array_expr",
                getattr(current, "array", None),
            )
            if index_node is None or array_node is None:
                return None, []
            indices.insert(0, index_node)
            current = array_node
        return current, indices

    def is_safe_query_return_index(self, index_expr, param_indices):
        """Return whether a returned-array index can be rendered in the caller."""
        if isinstance(index_expr, LiteralNode) or isinstance(index_expr, int):
            return True
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_index(index_expr.left, param_indices)
                and self.is_safe_query_return_index(index_expr.right, param_indices)
            )
        index_name = self.get_expression_name(index_expr)
        if index_name:
            return index_name in param_indices
        if isinstance(index_expr, str):
            return index_expr.lstrip("-").isdigit()
        return False

    def collect_resource_query_requirements(self, node):
        """Collect query metadata needs, resolving local CUDA resource aliases."""
        global_names, function_params = (
            ResourceQueryMixin.collect_resource_query_requirements(self, node)
        )

        functions = self.query_collect_functions(node)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        function_parameter_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        function_parameter_names = {
            func_name: {name for name in names if name}
            for func_name, names in function_parameter_names.items()
        }

        local_resource_names = {}
        local_alias_sources = {}
        local_assignment_sources = {}
        for func_name, func in functions_by_name.items():
            names, aliases = self.collect_function_resource_aliases(func)
            local_resource_names[func_name] = names
            local_alias_sources[func_name] = aliases
            local_assignment_sources[func_name] = (
                self.collect_function_resource_assignments(func, names)
            )
        local_metadata_snapshots = {func_name: set() for func_name in functions_by_name}

        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(node, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}

        def mark_resolved_resource(func_name, resource_expr, visited=None):
            if visited is None:
                visited = set()
            if isinstance(resource_expr, TernaryOpNode):
                if not self.is_safe_query_return_actual_index(resource_expr.condition):
                    return False
                true_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.true_expr,
                    visited.copy(),
                )
                false_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.false_expr,
                    visited.copy(),
                )
                return true_changed or false_changed

            if isinstance(resource_expr, FunctionCallNode):
                callee_name = self.raw_function_call_name(resource_expr)
                return_source = self.query_return_sources.get(callee_name)
                if return_source is None:
                    return False
                raw_args = getattr(
                    resource_expr,
                    "arguments",
                    getattr(resource_expr, "args", []),
                )
                return mark_return_source_resources(
                    func_name,
                    return_source,
                    raw_args,
                    visited,
                )

            resource_name = self.get_expression_name(resource_expr)
            if not resource_name:
                return False
            if resource_name in visited:
                return False
            visited.add(resource_name)

            aliases = local_alias_sources.get(func_name, {})
            assignments = local_assignment_sources.get(func_name, {})
            if resource_name in local_resource_names.get(func_name, set()):
                if resource_name in assignments:
                    snapshot_names = local_metadata_snapshots.setdefault(
                        func_name,
                        set(),
                    )
                    before = len(snapshot_names)
                    snapshot_names.add(resource_name)
                    changed = len(snapshot_names) != before
                    sources = []
                    if resource_name in aliases:
                        sources.append(aliases[resource_name])
                    sources.extend(assignments.get(resource_name, []))
                    for source in sources:
                        changed = (
                            mark_resolved_resource(
                                func_name,
                                source,
                                visited.copy(),
                            )
                            or changed
                        )
                    return changed
                if resource_name in aliases:
                    return mark_resolved_resource(
                        func_name,
                        aliases[resource_name],
                        visited,
                    )
                return False

            if resource_name in function_parameter_names.get(func_name, set()):
                before = len(function_params.setdefault(func_name, set()))
                function_params[func_name].add(resource_name)
                return len(function_params[func_name]) != before

            before = len(global_names)
            global_names.add(resource_name)
            return len(global_names) != before

        def mark_return_source_resources(func_name, return_source, raw_args, visited):
            kind = return_source["kind"]
            if kind == "global":
                before = len(global_names)
                global_names.add(return_source["name"])
                return len(global_names) != before
            if kind == "parameter":
                index = return_source["index"]
                if index >= len(raw_args):
                    return False
                return mark_resolved_resource(func_name, raw_args[index], visited)
            if kind == "ternary":
                true_changed = mark_return_source_resources(
                    func_name,
                    return_source["true_source"],
                    raw_args,
                    visited.copy(),
                )
                false_changed = mark_return_source_resources(
                    func_name,
                    return_source["false_source"],
                    raw_args,
                    visited.copy(),
                )
                return true_changed or false_changed
            return False

        struct_metadata_member_names = {
            member
            for members in self.struct_query_metadata_members.values()
            for member in members
        }

        def mark_struct_constructor_resources(func_name, call):
            struct_name = self.raw_function_call_name(call)
            metadata_members = self.struct_query_metadata_members.get(struct_name)
            if not metadata_members:
                return False
            raw_args = getattr(call, "arguments", getattr(call, "args", []))
            changed = False
            member_types = list(self.struct_member_types.get(struct_name, {}).items())
            for index, (member_name, _) in enumerate(member_types):
                if member_name not in metadata_members or index >= len(raw_args):
                    continue
                changed = (
                    mark_resolved_resource(
                        func_name,
                        raw_args[index],
                    )
                    or changed
                )
            return changed

        def metadata_assignment_member_name(target):
            while isinstance(target, ArrayAccessNode):
                target = getattr(
                    target,
                    "array_expr",
                    getattr(target, "array", None),
                )
            if isinstance(target, (MemberAccessNode, PointerAccessNode)):
                member = getattr(target, "member", None)
                return getattr(member, "name", member)
            return None

        for func_name, func in functions_by_name.items():
            for node in self.query_walk_nodes(getattr(func, "body", [])):
                if isinstance(node, FunctionCallNode):
                    changed = mark_struct_constructor_resources(func_name, node)
                    if changed:
                        continue
                if not isinstance(node, AssignmentNode):
                    continue
                member_name = metadata_assignment_member_name(
                    getattr(node, "target", None)
                )
                if member_name not in struct_metadata_member_names:
                    continue
                value = getattr(node, "value", None)
                if value is not None:
                    mark_resolved_resource(func_name, value)

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                func_call_name = self.raw_function_call_name(call)
                raw_args = getattr(call, "arguments", getattr(call, "args", []))
                if func_call_name in self.query_function_names and raw_args:
                    mark_resolved_resource(func_name, raw_args[0])

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_params.get(callee_name, set())
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue
                        changed = (
                            mark_resolved_resource(caller_name, raw_args[index])
                            or changed
                        )

        for names in local_resource_names.values():
            global_names.difference_update(names - global_resource_names)

        self.query_local_resource_names_by_function = {
            func_name: names
            for func_name, names in local_resource_names.items()
            if names
        }
        self.query_metadata_snapshot_locals_by_function = {
            func_name: names
            for func_name, names in local_metadata_snapshots.items()
            if names
        }
        return (
            global_names,
            {func_name: names for func_name, names in function_params.items() if names},
        )

    def register_query_metadata_alias(self, name, type_name, source_node=None):
        """Track local resource aliases that can reuse an existing metadata sidecar."""
        if not self.current_function_name or not name:
            return
        if not self.is_queryable_resource_type(type_name) or source_node is None:
            self.query_metadata_aliases.pop(name, None)
            return

        metadata_expr = self.query_metadata_expression(source_node)
        if metadata_expr is None:
            self.query_metadata_aliases.pop(name, None)
            return
        self.query_metadata_aliases[name] = metadata_expr

    def is_query_metadata_snapshot_local(self, name, type_name=None):
        """Return whether a local resource needs its own mutable metadata sidecar."""
        if not self.current_function_name or not name:
            return False
        snapshot_names = self.query_metadata_snapshot_locals_by_function.get(
            self.current_function_name,
            set(),
        )
        if name not in snapshot_names:
            return False
        if type_name is not None and not self.is_queryable_resource_type(type_name):
            return False
        return True

    def query_metadata_snapshot_declaration(self, name, type_name, initial_value=None):
        """Return a local metadata sidecar declaration for reassigned resources."""
        if not self.is_query_metadata_snapshot_local(name, type_name):
            return None

        declarator = self.format_typed_declarator(
            self.query_metadata_type(type_name),
            self.query_metadata_name(name),
            dynamic_array_as_pointer=False,
        )
        metadata_expr = None
        if initial_value is not None:
            metadata_expr = self.query_metadata_expression(initial_value)
        if metadata_expr is None:
            metadata_expr = "{}"
        return f"{declarator} = {metadata_expr}"

    def struct_member_has_query_metadata(self, struct_type, member_name):
        """Return whether a struct member has an embedded metadata sidecar."""
        struct_type = self.struct_member_lookup_type(struct_type)
        return member_name in self.struct_query_metadata_members.get(struct_type, set())

    def can_reuse_struct_metadata_object_expression(
        self, object_node, allow_function_call_object=False
    ):
        """Return whether a direct member sidecar read avoids duplicating calls."""
        if object_node is None:
            return False
        if isinstance(object_node, FunctionCallNode):
            return allow_function_call_object
        if isinstance(object_node, ArrayAccessNode):
            array_node = getattr(
                object_node,
                "array_expr",
                getattr(object_node, "array", None),
            )
            index_node = getattr(
                object_node,
                "index_expr",
                getattr(object_node, "index", None),
            )
            return self.can_reuse_struct_metadata_object_expression(
                array_node, allow_function_call_object
            ) and self.is_safe_query_return_actual_index(index_node)
        if isinstance(object_node, MemberAccessNode):
            return self.can_reuse_struct_metadata_object_expression(
                getattr(
                    object_node,
                    "object_expr",
                    getattr(object_node, "object", None),
                ),
                allow_function_call_object,
            )
        if isinstance(object_node, PointerAccessNode):
            return self.can_reuse_struct_metadata_object_expression(
                getattr(object_node, "pointer_expr", None),
                allow_function_call_object,
            )
        return True

    def query_struct_member_metadata_expression(
        self, resource_expr, allow_function_call_object=False
    ):
        """Return metadata paired with a struct resource-member expression."""
        if isinstance(resource_expr, PointerAccessNode):
            object_node = getattr(resource_expr, "pointer_expr", None)
            member_name = getattr(resource_expr, "member", None)
            object_type = self.resource_expression_type(object_node)
            pointer_info = self.cuda_indirect_type_info(object_type)
            if pointer_info is None:
                return None
            struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
            object_access = "->"
        elif isinstance(resource_expr, MemberAccessNode):
            object_node = getattr(
                resource_expr,
                "object_expr",
                getattr(resource_expr, "object", None),
            )
            member_name = getattr(resource_expr, "member", None)
            struct_type = self.struct_member_lookup_type(
                self.resource_expression_type(object_node)
            )
            object_access = "."
        else:
            return None

        if object_node is None or not isinstance(member_name, str):
            return None
        if not self.can_reuse_struct_metadata_object_expression(
            object_node, allow_function_call_object
        ):
            return None
        if not self.struct_member_has_query_metadata(struct_type, member_name):
            return None

        object_expr = self.visit(object_node)
        if not object_expr:
            return None
        return f"{object_expr}{object_access}{self.query_metadata_name(member_name)}"

    def is_struct_member_metadata_assignment_target(self, target_node):
        """Return whether an assignment target is a struct resource member."""
        if isinstance(target_node, ArrayAccessNode):
            target_node = getattr(
                target_node,
                "array_expr",
                getattr(target_node, "array", None),
            )
        return isinstance(target_node, (MemberAccessNode, PointerAccessNode))

    def format_query_metadata_assignment(self, node):
        """Return a metadata sidecar update for a reassigned local resource."""
        if getattr(node, "operator", "=") != "=":
            return None

        target_node = getattr(node, "target", None)
        if isinstance(target_node, (IdentifierNode, VariableNode, str)):
            target_name = self.get_expression_name(target_node)
            target_type = self.variable_types.get(target_name)
            if not self.is_query_metadata_snapshot_local(target_name, target_type):
                return None

            metadata_expr = self.query_metadata_expression(getattr(node, "value", None))
            if metadata_expr is None:
                resource_type = self.resource_base_type(target_type) or target_type
                resource_type = resource_type or "resource"
                metadata_expr = (
                    "/* unsupported CUDA resource query: metadata unavailable for "
                    f"{resource_type} assignment */ CglResourceQueryInfo{{}}"
                )
            return f"{self.query_metadata_name(target_name)} = {metadata_expr}"

        if not self.is_struct_member_metadata_assignment_target(target_node):
            return None
        target_metadata = self.query_metadata_expression(target_node)
        if target_metadata is None:
            return None
        target_type = self.resource_expression_type(target_node)
        if not self.is_queryable_resource_type(target_type):
            return None
        metadata_expr = self.query_metadata_expression(getattr(node, "value", None))
        if metadata_expr is None:
            resource_type = self.resource_base_type(target_type) or target_type
            resource_type = resource_type or "resource"
            metadata_expr = (
                "/* unsupported CUDA resource query: metadata unavailable for "
                f"{resource_type} assignment */ CglResourceQueryInfo{{}}"
            )
        return f"{target_metadata} = {metadata_expr}"

    def query_metadata_expression(
        self, resource_expr, allow_function_call_object=False
    ):
        """Return CUDA query metadata paired with a resource expression."""
        if isinstance(resource_expr, TernaryOpNode):
            if not self.is_safe_query_return_actual_index(resource_expr.condition):
                return None
            true_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.true_expr)
            )
            false_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.false_expr)
            )
            if true_type is None or false_type is None or true_type != false_type:
                return None
            true_metadata = self.query_metadata_expression(
                resource_expr.true_expr, allow_function_call_object
            )
            false_metadata = self.query_metadata_expression(
                resource_expr.false_expr, allow_function_call_object
            )
            if true_metadata is None or false_metadata is None:
                return None
            condition = self.visit(resource_expr.condition)
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if isinstance(resource_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(resource_expr)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                resource_expr,
                "arguments",
                getattr(resource_expr, "args", []),
            )
            return self.query_return_source_metadata_expression(
                return_source,
                raw_args,
                allow_function_call_object,
            )

        if isinstance(resource_expr, ArrayAccessNode):
            array_node = getattr(
                resource_expr,
                "array_expr",
                getattr(resource_expr, "array", None),
            )
            index_node = getattr(
                resource_expr,
                "index_expr",
                getattr(resource_expr, "index", None),
            )
            if index_node is None or not self.is_safe_query_return_actual_index(
                index_node
            ):
                return None
            base_expr = self.query_metadata_expression(
                array_node, allow_function_call_object
            )
            if base_expr is None:
                return None
            return f"{base_expr}[{self.visit(index_node)}]"

        member_metadata = self.query_struct_member_metadata_expression(
            resource_expr, allow_function_call_object
        )
        if member_metadata is not None:
            return member_metadata

        resource_name = self.get_expression_name(resource_expr)
        if not resource_name:
            return None

        alias_expr = self.query_metadata_aliases.get(resource_name)
        if alias_expr is not None:
            return alias_expr

        current_function = self.current_function_name
        local_names = self.query_local_resource_names_by_function.get(
            current_function,
            set(),
        )
        if resource_name in local_names:
            return None

        if current_function:
            query_params = self.query_metadata_function_params.get(
                current_function,
                set(),
            )
            if resource_name in query_params:
                return self.query_metadata_name(resource_name)

        if resource_name in self.query_resource_names:
            return self.query_metadata_name(resource_name)
        return None

    def query_return_source_metadata_expression(
        self, return_source, raw_args, allow_function_call_object=False
    ):
        """Render query metadata for a resource returned by a user function."""
        if return_source["kind"] == "ternary":
            condition = self.format_query_return_safe_expression(
                return_source["condition"],
                return_source.get("param_indices", {}),
                raw_args,
            )
            if condition is None:
                return None
            true_metadata = self.query_return_source_metadata_expression(
                return_source["true_source"],
                raw_args,
                allow_function_call_object,
            )
            false_metadata = self.query_return_source_metadata_expression(
                return_source["false_source"],
                raw_args,
                allow_function_call_object,
            )
            if true_metadata is None or false_metadata is None:
                return None
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if return_source["kind"] == "global":
            metadata_expr = self.query_metadata_name(return_source["name"])
        elif return_source["kind"] == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            metadata_expr = self.query_metadata_expression(
                raw_args[index], allow_function_call_object
            )
        elif return_source["kind"] == "member":
            member = return_source.get("member")
            object_type = return_source.get("object_type")
            if not self.struct_member_has_query_metadata(object_type, member):
                return None
            object_expr = return_source.get("object_expr")
            if object_expr is None:
                return None
            metadata_object = self.format_query_return_metadata_object_expression(
                object_expr,
                return_source.get("param_indices", {}),
                raw_args,
            )
            if metadata_object is None:
                return None
            metadata_expr = (
                f"{metadata_object}"
                f"{return_source.get('object_access', '.')}"
                f"{self.query_metadata_name(member)}"
            )
        else:
            return None
        if metadata_expr is None:
            return None

        param_indices = return_source.get("param_indices", {})
        for index_expr in return_source.get("indices", []):
            rendered_index = self.format_query_return_index(
                index_expr,
                param_indices,
                raw_args,
            )
            if rendered_index is None:
                return None
            metadata_expr = f"{metadata_expr}[{rendered_index}]"
        return metadata_expr

    def format_query_return_metadata_object_expression(
        self,
        expr,
        param_indices,
        raw_args,
    ):
        """Render a returned struct object expression for member sidecar access."""
        if isinstance(expr, FunctionCallNode):
            func_name = self.raw_function_call_name(expr)
            if not func_name:
                return None
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                argument_expr = self.format_query_return_safe_expression(
                    argument,
                    param_indices,
                    raw_args,
                    allow_free_identifiers=True,
                )
                if argument_expr is None:
                    return None
                arguments.append(argument_expr)
            return f"{func_name}({', '.join(arguments)})"

        return self.format_query_return_safe_expression(
            expr,
            param_indices,
            raw_args,
            allow_free_identifiers=True,
        )

    def format_query_return_safe_expression(
        self,
        expr,
        param_indices,
        raw_args,
        allow_free_identifiers=False,
    ):
        """Render a side-effect-safe returned-resource selector expression."""
        if isinstance(expr, BinaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator not in (
                self.query_return_index_binary_ops
                | {"&&", "||", "==", "!=", "<", "<=", ">", ">="}
            ):
                return None
            left = self.format_query_return_safe_expression(
                expr.left,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            right = self.format_query_return_safe_expression(
                expr.right,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"

        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if getattr(expr, "is_postfix", False) or operator not in {
                "!",
                "+",
                "-",
                "~",
            }:
                return None
            operand = self.format_query_return_safe_expression(
                expr.operand,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if operand is None:
                return None
            return f"{operator}{operand}"

        if isinstance(expr, TernaryOpNode):
            condition = self.format_query_return_safe_expression(
                expr.condition,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            true_expr = self.format_query_return_safe_expression(
                expr.true_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            false_expr = self.format_query_return_safe_expression(
                expr.false_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return f"({condition} ? {true_expr} : {false_expr})"

        if isinstance(expr, ArrayAccessNode):
            array_node = getattr(
                expr,
                "array_expr",
                getattr(expr, "array", None),
            )
            index_node = getattr(
                expr,
                "index_expr",
                getattr(expr, "index", None),
            )
            if array_node is None or index_node is None:
                return None
            array_expr = self.format_query_return_safe_expression(
                array_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            index_expr = self.format_query_return_safe_expression(
                index_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if array_expr is None or index_expr is None:
                return None
            return f"{array_expr}[{index_expr}]"

        if isinstance(expr, MemberAccessNode):
            object_node = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", None),
            )
            member = getattr(expr, "member", None)
            member_name = getattr(member, "name", member)
            if object_node is None or not isinstance(member_name, str):
                return None
            object_expr = self.format_query_return_safe_expression(
                object_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if object_expr is None:
                return None
            return f"{object_expr}.{member_name}"

        if isinstance(expr, PointerAccessNode):
            pointer_node = getattr(expr, "pointer_expr", None)
            member = getattr(expr, "member", None)
            member_name = getattr(member, "name", member)
            if pointer_node is None or not isinstance(member_name, str):
                return None
            pointer_expr = self.format_query_return_safe_expression(
                pointer_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if pointer_expr is None:
                return None
            return f"{pointer_expr}->{member_name}"

        expr_name = self.get_expression_name(expr)
        if expr_name in param_indices:
            param_index = param_indices[expr_name]
            if param_index >= len(raw_args):
                return None
            return self.format_query_return_safe_expression(
                raw_args[param_index],
                {},
                [],
                allow_free_identifiers=True,
            )
        if allow_free_identifiers and expr_name:
            return self.visit(expr)
        if isinstance(expr, LiteralNode):
            return self.visit(expr)
        if isinstance(expr, bool):
            return "true" if expr else "false"
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, str):
            stripped = expr.strip()
            if stripped in {"true", "false"} or stripped.lstrip("-").isdigit():
                return stripped
            if allow_free_identifiers and stripped.isidentifier():
                return stripped
        return None

    def format_query_return_index(self, index_expr, param_indices, raw_args):
        """Render a returned-array index in the caller's argument context."""
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            if operator not in self.query_return_index_binary_ops:
                return None
            left = self.format_query_return_index(
                index_expr.left,
                param_indices,
                raw_args,
            )
            right = self.format_query_return_index(
                index_expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"
        index_name = self.get_expression_name(index_expr)
        if index_name in param_indices:
            param_index = param_indices[index_name]
            if param_index >= len(raw_args):
                return None
            actual_expr = raw_args[param_index]
            if not self.is_safe_query_return_actual_index(actual_expr):
                return None
            return self.visit(actual_expr)
        if isinstance(index_expr, LiteralNode):
            return self.visit(index_expr)
        if isinstance(index_expr, int):
            return str(index_expr)
        if isinstance(index_expr, str) and index_expr.lstrip("-").isdigit():
            return index_expr
        return None

    def is_safe_query_return_actual_index(self, actual_expr):
        """Return whether a caller argument can be reused as a metadata index."""
        if isinstance(actual_expr, LiteralNode) or isinstance(actual_expr, int):
            return True
        if isinstance(actual_expr, (IdentifierNode, VariableNode)):
            return True
        if isinstance(actual_expr, str):
            actual_expr = actual_expr.strip()
            return actual_expr.lstrip("-").isdigit() or actual_expr.isidentifier()
        if isinstance(actual_expr, BinaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_actual_index(actual_expr.left)
                and self.is_safe_query_return_actual_index(actual_expr.right)
            )
        if isinstance(actual_expr, UnaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                not getattr(actual_expr, "is_postfix", False)
                and operator in {"+", "-", "~"}
                and self.is_safe_query_return_actual_index(actual_expr.operand)
            )
        if isinstance(actual_expr, TernaryOpNode):
            return (
                self.is_safe_query_return_actual_index(actual_expr.condition)
                and self.is_safe_query_return_actual_index(actual_expr.true_expr)
                and self.is_safe_query_return_actual_index(actual_expr.false_expr)
            )
        if isinstance(actual_expr, ArrayAccessNode):
            array_node = getattr(
                actual_expr,
                "array_expr",
                getattr(actual_expr, "array", None),
            )
            index_node = getattr(
                actual_expr,
                "index_expr",
                getattr(actual_expr, "index", None),
            )
            return (
                array_node is not None
                and index_node is not None
                and self.is_safe_query_return_actual_index(array_node)
                and self.is_safe_query_return_actual_index(index_node)
            )
        if isinstance(actual_expr, MemberAccessNode):
            object_expr = getattr(
                actual_expr,
                "object_expr",
                getattr(actual_expr, "object", None),
            )
            return object_expr is not None and self.is_safe_query_return_actual_index(
                object_expr
            )
        return False

    def convert_builtin_function(self, func_name):
        """Convert CrossGL built-in functions to CUDA equivalents"""
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
            "abs": "fabsf",
            "round": "roundf",
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
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
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

        return function_mapping.get(func_name, func_name)

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
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
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

    def cuda_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.cuda_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_cuda_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.cuda_resource_binding_cursors.get(key, 0)
        while self.cuda_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.cuda_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_cuda_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.cuda_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise ValueError(
                    "Conflicting CUDA resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
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
        namespace = self.cuda_resource_binding_namespace(resource_kind)
        set_index = self.explicit_cuda_resource_set_index(node)
        binding = self.explicit_cuda_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_cuda_resource_binding(
                namespace, set_index, count
            )
            binding_source = "automatic"
            self.reserve_cuda_resource_binding(
                namespace, set_index, binding, count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_cuda_resource_binding(
                namespace, set_index, binding, count, name
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
        namespace = self.cuda_resource_binding_namespace(resource_kind)
        set_index = self.explicit_cuda_resource_set_index(node)
        self.reserve_cuda_resource_binding(namespace, set_index, binding, count, name)

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

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def glsl_buffer_block_declaration_type(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not self.is_glsl_buffer_block_node(node):
            return type_name
        if not type_name or self.explicit_resource_access(node) != "readonly":
            return type_name
        if type_name.startswith("const "):
            return type_name
        if "[" not in type_name or "]" not in type_name:
            return f"const {type_name}"
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket].strip()
        array_suffix = type_name[open_bracket:]
        return f"const {base_type}{array_suffix}"

    def glsl_buffer_block_metadata_comment(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or not self.is_glsl_buffer_block_node(node):
            return ""
        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            "kind=glsl_buffer_block",
        ]
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.explicit_resource_access(node)
        if access:
            parts.append(f"access={access}")
        return " ".join(parts)

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.glsl_buffer_block_root_name(array_expr)
        if isinstance(expr, (MemberAccessNode, PointerAccessNode)):
            object_expr = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", getattr(expr, "pointer_expr", None)),
            )
            return self.glsl_buffer_block_root_name(object_expr)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        return getattr(expr, "name", None)

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if root_name is None:
            return None, None
        return self.glsl_buffer_block_accesses.get(root_name), root_name

    def glsl_buffer_block_diagnostic(self, operation, expr, reason, fallback):
        _, resource_name = self.glsl_buffer_block_access(expr)
        resource_name = resource_name or "unknown"
        return (
            f"/* unsupported {self.resource_backend_name()} GLSL buffer block "
            f"{operation}: resource '{resource_name}' {reason} */ {fallback}"
        )

    def glsl_buffer_block_read_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "writeonly":
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is writeonly", fallback
        )

    def glsl_buffer_block_write_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "readonly":
            return None
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is readonly", "((void)0)"
        )

    def glsl_buffer_block_read_write_diagnostic(self, expr, operation, fallback):
        access, _ = self.glsl_buffer_block_access(expr)
        if access == "readonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is readonly", fallback
            )
        if access == "writeonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is writeonly", fallback
            )
        return None

    def buffer_resource_access(self, buffer_arg):
        """Return explicit readonly/writeonly access tracked for buffer resources."""
        if isinstance(buffer_arg, FunctionCallNode):
            callee_name = self.raw_function_call_name(buffer_arg)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                buffer_arg,
                "arguments",
                getattr(buffer_arg, "args", []),
            )
            return self.query_return_source_buffer_access(return_source, raw_args)

        if isinstance(buffer_arg, ArrayAccessNode):
            array_node = getattr(
                buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
            )
            return self.buffer_resource_access(array_node)

        if isinstance(buffer_arg, MemberAccessNode):
            object_node = getattr(
                buffer_arg,
                "object_expr",
                getattr(buffer_arg, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.expression_result_type(object_node)
            )
            member = getattr(buffer_arg, "member", None)
            access = self.struct_member_image_accesses.get(object_type, {}).get(member)
            if access is not None:
                return access
            return self.buffer_resource_access(object_node)

        if isinstance(buffer_arg, PointerAccessNode):
            pointer_expr = getattr(buffer_arg, "pointer_expr", None)
            pointer_type = self.expression_result_type(pointer_expr)
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is not None:
                object_type = self.struct_member_lookup_type(
                    pointer_info["pointee_type"]
                )
                member = getattr(buffer_arg, "member", None)
                access = self.struct_member_image_accesses.get(object_type, {}).get(
                    member
                )
                if access is not None:
                    return access
            return self.buffer_resource_access(pointer_expr)

        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def query_return_source_buffer_access(self, return_source, raw_args):
        """Return buffer access for a traceable returned resource."""
        kind = return_source.get("kind")
        if kind == "ternary":
            true_access = self.query_return_source_buffer_access(
                return_source["true_source"],
                raw_args,
            )
            false_access = self.query_return_source_buffer_access(
                return_source["false_source"],
                raw_args,
            )
            if true_access is None or true_access != false_access:
                return None
            return true_access

        if kind == "global":
            return self.buffer_resource_access(return_source.get("name"))

        if kind == "parameter":
            index = return_source.get("index")
            if index is None or index >= len(raw_args):
                return None
            return self.buffer_resource_access(raw_args[index])

        if kind == "member":
            object_type = return_source.get("object_type")
            member = return_source.get("member")
            return self.struct_member_image_accesses.get(object_type, {}).get(member)

        return None

    def unsupported_buffer_access_call(
        self, resource_label, operation, buffer_type, reason, fallback
    ):
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} {resource_label} "
            f"access: {operation} {reason} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_access_diagnostic(
        self, operation, buffer_type, buffer_expr, requirement, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if requirement == "read" and access == "writeonly":
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires readable buffer resource",
                fallback,
            )
        if requirement == "write" and access == "readonly":
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires writable buffer resource",
                fallback,
            )
        if requirement == "readwrite" and access in {"readonly", "writeonly"}:
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires readwrite buffer resource",
                fallback,
            )
        return None

    def byte_address_buffer_access_diagnostic(
        self, operation, buffer_type, buffer_expr, requirement, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if requirement == "read" and access == "writeonly":
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires readable buffer resource",
                fallback,
            )
        if requirement == "write" and access == "readonly":
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires writable buffer resource",
                fallback,
            )
        if requirement == "readwrite" and access in {"readonly", "writeonly"}:
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires readwrite buffer resource",
                fallback,
            )
        return None

    def structured_buffer_element_resource_type(self, expr):
        """Return the owning structured-buffer type for an element expression."""
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.expression_result_type(array_expr)
            array_type_string = self.type_name_string(array_type) or ""
            if (
                self.structured_buffer_type_parts(array_type) is not None
                and "[" not in array_type_string
            ):
                return array_type
            return None

        if isinstance(expr, (MemberAccessNode, PointerAccessNode)):
            object_expr = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", getattr(expr, "pointer_expr", None)),
            )
            return self.structured_buffer_element_resource_type(object_expr)

        return None

    def structured_buffer_element_read_diagnostic(self, expr, operation):
        buffer_type = self.structured_buffer_element_resource_type(expr)
        if buffer_type is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            expr,
            "read",
            fallback,
        )

    def structured_buffer_element_write_diagnostic(self, expr, operation, operator):
        buffer_type = self.structured_buffer_element_resource_type(expr)
        if buffer_type is None:
            return None
        requirement = "readwrite" if operator != "=" else "write"
        return self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            expr,
            requirement,
            "((void)0)",
        )

    def declaration_qualifier_names(self, node):
        """Return normalized qualifier and attribute names on declarations."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_readonly_qualifier_to_type(self, type_name, node):
        """Apply readonly declaration qualifiers to pointer/reference types."""
        type_string = self.type_name_string(type_name)
        if not type_string:
            return type_name

        if self.cuda_pointer_or_reference_type_info(type_string) is None:
            return type_string

        if not (
            self.declaration_qualifier_names(node) & {"const", "readonly", "constant"}
        ):
            return type_string

        if type_string.startswith("const "):
            return type_string
        return f"const {type_string}"

    def resource_type_with_access(self, type_name, node):
        """Apply resource access metadata to CUDA declaration type spelling."""
        type_string = self.apply_readonly_qualifier_to_type(type_name, node)
        type_string = self.type_name_string(type_string)
        if not type_string:
            return type_string

        access = self.explicit_resource_access(node)
        if access is None:
            return type_string

        base_type = type_string
        array_suffix = ""
        if "[" in type_string and "]" in type_string:
            open_bracket = type_string.find("[")
            base_type = type_string[:open_bracket]
            array_suffix = type_string[open_bracket:]

        parts = self.structured_buffer_type_parts(base_type)
        if parts is not None:
            base_name, element_type = parts
            if base_name in {"StructuredBuffer", "RWStructuredBuffer"}:
                mapped_base = (
                    "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
                )
                return f"{mapped_base}<{element_type}>{array_suffix}"

        byte_base = self.byte_address_buffer_base_type(base_type)
        if byte_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return f"{mapped_base}{array_suffix}"

        return type_string

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def is_buffer_resource_type(self, type_name):
        """Return whether a type names a CUDA-lowered buffer resource."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def register_variable_type(self, name, type_name, node=None, source_node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        type_name = self.apply_readonly_qualifier_to_type(type_name, node)
        self.variable_types[name] = type_name
        if self.is_glsl_buffer_block_node(node):
            access = self.explicit_resource_access(node) or "readwrite"
            self.glsl_buffer_block_accesses[name] = access
            self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)
        else:
            self.glsl_buffer_block_accesses.pop(name, None)
            self.glsl_buffer_block_layouts.pop(name, None)

        if self.is_buffer_resource_type(type_name):
            access = self.explicit_resource_access(node)
            if access is None and source_node is not None:
                access = self.buffer_resource_access(source_node)
            if access is None:
                self.buffer_resource_accesses.pop(name, None)
            else:
                self.buffer_resource_accesses[name] = access
        else:
            self.buffer_resource_accesses.pop(name, None)

        if not self.is_storage_image_type(type_name):
            self.image_resource_accesses.pop(name, None)
            return

        access = self.explicit_resource_access(node)
        if access is None and source_node is not None:
            access = self.image_resource_access(source_node)
        if access is None:
            self.image_resource_accesses.pop(name, None)
        else:
            self.image_resource_accesses[name] = access

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def next_cuda_temp_variable(self, prefix):
        index = self.match_temp_variable_index
        self.match_temp_variable_index += 1
        return f"__crossgl_{prefix}_{index}"

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def resource_expression_type(self, node):
        type_name = self.get_expression_type(node)
        if type_name is None:
            type_name = self.expression_result_type(node)
        if type_name is not None and not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def resource_query_metadata_unavailable_call(
        self, func_name, resource_type, fallback
    ):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} metadata unavailable on {resource_type} */ {fallback}"
        )

    def unsupported_dimension_metadata_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, fallback
        )

    def unsupported_scalar_metadata_query_call(self, func_name, resource_type):
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, "0"
        )

    def unsupported_dimension_resource_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ {fallback}"
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if resource_type is None:
            return None
        if func_name == "textureSize" and not self.is_sampled_resource_type(
            resource_type
        ):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        if func_name == "imageSize" and not self.is_storage_image_type(resource_type):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_dimension_metadata_query_call(
                func_name, resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        if spec["mip"]:
            self.ensure_query_prefix_helper()
        self.require_helper_function(
            helper_name, self.build_dimension_query_helper(helper_name, spec)
        )

        if spec["mip"]:
            lod = args[1] if len(args) > 1 else "0"
            return f"{helper_name}({metadata_expr}, {lod})"
        return f"{helper_name}({metadata_expr})"

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        expected_resource = (
            self.is_sampled_resource_type(resource_type)
            if func_name == "textureSamples"
            else self.is_storage_image_type(resource_type)
        )
        if not expected_resource:
            return None

        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(func_name, resource_type)

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_sample_count_query_helper(helper_name)
        )
        return f"{helper_name}({metadata_expr})"

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        spec = self.dimension_query_spec(resource_type)
        if (
            not self.is_sampled_resource_type(resource_type)
            or spec is None
            or not spec["mip"]
        ):
            return self.unsupported_scalar_resource_query_call(
                "textureQueryLevels", resource_type
            )

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(
                "textureQueryLevels", resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_textureQueryLevels_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_texture_query_levels_helper(helper_name, spec)
        )
        return f"{helper_name}({metadata_expr})"

    def map_vector_arithmetic_type(self, type_name):
        return self.convert_crossgl_type_to_cuda(type_name)

    def insert_helper_functions(self):
        if not self.helper_functions and not self.resource_query_info_required:
            return

        helper_lines = []
        if self.resource_query_info_required:
            helper_lines.extend(
                [
                    "struct CglResourceQueryInfo {",
                    "    int width;",
                    "    int height;",
                    "    int depth;",
                    "    int elements;",
                    "    int levels;",
                    "    int samples;",
                    "};",
                    "",
                ]
            )
        for helper in self.helper_functions.values():
            helper_lines.extend(helper.splitlines())
            helper_lines.append("")

        self.output[3:3] = helper_lines

    def require_cuda_ray_runtime_helpers(self):
        helper_name = "cgl_ray_runtime_helpers"
        if helper_name in self.helper_functions:
            return

        self.helper_functions[helper_name] = (
            "struct CglRayTracingAccelerationStructure\n"
            "{\n"
            "    unsigned long long handle;\n"
            "};\n"
            "\n"
            "struct CglRayDesc\n"
            "{\n"
            "    float3 origin;\n"
            "    float t_min;\n"
            "    float3 direction;\n"
            "    float t_max;\n"
            "\n"
            "    __host__ __device__ CglRayDesc()\n"
            "        : origin(make_float3(0.0f, 0.0f, 0.0f)),\n"
            "          t_min(0.0f),\n"
            "          direction(make_float3(0.0f, 0.0f, 1.0f)),\n"
            "          t_max(0.0f)\n"
            "    {\n"
            "    }\n"
            "\n"
            "    __host__ __device__ CglRayDesc(\n"
            "        float3 origin_value,\n"
            "        float t_min_value,\n"
            "        float3 direction_value,\n"
            "        float t_max_value)\n"
            "        : origin(origin_value),\n"
            "          t_min(t_min_value),\n"
            "          direction(direction_value),\n"
            "          t_max(t_max_value)\n"
            "    {\n"
            "    }\n"
            "};\n"
            "\n"
            "struct CglRayQuery\n"
            "{\n"
            "    unsigned int state;\n"
            "};\n"
            "\n"
            "struct CglBuiltInTriangleIntersectionAttributes\n"
            "{\n"
            "    float2 barycentrics;\n"
            "};\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_id()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_size()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_hit_t()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_hit_kind()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_t_min()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_incoming_flags()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_instance_custom_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_geometry_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "template <typename AS, typename Flags, typename Mask, "
            "typename HitGroup, typename Multiplier, typename Miss, "
            "typename Ray, typename Payload>\n"
            "__device__ inline void cgl_trace_ray(\n"
            "    const AS&,\n"
            "    const Flags&,\n"
            "    const Mask&,\n"
            "    const HitGroup&,\n"
            "    const Multiplier&,\n"
            "    const Miss&,\n"
            "    const Ray&,\n"
            "    const Payload&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Index, typename Data>\n"
            "__device__ inline void cgl_call_shader(const Index&, const Data&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Distance, typename Kind, typename Attributes>\n"
            "__device__ inline bool cgl_report_hit(\n"
            "    const Distance&,\n"
            "    const Kind&,\n"
            "    const Attributes&)\n"
            "{\n"
            "    return false;\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_ignore_hit()\n"
            "{\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_accept_hit_and_end_search()\n"
            "{\n"
            "}\n"
        )

    def require_cuda_ray_query_helper(self, operation):
        self.require_cuda_ray_runtime_helpers()
        helper_name = f"cgl_ray_query_{self.cuda_snake_case_name(operation)}"
        if helper_name in self.helper_functions:
            return helper_name

        return_type = "bool" if operation == "Proceed" else "unsigned int"
        return_value = "false" if return_type == "bool" else "0u"
        self.helper_functions[helper_name] = (
            "template <typename Query, typename... Args>\n"
            f"__device__ inline {return_type} {helper_name}(Query&, Args&&...)\n"
            "{\n"
            f"    return {return_value};\n"
            "}"
        )
        return helper_name

    def cuda_snake_case_name(self, name):
        text = str(name)
        result = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0:
                previous = text[index - 1]
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if previous.islower() or previous.isdigit() or next_char.islower():
                    result.append("_")
            result.append(char.lower())
        return "".join(result).replace("__", "_").strip("_")

    def vector_component_count_for_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        scalar_types = {
            "bool",
            "double",
            "f16",
            "f32",
            "f64",
            "float",
            "half",
            "i32",
            "int",
            "u32",
            "uint",
        }
        if base_type in scalar_types:
            return 1

        vector_prefixes = (
            "bvec",
            "dvec",
            "ivec",
            "uvec",
            "vec",
            "bool",
            "double",
            "float",
            "half",
            "int",
            "uint",
        )
        for prefix in vector_prefixes:
            if not base_type.startswith(prefix):
                continue
            suffix = base_type[len(prefix) :]
            if suffix and suffix[0] in {"2", "3", "4"}:
                return int(suffix[0])
        return None

    def texel_fetch_coordinate_count(self, raw_coord):
        coord_type = self.get_expression_type(raw_coord)
        if coord_type is None:
            coord_type = self.expression_result_type(raw_coord)
        return self.vector_component_count_for_type(coord_type)

    def expected_texel_fetch_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
        }.get(texture_type)

    def expected_texture_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(texture_type)

    def texture_coordinate_rank_diagnostic(self, func_name, texture_type, raw_coord):
        expected_coord_count = self.expected_texture_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def texture_offset_argument_index(self, func_name):
        return {
            "textureOffset": 2,
            "textureLodOffset": 3,
            "textureGradOffset": 4,
            "textureProjOffset": 2,
            "textureProjLodOffset": 3,
            "textureProjGradOffset": 4,
            "texelFetchOffset": 3,
        }.get(func_name)

    def expected_texture_offset_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texture_offset_rank_diagnostic(self, func_name, texture_type, raw_offset):
        expected_offset_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} offset rank",
                texture_type,
                [],
            )
        return None

    def expected_texture_gradient_coordinate_counts(self, texture_type):
        return {
            "sampler1D": {1},
            "sampler1DArray": {1},
            "sampler2D": {2},
            "sampler2DArray": {2},
            "sampler3D": {3, 4},
            "samplerCube": {3, 4},
            "samplerCubeArray": {3, 4},
        }.get(texture_type)

    def texture_gradient_rank_diagnostic(self, texture_type, *raw_gradients):
        expected_gradient_counts = self.expected_texture_gradient_coordinate_counts(
            texture_type
        )
        if expected_gradient_counts is None:
            return None

        for raw_gradient in raw_gradients:
            actual_gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
            if (
                actual_gradient_count is not None
                and actual_gradient_count not in expected_gradient_counts
            ):
                return self.unsupported_sampled_resource_call(
                    "textureGrad derivative rank",
                    texture_type,
                    [],
                )
        return None

    def sampled_texture_sampler_argument_offset(self, raw_args):
        if len(raw_args) > 2 and self.is_sampler_state_argument(raw_args[1]):
            return 1
        return 0

    def cuda_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
        if texture_type not in {"sampler3D", "samplerCube", "samplerCubeArray"}:
            return gradient

        gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
        if gradient_count == 3:
            return (
                f"make_float4({self.coord_component(gradient, 'x')}, "
                f"{self.coord_component(gradient, 'y')}, "
                f"{self.coord_component(gradient, 'z')}, 0.0f)"
            )
        return gradient

    def cuda_texture_coord_components(self, texture_type, coord):
        """Return CUDA texture coordinate components for a sampled resource."""
        if texture_type == "sampler1D":
            return [coord]
        if texture_type == "sampler1DArray":
            return [self.coord_component(coord, "x"), self.coord_component(coord, "y")]
        if texture_type == "sampler2D":
            return [self.coord_component(coord, "x"), self.coord_component(coord, "y")]
        if texture_type == "sampler2DArray":
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
            ]
        if texture_type in {"sampler3D", "samplerCube"}:
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
            ]
        if texture_type == "samplerCubeArray":
            return [
                self.coord_component(coord, "x"),
                self.coord_component(coord, "y"),
                self.coord_component(coord, "z"),
                self.coord_component(coord, "w"),
            ]
        return None

    def cuda_texture_offset_components(self, texture_type, coord, offset):
        """Return texture coordinate components with a GLSL-style texel offset."""
        components = self.cuda_texture_coord_components(texture_type, coord)
        if components is None:
            return None

        offset_component_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        if offset_component_count is None:
            return None

        if offset_component_count == 1:
            components[0] = f"({components[0]} + {offset})"
            return components

        for index, component in enumerate(("x", "y", "z")[:offset_component_count]):
            components[index] = (
                f"({components[index]} + {self.coord_component(offset, component)})"
            )
        return components

    def cuda_projected_texture_coord_components(self, texture_type, raw_coord, coord):
        """Return projected CUDA texture coordinate components."""
        coord_count = self.texel_fetch_coordinate_count(raw_coord)

        if texture_type == "sampler1D" and coord_count in {2, 4}:
            divisor = "y" if coord_count == 2 else "w"
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, divisor)})"
            ]

        if texture_type == "sampler2D" and coord_count in {3, 4}:
            divisor = "z" if coord_count == 3 else "w"
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, divisor)})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, divisor)})",
            ]

        if texture_type == "sampler2DArray" and coord_count == 4:
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, 'w')})",
                self.coord_component(coord, "z"),
            ]

        if texture_type in {"sampler3D", "samplerCube"} and coord_count == 4:
            return [
                f"({self.coord_component(coord, 'x')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'y')} / "
                f"{self.coord_component(coord, 'w')})",
                f"({self.coord_component(coord, 'z')} / "
                f"{self.coord_component(coord, 'w')})",
            ]

        return None

    def cuda_apply_texture_offset_components(self, texture_type, components, offset):
        """Apply a texture offset to existing coordinate components."""
        offset_component_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        if offset_component_count is None:
            return None

        components = list(components)
        if offset_component_count == 1:
            components[0] = f"({components[0]} + {offset})"
            return components

        for index, component in enumerate(("x", "y", "z")[:offset_component_count]):
            components[index] = (
                f"({components[index]} + {self.coord_component(offset, component)})"
            )
        return components

    def cuda_texture_sample_call_from_components(
        self,
        func_name,
        texture_type,
        texture_name,
        components,
        lod=None,
        grad_x=None,
        grad_y=None,
    ):
        """Return a CUDA texture sample call from precomputed components."""
        if texture_type == "sampler1D" and len(components) >= 1:
            coord_args = f"{texture_name}, {components[0]}"
            if func_name == "texture":
                return f"tex1D<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex1DLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex1DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler1DArray" and len(components) >= 2:
            coord_args = f"{texture_name}, {components[0]}, {components[1]}"
            if func_name == "texture":
                return f"tex1DLayered<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex1DLayeredLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex1DLayeredGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler2D" and len(components) >= 2:
            coord_args = f"{texture_name}, {components[0]}, {components[1]}"
            if func_name == "texture":
                return f"tex2D<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex2DLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex2DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler2DArray" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"tex2DLayered<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex2DLayeredLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex2DLayeredGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler3D" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"tex3D<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"tex3DLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "samplerCube" and len(components) >= 3:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, {components[2]}"
            )
            if func_name == "texture":
                return f"texCubemap<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"texCubemapLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return f"texCubemapGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "samplerCubeArray" and len(components) >= 4:
            coord_args = (
                f"{texture_name}, {components[0]}, {components[1]}, "
                f"{components[2]}, {components[3]}"
            )
            if func_name == "texture":
                return f"texCubemapLayered<float4>({coord_args})"
            if func_name == "textureLod" and lod is not None:
                return f"texCubemapLayeredLod<float4>({coord_args}, {lod})"
            if func_name == "textureGrad" and grad_x is not None:
                return (
                    f"texCubemapLayeredGrad<float4>"
                    f"({coord_args}, {grad_x}, {grad_y})"
                )

        return None

    def cuda_texel_fetch_call_from_components(
        self, texture_type, texture_name, components
    ):
        """Return a CUDA texel-fetch expression from precomputed components."""
        if texture_type == "sampler1D" and len(components) >= 1:
            return f"tex1Dfetch<float4>({texture_name}, {components[0]})"
        if texture_type == "sampler1DArray" and len(components) >= 2:
            return (
                f"tex1DLayered<float4>"
                f"({texture_name}, {components[0]}, {components[1]})"
            )
        if texture_type == "sampler2D" and len(components) >= 2:
            return f"tex2D<float4>({texture_name}, {components[0]}, {components[1]})"
        if texture_type == "sampler2DArray" and len(components) >= 3:
            return (
                f"tex2DLayered<float4>"
                f"({texture_name}, {components[0]}, {components[1]}, {components[2]})"
            )
        if texture_type == "sampler3D" and len(components) >= 3:
            return (
                f"tex3D<float4>"
                f"({texture_name}, {components[0]}, {components[1]}, {components[2]})"
            )
        return None

    def generate_texture_offset_call(self, func_name, raw_args, args):
        """Lower textureOffset/textureLodOffset/textureGradOffset to CUDA calls."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )
        if texture_type not in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
        }:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        if len(args) <= coord_index:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            func_name, texture_type, raw_args[coord_index]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        texture_name = args[0]
        coord = args[coord_index]
        lod = None
        grad_x = None
        grad_y = None
        offset_index = None
        sample_func = "texture"

        if func_name == "textureOffset":
            offset_index = coord_index + 1
            if len(args) > offset_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureOffset bias", texture_type, args
                )
        elif func_name == "textureLodOffset":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            offset_index = coord_index + 2
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureGradOffset":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            offset_index = coord_index + 3
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )

        if offset_index is None or len(args) <= offset_index:
            return None

        offset_diagnostic = self.texture_offset_rank_diagnostic(
            func_name, texture_type, raw_args[offset_index]
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        components = self.cuda_texture_offset_components(
            texture_type, coord, args[offset_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        return self.cuda_texture_sample_call_from_components(
            sample_func,
            texture_type,
            texture_name,
            components,
            lod=lod,
            grad_x=grad_x,
            grad_y=grad_y,
        )

    def generate_texture_projected_call(self, func_name, raw_args, args):
        """Lower supported projected texture operations to CUDA texture calls."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )
        if texture_type not in {
            "sampler1D",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
            "samplerCube",
        }:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        if len(args) <= coord_index:
            return None

        components = self.cuda_projected_texture_coord_components(
            texture_type, raw_args[coord_index], args[coord_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank", texture_type, args
            )

        texture_name = args[0]
        sample_func = "texture"
        lod = None
        grad_x = None
        grad_y = None
        offset_index = None

        if func_name == "textureProj":
            if len(args) > coord_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureProj bias", texture_type, args
                )
        elif func_name == "textureProjOffset":
            offset_index = coord_index + 1
            if len(args) > offset_index + 1:
                return self.unsupported_sampled_resource_call(
                    "textureProjOffset bias", texture_type, args
                )
        elif func_name == "textureProjLod":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureProjLodOffset":
            sample_func = "textureLod"
            lod_index = coord_index + 1
            offset_index = coord_index + 2
            if len(args) <= lod_index:
                return None
            lod = args[lod_index]
        elif func_name == "textureProjGrad":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )
        elif func_name == "textureProjGradOffset":
            sample_func = "textureGrad"
            grad_x_index = coord_index + 1
            grad_y_index = coord_index + 2
            offset_index = coord_index + 3
            if len(args) <= grad_y_index:
                return None
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[grad_x_index],
                raw_args[grad_y_index],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_x_index],
                args[grad_x_index],
            )
            grad_y = self.cuda_texture_gradient_argument(
                texture_type,
                raw_args[grad_y_index],
                args[grad_y_index],
            )

        if offset_index is not None:
            if len(args) <= offset_index:
                return None
            if texture_type == "samplerCube":
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            offset_diagnostic = self.texture_offset_rank_diagnostic(
                func_name, texture_type, raw_args[offset_index]
            )
            if offset_diagnostic is not None:
                return offset_diagnostic
            components = self.cuda_apply_texture_offset_components(
                texture_type, components, args[offset_index]
            )
            if components is None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        return self.cuda_texture_sample_call_from_components(
            sample_func,
            texture_type,
            texture_name,
            components,
            lod=lod,
            grad_x=grad_x,
            grad_y=grad_y,
        )

    def generate_texel_fetch_offset_call(self, raw_args, args):
        """Lower texelFetchOffset by applying the offset before CUDA fetch."""
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type is None:
            return None
        if texture_type is not None and not self.is_sampled_resource_type(texture_type):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if texture_type in {"samplerCube", "samplerCubeArray"}:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )

        sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
        coord_index = 1 + sampler_arg_offset
        offset_index = coord_index + 2
        if len(args) <= offset_index:
            return None

        expected_coord_count = self.expected_texel_fetch_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_args[coord_index])
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset coordinate rank",
                texture_type,
                args,
            )

        offset_diagnostic = self.texture_offset_rank_diagnostic(
            "texelFetchOffset", texture_type, raw_args[offset_index]
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        components = self.cuda_texture_offset_components(
            texture_type, args[coord_index], args[offset_index]
        )
        if components is None:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        return self.cuda_texel_fetch_call_from_components(
            texture_type, args[0], components
        )

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
        if "3D" in image_type:
            return 3
        if "Cube" in image_type:
            return 3
        if "Array" in image_type:
            return 3
        if "2D" in image_type:
            return 2
        return None

    def unsupported_image_coordinate_rank_call(self, func_name, image_type):
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported CUDA image resource call: "
            f"{func_name} coordinate rank on {image_type} */ {fallback}"
        )

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, FunctionCallNode):
            callee_name = self.raw_function_call_name(image_arg)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                image_arg,
                "arguments",
                getattr(image_arg, "args", []),
            )
            return self.query_return_source_image_access(return_source, raw_args)

        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        if isinstance(image_arg, MemberAccessNode):
            object_node = getattr(
                image_arg,
                "object_expr",
                getattr(image_arg, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.expression_result_type(object_node)
            )
            member = getattr(image_arg, "member", None)
            access = self.struct_member_image_accesses.get(object_type, {}).get(member)
            if access is not None:
                return access

        if isinstance(image_arg, PointerAccessNode):
            pointer_expr = getattr(image_arg, "pointer_expr", None)
            pointer_type = self.expression_result_type(pointer_expr)
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is not None:
                object_type = self.struct_member_lookup_type(
                    pointer_info["pointee_type"]
                )
                member = getattr(image_arg, "member", None)
                access = self.struct_member_image_accesses.get(object_type, {}).get(
                    member
                )
                if access is not None:
                    return access

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def query_return_source_image_access(self, return_source, raw_args):
        """Return storage-image access for a traceable returned resource."""
        kind = return_source.get("kind")
        if kind == "ternary":
            true_access = self.query_return_source_image_access(
                return_source["true_source"],
                raw_args,
            )
            false_access = self.query_return_source_image_access(
                return_source["false_source"],
                raw_args,
            )
            if true_access is None or true_access != false_access:
                return None
            return true_access

        if kind == "global":
            return self.image_resource_access(return_source.get("name"))

        if kind == "parameter":
            index = return_source.get("index")
            if index is None or index >= len(raw_args):
                return None
            return self.image_resource_access(raw_args[index])

        if kind == "member":
            object_type = return_source.get("object_type")
            member = return_source.get("member")
            return self.struct_member_image_accesses.get(object_type, {}).get(member)

        return None

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported CUDA image access: "
            f"{func_name} {reason} on {image_type} */ {fallback}"
        )

    def image_access_diagnostic(self, func_name, image_type, raw_image):
        access = self.image_resource_access(raw_image)
        if func_name == "imageLoad" and access == "writeonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires readable image resource",
            )
        if func_name == "imageStore" and access == "readonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires writable image resource",
            )
        return None

    def image_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_coordinate_rank_call(func_name, image_type)
        return None

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported CUDA image atomic resource call: "
            f"{func_name} coordinate rank on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported CUDA image atomic resource call: "
            f"{func_name} requires readwrite image resource on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

    def image_atomic_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_atomic_coordinate_rank_call(
                func_name, image_type
            )
        return None

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ 0"
        )

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            sample_count_query = self.generate_sample_count_query(
                func_name, raw_args, args
            )
            if sample_count_query is not None:
                return sample_count_query
            if raw_args:
                resource_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                if resource_type is not None:
                    return self.unsupported_scalar_resource_query_call(
                        func_name, resource_type
                    )
            return None

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureCompareLodOffset",
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
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {"textureGather", "textureGatherOffset"} and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(
                func_name, raw_args, args
            )
            if texture_gather is not None:
                return texture_gather

        if (
            func_name
            in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
            }
            and raw_args
        ):
            return self.generate_texture_offset_call(func_name, raw_args, args)

        if (
            func_name
            in {
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
            }
            and raw_args
        ):
            return self.generate_texture_projected_call(func_name, raw_args, args)

        if func_name == "texelFetchOffset" and raw_args:
            return self.generate_texel_fetch_offset_call(raw_args, args)

        if (
            func_name
            in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                offset_arg_index = self.texture_offset_argument_index(func_name)
                if offset_arg_index is not None:
                    offset_arg_index += self.sampled_texture_sampler_argument_offset(
                        raw_args
                    )
                if offset_arg_index is not None and len(raw_args) > offset_arg_index:
                    offset_diagnostic = self.texture_offset_rank_diagnostic(
                        func_name,
                        texture_type,
                        raw_args[offset_arg_index],
                    )
                    if offset_diagnostic is not None:
                        return offset_diagnostic
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
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
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coord_index = 1 + sampler_arg_offset
            if len(args) <= coord_index:
                return None
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[coord_index]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            if func_name == "texture" and len(args) > coord_index + 1:
                return self.unsupported_sampled_resource_call(
                    "texture bias", texture_type, args
                )
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) > coord_index + 2:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[coord_index + 1],
                    raw_args[coord_index + 2],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[coord_index + 1],
                    args[coord_index + 1],
                )
                grad_y = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[coord_index + 2],
                    args[coord_index + 2],
                )

            texture_name = args[0]
            coord = args[coord_index]
            lod_index = coord_index + 1
            if texture_type == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<float4>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"tex1DLod<float4>"
                        f"({texture_name}, {coord}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DGrad<float4>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex1DLayeredLod<float4>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLod<float4>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex2DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLayeredLod<float4>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex2DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex3DLod<float4>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"texCubemapLod<float4>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"texCubemapGrad<float4>({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"texCubemapLayeredLod<float4>"
                        f"({coord_args}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"texCubemapLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_resource_type(
                texture_type
            ):
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            if texture_type in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coord_index = 1 + sampler_arg_offset
            lod_index = coord_index + 1
            if len(args) <= lod_index:
                return None
            texture_name = args[0]
            coord = args[coord_index]
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_type
            )
            actual_coord_count = self.texel_fetch_coordinate_count(
                raw_args[coord_index]
            )
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            components = self.cuda_texture_coord_components(texture_type, coord)
            if components is not None:
                return self.cuda_texel_fetch_call_from_components(
                    texture_type, texture_name, components
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "CubeArray" in image_type:
                layer_face = self.coord_component(coord, "z")
                return (
                    f"surfCubemapLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer_face})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapread<{value_type}>({image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredread<{value_type}>({image_name}, {x}, {layer})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return (
                    f"surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "1D" in image_type:
                x = f"{coord} * sizeof({value_type})"
                return f"surf1Dread<{value_type}>({image_name}, {x})"
            if "2D" in image_type:
                return f"surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            value = args[2]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "CubeArray" in image_type:
                layer_face = self.coord_component(coord, "z")
                return (
                    f"surfCubemapLayeredwrite"
                    f"({value}, {image_name}, {x}, {y}, {layer_face})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "1D" in image_type:
                x = f"{coord} * sizeof({value_type})"
                return f"surf1Dwrite({value}, {image_name}, {x})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

    def generate_texture_gather_call(self, func_name, raw_args, args):
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        offset_index = None
        component = None
        component_arg = None
        if func_name == "textureGather":
            if len(args) == 2:
                pass
            elif len(args) == 3:
                if self.is_sampler_state_argument(raw_args[1]):
                    coordinate_index = 2
                else:
                    component = args[2]
                    component_arg = raw_args[2]
            elif len(args) == 4 and self.is_sampler_state_argument(raw_args[1]):
                coordinate_index = 2
                component = args[3]
                component_arg = raw_args[3]
            else:
                return None
        elif func_name == "textureGatherOffset":
            sampler_arg_offset = self.sampled_texture_sampler_argument_offset(raw_args)
            coordinate_index = 1 + sampler_arg_offset
            offset_index = coordinate_index + 1
            if len(args) <= offset_index:
                return None
            if len(args) > offset_index + 1:
                component = args[offset_index + 1]
                component_arg = raw_args[offset_index + 1]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            func_name,
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        if offset_index is not None:
            offset_diagnostic = self.texture_offset_rank_diagnostic(
                func_name,
                texture_type,
                raw_args[offset_index],
            )
            if offset_diagnostic is not None:
                return offset_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        x = self.coord_component(coord, "x")
        y = self.coord_component(coord, "y")
        if offset_index is not None:
            offset = args[offset_index]
            x = f"({x} + {self.coord_component(offset, 'x')})"
            y = f"({y} + {self.coord_component(offset, 'y')})"
        gather_args = f"{texture_name}, " f"{x}, " f"{y}"
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False

    def literal_int_value(self, node):
        if isinstance(node, LiteralNode):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                literal_type = getattr(
                    getattr(node, "literal_type", None), "name", None
                )
                if literal_type in {"char", "string"}:
                    return None
                try:
                    return int(value, 0)
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator not in {"+", "-"}:
                return None
            value = self.literal_int_value(node.operand)
            if value is None:
                return None
            if operator == "-":
                return -value
            return value

        return None

    def is_sampler_state_argument(self, raw_arg):
        return (
            self.resource_base_type(self.resource_expression_type(raw_arg)) == "sampler"
        )

    def visit_cbuffer(self, cbuffer):
        """Visit constant buffer (convert to CUDA constant memory)"""
        metadata_comment = self.cuda_resource_metadata_comment(
            cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
        )
        if metadata_comment:
            self.emit(metadata_comment)
        self.emit(f"// Constant buffer: {cbuffer.name}")
        for member in cbuffer.members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            else:
                member_type = member.vtype
            declaration = self.format_typed_declarator(member_type, member.name)
            self.emit(f"__constant__ {declaration};")

    def visit_ArrayNode(self, node):
        """Visit array declaration"""
        if hasattr(node, "element_type"):
            element_type = self.convert_crossgl_type_to_cuda(node.element_type)
        else:
            element_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if node.size:
            self.emit(
                f"{element_type} {node.name}[{self.format_array_size(node.size)}];"
            )
        else:
            # Dynamic array - use pointer in CUDA
            self.emit(f"{element_type}* {node.name};")

    def visit_TernaryOpNode(self, node):
        """Visit ternary conditional operator"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_list(self, node_list):
        """Visit a list of nodes"""
        results = []
        for node in node_list:
            result = self.visit(node)
            if result:
                results.append(result)
        return results

    def visit_str(self, node):
        """Visit string literals"""
        return node

    def visit_int(self, node):
        """Visit integer literals"""
        return str(node)

    def visit_float(self, node):
        """Visit float literals"""
        return str(node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if (
                        hasattr(arg, "name")
                        or hasattr(arg, "element_type")
                        or hasattr(arg, "pointee_type")
                        or hasattr(arg, "referenced_type")
                    ):
                        converted = self.convert_type_node_to_string(arg)
                    elif hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                    if converted is None:
                        if hasattr(arg, "value"):
                            converted = str(arg.value)
                        else:
                            converted = str(arg)
                    converted_args.append(converted)
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                prefix = "dmat" if element_type == "double" else "mat"
                return f"{prefix}{type_node.rows}x{type_node.cols}"
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{self.format_array_size(type_node.size)}]"
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
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)
