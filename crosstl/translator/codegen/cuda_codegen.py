"""CrossGL-to-CUDA code generator."""

from ..ast import (
    EnumNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    StructNode,
    VariableNode,
    WaveOpNode,
)
from ..refusals import CudaRefusal
from ..semantics.declared_types import declared_type_names
from ..semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)
from .c_narrow_integer_vectors import generic_narrow_vector_spelling
from .c_reference_parameters import is_output_parameter, reference_declarator
from .c_wide_integer_vectors import wide_integer_vector_type
from .comma_expressions import (
    CommaExpressionVisitorMixin,
    comma_result_operand,
    is_comma_expression,
)
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .cuda_bindings import CudaBindingMixin
from .cuda_buffer_calls import CudaBufferCallMixin
from .cuda_expression_types import CudaExpressionTypeMixin
from .cuda_expressions import CudaExpressionMixin
from .cuda_function_calls import CudaFunctionCallMixin
from .cuda_gather import CudaGatherMixin
from .cuda_library_functions import rename_cuda_library_functions
from .cuda_math_calls import CudaMathCallMixin
from .cuda_query_metadata import CudaQueryMetadataMixin
from .cuda_query_returns import CudaQueryReturnMixin
from .cuda_refusals import CudaLoweringRefused
from .cuda_resource_access import CudaResourceAccessMixin
from .cuda_semantics import CudaSemanticMixin
from .cuda_stage_attributes import CudaStageAttributeMixin
from .cuda_stage_checks import CudaStageCheckMixin
from .cuda_statements import CudaStatementMixin
from .cuda_tables import (
    CUDA_64BIT_BITCAST_FUNCTIONS,
    CUDA_BITCAST_FUNCTION_TARGETS,
    CUDA_FP16_VECTOR_TYPES,
    CUDA_INTEGER_BIT_FUNCTION_ALIASES,
    CUDA_SCALAR_CONSTRUCTOR_TYPE_ALIASES,
    CUDA_WARP_SYNC_BUILTIN_ARITIES,
    CUDA_WAVE_BOOL_RESULT_OPS,
    CUDA_WAVE_OP_ARITIES,
    CUDA_WAVE_PREDICATE_ARGUMENT_OPS,
    CUDA_WAVE_UINT_RESULT_OPS,
    CUDA_WAVE_UVEC4_RESULT_OPS,
)
from .cuda_textures import CudaTextureMixin
from .cuda_types import CudaTypeMixin
from .cuda_vector_types import CudaVectorTypeMixin
from .cuda_visitor import CudaVisitorMixin
from .cuda_wave_ops import CudaWaveOpMixin
from .float_remainder_assignment import float_remainder_assignment
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    iter_function_nodes,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .gpu_device_globals import is_constant_value_declaration
from .lowering_refusals import LoweringRefusalMixin
from .precise_evaluation import PreciseEvaluationMixin
from .relational_builtins import lower_relational_builtins
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .sampler_pairs import canonicalize_separate_textures
from .stage_local_structs import module_structs
from .stage_utils import is_fragment_output_parameter
from .value_array_parameters import (
    collect_function_parameters_by_name,
    collect_struct_names,
    fixed_array_dimensions,
    plan_value_array_copies,
    value_array_copy_lines,
)
from .vector_arithmetic import VectorArithmeticMixin
from .written_function_names import (
    CUDA_CALLABLE_FUNCTION_NAMES,
    refuse_written_foreign_calls,
)
from .written_type_names import (
    CUDA_TYPE_NAMES,
    refuse_written_foreign_types,
    written_type_name,
)

# Most generator members live in the cuda_*.py mixin modules next to this one. These names stay importable from here.
__all__ = [
    "CUDA_64BIT_BITCAST_FUNCTIONS",
    "CUDA_BITCAST_FUNCTION_TARGETS",
    "CUDA_FP16_VECTOR_TYPES",
    "CUDA_INTEGER_BIT_FUNCTION_ALIASES",
    "CUDA_SCALAR_CONSTRUCTOR_TYPE_ALIASES",
    "CUDA_WARP_SYNC_BUILTIN_ARITIES",
    "CUDA_WAVE_BOOL_RESULT_OPS",
    "CUDA_WAVE_OP_ARITIES",
    "CUDA_WAVE_PREDICATE_ARGUMENT_OPS",
    "CUDA_WAVE_UINT_RESULT_OPS",
    "CUDA_WAVE_UVEC4_RESULT_OPS",
    "CudaCodeGen",
]


def _refuse_buffer_counter(reason):
    raise CudaRefusal(f"CUDA buffer counter: {reason}")


class CudaCodeGen(
    CudaStageCheckMixin,
    CudaVisitorMixin,
    CudaSemanticMixin,
    CudaStageAttributeMixin,
    CudaStatementMixin,
    CudaExpressionMixin,
    CudaFunctionCallMixin,
    CudaWaveOpMixin,
    CudaBufferCallMixin,
    CudaMathCallMixin,
    CudaTypeMixin,
    CudaExpressionTypeMixin,
    CudaQueryReturnMixin,
    CudaQueryMetadataMixin,
    CudaBindingMixin,
    CudaResourceAccessMixin,
    CudaVectorTypeMixin,
    CudaTextureMixin,
    CudaGatherMixin,
    LoweringRefusalMixin,
    PreciseEvaluationMixin,
    CommaExpressionVisitorMixin,
    VectorArithmeticMixin,
    ResourceQueryMixin,
    ResourceDiagnosticMixin,
):
    """Emit CUDA source from the shared CrossGL translator AST."""

    LOWERING_REFUSAL_TYPE = CudaLoweringRefused
    # nvcc never merges these into a multiply-add (CUDA Math API); see precise_evaluation.
    PRECISE_MULTIPLY_INTRINSICS = {
        "float": "__fmul_rn",
        "double": "__dmul_rn",
        "half": "__hmul_rn",
    }
    PRECISE_TARGET_LABEL = "CUDA"

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
        "workgroupExecutionBarrier",
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
    # The type names the program declares, and the foreign built-in type names written so far (``None`` outside
    # a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None

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
        self.structs_by_name = {}
        self.struct_member_semantics = {}
        self.struct_member_image_accesses = {}
        self.function_return_types = {}
        self.function_parameters_by_name = {}
        self.value_array_struct_names = frozenset()
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
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
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
        """Generate complete CUDA source for a CrossGL AST, or raise :class:`CudaLoweringRefused`.

        A construct with no faithful CUDA lowering is refused; no program is returned while a refusal is
        pending, and a rendered program that still carries a ``/* unsupported ... */`` loss marker is refused.
        """
        refuse_cooperative_matrices(ast_node, label="CUDA", target="cuda")
        return self.refusals.run(self.render_checked_program, ast_node)

    def render_checked_program(self, ast_node):
        """Render the program and refuse each built-in type and function of another language it still names."""
        self.program_type_names = declared_type_names(ast_node)
        self.written_foreign_type_names = set()
        try:
            output = self.render_program(ast_node)
            refuse_written_foreign_types(
                output, self.written_foreign_type_names, "CUDA"
            )
            refuse_written_foreign_calls(output, CUDA_CALLABLE_FUNCTION_NAMES, "CUDA")
        finally:
            self.written_foreign_type_names = None
        return output

    def render_program(self, ast_node):
        """Render complete CUDA source for a CrossGL AST."""
        # A CUDA texture object is the image whatever sampler state a separate CrossGL texture is paired with.
        ast_node = canonicalize_separate_textures(ast_node)
        # A user function named like a device library function would redefine it or make its calls ambiguous.
        ast_node = rename_cuda_library_functions(ast_node)
        # GLSL's lessThan, equal, not and the other relational built-ins are the componentwise operators.
        lower_relational_builtins(ast_node)
        # CUDA has no hidden buffer counter, so a counted RWStructuredBuffer gets an explicit one.
        ast_node = materialize_structured_buffer_counters(
            ast_node, _refuse_buffer_counter
        )
        self.output = []
        self.indent_level = 0
        self.validate_supported_stage_types(ast_node)
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
        self.structs_by_name = self.collect_structs_by_name(ast_node)
        (
            self.struct_member_types,
            self.struct_member_image_accesses,
        ) = self.collect_struct_member_metadata(ast_node)
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(ast_node)
        self.function_parameters_by_name = collect_function_parameters_by_name(
            ast_node, self.type_name_string
        )
        self.value_array_struct_names = collect_struct_names(ast_node)
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            list(iter_function_nodes(ast_node)),
            collect_constant_and_cbuffer_member_types(self, ast_node),
            ast=ast_node,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        self.reject_unsupported_generic_functions(ast_node)
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
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
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
        result = "\n".join(self.output)
        self.reject_unresolved_generic_output(result)
        return result

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

    def collect_structs_by_name(self, root):
        """Return plain struct declarations visible to CUDA expression lowering."""
        structs = list(getattr(root, "structs", []) or [])
        for stage in (getattr(root, "stages", {}) or {}).values():
            structs.extend(getattr(stage, "local_structs", []) or [])
        return {
            struct.name: struct
            for struct in structs
            if isinstance(struct, StructNode) and getattr(struct, "name", None)
        }

    def visit_ShaderNode(self, node):
        """Render a full shader/program AST as a CUDA translation unit."""
        self.emit("#include <cuda_runtime.h>")
        self.emit("#include <device_launch_parameters.h>")
        self.emit("#include <cuda_fp16.h>")
        # CrossGL's fixed-width integers lower to uint8_t, int64_t and friends, which none of the
        # CUDA headers above declares.
        self.emit("#include <cstdint>")
        self.emit("")
        self.emit_generated_code(self.generate_cuda_matrix_type_helpers())
        self.emit("")
        if self.has_geometry_stage(node):
            self.emit_generated_code(self.generate_cuda_geometry_stream_helpers())
        if self.has_tessellation_stage(node):
            self.emit_generated_code(self.generate_cuda_tessellation_patch_helpers())
        if self.has_mesh_task_stage(node):
            self.emit_generated_code(self.generate_cuda_mesh_task_helpers())

        for struct in module_structs(node, "CUDA"):
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
        for stage_type, stage in (getattr(node, "stages", {}) or {}).items():
            stage_name = (
                str(stage_type).split(".")[-1].lower()
                if hasattr(stage_type, "name")
                else str(stage_type).lower()
            )
            for var in getattr(stage, "local_variables", []) or []:
                if self.is_cuda_shared_variable(var):
                    continue
                if self.cuda_stage_local_builtin_role(stage_name, var) is not None:
                    # A stage-local input that names a compute built-in is the
                    # built-in itself, not a variable to declare.
                    continue
                var_name = getattr(var, "name", None)
                var_type = self.type_name_string(self.get_variable_node_type(var))
                if not var_name:
                    continue
                previous_type = emitted_stage_local_variables.get(var_name)
                if previous_type is not None:
                    if previous_type != var_type:
                        raise CudaRefusal(
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
            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    self.visit(specialized_func)
                    self.emit("")
                continue
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
                            if generic_function_parameters(func):
                                for specialized_func in generic_function_emission_list(
                                    self,
                                    func,
                                ):
                                    self.visit(specialized_func)
                                    self.emit("")
                                emitted_local_functions.add(id(func))
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
        saved_stage_builtin_alias_types = self.stage_builtin_alias_types
        saved_current_function_is_kernel_entry = self.current_function_is_kernel_entry
        saved_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_generic_function_substitutions = (
            getattr(node, "_generic_substitutions", {}) or {}
        )
        self.current_structured_buffer_length_parameters = {}
        self.query_metadata_aliases = {}
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
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

        param_list = list(getattr(node, "parameters", getattr(node, "params", [])))
        param_list.extend(self.cuda_function_capture_params.get(node.name, []))
        return_semantic = self.semantic_from_node(node)
        fragment_output_param = None
        fragment_output_params = [
            param
            for param in param_list
            if is_fragment_output_parameter(
                stage_name,
                param,
                self.semantic_from_node(param),
            )
        ]
        if (
            fragment_output_params
            and return_semantic is None
            and self.convert_crossgl_type_to_cuda(self.current_function_return_type)
            == "void"
        ):
            fragment_output_param = fragment_output_params[0]
            self.current_function_return_type = self.get_parameter_type(
                fragment_output_param
            )
            return_type = self.convert_crossgl_type_to_cuda(
                self.current_function_return_type
            )
            return_semantic = self.semantic_from_node(fragment_output_param)
            param_list = [
                param for param in param_list if param is not fragment_output_param
            ]

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
        self.validate_cuda_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_cuda_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_cuda_tessellation_stage(node, stage_name)
        if stage_name in self.cuda_mesh_task_stage_names():
            self.validate_cuda_mesh_task_stage(node, stage_name)

        value_array_copies = (
            {}
            if stage_name is not None
            else plan_value_array_copies(
                node,
                param_list,
                self.type_name_string,
                self.function_parameters_by_name,
                self.value_array_struct_names,
            )
        )
        value_array_copy_statements = []
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
                self.stage_builtin_alias_types[param_name] = declaration_type
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
            elif param_name in value_array_copies:
                incoming = value_array_copies[param_name]
                self.register_variable_type(param_name, declaration_type, param)
                params.append(self.format_typed_declarator(declaration_type, incoming))
                value_array_copy_statements.extend(
                    value_array_copy_lines(
                        self.format_typed_declarator(declaration_type, param_name),
                        param_name,
                        incoming,
                        fixed_array_dimensions(self.type_name_string(declaration_type)),
                    )
                )
            else:
                self.register_variable_type(param_name, declaration_type, param)
                declarator = self.format_typed_declarator(declaration_type, param_name)
                if stage_name is None and is_output_parameter(param):
                    # out and inout copy back to the caller; a C++ reference binds it.
                    declarator = reference_declarator(declarator, param_name)
                declarator = self.apply_restrict_qualifier_to_declaration(
                    declarator, param_name, param
                )
                params.append(declarator)
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
            self.stage_builtin_alias_types = saved_stage_builtin_alias_types
            self.current_function_is_kernel_entry = (
                saved_current_function_is_kernel_entry
            )
            self.current_generic_function_substitutions = (
                saved_generic_function_substitutions
            )
            self.current_structured_buffer_length_parameters = (
                saved_structured_buffer_length_parameters
            )
            return
        self.emit(f"{signature_prefix} {return_type} {function_name}({param_str}) {{")

        self.indent_level += 1
        for statement in value_array_copy_statements:
            self.emit(statement)

        fragment_output_name = None
        if fragment_output_param is not None:
            fragment_output_name = getattr(
                fragment_output_param,
                "name",
                getattr(fragment_output_param, "param_name", None),
            )
            fragment_output_type = self.resource_type_with_access(
                self.get_parameter_type(fragment_output_param),
                fragment_output_param,
            )
            self.register_variable_type(
                fragment_output_name,
                fragment_output_type,
                fragment_output_param,
            )
            self.emit(
                f"{self.format_typed_declarator(fragment_output_type, fragment_output_name)};"
            )

        for local_var in getattr(self, "current_stage_local_variables", []) or []:
            if self.is_cuda_shared_variable(local_var):
                self.visit(local_var)

        precise_state = self.enter_precise_evaluation(node)
        try:
            self.emit_body(body)
        finally:
            self.leave_precise_evaluation(precise_state)
        if fragment_output_name is not None and not self.statement_body_terminates(
            body
        ):
            self.emit(f"return {fragment_output_name};")

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
        self.stage_builtin_alias_types = saved_stage_builtin_alias_types
        self.current_function_is_kernel_entry = saved_current_function_is_kernel_entry
        self.current_generic_function_substitutions = (
            saved_generic_function_substitutions
        )
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )

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
                    if qualifier == "const" or (
                        qualifier == "constant"
                        and is_constant_value_declaration(node)
                        and self.cuda_pointer_or_reference_type_info(
                            self.type_name_string(var_type)
                        )
                        is None
                    ):
                        if "const" not in qualifiers:
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
            declaration = self.apply_restrict_qualifier_to_declaration(
                declaration, node.name, node
            )
            if initial_value is not None:
                declaration += f" = {self.visit(initial_value)}"
            return declaration

        return node.name

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
        # C++ has no floating %=; x = x % y lowers the remainder through fmodf.
        remainder = float_remainder_assignment(
            node.target,
            node.value,
            operator,
            type_of=self.expression_result_type,
            is_floating=self.is_floating_value_type,
            backend="cuda",
        )
        if remainder is not None:
            return self.format_assignment_expression(remainder)
        precise_product = self.precise_multiply_assignment(
            node, node.target, node.value, operator
        )
        if precise_product is not None:
            return self.format_assignment_expression(precise_product)
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

    def visit_ConstructorNode(self, node):
        """Lower braced AST constructors to CUDA aggregate/value constructors."""
        constructor_type = self.type_name_string(
            getattr(node, "constructor_type", None)
        )
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]

        if constructor_type in self.structs_by_name:
            args = self.cuda_struct_constructor_brace_arguments(node, constructor_type)
            mapped_type = self.convert_crossgl_type_to_cuda(constructor_type)
            return f"{mapped_type}{{{', '.join(args)}}}"

        vector_info = self.vector_type_info(constructor_type)
        if vector_info:
            raw_args, args = self.cuda_vector_constructor_arguments(
                node,
                vector_info,
                raw_args,
                args,
            )
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
            return f"{vector_info['constructor']}({', '.join(args)})"

        half_constructor = self.cuda_half_constructor_expression(
            constructor_type, raw_args, args
        )
        if half_constructor is not None:
            return half_constructor

        scalar_alias_constructor = self.cuda_scalar_alias_constructor_expression(
            constructor_type, args
        )
        if scalar_alias_constructor is not None:
            return scalar_alias_constructor

        mapped_type = self.convert_crossgl_type_to_cuda(constructor_type)
        return f"{mapped_type}({', '.join(args)})"

    def cuda_half_constructor_expression(self, constructor_type, raw_args, args):
        """Lower CrossGL FP16 constructors to CUDA's documented half intrinsics."""
        if constructor_type in {"f16", "half", "float16"}:
            if not args:
                return "half{}"
            return f"__float2half({args[0]})"

        if constructor_type not in {"vec2<f16>", "half2", "f16vec2"}:
            return None

        if not args:
            return "__float2half2_rn(0.0f)"

        if len(args) == 1:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type in {"vec2<f16>", "half2", "f16vec2"}:
                return args[0]
            return f"__float2half2_rn({args[0]})"

        return f"__floats2half2_rn({args[0]}, {args[1]})"

    def convert_crossgl_type_to_cuda(self, crossgl_type):
        """Convert CrossGL types to CUDA equivalents"""
        crossgl_type = self.type_name_string(crossgl_type)
        if crossgl_type is None:
            return "void"
        crossgl_type = generic_narrow_vector_spelling(crossgl_type) or crossgl_type

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
            "string": "const char*",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "int64": "long long",
            "uint64": "unsigned long long",
            # The other sized and minimum-precision scalar spellings of the IR (hub_types): exact widths, except
            # that a minimum-precision type may be computed at full precision (HLSL "Minimum precision").
            "int8": "char",
            "uint8": "unsigned char",
            "uchar": "unsigned char",
            "int16": "short",
            "uint16": "unsigned short",
            "int32": "int",
            "uint32": "unsigned int",
            "ulong": "unsigned long long",
            "float16_t": "half",
            "min16float": "float",
            "min16int": "int",
            "min16uint": "unsigned int",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "int": "int",
            "float": "float",
            "float16": "half",
            "half": "half",
            "double": "double",
            # Vector types (with generics)
            "vec2<f16>": "half2",
            "vec2<i8>": "char2",
            "vec3<i8>": "char3",
            "vec4<i8>": "char4",
            "vec2<u8>": "uchar2",
            "vec3<u8>": "uchar3",
            "vec4<u8>": "uchar4",
            "vec2<i16>": "short2",
            "vec3<i16>": "short3",
            "vec4<i16>": "short4",
            "vec2<u16>": "ushort2",
            "vec3<u16>": "ushort3",
            "vec4<u16>": "ushort4",
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
            "f16vec2": "half2",
            "f16vec3": "cgl_half3",
            "f16vec4": "cgl_half4",
            "float16vec3": "cgl_half3",
            "float16vec4": "cgl_half4",
            "half3": "cgl_half3",
            "half4": "cgl_half4",
            "half2": "half2",
            "vec3<f16>": "cgl_half3",
            "vec3<float16>": "cgl_half3",
            "vec3<half>": "cgl_half3",
            "vec4<f16>": "cgl_half4",
            "vec4<float16>": "cgl_half4",
            "vec4<half>": "cgl_half4",
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
            "isampler1D": "cudaTextureObject_t",
            "isampler1DArray": "cudaTextureObject_t",
            "isampler2D": "cudaTextureObject_t",
            "isampler3D": "cudaTextureObject_t",
            "isamplerCube": "cudaTextureObject_t",
            "isampler2DArray": "cudaTextureObject_t",
            "isamplerCubeArray": "cudaTextureObject_t",
            "isampler2DMS": "cudaTextureObject_t",
            "isampler2DMSArray": "cudaTextureObject_t",
            "usampler1D": "cudaTextureObject_t",
            "usampler1DArray": "cudaTextureObject_t",
            "usampler2D": "cudaTextureObject_t",
            "usampler3D": "cudaTextureObject_t",
            "usamplerCube": "cudaTextureObject_t",
            "usampler2DArray": "cudaTextureObject_t",
            "usamplerCubeArray": "cudaTextureObject_t",
            "usampler2DMS": "cudaTextureObject_t",
            "usampler2DMSArray": "cudaTextureObject_t",
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

        # Every spelling the two alias tables produce, with its i/u image variants, is a type_mapping key.
        sampled_resource_type = self.canonical_sampled_resource_type(crossgl_type)
        if sampled_resource_type:
            return type_mapping[sampled_resource_type]

        storage_resource_type = self.canonical_storage_resource_type(crossgl_type)
        if storage_resource_type:
            return type_mapping[storage_resource_type]

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

        wide_vector_type = wide_integer_vector_type(crossgl_type)
        if wide_vector_type is not None:
            return wide_vector_type

        if crossgl_type in type_mapping:
            return self.cuda_mapped_type_result(type_mapping[crossgl_type])
        # A program type or a CUDA type is written as it is; a built-in type of another language is recorded and
        # refused by render_checked_program if the output still names it.
        return self.cuda_mapped_type_result(
            written_type_name(
                crossgl_type,
                CUDA_TYPE_NAMES,
                self.program_type_names,
                self.written_foreign_type_names,
            )
        )

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
        if is_comma_expression(node):
            return self.expression_result_type(comma_result_operand(node))
        if isinstance(node, (IdentifierNode, VariableNode)):
            name = getattr(node, "name", None)
            if not self.cuda_identifier_shadows_compute_builtin(name):
                builtin_type = self.cuda_compute_builtin_type(name)
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
            if operation in {"CandidateTriangleRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, MemberAccessNode):
            object_node = getattr(
                node,
                "object_expr",
                getattr(node, "object", None),
            )
            object_name = getattr(object_node, "name", None)
            if not self.cuda_identifier_shadows_compute_builtin(object_name):
                builtin_member_type = self.cuda_compute_builtin_member_type(
                    object_name,
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
            numeric_result_type = numeric_trait_method_result_type(self, node)
            if numeric_result_type is not None:
                return numeric_result_type
            raw_args = getattr(node, "arguments", getattr(node, "args", [])) or []
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                raw_args,
            )
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            if self.is_user_defined_function(func_name):
                return self.function_return_types.get(func_name)
            if func_name in CUDA_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    getattr(node, "arguments", getattr(node, "args", [])) or [],
                )
            if func_name == "ReportHit":
                return "bool"
            scalar_boolean_type = self.scalar_boolean_builtin_result_type(
                func_name, raw_args
            )
            if scalar_boolean_type is not None:
                return scalar_boolean_type
            if func_name in self.CUDA_SHADER_CLOCK_RESULT_TYPES:
                return self.CUDA_SHADER_CLOCK_RESULT_TYPES[func_name]
            if (
                func_name in {"inverseSqrt", "rsqrt"}
                and raw_args
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0])
            reflection_result_type = self.reflection_result_type(func_name, raw_args)
            if reflection_result_type is not None:
                return reflection_result_type
            bitcast_result_type = self.cuda_bitcast_result_type(func_name, raw_args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.cuda_integer_bit_result_type(
                func_name,
                raw_args,
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if func_name in {"fma", "mad"}:
                cached_type = getattr(node, "expression_type", None) or getattr(
                    node, "vtype", None
                )
                if cached_type is not None:
                    return cached_type
                return self.fused_multiply_add_result_type(raw_args)
            if func_name == "step" and len(raw_args) == 2:
                return self.step_result_type(raw_args)
            if func_name == "lerp" and raw_args:
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        return super().expression_result_type(node)

    def resource_call_result_type(self, func_name, raw_args):
        """Infer CUDA-specific resource diagnostic result types."""
        if func_name in self.cuda_sampled_vector_result_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if resource_type is not None:
                if self.is_shadow_resource_type(resource_type):
                    return "float"
                if self.is_sampled_texture_family_type(resource_type):
                    return self.sampled_texture_value_type(resource_type)

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
            return self.sampled_texture_value_type(resource_type)

        if func_name in self.cuda_image_atomic_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            return self.cuda_image_atomic_result_type(resource_type)

        return None

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def resource_query_metadata_unavailable_call(
        self, func_name, resource_type, fallback
    ):
        resource_type = resource_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} metadata unavailable on {resource_type}",
            fallback,
        )

    def vector_type_info(self, type_name):
        half_vector_info = self.cuda_half_vector_type_info(type_name)
        if half_vector_info is not None:
            return half_vector_info
        narrow_info = self.cuda_narrow_integer_vector_type_info(type_name)
        if narrow_info is not None:
            return narrow_info
        return super().vector_type_info(type_name)

    def vector_type_for_components(self, component_type, component_count):
        # half has no vecN spelling: f16vec3 and f16vec4 lower to the
        # cgl_half helpers and f16vec2 to the native half2
        if component_type == "half" and 2 <= component_count <= 4:
            return f"f16vec{component_count}"
        return super().vector_type_for_components(component_type, component_count)

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
            func_name in {
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
            func_name in {
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
            func_name in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
            }
            and raw_args
        ):
            return self.generate_texture_offset_call(func_name, raw_args, args)

        if (
            func_name in {
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
            func_name in {
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
            texture_shape = self.sampled_texture_shape_type(texture_type)
            value_type = self.sampled_texture_value_type(texture_type)
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
            if texture_shape == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<{value_type}>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"tex1DLod<{value_type}>"
                        f"({texture_name}, {coord}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DGrad<{value_type}>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex1DLayeredLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex1DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex2DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex2DLayeredLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"tex2DLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_shape == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return f"tex3DLod<{value_type}>({coord_args}, {args[lod_index]})"
                if func_name == "textureGrad" and grad_x is not None:
                    return f"tex3DGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"texCubemapLod<{value_type}>({coord_args}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return f"texCubemapGrad<{value_type}>({coord_args}, {grad_x}, {grad_y})"

            if texture_shape == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<{value_type}>({coord_args})"
                if func_name == "textureLod" and len(args) > lod_index:
                    return (
                        f"texCubemapLayeredLod<{value_type}>"
                        f"({coord_args}, {args[lod_index]})"
                    )
                if func_name == "textureGrad" and grad_x is not None:
                    return (
                        f"texCubemapLayeredGrad<{value_type}>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_texture_family_type(
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
            texture_shape = self.sampled_texture_shape_type(texture_type)
            if texture_shape in {"samplerCube", "samplerCubeArray"}:
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
                texture_shape
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
            components = self.cuda_texture_coord_components(texture_shape, coord)
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
