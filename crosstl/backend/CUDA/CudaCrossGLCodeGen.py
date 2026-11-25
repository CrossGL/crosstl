"""CUDA to CrossGL Code Generator"""

from ...translator.semantics.warp import (
    ORDERED_WORKGROUP_BARRIER,
    OTHER_WARP_INTRINSICS,
    SHUFFLES,
)
from ..import_refusals import CudaImportRefused, ImportRefusalMixin
from ..no_contraction import (
    PRECISE_FUNCTION_ATTRIBUTE,
    function_requires_no_contraction,
)
from ..thread_hierarchy import (
    kernel_workgroup_size_attribute,
    normalize_threadgroup_size_option,
)
from .array_extents import IntegerConstants
from .block_votes import BlockVoteLibrary, insert_block_vote_functions
from .buffer_layout_import import EmittedBufferLayouts
from .buffer_memory_qualifiers import buffer_memory_attributes
from .crossgl_codegen.cooperative_groups import CudaImportCooperativeGroupMixin
from .crossgl_codegen.declarations import CudaImportDeclarationMixin
from .crossgl_codegen.intrinsics import CudaImportIntrinsicMixin
from .crossgl_codegen.names import CudaImportNameMixin
from .crossgl_codegen.resource_hints import CudaImportResourceHintMixin
from .crossgl_codegen.resources import CudaImportResourceMixin
from .crossgl_codegen.runtime_calls import CudaImportRuntimeCallMixin
from .crossgl_codegen.runtime_execution_calls import CudaImportRuntimeExecutionCallMixin
from .crossgl_codegen.runtime_graph_calls import CudaImportRuntimeGraphCallMixin
from .crossgl_codegen.runtime_memory_calls import CudaImportRuntimeMemoryCallMixin
from .crossgl_codegen.statements import CudaImportStatementMixin
from .crossgl_codegen.types import CudaImportTypeMixin
from .CudaAst import (
    ArrayAccessNode,
    CastNode,
    FunctionCallNode,
    KernelLaunchNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .directed_rounding import (
    DirectedRoundingLibrary,
    insert_directed_rounding_functions,
)
from .function_templates import (
    collect_function_templates,
    generic_prefix,
    template_parameters,
)
from .keyword_renames import source_identifier_names
from .parameter_defaults import parameter_default_suffix
from .struct_methods import lower_struct_methods
from .surface_formats import (
    SurfaceAccessMixin,
    SurfaceElementKey,
    declared_variable_types,
    is_address_of,
    surface_value_type,
)
from .union_layout_import import UnionLayoutContracts

# Warp intrinsics CrossGL keeps as CUDA calls; __syncwarp is a statement (visit_SyncNode).
CUDA_WARP_INTRINSIC_CALLS = (frozenset(SHUFFLES) | OTHER_WARP_INTRINSICS) - {
    "__syncwarp"
}


class CudaToCrossGLConverter(
    CudaImportNameMixin,
    CudaImportResourceHintMixin,
    CudaImportRuntimeCallMixin,
    CudaImportRuntimeMemoryCallMixin,
    CudaImportRuntimeExecutionCallMixin,
    CudaImportRuntimeGraphCallMixin,
    CudaImportDeclarationMixin,
    CudaImportIntrinsicMixin,
    CudaImportCooperativeGroupMixin,
    CudaImportResourceMixin,
    CudaImportStatementMixin,
    CudaImportTypeMixin,
    SurfaceAccessMixin,
    ImportRefusalMixin,
):
    """Serialize CUDA backend AST nodes back into CrossGL source.

    A CUDA construct with no faithful CrossGL form is refused with :class:`CudaImportRefused`; no CrossGL program is
    returned while a refusal is pending.
    """

    IMPORT_REFUSAL_TYPE = CudaImportRefused

    CUDA_RUNTIME_ERROR_WRAPPER_NAMES = {
        "CHECK_CUDA",
        "CUDA_CHECK",
        "checkCuda",
        "checkCudaErrors",
        "gpuErrchk",
    }

    VECTOR_TYPE_MAPPING = {
        "half2": "vec2<f16>",
        "float2": "vec2<f32>",
        "float3": "vec3<f32>",
        "float4": "vec4<f32>",
        "double2": "vec2<f64>",
        "double3": "vec3<f64>",
        "double4": "vec4<f64>",
        "int2": "vec2<i32>",
        "int3": "vec3<i32>",
        "int4": "vec4<i32>",
        "uint2": "vec2<u32>",
        "uint3": "vec3<u32>",
        "uint4": "vec4<u32>",
        "char2": "vec2<i8>",
        "char3": "vec3<i8>",
        "char4": "vec4<i8>",
        "uchar2": "vec2<u8>",
        "uchar3": "vec3<u8>",
        "uchar4": "vec4<u8>",
        "short2": "vec2<i16>",
        "short3": "vec3<i16>",
        "short4": "vec4<i16>",
        "ushort2": "vec2<u16>",
        "ushort3": "vec3<u16>",
        "ushort4": "vec4<u16>",
        "long2": "vec2<i64>",
        "long3": "vec3<i64>",
        "long4": "vec4<i64>",
        "ulong2": "vec2<u64>",
        "ulong3": "vec3<u64>",
        "ulong4": "vec4<u64>",
        "longlong2": "vec2<i64>",
        "longlong3": "vec3<i64>",
        "longlong4": "vec4<i64>",
        "ulonglong2": "vec2<u64>",
        "ulonglong3": "vec3<u64>",
        "ulonglong4": "vec4<u64>",
    }
    VECTOR1_TYPE_MAPPING = {
        "char1": "i8",
        "uchar1": "u8",
        "short1": "i16",
        "ushort1": "u16",
        "int1": "i32",
        "uint1": "u32",
        "long1": "i64",
        "ulong1": "u64",
        "longlong1": "i64",
        "ulonglong1": "u64",
        "float1": "f32",
        "double1": "f64",
    }
    VECTOR_CONSTRUCTOR_MAPPING = {
        **VECTOR1_TYPE_MAPPING,
        **VECTOR_TYPE_MAPPING,
        **{
            f"make_{name}": mapped
            for name, mapped in {
                **VECTOR1_TYPE_MAPPING,
                **VECTOR_TYPE_MAPPING,
            }.items()
        },
    }
    CUDA_TEXTURE_TYPE_MAPPING = {
        "1": "sampler1D",
        "2": "sampler2D",
        "3": "sampler3D",
        "cudaTextureType1D": "sampler1D",
        "cudaTextureType1DLayered": "sampler1DArray",
        "cudaTextureType2D": "sampler2D",
        "cudaTextureType2DLayered": "sampler2DArray",
        "cudaTextureType3D": "sampler3D",
        "cudaTextureTypeCubemap": "samplerCube",
        "cudaTextureTypeCubemapLayered": "samplerCubeArray",
    }
    CUDA_SURFACE_TYPE_MAPPING = {
        "1": "image1D",
        "2": "image2D",
        "3": "image3D",
        "cudaSurfaceType1D": "image1D",
        "cudaSurfaceType1DLayered": "image1DArray",
        "cudaSurfaceType2D": "image2D",
        "cudaSurfaceType2DLayered": "image2DArray",
        "cudaSurfaceType3D": "image3D",
        "cudaSurfaceTypeCubemap": "imageCube",
        "cudaSurfaceTypeCubemapLayered": "imageCubeArray",
    }
    CUDA_TEXTURE_CALL_TYPE_HINTS = {
        "tex1D": "sampler1D",
        "tex1Dfetch": "sampler1D",
        "tex1DLod": "sampler1D",
        "tex1DGrad": "sampler1D",
        "tex2D": "sampler2D",
        "tex2DLod": "sampler2D",
        "tex2DGrad": "sampler2D",
        "tex2Dgather": "sampler2D",
        "tex3D": "sampler3D",
        "tex3DLod": "sampler3D",
        "tex3DGrad": "sampler3D",
        "texCubemap": "samplerCube",
        "texCubemapLod": "samplerCube",
        "texCubemapGrad": "samplerCube",
        "tex1DLayered": "sampler1DArray",
        "tex1DLayeredLod": "sampler1DArray",
        "tex1DLayeredGrad": "sampler1DArray",
        "tex2DLayered": "sampler2DArray",
        "tex2DLayeredLod": "sampler2DArray",
        "tex2DLayeredGrad": "sampler2DArray",
        "texCubemapLayered": "samplerCubeArray",
        "texCubemapLayeredLod": "samplerCubeArray",
        "texCubemapLayeredGrad": "samplerCubeArray",
    }
    CUDA_SURFACE_CALL_TYPE_HINTS = {
        "surf1Dread": "image1D",
        "surf1Dwrite": "image1D",
        "surf1DLayeredread": "image1DArray",
        "surf1DLayeredwrite": "image1DArray",
        "surf2Dread": "image2D",
        "surf2Dwrite": "image2D",
        "surf3Dread": "image3D",
        "surf3Dwrite": "image3D",
        "surf2DLayeredread": "image2DArray",
        "surf2DLayeredwrite": "image2DArray",
        "surfCubemapread": "imageCube",
        "surfCubemapwrite": "imageCube",
        "surfCubemapLayeredread": "imageCubeArray",
        "surfCubemapLayeredwrite": "imageCubeArray",
    }
    CUDA_SURFACE_BOUNDARY_MODES = {
        "cudaBoundaryModeTrap": "trap",
        "cudaBoundaryModeClamp": "clamp",
        "cudaBoundaryModeZero": "zero",
    }
    # The coordinates of each surface shape that are texel coordinates, before a layer index or cubemap face.
    CUDA_SURFACE_TEXEL_DIMENSIONS = {
        "1D": 1,
        "1DLayered": 1,
        "2D": 2,
        "2DLayered": 2,
        "3D": 3,
        "Cubemap": 2,
        "CubemapLayered": 2,
    }
    CUDA_DEVICE_GRAPH_LAUNCH_MODES = {
        "cudaStreamGraphFireAndForget": "fire-and-forget",
        "cudaStreamGraphTailLaunch": "tail",
        "cudaStreamGraphFireAndForgetAsSibling": "fire-and-forget sibling",
    }
    CUDA_INTRINSIC_OVERRIDE_NAMES = {"__usad4"}
    # ``let`` and ``match`` are CrossGL lexer keywords like ``kernel`` and ``var``.
    CROSSGL_RESERVED_IDENTIFIERS = {
        "buffer",
        "in",
        "kernel",
        "let",
        "match",
        "precision",
        "var",
    }

    def __init__(self, threadgroup_size=None):
        # The host picks a CUDA kernel's block size at launch time; this option names it.
        self.threadgroup_size = normalize_threadgroup_size_option(threadgroup_size)
        self.indent_level = 0
        self.output = []
        self.block_votes = BlockVoteLibrary(())
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.user_function_names = set()
        self.global_resource_object_type_hints = {}
        self.global_surface_variable_types = {}
        self.surface_struct_members = {}
        self.struct_resource_member_hints = {}
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.cuda_async_sync_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.integer_constants = IntegerConstants()
        self.warp_mask_value_scopes = [{}]
        self.namespace_aliases = {}
        self.cuda_record_names = set()
        self.source_identifier_names = frozenset()
        self.generated_matrix_helper_types = set()
        self.global_resource_binding_count = 0
        self.function_body_depth = 0
        self.anonymous_enum_count = 0
        self.namespace_scope_variable = False
        # Inside a __global__ kernel or a __device__ function, where host runtime calls are refused.
        self.in_device_code = False
        self.namespace_constant_names = set()
        self.function_templates = {}
        self.directed_rounding = None

    def generate(self, ast_node):
        """Render ``ast_node`` as CrossGL, or raise :class:`CudaImportRefused` listing every construct it cannot carry."""
        return self.refusals.run(self.render_program, ast_node)

    def render_program(self, ast_node):
        self.output = []
        self.indent_level = 0
        lower_struct_methods(ast_node, self.refuse_import)
        self.source_identifier_names = source_identifier_names(ast_node)
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.generated_matrix_helper_types = self.collect_generated_matrix_helper_types(
            ast_node
        )
        self.user_function_names = self.collect_user_function_names(ast_node)
        self.global_surface_variable_types = (
            self.collect_global_declared_variable_types(ast_node)
        )
        self.surface_struct_members = self.collect_struct_member_types(ast_node)
        self.global_resource_object_type_hints = (
            self.collect_global_resource_object_type_hints(ast_node)
        )
        self.struct_resource_member_hints = self.collect_struct_resource_member_hints(
            ast_node
        )
        self.cuda_record_names = self.collect_cuda_record_names(ast_node)
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.cuda_async_sync_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.integer_constants = IntegerConstants()
        self.warp_mask_value_scopes = [{}]
        self.namespace_aliases = getattr(ast_node, "namespace_aliases", {}) or {}
        self.global_resource_binding_count = 0
        self.function_body_depth = 0
        self.anonymous_enum_count = 0
        self.namespace_constant_names = set()
        self.function_templates = collect_function_templates(ast_node)
        self.emitted_buffer_layouts = EmittedBufferLayouts("CUDA")
        self.union_layout_contracts = UnionLayoutContracts("CUDA")
        self.block_votes = BlockVoteLibrary(self.source_identifier_names)
        self.directed_rounding = DirectedRoundingLibrary(self.source_identifier_names)
        self.visit(ast_node)
        self.union_layout_contracts.apply(
            self.output,
            self.emitted_buffer_layouts.structs,
            self.emitted_buffer_layouts.aliases,
            self.refuse_import,
        )
        self.emitted_buffer_layouts.apply(self.output)
        insert_block_vote_functions(
            self.output, self.block_votes, "// CUDA to CrossGL conversion"
        )
        insert_directed_rounding_functions(
            self.output, self.directed_rounding, "// CUDA to CrossGL conversion"
        )
        return "\n".join(self.output)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if isinstance(node, str):
            if self.is_cuda_numeric_literal_text(node):
                return self.convert_cuda_numeric_literal_to_crossgl(node)
            if self.is_cuda_char_literal_text(node):
                return self.convert_cuda_char_literal_to_crossgl(node)
            if self.is_cuda_string_literal_text(node):
                return self.convert_cuda_string_literal_to_crossgl(node)
            if self.is_cuda_qualified_identifier_text(node):
                return self.sanitize_crossgl_type_identifier(node)
            return self.output_identifier_name(node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def collect_resource_object_type_hints(self, node, declared_names=None):
        hints = {}
        variable_types = declared_variable_types(node)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name, current.args)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        self.add_resource_object_type_hint(
                            hints,
                            self.get_resource_object_expression_name(
                                current.args[arg_index]
                            ),
                            resource_type,
                        )
                # A global surface's element is merged over the whole program instead.
                self.collect_surface_element_hint(
                    hints, current, variable_types, declared_names
                )

            for value in vars(current).values():
                collect(value)

        collect(node)
        for name in declared_names or []:
            hints.setdefault(name, None)
            # A local name hides the element of a global surface of the same name.
            hints.setdefault(SurfaceElementKey(name), None)
        return hints

    def collect_global_resource_object_type_hints_from_function(
        self, node, global_names, hints
    ):
        local_names = self.collect_declared_variable_names(node)
        variable_types = {
            **self.global_surface_variable_types,
            **declared_variable_types(node),
        }
        surface_names = set(global_names) - set(local_names)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name, current.args)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        name = self.get_resource_object_expression_name(
                            current.args[arg_index]
                        )
                        if name in global_names and name not in local_names:
                            self.add_resource_object_type_hint(
                                hints, name, resource_type
                            )
                self.collect_surface_element_hint(
                    hints, current, variable_types, surface_names
                )

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))

    def collect_declared_variable_names(self, node):
        names = {
            param.name
            for param in getattr(node, "params", [])
            if isinstance(param, VariableNode)
        }

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, VariableNode):
                names.add(current.name)

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))
        return names

    def walk_ast_values(self, root):
        visited = set()

        def walk(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    yield from walk(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    yield from walk(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)
            yield current

            for value in vars(current).values():
                yield from walk(value)

        yield from walk(root)

    def collect_struct_resource_member_hints(self, node):
        struct_member_types = self.collect_struct_member_types(node)
        struct_names = set(struct_member_types)
        if not struct_names:
            return {}

        hints = {}
        global_variable_types = self.collect_global_declared_variable_types(node)
        for function in [
            *getattr(node, "functions", []),
            *getattr(node, "kernels", []),
        ]:
            variable_types = dict(global_variable_types)
            variable_types.update(self.collect_declared_variable_types(function))
            for current in self.walk_ast_values(getattr(function, "body", [])):
                if not isinstance(current, FunctionCallNode):
                    continue
                call_hint = self.get_resource_object_call_hint(
                    current.name, current.args
                )
                if call_hint is None:
                    continue
                arg_index, resource_type = call_hint
                if len(current.args) <= arg_index:
                    continue

                member_access = self.resource_member_access_target(
                    current.args[arg_index]
                )
                if member_access is None:
                    continue

                struct_name = self.struct_type_for_resource_member_object(
                    member_access.object,
                    variable_types,
                    struct_names,
                    struct_member_types,
                )
                if struct_name is None:
                    continue
                self.add_struct_resource_member_hint(
                    hints, struct_name, member_access.member, resource_type
                )

        return hints

    def get_resource_object_call_hint(self, function_name, args=None):
        """``(index of the resource argument, its CrossGL type)`` of a texture or surface call, or ``None``."""
        base_name, _ = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None
        if base_name in self.CUDA_TEXTURE_CALL_TYPE_HINTS:
            return 0, self.CUDA_TEXTURE_CALL_TYPE_HINTS[base_name]
        if base_name in self.CUDA_SURFACE_CALL_TYPE_HINTS:
            # surf*write(value, surface, ...) and the pointer form surf*read(&value, surface, ...).
            if base_name.endswith("write") or (args and is_address_of(args[0])):
                return 1, self.CUDA_SURFACE_CALL_TYPE_HINTS[base_name]
            return 0, self.CUDA_SURFACE_CALL_TYPE_HINTS[base_name]
        return None

    def get_resource_object_expression_name(self, expression):
        if isinstance(expression, str):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.get_resource_object_expression_name(expression.array)
        if isinstance(expression, CastNode):
            return self.get_resource_object_expression_name(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.get_resource_object_expression_name(expression.operand)
        return None

    def emit_statement(self, stmt):
        if stmt is None:
            return
        if isinstance(stmt, list):
            for item in stmt:
                self.emit_statement(item)
            return

        if self.emit_cuda_runtime_call_statement(stmt):
            return
        if self.emit_cuda_output_parameter_call_statement(stmt):
            return
        if self.emit_cooperative_group_sync_statement(stmt):
            return

        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_cuda_runtime_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False
        if self.is_user_defined_function(stmt.name):
            return False
        if self.in_device_code:
            if self.format_cuda_runtime_expression(stmt) is not None:
                return True
            return False

        runtime_wrapper = self.format_cuda_runtime_wrapper_expression(stmt)
        if runtime_wrapper is not None:
            comments, _ = runtime_wrapper
            for comment in comments:
                self.emit(comment)
            return True

        runtime_value = self.format_cuda_runtime_value_expression(stmt)
        if runtime_value is not None:
            comments, _ = runtime_value
            for comment in comments:
                self.emit(comment)
            return True

        comments = self.format_cuda_runtime_call(stmt)
        if comments is None:
            return False

        for comment in comments:
            self.emit(comment)
        return True

    def format_cuda_runtime_expression(self, value):
        """Host runtime calls become comments plus their success value; in device code they are refused."""
        if isinstance(value, KernelLaunchNode):
            runtime = self.format_kernel_launch_comments(value), "cudaSuccess"
        else:
            runtime = self.format_cuda_runtime_value_expression(value)
            if runtime is None:
                runtime = self.format_cuda_runtime_status_expression(value)
        if runtime is not None and self.in_device_code:
            return [], self.refuse_device_runtime_call(value)
        return runtime

    def refuse_device_runtime_call(self, value):
        """Refuse a CUDA runtime or driver call, or a kernel launch, made from device code."""
        if isinstance(value, KernelLaunchNode):
            what = f"kernel launch {self.visit(value.kernel_name)}<<<...>>>"
        else:
            name = value.name if isinstance(value.name, str) else self.visit(value.name)
            what = f"runtime call {name}()"
        return self.refuse_import(
            f"CUDA {what} in device code: CrossGL kernels have no device-side runtime (allocation, memory "
            "operations, graphs or launches); only the host can perform it",
            "0",
        )

    def format_cuda_runtime_call(self, node):
        args = [self.visit(arg) for arg in node.args]
        name = node.name

        for describe in (
            self.describe_cuda_memory_allocation_call,
            self.describe_cuda_memory_sharing_call,
            self.describe_cuda_array_call,
            self.describe_cuda_texture_reference_call,
            self.describe_cuda_graphics_interop_call,
            self.describe_cuda_driver_memory_copy_call,
            self.describe_cuda_module_call,
            self.describe_cuda_device_call,
            self.describe_cuda_stream_call,
            self.describe_cuda_external_resource_call,
            self.describe_cuda_graph_lifecycle_call,
            self.describe_cuda_driver_graph_node_call,
            self.describe_cuda_driver_graph_parameter_call,
            self.describe_cuda_runtime_graph_node_call,
            self.describe_cuda_runtime_graph_parameter_call,
            self.describe_cuda_graph_update_call,
        ):
            comments = describe(node, name, args)
            if comments is not None:
                return comments
        return None

    def visit_FunctionNode(self, node):
        """Render a CUDA function node as a CrossGL function, tracking whether its body is device code."""
        previous = self.in_device_code
        self.in_device_code = previous or "__device__" in (
            getattr(node, "qualifiers", None) or []
        )
        try:
            return self.render_function_node(node)
        finally:
            self.in_device_code = previous

    def render_function_node(self, node):
        """Render a CUDA function node as a CrossGL function."""
        return_type = self.convert_cuda_type_to_crossgl(node.return_type)
        function_name = self.convert_cuda_function_name_to_crossgl(node.name)
        params = []

        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                node, self.collect_declared_variable_names(node)
            )
        )
        self.push_identifier_name_scope()
        self.push_vector1_name_scope()
        try:
            for index, param in enumerate(node.params):
                param_type = self.surface_declaration_type(
                    self.convert_cuda_variable_type_to_crossgl(param.vtype, param.name),
                    param.name,
                )
                self.register_vector1_name(param.name, param.vtype)
                param_name = self.crossgl_parameter_name(param, index)
                mode = self.function_reference_parameters().mode(node, index)
                if mode:
                    param_type = f"{mode} {param_type}"
                default = parameter_default_suffix(
                    self, node.name, param.name, param, "CUDA"
                )
                params.append(f"{param_type} {param_name}{default}")

            param_str = ", ".join(params)
            if function_requires_no_contraction(node):
                self.emit(PRECISE_FUNCTION_ATTRIBUTE)
            prefix = generic_prefix(
                self,
                node,
                "CUDA",
                self.convert_cuda_type_to_crossgl,
                f"{return_type} {param_str}",
            )
            self.emit(f"{prefix}{return_type} {function_name}({param_str}) {{")

            self.indent_level += 1
            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            self.push_cuda_async_sync_scope()
            self.push_variable_type_scope()
            self.push_warp_mask_value_scope()
            for parameter in template_parameters(node) or []:
                # A template value parameter hides a namespace constant of its name.
                self.integer_constants.declare(parameter.name)
            for param in node.params:
                self.register_variable_type(param.name, param.vtype)
                self.register_unique_ptr_name(param.name, param.vtype)
                self.register_cooperative_group_parameter(param)
                self.register_cuda_async_sync_parameter(param)
            try:
                self.function_body_depth += 1
                for stmt in node.body:
                    self.emit_statement(stmt)
            finally:
                self.function_body_depth -= 1
                self.pop_warp_mask_value_scope()
                self.pop_variable_type_scope()
                self.pop_cuda_async_sync_scope()
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_vector1_name_scope()
            self.pop_identifier_name_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Render a ``__global__`` kernel as a CrossGL compute entry point; its body is device code."""
        previous = self.in_device_code
        self.in_device_code = True
        try:
            return self.render_kernel_as_compute_shader(kernel)
        finally:
            self.in_device_code = previous

    def render_kernel_as_compute_shader(self, kernel):
        for attribute in getattr(kernel, "attributes", []) or []:
            attribute_text = str(attribute)
            if attribute_text.startswith("__launch_bounds__"):
                bounds = attribute_text[len("__launch_bounds__") :]
                self.emit(f"// CUDA launch bounds: {bounds}")
            elif attribute_text.startswith("__cluster_dims__"):
                dims = attribute_text[len("__cluster_dims__") :]
                self.emit(f"// CUDA cluster dims: {dims}")
            elif attribute_text.startswith("__block_size__"):
                size = attribute_text[len("__block_size__") :]
                self.emit(f"// CUDA block size: {size}")
        for param in kernel.params:
            if "__grid_constant__" in str(param.vtype).split():
                self.emit(f"// CUDA grid constant parameter: {param.name}")

        self.emit("@compute")
        if function_requires_no_contraction(kernel):
            self.emit(PRECISE_FUNCTION_ATTRIBUTE)
        self.emit(
            kernel_workgroup_size_attribute(self.threadgroup_size, whole_groups=True)
        )

        params = []
        buffer_params = (
            {}
        )  # index in params -> (context, element type) of a buffer parameter
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_identifier_name_scope()
        self.push_vector1_name_scope()
        try:
            for index, param in enumerate(kernel.params):
                self.register_vector1_name(param.name, param.vtype)
                param_name = self.crossgl_parameter_name(param, index)
                if "*" in param.vtype:
                    element_type = self.cuda_kernel_buffer_element_type(param.vtype)
                    storage_type = self.format_crossgl_type_syntax(
                        f"array<{element_type}>"
                    )
                    binding = self.global_resource_binding_count + len(params)
                    buffer_params[len(params)] = (
                        f"CUDA kernel {kernel.name} parameter {param.name}",
                        element_type,
                    )
                    params.append(
                        f"{buffer_memory_attributes(param.vtype)}@group(0) @binding({binding}) "
                        f"var<storage, read_write> {param_name}: {storage_type}"
                    )
                else:
                    param_type = self.convert_cuda_variable_type_to_crossgl(
                        param.vtype, param.name
                    )
                    param_type = self.surface_declaration_type(
                        self.format_crossgl_type_syntax(param_type), param.name
                    )
                    params.append(f"{param_type} {param_name}")

            self.emit(f"fn {kernel.name}(")
            self.indent_level += 1
            layouts = getattr(self, "emitted_buffer_layouts", None)
            for i, param in enumerate(params):
                if layouts is not None and i in buffer_params:
                    layouts.buffer(*buffer_params[i], len(self.output))
                if i == len(params) - 1:
                    self.emit(f"{param}")
                else:
                    self.emit(f"{param},")
            self.indent_level -= 1
            self.emit(") {")

            self.indent_level += 1

            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            self.push_cuda_async_sync_scope()
            self.push_variable_type_scope()
            self.push_warp_mask_value_scope()
            for param in kernel.params:
                self.register_variable_type(param.name, param.vtype)
                self.register_unique_ptr_name(param.name, param.vtype)
                self.register_cooperative_group_parameter(param)
                self.register_cuda_async_sync_parameter(param)
            try:
                self.function_body_depth += 1
                for stmt in kernel.body:
                    self.emit_statement(stmt)
            finally:
                self.function_body_depth -= 1
                self.pop_warp_mask_value_scope()
                self.pop_variable_type_scope()
                self.pop_cuda_async_sync_scope()
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_vector1_name_scope()
            self.pop_identifier_name_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_KernelLaunchNode(self, node):
        if self.in_device_code:
            self.refuse_device_runtime_call(node)
            return
        for comment in self.format_kernel_launch_comments(node):
            self.emit(comment)

    def format_cuda_warp_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        # Shuffles, votes, reductions and matches act on a warp of 32 consecutive threads. Wave operations act on a
        # subgroup of implementation-defined size and layout, so they computed something else wherever a subgroup is
        # not exactly such a warp (lavapipe's has 4 or 8 lanes, WARP's 4). Every warp intrinsic stays a CUDA call:
        # crosstl.translator.semantics.warp emulates it exactly or refuses it for every target without CUDA warps.
        if function_name in CUDA_WARP_INTRINSIC_CALLS:
            if args and function_name.endswith("_sync"):
                args = [self.literal_full_warp_mask(args[0]), *args[1:]]
            return self.format_unsupported_cuda_warp_intrinsic(function_name, args)

        return None

    def literal_full_warp_mask(self, mask):
        """``0xFFFFFFFFu`` for a mask variable that holds the full-warp literal, else the mask as written."""
        resolved = self.normalize_warp_mask_expression(
            self.resolve_warp_mask_expression(mask)
        )
        literal = resolved.rstrip("ul")
        if resolved != self.normalize_warp_mask_expression(mask) and (
            literal in {"0xffffffff", "0xffffffffffffffff", "4294967295"}
            or resolved == "uint_max"
        ):
            return "0xFFFFFFFFu"
        return mask

    def is_full_warp_shuffle_width(self, width_args):
        if not width_args:
            return True
        if len(width_args) != 1:
            return False
        return self.is_full_warp_width(width_args[0])

    def cooperative_group_sync_metadata(self, stmt):
        """The group a ``g.sync()`` or ``cg::sync(g)`` statement synchronizes, or None for any other statement."""
        if not isinstance(stmt, FunctionCallNode):
            return None
        if isinstance(stmt.name, MemberAccessNode):
            member_base_name, _ = self.parse_cpp_template(stmt.name.member)
            member_name = (
                self.cooperative_group_base_name(member_base_name) or stmt.name.member
            )
            if member_name != "sync" or stmt.args:
                return None
            return self.resolve_cooperative_group_metadata(stmt.name.object)
        if not isinstance(stmt.name, str) or len(stmt.args) != 1:
            return None
        base_call_name, _ = self.parse_cpp_template(stmt.name)
        if self.cooperative_group_base_name(base_call_name) != "sync":
            return None
        return self.resolve_cooperative_group_metadata(stmt.args[0])

    def emit_cooperative_group_sync_statement(self, stmt):
        """Emit a thread block's or a warp tile's ``sync()``; other groups' syncs are refused as values are.

        ``thread_block::sync()`` is ``__syncthreads()`` (CUDA C++ Programming Guide, "Cooperative Groups", Thread
        Block Group). A ``thread_block_tile<N>`` of a thread block with ``N`` a power of two up to 32 lies inside one
        warp, so its ``sync()`` becomes ``__syncwarp()``, which waits for the whole warp; crosstl.translator.semantics
        .warp emulates it only where every lane of the warp reaches it, and there waiting for the whole warp also
        waits for the tile.
        """
        group_metadata = self.cooperative_group_sync_metadata(stmt)
        if group_metadata is None:
            return False
        group_kind = group_metadata["kind"]
        if group_kind == "thread_block":
            self.emit_block_barrier()
            return True
        tile_size = self.literal_warp_tile_size(group_metadata)
        if (
            group_kind == "thread_block_tile"
            and group_metadata.get("parent_kind") == "thread_block"
            and tile_size is not None
        ):
            self.emit("__syncwarp();")
            return True
        if group_kind == "thread_block_tile":
            self.refuse_import(
                f"CUDA cooperative_groups thread_block_tile<{group_metadata.get('tile_size', '?')}>.sync(): "
                "CrossGL has barriers over a workgroup and over a 32-lane warp only, so a tile must be a literal "
                "power of two up to 32 partitioned from a thread block"
            )
            return True
        return False

    def literal_warp_tile_size(self, group_metadata):
        """The tile size as an int when it is a literal power of two up to 32, else None."""
        text = str(group_metadata.get("tile_size") or "").strip().rstrip("uU")
        try:
            size = int(text, 0)
        except ValueError:
            return None
        if 0 < size <= 32 and size & (size - 1) == 0:
            return size
        return None

    def format_cooperative_group_member_call(self, group_metadata, member, args):
        group_kind = group_metadata["kind"]
        member_base_name, _ = self.parse_cpp_template(member)
        member_name = self.cooperative_group_base_name(member_base_name) or member
        # A group's sync() is a statement (emit_cooperative_group_sync_statement); as a value it is refused below.
        if group_kind == "thread_block" and not args:
            if member_name == "thread_rank":
                return "gl_LocalInvocationIndex"
            if member_name in {"size", "num_threads"}:
                return self.format_thread_block_size_expression()
            if member_name == "thread_index":
                return "gl_LocalInvocationID"
            if member_name == "dim_threads":
                return "gl_WorkGroupSize"

        if group_kind == "thread_block_tile" and not args:
            tile_size = group_metadata.get("tile_size")
            if member_name in {"size", "num_threads"} and tile_size:
                return tile_size
            if (
                member_name == "thread_rank"
                and tile_size
                and group_metadata.get("parent_kind") == "thread_block"
            ):
                return f"(gl_LocalInvocationIndex % {tile_size})"

        if group_kind == "thread_block_tile" and args:
            tile_collective = self.format_thread_block_tile_collective_call(
                group_metadata, member_name, args
            )
            if tile_collective is not None:
                return tile_collective

        if member_name in {"sync", "memcpy_async", "wait", "wait_prior"}:
            return self.format_unsupported_cooperative_group_statement(
                group_kind, member_name, args
            )
        if member_name in {"thread_rank", "size", "num_threads", "block_rank"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name
            )
        if member_name in {"thread_index", "dim_threads", "dim_blocks"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, "vec3<u32>(0, 0, 0)"
            )
        if member_name == "map_shared_rank" and args:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, self.visit(args[0]), args=args
            )
        return self.format_unsupported_cooperative_group_expression(
            group_kind, member_name, args=args
        )

    def format_thread_block_tile_collective_call(
        self, group_metadata, member_name, args
    ):
        if (
            group_metadata.get("tile_size") != "32"
            or group_metadata.get("parent_kind") != "thread_block"
        ):
            return None

        # A 32-thread tile of a thread block is a warp (its threads are the warp's lanes in rank order), so each
        # collective is the full-mask warp intrinsic, which crosstl.translator.semantics.warp emulates or refuses.
        formatted_args = [self.visit(arg) for arg in args]
        full = "0xFFFFFFFFu"
        votes = {"any": "__any_sync", "all": "__all_sync", "ballot": "__ballot_sync"}
        if member_name in votes and len(formatted_args) == 1:
            return f"{votes[member_name]}({full}, {formatted_args[0]})"
        shuffles = {
            "shfl": "__shfl_sync",
            "shfl_up": "__shfl_up_sync",
            "shfl_down": "__shfl_down_sync",
            "shfl_xor": "__shfl_xor_sync",
        }
        if member_name in shuffles and len(formatted_args) == 2:
            value, operand = formatted_args
            return f"{shuffles[member_name]}({full}, {value}, {operand})"

        return None

    # cooperative_groups::reduce operators over 32-bit integers, and the warp reductions computing them.
    CUDA_TILE_REDUCTIONS = {
        "plus": "__reduce_add_sync",
        "less": "__reduce_min_sync",
        "greater": "__reduce_max_sync",
        "bit_and": "__reduce_and_sync",
        "bit_or": "__reduce_or_sync",
        "bit_xor": "__reduce_xor_sync",
    }
    CUDA_TILE_REDUCTION_TYPES = {
        "int": "int",
        "int32_t": "int",
        "signed": "int",
        "signed int": "int",
        "unsigned": "unsigned",
        "unsigned int": "unsigned",
        "uint32_t": "unsigned",
    }

    def format_cooperative_group_reduce_call(self, group_metadata, args):
        group_kind = group_metadata["kind"]
        operator, value_type = None, None
        if len(args) == 2 and isinstance(args[1], FunctionCallNode):
            operator = self.cooperative_group_reduce_operator(args[1])
            _, template = self.parse_cpp_template(args[1].name)
            if len(template) == 1:
                value_type = self.CUDA_TILE_REDUCTION_TYPES.get(
                    " ".join(str(template[0]).split())
                )
        if (
            group_kind == "thread_block_tile"
            and group_metadata.get("tile_size") == "32"
            and group_metadata.get("parent_kind") == "thread_block"
            and operator in self.CUDA_TILE_REDUCTIONS
            and value_type is not None
            and (value_type == "unsigned" or not operator.startswith("bit_"))
        ):
            # A warp of 32 consecutive threads, whatever the target's subgroup size; floating-point reductions,
            # whose association order cg::reduce leaves unspecified, are refused.
            value = self.visit(args[0])
            crossgl_type = self.convert_cuda_type_to_crossgl(value_type)
            reduction = self.CUDA_TILE_REDUCTIONS[operator]
            return f"{reduction}(0xFFFFFFFFu, {crossgl_type}({value}))"

        return self.format_unsupported_cooperative_group_expression(
            group_kind, "reduce", args=args
        )

    def format_cuda_resource_call(self, function_name, args, raw_args=None):
        base_name, template_args = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None

        value_type = template_args[0] if template_args else None
        if base_name == "tex1Dfetch":
            return self.format_cuda_texture_fetch_call(args)
        if base_name in {"tex1D", "tex1DLod", "tex1DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec1", 1)
        if base_name in {"tex2D", "tex2DLod", "tex2DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec2", 2)
        if base_name == "tex2Dgather":
            return self.format_cuda_texture_gather_call(args)
        if base_name in {"tex3D", "tex3DLod", "tex3DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {"texCubemap", "texCubemapLod", "texCubemapGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {
            "tex1DLayered",
            "tex1DLayeredLod",
            "tex1DLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec2", 2)
        if base_name in {
            "tex2DLayered",
            "tex2DLayeredLod",
            "tex2DLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {
            "texCubemapLayered",
            "texCubemapLayeredLod",
            "texCubemapLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec4", 4)

        if base_name in {
            "surf1Dread",
            "surf1DLayeredread",
            "surf2Dread",
            "surf3Dread",
            "surf2DLayeredread",
            "surfCubemapread",
            "surfCubemapLayeredread",
        }:
            dimensions = {
                "surf1Dread": 1,
                "surf1DLayeredread": 2,
                "surf2Dread": 2,
                "surf3Dread": 3,
                "surf2DLayeredread": 3,
                "surfCubemapread": 3,
                "surfCubemapLayeredread": 3,
            }[base_name]
            return self.format_cuda_surface_read(
                args, dimensions, value_type, base_name, raw_args
            )

        if base_name in {
            "surf1Dwrite",
            "surf1DLayeredwrite",
            "surf2Dwrite",
            "surf3Dwrite",
            "surf2DLayeredwrite",
            "surfCubemapwrite",
            "surfCubemapLayeredwrite",
        }:
            dimensions = {
                "surf1Dwrite": 1,
                "surf1DLayeredwrite": 2,
                "surf2Dwrite": 2,
                "surf3Dwrite": 3,
                "surf2DLayeredwrite": 3,
                "surfCubemapwrite": 3,
                "surfCubemapLayeredwrite": 3,
            }[base_name]
            return self.format_cuda_surface_write(
                args, dimensions, value_type, raw_args, base_name
            )

        return None

    def format_cuda_surface_read(
        self, args, dimensions, value_type, base_name, raw_args=None
    ):
        """``imageLoad`` of a surface read: ``surf2Dread<T>(s, x, y)`` or ``surf2Dread(&v, s, x, y)``.

        The pointer form stores the texel in its destination, so it becomes an assignment; CUDA declares it
        ``void``, so it only stands as a statement. :mod:`..surface_formats` gives the element type ``T``, the
        image's format, and the texel column ``x / sizeof(T)``.
        """
        raw_args = list(raw_args or [])
        pointer_form = bool(raw_args) and is_address_of(raw_args[0])
        start = 1 if pointer_form else 0
        if len(args) < start + dimensions + 1 or len(raw_args) < len(args):
            return None
        surface_name = args[start]
        if pointer_form and value_type is None:
            value_type = self.cuda_surface_value_type(raw_args[0].operand)
        access = self.surface_access_element(base_name, raw_args[start], value_type)
        if access is None:
            return "0"
        element, size = access
        coord_args = [
            self.cuda_surface_texel_column(base_name, raw_args[start + 1], size)
        ]
        coord_args.extend(args[start + 2 : start + dimensions + 1])
        coord = self.format_cuda_surface_coordinate(
            base_name,
            surface_name,
            coord_args,
            dimensions,
            args[start + dimensions + 1 :],
        )
        if coord is None:
            return None
        texel = f"imageLoad({surface_name}, {coord})"
        if element is not None:
            texel = element.load(texel)
        if pointer_form:
            return f"{self.visit(raw_args[0].operand)} = {texel}"
        return texel

    def format_cuda_surface_write(
        self, args, dimensions, value_type, raw_args=None, base_name=None
    ):
        """``imageStore`` of ``surf2Dwrite(v, s, x, y)``: the element of ``v`` widened to the image's texel."""
        raw_args = list(raw_args or [])
        if len(args) < dimensions + 2 or len(raw_args) < len(args):
            return None
        if value_type is None:
            value_type = self.cuda_surface_value_type(raw_args[0])
        surface_name = args[1]
        access = self.surface_access_element(base_name, raw_args[1], value_type)
        if access is None:
            return "0"
        element, size = access
        coord_args = [self.cuda_surface_texel_column(base_name, raw_args[2], size)]
        coord_args.extend(args[3 : dimensions + 2])
        coord = self.format_cuda_surface_coordinate(
            base_name, surface_name, coord_args, dimensions, args[dimensions + 2 :]
        )
        if coord is None:
            return None
        value = args[0] if element is None else element.store(args[0])
        return f"imageStore({surface_name}, {coord}, {value})"

    def cuda_surface_value_type(self, expression):
        """The CUDA type of a surface write's value or a pointer-form read's destination, or ``None``."""
        return surface_value_type(
            expression, self.lookup_variable_type, self.surface_member_type
        )

    def surface_member_type(self, object_type, member):
        """The declared type of member ``member`` of the program's structure ``object_type``, or ``None``."""
        name = self.strip_type_qualifiers(self.resolve_type_alias(str(object_type)))
        name = name.replace("&", "").strip().removeprefix("struct ").strip()
        return self.surface_struct_members.get(name, {}).get(member)

    def cuda_surface_texel_column(self, base_name, raw_offset, element_size):
        """The texel column of a surface access's x byte offset; an offset not provably a multiple is refused."""
        divided = self.divide_surface_byte_offset(raw_offset, element_size, None)
        if divided is not None:
            return divided[0]
        offset = self.visit(raw_offset) if raw_offset is not None else "?"
        return self.refuse_import(
            f"CUDA {base_name} x byte offset {offset} is not provably a multiple of the "
            f"{element_size}-byte surface element; write it as a multiple of the element "
            "size, for example x * sizeof(T)",
            "0",
        )

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            self.emit_block_barrier()
        elif node.sync_type == "__syncwarp" and self.is_full_warp_mask(node.args):
            # __syncwarp waits for the 32 lanes of the warp; a subgroup barrier waits for a subgroup of another size
            # on most targets. It stays a CUDA call for crosstl.translator.semantics.warp to emulate or refuse.
            self.emit("__syncwarp();")
        elif node.sync_type == "__syncwarp" and len(node.args) == 1:
            # A mask that may name only some lanes stays a CUDA call too: CUDA output writes it back, and
            # crosstl.translator.semantics.warp refuses it for the targets without CUDA warps.
            self.emit(f"__syncwarp({self.visit(node.args[0])});")
        elif node.sync_type == "__syncwarp":
            args = ", ".join(self.visit(arg) for arg in node.args)
            self.refuse_import(f"CUDA __syncwarp({args}) takes one mask")
        else:
            self.refuse_import(f"CUDA {node.sync_type}() has no CrossGL form")

    def emit_block_barrier(self):
        """Emit ``__syncthreads()``: an execution barrier for the block that orders global and shared memory.

        ``__syncthreads()`` makes every global and shared memory access made before it visible to every thread of
        the block (CUDA C++ Programming Guide, "Synchronization Functions"). CrossGL's workgroupBarrier orders
        workgroup memory only, so a workgroup-scope fence over all memory (groupMemoryBarrier, GLSL 4.60 section
        8.16) releases the earlier accesses before it and acquires them after it.
        """
        for function_name in ORDERED_WORKGROUP_BARRIER:
            self.emit(f"{function_name}();")

    def is_full_warp_mask(self, args):
        """Whether ``__syncwarp`` arguments name every lane: none, or the literal mask ``0xffffffff``."""
        if not args:
            return True
        if len(args) != 1:
            return False
        value = self.visit(args[0]).strip().rstrip("uUlL")
        try:
            return int(value, 0) == 0xFFFFFFFF
        except ValueError:
            return False

    def visit_CastNode(self, node):
        target_type = self.convert_cuda_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_cuda_type_to_crossgl(self, cuda_type):
        cuda_type = self.strip_dependent_template_disambiguators(cuda_type)
        cuda_type = self.strip_type_qualifiers(cuda_type)

        matrix_type = self.convert_native_matrix_helper_name_to_crossgl(cuda_type)
        if matrix_type is not None:
            return matrix_type

        elaborated_type = self.convert_cuda_elaborated_type_to_crossgl(cuda_type)
        if elaborated_type is not None:
            return elaborated_type

        parameter_pack_type = self.convert_cuda_parameter_pack_type_to_crossgl(
            cuda_type
        )
        if parameter_pack_type is not None:
            return parameter_pack_type

        type_mapping = {
            "void": "void",
            "bool": "bool",
            "char": "i8",
            "signed char": "i8",
            "unsigned char": "u8",
            "short": "i16",
            "signed short": "i16",
            "short int": "i16",
            "signed short int": "i16",
            "unsigned short": "u16",
            "unsigned short int": "u16",
            "int": "i32",
            "signed": "i32",
            "signed int": "i32",
            "unsigned": "u32",
            "uint": "u32",
            "unsigned int": "u32",
            "long": "i64",
            "signed long": "i64",
            "long int": "i64",
            "signed long int": "i64",
            "unsigned long": "u64",
            "unsigned long int": "u64",
            "long unsigned int": "u64",
            "long long": "i64",
            "signed long long": "i64",
            "long long int": "i64",
            "signed long long int": "i64",
            "unsigned long long": "u64",
            "unsigned long long int": "u64",
            "long long unsigned int": "u64",
            "int8_t": "i8",
            "uint8_t": "u8",
            "int16_t": "i16",
            "uint16_t": "u16",
            "int32_t": "i32",
            "uint32_t": "u32",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "__half": "f16",
            "__half2": "vec2<f16>",
            "__nv_bfloat16": "bfloat16",
            "__nv_bfloat162": "vec2<bfloat16>",
            "float": "f32",
            "double": "f64",
            "size_t": "u64",
            "ptrdiff_t": "i64",
            # clock_t is long on the LP64 hosts CUDA and ROCm compile device code for (C11 7.27.1).
            "clock_t": "i64",
            **self.VECTOR1_TYPE_MAPPING,
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }
        unqualified_type = self.unqualified_leading_cuda_std_type_name(cuda_type)
        if unqualified_type is not None and unqualified_type in type_mapping:
            return type_mapping[unqualified_type]

        cooperative_group_type = self.convert_cooperative_group_type(cuda_type)
        if cooperative_group_type is not None:
            return cooperative_group_type

        resource_type = self.convert_cuda_resource_type(cuda_type)
        if resource_type is not None:
            return resource_type

        unique_ptr_type = self.convert_unique_ptr_type(cuda_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        span_type = self.convert_cuda_span_type(cuda_type)
        if span_type is not None:
            return span_type

        std_array_type = self.convert_cuda_std_array_type(cuda_type)
        if std_array_type is not None:
            return std_array_type

        nested_scoped_template_type = (
            self.convert_nested_scoped_template_type_to_crossgl(cuda_type)
        )
        if nested_scoped_template_type is not None:
            return nested_scoped_template_type

        enable_if_type = self.convert_enable_if_type(cuda_type)
        if enable_if_type is not None:
            return enable_if_type

        wmma_type = self.convert_cuda_wmma_type(cuda_type)
        if wmma_type is not None:
            return wmma_type

        if self.is_decltype_type_name(cuda_type):
            return "auto"

        expression_template_type = (
            self.convert_cuda_expression_template_type_to_crossgl(cuda_type)
        )
        if expression_template_type is not None:
            return expression_template_type

        if self.has_array_suffix(cuda_type):
            return self.convert_cuda_array_type(cuda_type, type_mapping)

        if "*" in cuda_type:
            return self.convert_cuda_pointer_type(cuda_type)

        template_type = None
        if cuda_type in self.cuda_record_names:
            template_type = self.convert_unknown_cuda_template_type_to_crossgl(
                cuda_type
            )
        if template_type is not None:
            return template_type

        mapped_type = type_mapping.get(cuda_type)
        if mapped_type is not None:
            return mapped_type
        scoped_type = self.convert_cuda_scoped_type_to_crossgl(cuda_type)
        if scoped_type is not None:
            return scoped_type
        if cuda_type in self.CROSSGL_RESERVED_IDENTIFIERS:
            # A struct named after a CrossGL keyword is declared under its renamed name.
            return self.sanitize_identifier_name(cuda_type)
        return cuda_type
