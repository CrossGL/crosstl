"""OpenCL to CrossGL code generator."""

import re

from crosstl.backend.HIP.HipAst import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    DesignatedInitializerNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    KernelNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from ..assignment_operands import grouped_operand
from ..thread_hierarchy import kernel_workgroup_size_attribute
from .conversions import format_conversion, is_conversion_name
from .OpenCLAst import OpenCLBlockLiteralNode, OpenCLMacroBlockNode, OpenCLProgramNode

_DYNAMIC_LOCAL_MEMORY_FALLBACK_SIZE = 1024
_OPENCL_WORK_ITEM_QUERIES = frozenset(
    {
        "get_enqueued_local_size",
        "get_global_id",
        "get_global_linear_id",
        "get_global_offset",
        "get_global_size",
        "get_group_id",
        "get_local_id",
        "get_local_linear_id",
        "get_local_size",
        "get_num_groups",
        "get_work_dim",
    }
)


class OpenCLRecordSemanticError(ValueError):
    """Raised when used OpenCL record semantics cannot survive CrossGL lowering."""

    project_diagnostic_code = "project.translate.opencl-record-semantics-unsupported"
    missing_capabilities = ("opencl.record-lifecycle-layout-lowering",)

    def __init__(self, feature, record_name, context):
        self.feature = feature
        self.record_name = record_name
        self.context = context
        super().__init__(
            f"OpenCL {feature} is unsupported for record '{record_name}' in {context}"
        )


class OpenCLAtomicOrderUnsupportedError(TranslationDiagnostic):
    """An OpenCL C ``*_explicit`` atomic whose memory order or scope CrossGL atomics cannot keep."""

    code = CODE_PREFIX + "opencl-atomic-order-unsupported"


class OpenCLFenceScopeUnsupportedError(TranslationDiagnostic):
    """An OpenCL C fence or sub-group barrier whose memory scope no CrossGL fence covers."""

    code = CODE_PREFIX + "opencl-fence-scope-unsupported"


class OpenCLImageWriteUnsupportedError(TranslationDiagnostic):
    """An OpenCL C ``write_image*`` call that no CrossGL ``imageStore`` keeps."""

    code = CODE_PREFIX + "opencl-image-write-unsupported"


class OpenCLImageReadUnsupportedError(TranslationDiagnostic):
    """An OpenCL C ``read_imagef`` call or ``sampler_t`` value that CrossGL cannot carry exactly."""

    code = CODE_PREFIX + "opencl-image-read-unsupported"


class OpenCLToCrossGLConverter(HipToCrossGLConverter):
    """Serialize OpenCL backend AST nodes into CrossGL source."""

    SOURCE_LANGUAGE_NAME = "OpenCL"
    ERASE_CONCRETE_TYPE_ALIASES = False
    RECORD_SEMANTIC_ERROR = OpenCLRecordSemanticError

    CROSSGL_RESERVED_IDENTIFIERS = {
        *HipToCrossGLConverter.CROSSGL_RESERVED_IDENTIFIERS,
        "shared",
    }
    OPENCL_ADDRESS_SPACE_QUALIFIERS = {
        "__global__",
        "global",
        "__local",
        "__local__",
        "local",
        "__shared__",
        "__constant",
        "__constant__",
        "constant",
        "__private",
        "__private__",
        "private",
        "__generic",
        "__generic__",
        "generic",
        "read_only",
        "write_only",
        "read_write",
    }
    OPENCL_SCALAR_TYPE_MAPPING = {
        "signed char": "i8",
        "unsigned char": "u8",
        "uchar": "u8",
        "_Bool": "bool",
        "signed short": "i16",
        "unsigned short": "u16",
        "ushort": "u16",
        "signed int": "i32",
        "unsigned int": "u32",
        "uint": "u32",
        "signed long": "i64",
        "unsigned long": "u64",
        "ulong": "u64",
        "intptr_t": "i64",
        "uintptr_t": "u64",
        "ptrdiff_t": "i64",
        "__SIZE_TYPE__": "u64",
        "__PTRDIFF_TYPE__": "i64",
        "__INTPTR_TYPE__": "i64",
        "__UINTPTR_TYPE__": "u64",
        "dim_t": "i64",
        "cl_char": "i8",
        "cl_uchar": "u8",
        "cl_short": "i16",
        "cl_ushort": "u16",
        "cl_int": "i32",
        "cl_uint": "u32",
        "cl_long": "i64",
        "cl_ulong": "u64",
        "cl_bool": "bool",
        "cl_bitfield": "u64",
        "cl_device_type": "u64",
        "cl_platform_info": "u32",
        "cl_device_info": "u32",
        "cl_command_queue_info": "u32",
        "cl_context_info": "u32",
        "cl_mem_flags": "u64",
        "cl_mem_object_type": "u32",
        "cl_map_flags": "u64",
        "cl_event_info": "u32",
        "cl_command_type": "u32",
        "cl_profiling_info": "u32",
        "cl_sampler_info": "u32",
        "cl_channel_order": "u32",
        "cl_channel_type": "u32",
        "cl_context": "u64",
        "cl_command_queue": "u64",
        "cl_mem": "u64",
        "cl_program": "u64",
        "cl_kernel": "u64",
        "cl_event": "u64",
        "cl_sampler": "sampler",
        "cl_device_id": "u64",
        "cl_platform_id": "u64",
        "atomic_int": "i32",
        "atomic_uint": "u32",
        "atomic_long": "i64",
        "atomic_ulong": "u64",
        "atomic_float": "f32",
        "atomic_double": "f64",
        "atomic_intptr_t": "i64",
        "atomic_uintptr_t": "u64",
        "cl_float2": "vec2<f32>",
        "cl_float4": "vec4<f32>",
        "cl_double2": "vec2<f64>",
        "cl_double4": "vec4<f64>",
        "real": "f32",
        "real_arg": "f32",
        "sampler_t": "sampler",
        "event_t": "u64",
        "clk_event_t": "u64",
        "image1d_t": "image1D",
        "image1d_array_t": "image1DArray",
        "image1d_buffer_t": "imageBuffer",
        "image2d_t": "image2D",
        "image2d_array_t": "image2DArray",
        "image2d_depth_t": "image2D",
        "image2d_array_depth_t": "image2DArray",
        "image3d_t": "image3D",
    }
    OPENCL_VECTOR_TYPE_MAPPING = {
        **HipToCrossGLConverter.VECTOR_TYPE_MAPPING,
        **HipToCrossGLConverter.VECTOR1_TYPE_MAPPING,
    }
    for _scalar, _mapped in {
        "char": "i8",
        "uchar": "u8",
        "short": "i16",
        "ushort": "u16",
        "int": "i32",
        "uint": "u32",
        "long": "i64",
        "ulong": "u64",
        "float": "f32",
        "double": "f64",
        "half": "f16",
    }.items():
        for _width in (2, 3, 4):
            OPENCL_VECTOR_TYPE_MAPPING[f"{_scalar}{_width}"] = f"vec{_width}<{_mapped}>"
        for _width in (8, 16):
            OPENCL_VECTOR_TYPE_MAPPING[f"{_scalar}{_width}"] = (
                f"array<{_mapped}, {_width}>"
            )

    OPENCL_BUILTIN_ID_MAP = {
        "get_global_id": "gl_GlobalInvocationID",
        "get_local_id": "gl_LocalInvocationID",
        "get_group_id": "gl_WorkGroupID",
        "get_local_size": "gl_WorkGroupSize",
        "get_num_groups": "gl_NumWorkGroups",
    }
    # OpenCL 2.0 defines get_local_linear_id() as
    # (lid.z * lsize.y + lid.y) * lsize.x + lid.x, which is the local
    # invocation index every compute target already exposes.
    OPENCL_LINEAR_ID_MAP = {
        "get_local_linear_id": "gl_LocalInvocationIndex",
    }
    # OpenCL fence flags name the address spaces a fence orders; each maps to
    # the CrossGL fence over the same memory. More than one flag, or a flag
    # that is not a literal, needs the fence over all memory.
    OPENCL_MEM_FENCE_FLAG_BARRIERS = {
        "CLK_LOCAL_MEM_FENCE": "memoryBarrierShared",
        "CLK_GLOBAL_MEM_FENCE": "memoryBarrierBuffer",
        "CLK_IMAGE_MEM_FENCE": "memoryBarrierImage",
    }
    OPENCL_MEM_FENCE_FUNCTIONS = frozenset(
        {"mem_fence", "read_mem_fence", "write_mem_fence", "atomic_work_item_fence"}
    )
    # The CrossGL fences order memory at device scope at most, the scopes in
    # OPENCL_ATOMIC_DEVICE_SCOPES, and subgroupBarrier() orders it for the
    # sub-group only.
    OPENCL_SUB_GROUP_FENCE_SCOPES = frozenset(
        {"memory_scope_work_item", "memory_scope_sub_group"}
    )
    OPENCL_ATOMIC_MAP = {
        "atomic_add": "atomicAdd",
        "atomic_fetch_add": "atomicAdd",
        "atomic_fetch_add_explicit": "atomicAdd",
        "atomic_sub": "atomicAdd",
        "atomic_fetch_sub": "atomicAdd",
        "atomic_fetch_sub_explicit": "atomicAdd",
        "atomic_xchg": "atomicExchange",
        "atomic_exchange": "atomicExchange",
        "atomic_exchange_explicit": "atomicExchange",
        "atomic_cmpxchg": "atomicCompareExchange",
        "atomic_compare_exchange_strong": "atomicCompareExchange",
        "atomic_compare_exchange_weak": "atomicCompareExchange",
        "atomic_compare_exchange_strong_explicit": "atomicCompareExchange",
        "atomic_compare_exchange_weak_explicit": "atomicCompareExchange",
        "atomic_min": "atomicMin",
        "atomic_fetch_min": "atomicMin",
        "atomic_fetch_min_explicit": "atomicMin",
        "atomic_max": "atomicMax",
        "atomic_fetch_max": "atomicMax",
        "atomic_fetch_max_explicit": "atomicMax",
        "atomic_and": "atomicAnd",
        "atomic_fetch_and": "atomicAnd",
        "atomic_fetch_and_explicit": "atomicAnd",
        "atomic_or": "atomicOr",
        "atomic_fetch_or": "atomicOr",
        "atomic_fetch_or_explicit": "atomicOr",
        "atomic_xor": "atomicXor",
        "atomic_fetch_xor": "atomicXor",
        "atomic_fetch_xor_explicit": "atomicXor",
        "atomic_inc": "atomicAdd",
        "atomic_dec": "atomicAdd",
    }
    OPENCL_SIZEOF_SCALAR_SIZES = {
        "bool": 1,
        "char": 1,
        "signed char": 1,
        "unsigned char": 1,
        "uchar": 1,
        "short": 2,
        "signed short": 2,
        "unsigned short": 2,
        "ushort": 2,
        "half": 2,
        "int": 4,
        "signed int": 4,
        "unsigned int": 4,
        "uint": 4,
        "float": 4,
        "size_t": 8,
        "ptrdiff_t": 8,
        "long": 8,
        "signed long": 8,
        "unsigned long": 8,
        "ulong": 8,
        "double": 8,
        "intptr_t": 8,
        "uintptr_t": 8,
        "cl_char": 1,
        "cl_uchar": 1,
        "cl_short": 2,
        "cl_ushort": 2,
        "cl_int": 4,
        "cl_uint": 4,
        "cl_float": 4,
        "cl_long": 8,
        "cl_ulong": 8,
        "cl_double": 8,
        "atomic_int": 4,
        "atomic_uint": 4,
        "atomic_long": 8,
        "atomic_ulong": 8,
        "atomic_float": 4,
        "atomic_double": 8,
        "atomic_intptr_t": 8,
        "atomic_uintptr_t": 8,
    }
    OPENCL_SIZEOF_POINTER_SIZE = 8
    OPENCL_HALF_FLOAT_LITERAL = re.compile(
        r"^(?P<body>"
        r"0[xX](?:"
        r"[0-9a-fA-F](?:'?[0-9a-fA-F])*"
        r"(?:\.(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)?)?"
        r"|\.(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)"
        r")"
        r"[pP][+-]?\d(?:'?\d)*"
        r"|(?:\d(?:'?\d)*\.(?:\d(?:'?\d)*)?|\.(?:\d(?:'?\d)*))(?:[eE][+-]?\d(?:'?\d)*)?"
        r"|\d(?:'?\d)*[eE][+-]?\d(?:'?\d)*)"
        r"[hH]$"
    )

    def __init__(self, normalize_target_safe_cgl=False, threadgroup_size=None):
        super().__init__(threadgroup_size=threadgroup_size)
        self.normalize_target_safe_cgl = normalize_target_safe_cgl
        self.opencl_image_access = {}
        self.opencl_sampler_states = {}

    def generate(self, ast_node):
        self.opencl_sizeof_symbols = {}
        self.opencl_current_function_name = None
        self.opencl_event_tokens = set()
        self.opencl_workgroup_pointer_helper_params = {}
        self.opencl_sdk_reduce_materialize_op = False
        self.opencl_image_access = {}
        self.opencl_sampler_states = {}
        return super().generate(ast_node)

    def generic_visit(self, node):
        if isinstance(node, str):
            literal = self.normalize_opencl_numeric_literal(node)
            if literal != node:
                return literal
        return super().generic_visit(node)

    def normalize_opencl_numeric_literal(self, value):
        half_literal = self.OPENCL_HALF_FLOAT_LITERAL.match(value)
        if half_literal:
            return half_literal.group("body").replace("'", "")
        return value

    def visit_InitializerListNode(self, node):
        depth = getattr(self, "opencl_initializer_list_depth", 0)
        self.opencl_initializer_list_depth = depth + 1
        try:
            elements = []
            for element in node.elements:
                strip_designator = (
                    depth > 0
                    or getattr(self, "opencl_strip_designated_initializers", 0) > 0
                )
                if strip_designator and isinstance(element, DesignatedInitializerNode):
                    elements.append(self.visit(element.value))
                else:
                    elements.append(self.visit(element))
            return f"{{{', '.join(elements)}}}"
        finally:
            self.opencl_initializer_list_depth = depth

    def format_variable_initializer_value(self, value):
        if self.indent_level != 0:
            return super().format_variable_initializer_value(value)

        depth = getattr(self, "opencl_strip_designated_initializers", 0)
        self.opencl_strip_designated_initializers = depth + 1
        try:
            return super().format_variable_initializer_value(value)
        finally:
            self.opencl_strip_designated_initializers = depth

    def visit_OpenCLProgramNode(self, node):
        """Render an OpenCL program AST as a CrossGL shader block."""
        self.opencl_sdk_reduce_materialize_op = self.is_opencl_sdk_reduce_shape(node)
        if self.opencl_sdk_reduce_materialize_op:
            self.opencl_event_tokens = self.collect_opencl_event_tokens(node)
        else:
            self.opencl_event_tokens = set()
        if self.normalize_target_safe_cgl and self.opencl_sdk_reduce_materialize_op:
            self.opencl_workgroup_pointer_helper_params = (
                self.collect_opencl_workgroup_pointer_helper_params(node)
            )
        else:
            self.opencl_workgroup_pointer_helper_params = {}
        self.emit("// OpenCL to CrossGL conversion")

        for stmt in node.statements:
            if isinstance(stmt, KernelNode) or (
                isinstance(stmt, FunctionNode)
                and "__global__" in getattr(stmt, "qualifiers", [])
            ):
                self.emit_function_local_record_definitions(stmt)
                self.emit(f"// Kernel: {stmt.name}")
                self.visit_kernel_as_compute_shader(stmt)
                self.emit("")
            elif isinstance(stmt, FunctionNode):
                self.emit_function_local_record_definitions(stmt)
                self.emit(f"// Function: {stmt.name}")
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, (StructNode, VariableNode, TypeAliasNode, EnumNode)):
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, OpenCLMacroBlockNode):
                self.visit(stmt)
                self.emit("")
            else:
                self.emit(
                    f"// skipped top-level OpenCL fragment: {type(stmt).__name__}"
                )

    def iter_opencl_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            for value in node.values():
                yield from self.iter_opencl_ast_children(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                yield from self.iter_opencl_ast_children(value)
            return
        yield node
        for value in getattr(node, "__dict__", {}).values():
            yield from self.iter_opencl_ast_children(value)

    def iter_opencl_function_calls(self, node):
        for child in self.iter_opencl_ast_children(node):
            if isinstance(child, FunctionCallNode):
                yield child

    def opencl_program_functions(self, node):
        return [
            statement
            for statement in getattr(node, "statements", []) or []
            if isinstance(statement, FunctionNode)
        ]

    def opencl_program_kernels(self, node):
        return [
            statement
            for statement in getattr(node, "statements", []) or []
            if isinstance(statement, KernelNode)
        ]

    def opencl_parameter_name(self, parameter):
        if isinstance(parameter, dict):
            return parameter.get("name")
        return getattr(parameter, "name", None)

    def opencl_parameter_type(self, parameter):
        if isinstance(parameter, dict):
            return parameter.get("type", "int")
        return getattr(parameter, "vtype", "int")

    def collect_opencl_event_tokens(self, node):
        names = set()
        for call in self.iter_opencl_function_calls(node):
            raw_args = getattr(call, "args", []) or []
            if call.name == "async_work_group_copy" and len(raw_args) >= 4:
                token_name = self.opencl_event_token_name(raw_args[3])
                if token_name:
                    names.add(token_name)
            elif call.name == "wait_group_events":
                for arg in raw_args[1:]:
                    token_name = self.opencl_event_token_name(arg)
                    if token_name:
                        names.add(token_name)
        return names

    def opencl_event_token_name(self, argument):
        if isinstance(argument, str):
            return argument
        if isinstance(argument, UnaryOpNode):
            return self.opencl_event_token_name(argument.operand)
        return getattr(argument, "name", None)

    def collect_opencl_workgroup_pointer_helper_params(self, node):
        functions = self.opencl_program_functions(node)
        function_by_name = {function.name: function for function in functions}
        workgroup_names = set()
        for kernel in self.opencl_program_kernels(node):
            for parameter in getattr(kernel, "params", []) or []:
                raw_type = self.opencl_parameter_type(parameter)
                if self.is_opencl_local_pointer_type(raw_type):
                    parameter_name = self.opencl_parameter_name(parameter)
                    if parameter_name:
                        workgroup_names.add(parameter_name)

        specialized = {}
        if not workgroup_names:
            return specialized

        for kernel in self.opencl_program_kernels(node):
            for call in self.iter_opencl_function_calls(kernel):
                callee = function_by_name.get(call.name)
                if callee is None:
                    continue
                callee_params = getattr(callee, "params", []) or []
                for index, argument in enumerate(getattr(call, "args", []) or []):
                    if index >= len(callee_params) or argument not in workgroup_names:
                        continue
                    raw_type = self.opencl_parameter_type(callee_params[index])
                    if not self.is_opencl_local_pointer_type(raw_type):
                        continue
                    parameter_name = self.opencl_parameter_name(callee_params[index])
                    if parameter_name:
                        specialized.setdefault(callee.name, set()).add(parameter_name)
        return specialized

    def is_opencl_sdk_reduce_shape(self, node):
        functions = self.opencl_program_functions(node)
        function_names = {function.name for function in functions}
        kernels = {kernel.name: kernel for kernel in self.opencl_program_kernels(node)}
        reduce_kernel = kernels.get("reduce")
        if reduce_kernel is None:
            return False
        called_names = {
            call.name for call in self.iter_opencl_function_calls(reduce_kernel)
        }
        if not {"op", "read_local", "zmin"}.issubset(called_names):
            return False
        if "op" not in function_names or "read_local" not in function_names:
            return False
        parameter_names = {
            self.opencl_parameter_name(parameter)
            for parameter in getattr(reduce_kernel, "params", []) or []
        }
        if not {"front", "back", "length", "zero_elem"}.issubset(parameter_names):
            return False
        return any(str(name or "").startswith("shared") for name in parameter_names)

    def visit_HipProgramNode(self, node):
        node.__class__ = OpenCLProgramNode
        return self.visit_OpenCLProgramNode(node)

    def visit_OpenCLStatementExpressionNode(self, node):
        for stmt in getattr(node, "statements", []) or []:
            self.emit_statement(stmt)

    def visit_OpenCLMacroBlockNode(self, node):
        args = ", ".join(arg for arg in getattr(node, "args", []) if arg)
        suffix = f"({args})" if args else "()"
        self.emit(f"// OpenCL macro block: {node.name}{suffix}")

    def visit_OpenCLBlockLiteralNode(self, node):
        return "0 /* unsupported OpenCL block literal */"

    def visit_StructNode(self, node):
        materialized = self.clone_ast_node(node)
        original_name = getattr(node, "name", None)
        if original_name:
            materialized.name = self.sanitize_opencl_type_identifier(original_name)
        return super().visit_StructNode(materialized)

    def sanitize_opencl_type_identifier(self, name):
        parts = []
        for char in str(name):
            if char.isalnum() or char == "_":
                parts.append(char)
            elif not parts or parts[-1] != "_":
                parts.append("_")

        sanitized = "".join(parts).strip("_") or "anonymous"
        if sanitized[0].isdigit():
            sanitized = f"type_{sanitized}"
        return self.sanitize_identifier_name(sanitized)

    def visit_kernel_as_compute_shader(self, kernel):
        self.emit("@compute")
        self.emit(self.opencl_workgroup_size_attribute(kernel))

        params = []
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        sampler_states = dict(self.opencl_sampler_states)
        try:
            # Program-scope __constant variables take the first bindings, so
            # the kernel's buffers start after them, as in HIP input.
            resource_binding = self.global_resource_binding_count
            for index, param in enumerate(getattr(kernel, "params", []) or []):
                if isinstance(param, dict):
                    raw_type = param.get("type", "int")
                    param_name = param.get("name", "param")
                else:
                    raw_type = getattr(param, "vtype", "int")
                    param_name = getattr(param, "name", "param")
                param_name = param_name or f"_param{index}"

                if "*" in raw_type:
                    element_type = self.convert_hip_pointer_element_type(raw_type)
                    output_name = self.register_identifier_name(param_name)
                    self.register_variable_type(param_name, f"array<{element_type}>")
                    params.append(
                        self.format_opencl_kernel_pointer_parameter(
                            raw_type, output_name, element_type, resource_binding
                        )
                    )
                    if not self.is_opencl_local_pointer_type(raw_type):
                        resource_binding += 1
                else:
                    param_type = self.convert_hip_variable_type_to_crossgl(
                        raw_type, param_name
                    )
                    output_name = self.register_identifier_name(param_name)
                    self.register_vector1_name(param_name, raw_type)
                    self.register_variable_type(param_name, param_type)
                    params.append(f"{param_type} {output_name}")

            self.emit(f"fn {kernel.name}(")
            self.indent_level += 1
            for index, param in enumerate(params):
                suffix = "," if index < len(params) - 1 else ""
                self.emit(f"{param}{suffix}")
            self.indent_level -= 1
            self.emit(") {")

            self.indent_level += 1

            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            try:
                for param in getattr(kernel, "params", []) or []:
                    self.register_unique_ptr_parameter(param)
                    self.register_cooperative_group_parameter(param)
                for stmt in getattr(kernel, "body", []) or []:
                    self.emit_statement(stmt)
            finally:
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()

            self.indent_level -= 1
            self.emit("}")
        finally:
            self.opencl_sampler_states = sampler_states
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

    def opencl_workgroup_size_attribute(self, kernel):
        """``reqd_work_group_size`` fixes the size; otherwise the host passes it to ``clEnqueueNDRangeKernel``."""
        for attribute in getattr(kernel, "attributes", []) or []:
            match = re.search(r"reqd_work_group_size\((.*?)\)", str(attribute))
            if match:
                size = re.sub(r"\s*,\s*", ", ", match.group(1).strip())
                return f"@workgroup_size({size})"
        return kernel_workgroup_size_attribute(self.threadgroup_size)

    def format_opencl_kernel_pointer_parameter(
        self, raw_type, output_name, element_type, binding
    ):
        if self.is_opencl_local_pointer_type(raw_type):
            return f"var<workgroup> {output_name}: array<{element_type}>"

        access = self.opencl_kernel_pointer_access(raw_type)
        return (
            "@group(0) "
            f"@binding({binding}) "
            f"var<storage, {access}> "
            f"{output_name}: array<{element_type}>"
        )

    def is_opencl_local_pointer_type(self, raw_type):
        qualifiers = set(str(raw_type).split())
        return bool(qualifiers & {"__shared__", "__local", "__local__", "local"})

    def opencl_kernel_pointer_access(self, raw_type):
        qualifiers = set(str(raw_type).split())
        if qualifiers & {"read_write"}:
            return "read_write"
        if qualifiers & {"write_only"}:
            return "write"
        if qualifiers & {
            "const",
            "read_only",
            "__constant",
            "__constant__",
            "constant",
        }:
            return "read"
        return "read_write"

    def visit_PreprocessorNode(self, node):
        content = self.format_preprocessor_content(node.content)
        if node.directive == "include":
            self.emit(f"// include {content}".strip())
        elif content:
            self.emit(f"// {node.directive} {content}")
        else:
            self.emit(f"// {node.directive}")

    def visit_EnumNode(self, node):
        if getattr(node, "name", None):
            enum_type = getattr(node, "underlying_type", None) or "uint"
            self.register_type_alias(
                node.name, self.convert_hip_type_to_crossgl(enum_type)
            )
            return super().visit_EnumNode(node)

        next_value = 0
        members = getattr(node, "members", None) or getattr(node, "variants", [])
        for member in members:
            if isinstance(member, tuple):
                member_name, member_value = member
            else:
                member_name = getattr(member, "name", str(member))
                member_value = getattr(member, "value", None)

            if member_value is None:
                value = str(next_value if next_value is not None else 0)
            else:
                value = self.visit(member_value)

            self.emit(f"const i32 {member_name} = {value};")
            next_value = self.next_anonymous_enum_value(value)

    def next_anonymous_enum_value(self, value):
        try:
            return int(str(value), 0) + 1
        except ValueError:
            return None

    def visit_SyncNode(self, node):
        if node.sync_type == "barrier":
            for statement in self.format_opencl_barrier_statements(node.args or []):
                self.emit(f"{statement};")
        elif node.sync_type == "mem_fence" and node.args:
            # mem_fence(flags) orders loads and stores over the flagged memory,
            # as read_mem_fence and write_mem_fence do for one of them.
            self.emit(f"{self.format_opencl_fence_call('mem_fence', node.args)};")
        else:
            super().visit_SyncNode(node)

    def visit_VariableNode(self, node):
        if (
            self.can_lower_opencl_event_local_memory()
            and node.name in self.opencl_event_tokens
            and self.is_opencl_event_type(getattr(node, "vtype", ""))
        ):
            return
        if self.is_opencl_block_declaration(node):
            self.emit(f"// unsupported OpenCL block declaration: {node.name}")
            return
        if self.is_host_embedded_source_string(node):
            self.emit(f"// skipped host OpenCL source string: {node.name}")
            return
        self.register_opencl_sizeof_symbol(node)
        if (
            getattr(node, "value", None) is not None
            and self.convert_hip_type_to_crossgl(getattr(node, "vtype", ""))
            == "sampler"
        ):
            # A sampler_t constant fixes its state in source with CLK_* flags,
            # as a CrossGL sampler(...) constructor does.
            state = self.format_opencl_sampler_state(node.value)
            output_name = self.register_variable_declaration_name(node.name)
            for name in (node.name, output_name):
                self.register_variable_type(name, "sampler")
                self.opencl_sampler_states[name] = state
            self.emit(f"const sampler {output_name} = {state};")
            return
        if self.is_opencl_constant_pointer_local(node):
            # In `__constant float *p = table;` __constant is the address space
            # p points into. p itself is a private local, like a `__global
            # float *` local, not a resource with a binding of its own.
            qualifiers = node.qualifiers
            node.qualifiers = [q for q in qualifiers if q != "__constant__"]
            try:
                super().visit_VariableNode(node)
            finally:
                node.qualifiers = qualifiers
            return
        super().visit_VariableNode(node)

    def is_opencl_constant_pointer_local(self, node):
        return (
            self.indent_level > 0
            and "__constant__" in (getattr(node, "qualifiers", None) or [])
            and "*" in str(getattr(node, "vtype", ""))
        )

    def visit_FunctionNode(self, node):
        previous_function_name = self.opencl_current_function_name
        self.opencl_current_function_name = getattr(node, "name", None)
        sampler_states = dict(self.opencl_sampler_states)
        original_body = getattr(node, "body", None)
        materialized_op = False
        if self.should_materialize_opencl_sdk_op_helper(node):
            lhs, rhs = (self.opencl_parameter_name(param) for param in node.params[:2])
            node.body = [ReturnNode(FunctionCallNode("min", [lhs, rhs]))]
            materialized_op = True
        try:
            return super().visit_FunctionNode(node)
        finally:
            if materialized_op:
                node.body = original_body
            self.opencl_current_function_name = previous_function_name
            self.opencl_sampler_states = sampler_states

    def should_materialize_opencl_sdk_op_helper(self, node):
        return (
            self.opencl_sdk_reduce_materialize_op
            and getattr(node, "name", None) == "op"
            and not getattr(node, "body", None)
            and len(getattr(node, "params", []) or []) == 2
            and self.convert_hip_type_to_crossgl(getattr(node, "return_type", ""))
            == "i32"
            and all(
                self.convert_hip_type_to_crossgl(self.opencl_parameter_type(param))
                == "i32"
                for param in node.params
            )
        )

    def visit_TypeAliasNode(self, node):
        if self.is_opencl_block_type(getattr(node, "alias_type", "")):
            self.register_type_alias(node.name, node.alias_type)
            self.emit(f"// unsupported OpenCL block typedef: {node.name}")
            return
        if self.is_opencl_pipe_type(getattr(node, "alias_type", "")):
            self.register_type_alias(node.name, node.alias_type)
            mapped_type = self.convert_hip_type_to_crossgl(node.alias_type)
            self.emit(f"// OpenCL pipe typedef: {mapped_type} {node.name}")
            return
        return super().visit_TypeAliasNode(node)

    def is_opencl_block_declaration(self, node):
        if self.is_opencl_block_type(getattr(node, "vtype", "")):
            return True
        return isinstance(getattr(node, "value", None), OpenCLBlockLiteralNode)

    def is_opencl_block_type(self, type_name):
        if not type_name:
            return False
        text = str(type_name).strip()
        if text.startswith("__opencl_block"):
            return True
        resolved = self.resolve_type_alias(text)
        return isinstance(resolved, str) and resolved.startswith("__opencl_block")

    def is_host_embedded_source_string(self, node):
        if self.indent_level != 0:
            return False

        compact_type = str(getattr(node, "vtype", "")).replace(" ", "")
        if compact_type not in {"constchar*", "charconst*", "staticconstchar*"}:
            return False

        value = getattr(node, "value", None)
        return isinstance(value, str) and "\n" in value

    def visit_FunctionCallNode(self, node):
        func_name = getattr(node, "name", None)
        if isinstance(func_name, str):
            raw_args = getattr(node, "args", []) or []
            image_write = self.format_opencl_image_write(func_name, raw_args)
            if image_write is not None:
                return image_write
            image_read = self.format_opencl_image_read(func_name, raw_args)
            if image_read is not None:
                return image_read
            if is_conversion_name(func_name) and len(raw_args) == 1:
                return format_conversion(
                    func_name, self.visit(raw_args[0]), self.convert_hip_type_to_crossgl
                )
            if func_name == "sizeof":
                sizeof_value = self.format_opencl_sizeof_call(raw_args)
                if sizeof_value is not None:
                    return str(sizeof_value)
                return self.format_unsupported_opencl_sizeof_expression(raw_args)
            builtin = self.format_opencl_builtin_call(func_name, raw_args)
            if builtin is not None:
                return builtin
            fence = self.format_opencl_fence_call(func_name, raw_args)
            if fence is not None:
                return fence
        return super().visit_FunctionCallNode(node)

    def format_opencl_fence_call(self, func_name, args):
        if func_name == "sub_group_barrier" and args:
            # sub_group_barrier(flags[, scope]) synchronizes the sub-group and
            # orders the flagged memory, which subgroupBarrier() does for the
            # default sub-group scope.
            self.require_opencl_fence_scope(
                func_name, args[1:], self.OPENCL_SUB_GROUP_FENCE_SCOPES
            )
            return "subgroupBarrier()"
        if func_name not in self.OPENCL_MEM_FENCE_FUNCTIONS or not args:
            return None
        if func_name == "atomic_work_item_fence":
            # The memory order is dropped: the CrossGL fences are acquire-release
            # fences and no target has a stronger one, so seq_cst lowers the same.
            self.require_opencl_fence_scope(
                func_name, args[2:], self.OPENCL_ATOMIC_DEVICE_SCOPES
            )
        flags = set(re.findall(r"\bCLK_\w+\b", self.visit(args[0])))
        if len(flags) == 1:
            barrier = self.OPENCL_MEM_FENCE_FLAG_BARRIERS.get(flags.pop())
            if barrier is not None:
                return f"{barrier}()"
        return "memoryBarrier()"

    # barrier(flags) and work_group_barrier(flags[, scope]) synchronize the
    # work-group and order the memory their flags name. workgroupBarrier()
    # orders work-group (local) memory; global and image memory need the
    # CrossGL fence over that memory on each side of the execution barrier,
    # the same spelling the HLSL importer gives DeviceMemoryBarrierWithGroupSync.
    OPENCL_BARRIER_DEVICE_FENCES = {
        frozenset({"CLK_GLOBAL_MEM_FENCE"}): "memoryBarrierBuffer",
        frozenset({"CLK_IMAGE_MEM_FENCE"}): "memoryBarrierImage",
        frozenset(
            {"CLK_GLOBAL_MEM_FENCE", "CLK_IMAGE_MEM_FENCE"}
        ): "deviceMemoryBarrier",
    }

    def format_opencl_barrier_statements(self, args):
        if not args:
            return ["workgroupBarrier()"]
        func_name = "work_group_barrier" if len(args) > 1 else "barrier"
        self.require_opencl_fence_scope(
            func_name, args[1:], self.OPENCL_ATOMIC_DEVICE_SCOPES
        )
        flags_text = self.visit(args[0])
        flags = set(re.findall(r"\bCLK_\w+\b", flags_text))
        known_flags = set(self.OPENCL_MEM_FENCE_FLAG_BARRIERS)
        # Only CLK_ names joined with | are literal flags, so a value such as
        # CLK_LOCAL_MEM_FENCE | mask may name more memory than its CLK_ names.
        literal = not re.sub(r"\bCLK_\w+\b|[\s()|]", "", flags_text)
        if re.sub(r"[\s()]", "", flags_text) == "0":
            # No memory is ordered: an execution barrier only.
            return ["workgroupExecutionBarrier()"]
        if not literal or not flags or not flags <= known_flags:
            # Flags that are not literals may name any memory.
            return ["allMemoryBarrier()", "workgroupBarrier()", "allMemoryBarrier()"]
        device_flags = frozenset(flags - {"CLK_LOCAL_MEM_FENCE"})
        if not device_flags:
            return ["workgroupBarrier()"]
        fence = f"{self.OPENCL_BARRIER_DEVICE_FENCES[device_flags]}()"
        barrier = (
            "workgroupBarrier()"
            if "CLK_LOCAL_MEM_FENCE" in flags
            else "workgroupExecutionBarrier()"
        )
        return [fence, barrier, fence]

    def require_opencl_fence_scope(self, func_name, scope_args, scopes):
        """Refuse a fence whose memory_scope argument is wider than ``scopes``."""
        for arg in scope_args:
            scope = self.visit(arg).strip()
            if scope not in scopes:
                raise OpenCLFenceScopeUnsupportedError(
                    f"{func_name} with {scope}: CrossGL has no {func_name} "
                    "lowering that orders memory at that scope"
                )

    # OpenCL images carry no element type; the write function names it, so an
    # image written with write_imagei or write_imageui is an integer image.
    OPENCL_IMAGE_WRITE_PREFIXES = {
        "write_imagef": "image",
        "write_imagei": "iimage",
        "write_imageui": "uimage",
    }
    OPENCL_IMAGE_STORE_SHAPES = {"1D", "1DArray", "Buffer", "2D", "2DArray", "3D"}
    # OpenCL addresses 3D images and 2D image arrays with a four-component
    # coordinate; imageStore and the reads take the three the image has.
    OPENCL_IMAGE_INT4_COORD_SHAPES = {"3D", "2DArray"}
    OPENCL_DEPTH_IMAGE_TYPES = {"image2d_depth_t", "image2d_array_depth_t"}

    def opencl_image_write_prefix(self, func_name):
        if not isinstance(func_name, str) or self.is_user_defined_function(func_name):
            return None
        return self.OPENCL_IMAGE_WRITE_PREFIXES.get(func_name)

    def get_resource_object_call_hint(self, function_name, args=None):
        prefix = self.opencl_image_write_prefix(function_name)
        if prefix is not None:
            return 0, prefix
        return super().get_resource_object_call_hint(function_name, args)

    def convert_hip_resource_object_base_type(self, hip_type, hint):
        if hint not in self.OPENCL_IMAGE_WRITE_PREFIXES.values():
            return super().convert_hip_resource_object_base_type(hip_type, hint)
        hip_type = self.strip_type_qualifiers(hip_type)
        if hip_type in self.OPENCL_DEPTH_IMAGE_TYPES:
            raise OpenCLImageWriteUnsupportedError(
                f"writing to {hip_type}: CrossGL has no depth storage image"
            )
        image_type = self.OPENCL_SCALAR_TYPE_MAPPING.get(hip_type, "")
        if not image_type.startswith("image"):
            return None
        return hint + image_type[len("image") :]

    def format_opencl_image_write(self, func_name, args):
        """Lower write_image{f,i,ui}(image, coord, color) to imageStore."""
        prefix = self.opencl_image_write_prefix(func_name)
        if prefix is None:
            return None
        if len(args) != 3:
            raise OpenCLImageWriteUnsupportedError(
                f"{func_name} with {len(args)} arguments: only "
                f"{func_name}(image, coord, color) has an imageStore lowering"
            )
        image, coord, color = args
        image_name = image if isinstance(image, str) else getattr(image, "name", None)
        image_type = (self.lookup_variable_type(image_name) if image_name else "") or ""
        shape = image_type[len(prefix) :] if image_type.startswith(prefix) else None
        if shape not in self.OPENCL_IMAGE_STORE_SHAPES:
            raise OpenCLImageWriteUnsupportedError(
                f"{func_name} on {image_name or 'an image expression'} of type "
                f"{image_type or 'unknown'}: {func_name} needs an {prefix}* image "
                "(an image written with more than one of write_imagef, "
                "write_imagei and write_imageui has no single type)"
            )
        if prefix != "image" and self.opencl_current_function_name is not None:
            raise OpenCLImageWriteUnsupportedError(
                f"{func_name} on parameter {image_name} of helper function "
                f"{self.opencl_current_function_name}: an integer image type is "
                "inferred from the writes in the kernel that declares the image"
            )
        rendered_coord = self.visit(coord)
        if shape in self.OPENCL_IMAGE_INT4_COORD_SHAPES:
            rendered_coord = f"({rendered_coord}).xyz"
        return f"imageStore({self.visit(image)}, {rendered_coord}, {self.visit(color)})"

    # A read_only image (the OpenCL default access) is read through a sampler
    # or a texel fetch, so it is a sampled texture rather than a storage image.
    # Depth and buffer images keep their image type; reading them is refused.
    OPENCL_SAMPLED_IMAGE_TYPES = {
        "image1d_t": "texture1D",
        "image1d_array_t": "texture1DArray",
        "image2d_t": "texture2D",
        "image2d_array_t": "texture2DArray",
        "image3d_t": "texture3D",
    }
    # The CrossGL sampler state each OpenCL C sampler flag selects. A field no
    # flag names is zero: unnormalized coordinates and CLK_ADDRESS_NONE.
    # CLK_ADDRESS_NONE leaves out-of-range reads undefined, so clamp_to_edge
    # keeps it; CLK_ADDRESS_CLAMP returns (0, 0, 0, 0), or (0, 0, 0, 1) for a
    # format without alpha, which is clamp_to_zero.
    OPENCL_SAMPLER_FLAGS = {
        "CLK_NORMALIZED_COORDS_TRUE": ("coord", "normalized"),
        "CLK_NORMALIZED_COORDS_FALSE": ("coord", "pixel"),
        "CLK_ADDRESS_NONE": ("address", "clamp_to_edge"),
        "CLK_ADDRESS_CLAMP_TO_EDGE": ("address", "clamp_to_edge"),
        "CLK_ADDRESS_CLAMP": ("address", "clamp_to_zero"),
        "CLK_ADDRESS_REPEAT": ("address", "repeat"),
        "CLK_ADDRESS_MIRRORED_REPEAT": ("address", "mirrored_repeat"),
        "CLK_FILTER_NEAREST": ("filter", "nearest"),
        "CLK_FILTER_LINEAR": ("filter", "linear"),
    }

    def opencl_resource_parameter_type(self, raw_type, name, crossgl_type):
        """Record an image parameter's access; a read_only image is a texture."""
        image_type = self.resolve_opencl_type_alias_chain(str(raw_type))
        if image_type == "sampler_t":
            # A sampler parameter has its caller's state, not a constant's.
            self.opencl_sampler_states.pop(name, None)
        if not image_type.startswith("image"):
            return crossgl_type
        words = {word.lstrip("_") for word in str(raw_type).split()}
        access = next(
            (access for access in ("read_write", "write_only") if access in words),
            "read_only",
        )
        self.opencl_image_access[name] = (image_type, access)
        if access != "read_only":
            return crossgl_type
        return self.OPENCL_SAMPLED_IMAGE_TYPES.get(image_type, crossgl_type)

    def format_opencl_sampler_state(self, value):
        """The CrossGL ``sampler(...)`` constructor for a CLK_* sampler value."""
        flags = {}
        for flag in self.opencl_sampler_flags(value):
            field, _setting = self.OPENCL_SAMPLER_FLAGS[flag]
            if flags.setdefault(field, flag) != flag:
                raise OpenCLImageReadUnsupportedError(
                    f"sampler_t value {self.visit(value)} names two {field} flags"
                )
        state = {"coord": "pixel", "address": "clamp_to_edge"}
        state.update(self.OPENCL_SAMPLER_FLAGS[flag] for flag in flags.values())
        if "filter" not in state:
            raise OpenCLImageReadUnsupportedError(
                f"sampler_t value {self.visit(value)} names no CLK_FILTER_* flag"
            )
        if state["coord"] == "pixel" and "repeat" in state["address"]:
            raise OpenCLImageReadUnsupportedError(
                f"sampler_t value {self.visit(value)} repeats unnormalized "
                "coordinates; OpenCL repeats only with CLK_NORMALIZED_COORDS_TRUE"
            )
        return (
            f"sampler(coord::{state['coord']}, address::{state['address']}, "
            f"filter::{state['filter']})"
        )

    def opencl_sampler_flags(self, value):
        if isinstance(value, BinaryOpNode) and value.op == "|":
            return [
                *self.opencl_sampler_flags(value.left),
                *self.opencl_sampler_flags(value.right),
            ]
        if isinstance(value, str) and value in self.OPENCL_SAMPLER_FLAGS:
            return [value]
        raise OpenCLImageReadUnsupportedError(
            f"sampler_t value {self.visit(value)}: only CLK_* sampler flags joined "
            "with | have CrossGL sampler state"
        )

    def format_opencl_image_read(self, func_name, args):
        """Lower read_imagef to textureLod, texelFetch or imageLoad.

        A read through a sampler samples level 0 (a kernel has no derivatives)
        with that sampler's state, so CLK_NORMALIZED_COORDS_FALSE keeps pixel
        coordinates. A read without a sampler takes integer coordinates and
        fetches one texel; a read_write image is read with imageLoad.
        """
        if func_name != "read_imagef" or self.is_user_defined_function(func_name):
            return None
        image = args[0] if args else None
        image_name = image if isinstance(image, str) else getattr(image, "name", None)
        image_type, access = self.opencl_image_access.get(image_name, ("", None))
        crossgl_type = str(self.lookup_variable_type(image_name) or "")
        sampled = crossgl_type.startswith("texture")
        shape = re.sub(r"^(?:texture|[iu]?image)", "", crossgl_type)

        def refuse(reason):
            return OpenCLImageReadUnsupportedError(
                f"read_imagef on {image_name or 'an image expression'}: {reason}"
            )

        if len(args) not in (2, 3):
            raise refuse(
                f"the {len(args)}-argument form has no CrossGL lowering, only "
                "read_imagef(image, coord) and read_imagef(image, sampler, coord)"
            )
        if "depth" in image_type or "buffer" in image_type:
            raise refuse(
                f"{image_type} has no CrossGL read (a depth image read returns "
                "one float, and a buffer image is not a texture)"
            )
        if crossgl_type.startswith(("iimage", "uimage")):
            raise refuse(
                "write_imagei or write_imageui makes it an integer image, and "
                "read_imagef reads only float images"
            )
        if not sampled and access != "read_write":
            raise refuse(
                f"{access or 'an unknown'} {image_type or crossgl_type or 'image'} "
                "is not a read_only or read_write image argument"
            )
        kind = self.opencl_image_coordinate_kind(args[-1])
        coord = self.visit(args[-1])
        if kind is None:
            raise refuse(f"the element type of coordinate {coord} is unknown")
        if shape in self.OPENCL_IMAGE_INT4_COORD_SHAPES:
            coord = f"({coord}).xyz"
        rendered_image = self.visit(image)
        if len(args) == 2:
            if kind == "float":
                raise refuse("a read without a sampler takes integer coordinates")
            if sampled:
                return f"texelFetch({rendered_image}, {coord}, 0)"
            return f"imageLoad({rendered_image}, {coord})"
        if not sampled:
            raise refuse("a read_write image is read without a sampler")
        sampler = args[1]
        if isinstance(sampler, str) and self.lookup_variable_type(sampler) == "sampler":
            state = self.opencl_sampler_states.get(sampler, "")
            sampler = self.visit(sampler)
        else:
            # The sampler argument may be the CLK_* flags themselves.
            sampler = state = self.format_opencl_sampler_state(sampler)
        # OpenCL reads integer coordinates through a sampler with unnormalized
        # coordinates and nearest filtering, whose addressing still applies, so
        # they are sampled as pixel coordinates.
        if (kind == "int" or "coord::pixel" in state) and shape not in {"1D", "2D"}:
            raise refuse(
                f"pixel coordinates on a {shape} image have no CrossGL sampler: "
                "Metal and Vulkan sample pixel coordinates only on 1D and 2D "
                "images, and Direct3D has none"
            )
        if kind == "int":
            coord = format_conversion(
                "convert_float2" if shape == "2D" else "convert_float",
                coord,
                self.convert_hip_type_to_crossgl,
            )
        return f"textureLod({rendered_image}, {sampler}, {coord}, 0.0)"

    def opencl_image_coordinate_kind(self, coord):
        """``"float"`` or ``"int"`` for an image coordinate, ``None`` if unknown."""
        if isinstance(coord, (BinaryOpNode, UnaryOpNode)):
            operands = (
                (coord.left, coord.right)
                if isinstance(coord, BinaryOpNode)
                else (coord.operand,)
            )
            kinds = [self.opencl_image_coordinate_kind(item) for item in operands]
            # An integer operand converts to a float one, never the reverse.
            if "float" in kinds:
                return "float"
            return "int" if all(kind == "int" for kind in kinds) else None
        type_name = self.expression_declared_type(coord)
        if type_name is None and isinstance(coord, MemberAccessNode):
            # A swizzle keeps the element type of the vector it reads.
            return self.opencl_image_coordinate_kind(coord.object)
        if type_name is None and isinstance(coord, FunctionCallNode):
            # (float2)(x, y) is a call named after its type, as is the result
            # of convert_<type> and as_<type>.
            name = str(coord.name)
            type_name = re.sub(r"^(?:convert|as)_", "", name).split("_")[0]
        if type_name is None and isinstance(coord, str):
            type_name = coord
        if not type_name:
            return None
        text = self.convert_hip_type_to_crossgl(str(type_name))
        if re.fullmatch(r"(?:vec\d<)?f\d+>?", text) or re.fullmatch(
            r"-?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?[fF]?", text
        ):
            return "float"
        if re.fullmatch(r"(?:vec\d<)?[iu]\d+>?", text) or re.fullmatch(
            r"-?\d+[uU]?", text
        ):
            return "int"
        return None

    def emit_statement(self, stmt):
        if self.is_opencl_prefetch_call(stmt):
            for arg in stmt.args:
                self.emit_opencl_discarded_expression_effects(arg)
            return
        can_lower_event_local = self.can_lower_opencl_event_local_memory()
        if (
            can_lower_event_local
            and isinstance(stmt, VariableNode)
            and stmt.name in self.opencl_event_tokens
        ):
            if self.is_opencl_event_type(getattr(stmt, "vtype", "")):
                return
        if can_lower_event_local and isinstance(stmt, FunctionCallNode):
            if stmt.name == "wait_group_events":
                return
            if stmt.name == "async_work_group_copy":
                if self.emit_opencl_async_work_group_copy(stmt):
                    return
        return super().emit_statement(stmt)

    def is_opencl_prefetch_call(self, stmt):
        # prefetch(p, n) only hints that the global range [p, p + n) will be
        # read soon and has no observable effect. Copying it into CrossGL
        # leaves a call, and usually a pointer cast, that no target declares,
        # so drop the call and keep only the argument side effects.
        return (
            isinstance(stmt, FunctionCallNode)
            and stmt.name == "prefetch"
            and len(getattr(stmt, "args", []) or []) == 2
        )

    def emit_opencl_discarded_expression_effects(self, expression):
        # The value is unused, so only its side effects are kept: each
        # assignment, ++/-- or possibly side-effecting call becomes its own
        # statement, without the casts, arithmetic and subscripts around it.
        if self.opencl_expression_is_side_effect_free(expression):
            return
        operands = self.opencl_always_evaluated_operands(expression)
        if operands is None:
            self.emit_statement(expression)
            return
        for operand in operands:
            self.emit_opencl_discarded_expression_effects(operand)

    def opencl_expression_is_side_effect_free(self, expression):
        if expression is None or isinstance(expression, (str, int, float, bool)):
            return True
        if isinstance(expression, TernaryOpNode):
            operands = [
                expression.condition,
                expression.true_expr,
                expression.false_expr,
            ]
        elif isinstance(expression, BinaryOpNode) and expression.op in ("&&", "||"):
            operands = [expression.left, expression.right]
        else:
            operands = self.opencl_always_evaluated_operands(expression)
            if operands is None:
                return False
        return all(
            self.opencl_expression_is_side_effect_free(operand) for operand in operands
        )

    def opencl_always_evaluated_operands(self, expression):
        """The operands evaluated whenever ``expression`` is, or ``None`` when
        it may have a side effect itself or evaluates an operand conditionally."""
        if isinstance(expression, FunctionCallNode):
            if expression.name == "sizeof":
                return []  # sizeof does not evaluate its operand
            if expression.name in _OPENCL_WORK_ITEM_QUERIES:
                return list(getattr(expression, "args", []) or [])
            return None
        if isinstance(expression, UnaryOpNode):
            if "++" in str(expression.op) or "--" in str(expression.op):
                return None
            return [expression.operand]
        if isinstance(expression, BinaryOpNode):
            if expression.op in ("&&", "||"):
                return None
            return [expression.left, expression.right]
        if isinstance(expression, CastNode):
            return [getattr(expression, "expression", None)]
        if isinstance(expression, ArrayAccessNode):
            return [expression.array, expression.index]
        if isinstance(expression, MemberAccessNode):
            return [expression.object]
        return None

    def can_lower_opencl_event_local_memory(self):
        return self.opencl_sdk_reduce_materialize_op

    def is_opencl_event_type(self, type_name):
        return str(type_name).strip() in {"event_t", "clk_event_t"}

    def emit_opencl_async_work_group_copy(self, node):
        raw_args = getattr(node, "args", []) or []
        if len(raw_args) < 3:
            return False

        local_index = "lid"
        destination = self.format_opencl_local_copy_access(raw_args[0], local_index)
        source = self.format_opencl_local_copy_access(raw_args[1], local_index)
        count = self.visit(raw_args[2])
        self.emit(f"if (({local_index} < {count})) {{")
        self.indent_level += 1
        self.emit(f"{destination} = {source};")
        self.indent_level -= 1
        self.emit("}")
        return True

    def format_opencl_local_copy_access(self, expression, local_index):
        if (
            isinstance(expression, BinaryOpNode)
            and getattr(expression, "op", None) == "+"
        ):
            array = self.visit(expression.left)
            offset = self.visit(expression.right)
            return f"{array}[({offset} + {local_index})]"
        return f"{self.visit(expression)}[{local_index}]"

    def visit_BinaryOpNode(self, node):
        flattened = self.format_flat_opencl_binary_chain(node)
        if flattened is not None:
            return flattened
        return super().visit_BinaryOpNode(node)

    def format_flat_opencl_binary_chain(self, node):
        op = getattr(node, "op", None)
        if op not in {"+", "*", "&", "|", "^", "&&", "||"}:
            return None

        operands = self.collect_opencl_binary_chain_operands(node, op)
        if len(operands) < 4:
            return None

        rendered = [grouped_operand(self, operand) for operand in operands]
        return f"({f' {op} '.join(rendered)})"

    def collect_opencl_binary_chain_operands(self, node, op):
        operands = []
        current = node

        while isinstance(current, BinaryOpNode) and getattr(current, "op", None) == op:
            if isinstance(current.right, BinaryOpNode) and current.right.op == op:
                return [node]
            operands.append(current.right)
            current = current.left

        operands.append(current)
        operands.reverse()
        return operands

    def register_opencl_sizeof_symbol(self, node):
        name = getattr(node, "name", None)
        vtype = getattr(node, "vtype", None)
        if not name or not vtype:
            return

        size, element_size = self.opencl_variable_sizeof_info(
            vtype, getattr(node, "value", None)
        )
        if size is None and element_size is None:
            return

        symbols = getattr(self, "opencl_sizeof_symbols", None)
        if symbols is None:
            symbols = {}
            self.opencl_sizeof_symbols = symbols
        symbols[name] = {"size": size, "element_size": element_size}

    def opencl_variable_sizeof_info(self, type_name, value=None):
        base_type, dimensions, pointer_depth = self.opencl_sizeof_type_parts(type_name)
        if pointer_depth:
            return (
                self.OPENCL_SIZEOF_POINTER_SIZE,
                self.opencl_sizeof_type_name(base_type),
            )

        if dimensions:
            element_size = self.opencl_sizeof_type_with_dimensions(
                base_type, dimensions[1:], value
            )
            outer_count = self.opencl_sizeof_array_count(dimensions[0], value)
            total_size = (
                element_size * outer_count
                if element_size is not None and outer_count is not None
                else None
            )
            return total_size, element_size

        return self.opencl_sizeof_type_name(base_type), None

    def format_opencl_sizeof_call(self, args):
        if len(args) != 1:
            return None
        return self.opencl_sizeof_operand(args[0])

    def format_unsupported_opencl_sizeof_expression(self, args):
        operand = "unknown"
        if args:
            try:
                operand = self.visit(args[0])
            except Exception:  # noqa: BLE001 - best-effort diagnostic text only.
                operand = str(args[0])
        operand = " ".join(str(operand).replace("*/", "* /").split())
        return f"(/* unsupported OpenCL size query: {operand} */ 0)"

    def opencl_sizeof_operand(self, operand):
        if isinstance(operand, UnaryOpNode) and operand.op == "*":
            return self.opencl_sizeof_dereferenced_operand(operand.operand)

        if isinstance(operand, ArrayAccessNode):
            return self.opencl_sizeof_dereferenced_operand(operand.array)

        if isinstance(operand, str):
            symbols = getattr(self, "opencl_sizeof_symbols", {})
            if operand in symbols:
                return symbols[operand].get("size")
            return self.opencl_sizeof_type_name(operand)

        target_type = getattr(operand, "target_type", None)
        if target_type:
            return self.opencl_sizeof_type_name(target_type)

        type_name = getattr(operand, "type_name", None)
        if type_name:
            return self.opencl_sizeof_type_name(type_name)

        return None

    def opencl_sizeof_dereferenced_operand(self, operand):
        if isinstance(operand, str):
            symbols = getattr(self, "opencl_sizeof_symbols", {})
            if operand in symbols:
                return symbols[operand].get("element_size")
            return self.opencl_sizeof_type_name(operand)

        if isinstance(operand, ArrayAccessNode):
            return self.opencl_sizeof_dereferenced_operand(operand.array)

        return self.opencl_sizeof_operand(operand)

    def opencl_sizeof_type_with_dimensions(self, base_type, dimensions, value=None):
        base_size = self.opencl_sizeof_type_name(base_type)
        if base_size is None:
            return None

        size = base_size
        for dimension in reversed(dimensions):
            count = self.opencl_sizeof_array_count(dimension, value)
            if count is None:
                return None
            size *= count
        return size

    def opencl_sizeof_type_name(self, type_name):
        if not type_name:
            return None

        base_type, dimensions, pointer_depth = self.opencl_sizeof_type_parts(type_name)
        if pointer_depth:
            return self.OPENCL_SIZEOF_POINTER_SIZE

        if dimensions:
            return self.opencl_sizeof_type_with_dimensions(base_type, dimensions)

        normalized = self.resolve_opencl_type_alias_chain(base_type)
        normalized = self.normalize_opencl_sizeof_type_name(normalized)

        scalar_size = self.OPENCL_SIZEOF_SCALAR_SIZES.get(normalized)
        if scalar_size is not None:
            return scalar_size

        vector_match = re.fullmatch(
            (
                r"(char|uchar|short|ushort|int|uint|long|ulong|float|double|half)"
                r"([234816])"
            ),
            normalized,
        )
        if vector_match:
            scalar_type, width = vector_match.groups()
            scalar_size = self.OPENCL_SIZEOF_SCALAR_SIZES.get(scalar_type)
            if scalar_size is not None:
                return scalar_size * int(width)

        cl_vector_match = re.fullmatch(
            r"cl_(char|uchar|short|ushort|int|uint|long|ulong|float|double)([234816])",
            normalized,
        )
        if cl_vector_match:
            scalar_type, width = cl_vector_match.groups()
            scalar_size = self.OPENCL_SIZEOF_SCALAR_SIZES.get(f"cl_{scalar_type}")
            if scalar_size is not None:
                return scalar_size * int(width)

        return None

    def opencl_sizeof_type_parts(self, type_name):
        normalized = self.normalize_opencl_sizeof_type_name(type_name)
        dimensions = re.findall(r"\[([^\]]*)\]", normalized)
        base_type = re.sub(r"\s*\[[^\]]*\]", "", normalized).strip()
        pointer_depth = base_type.count("*")
        base_type = " ".join(base_type.replace("*", " ").split())
        return base_type, dimensions, pointer_depth

    def normalize_opencl_sizeof_type_name(self, type_name):
        normalized = self.strip_type_qualifiers(str(type_name).strip())
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def opencl_sizeof_array_count(self, dimension, value=None):
        text = str(dimension).strip()
        if text:
            try:
                return int(text, 0)
            except ValueError:
                return None

        elements = getattr(value, "elements", None)
        if elements is not None:
            return len(elements)
        return None

    # CrossGL atomics are relaxed and atomic at device scope (work-group scope on
    # local memory), so an *_explicit atomic can drop its memory_order and
    # memory_scope arguments only when it asks for no more than that.
    OPENCL_ATOMIC_DEVICE_SCOPES = frozenset(
        {
            "memory_scope_work_item",
            "memory_scope_sub_group",
            "memory_scope_work_group",
            "memory_scope_device",
        }
    )

    def visit_AtomicOperationNode(self, node):
        node_args = list(node.args)
        if node.operation.endswith("_explicit"):
            node_args = self.opencl_relaxed_atomic_arguments(node.operation, node_args)
        args = [self.format_atomic_argument(arg, i) for i, arg in enumerate(node_args)]
        operation = self.OPENCL_ATOMIC_MAP.get(node.operation, node.operation)
        if node.operation == "atomic_inc" and len(args) == 1:
            args.append("1")
        elif node.operation == "atomic_dec" and len(args) == 1:
            args.append(self.negate_atomic_operand(node.args[0], "1"))
        elif node.operation in self.OPENCL_ATOMIC_SUB_OPERATIONS and len(args) >= 2:
            # CrossGL has no atomicSub, so subtract by adding the negated
            # operand; both return the value the memory held before.
            args[1] = self.negate_atomic_operand(node.args[0], args[1])
        return f"{operation}({', '.join(args)})"

    OPENCL_ATOMIC_SUB_OPERATIONS = frozenset(
        {"atomic_sub", "atomic_fetch_sub", "atomic_fetch_sub_explicit"}
    )
    OPENCL_ATOMIC_UINT_TYPES = frozenset({"u32", "uint", "unsigned int", "atomic_uint"})

    def negate_atomic_operand(self, pointer, operand):
        """``-operand``, or ``0u - operand`` on uint memory: WGSL has no unary minus on u32."""
        text = operand.strip()
        pointee = self.indexed_expression_type(self.expression_declared_type(pointer))
        if (
            pointee
            and self.strip_type_qualifiers(pointee) in self.OPENCL_ATOMIC_UINT_TYPES
        ):
            if text.isdigit():
                text += "u"
            return f"0u - {self.parenthesize_atomic_operand(text)}"
        return f"-{self.parenthesize_atomic_operand(text)}"

    @staticmethod
    def parenthesize_atomic_operand(text):
        if text.isidentifier() or re.fullmatch(r"[0-9][0-9A-Za-z_.]*", text):
            return text
        if text.startswith("(") and text.endswith(")"):
            depth = 0
            for index, char in enumerate(text):
                depth += char == "("
                depth -= char == ")"
                if depth == 0 and index < len(text) - 1:
                    break
            else:
                return text
        return f"({text})"

    def opencl_relaxed_atomic_arguments(self, operation, args):
        """The plain-form arguments of an ``*_explicit`` atomic, refusing an order or scope CrossGL cannot keep."""
        arity, orders = (3, 2) if "compare_exchange" in operation else (2, 1)
        if len(args) not in (arity + orders, arity + orders + 1):
            raise OpenCLAtomicOrderUnsupportedError(
                f"{operation} takes {arity + orders} or {arity + orders + 1} "
                f"arguments, got {len(args)}"
            )
        names = [self.visit(arg).strip() for arg in args[arity:]]
        for order in names[:orders]:
            if order != "memory_order_relaxed":
                raise OpenCLAtomicOrderUnsupportedError(
                    f"{operation} with {order}: CrossGL atomics are relaxed, so "
                    "only memory_order_relaxed keeps its ordering"
                )
        for scope in names[orders:]:
            if scope not in self.OPENCL_ATOMIC_DEVICE_SCOPES:
                raise OpenCLAtomicOrderUnsupportedError(
                    f"{operation} with {scope}: CrossGL atomics are atomic at "
                    "device scope at most"
                )
        return args[:arity]

    def format_opencl_builtin_call(self, func_name, args):
        if func_name == "get_global_size" and len(args) == 1:
            component = self.opencl_dimension_component(args[0])
            if component is not None:
                return f"(gl_NumWorkGroups.{component} * gl_WorkGroupSize.{component})"
        if func_name in self.OPENCL_LINEAR_ID_MAP and not args:
            return self.OPENCL_LINEAR_ID_MAP[func_name]
        base = self.OPENCL_BUILTIN_ID_MAP.get(func_name)
        if base is None or len(args) != 1:
            return None
        component = self.opencl_dimension_component(args[0])
        if component is None:
            rendered = self.visit(args[0])
            return f"{func_name}({rendered})"
        return f"{base}.{component}"

    def opencl_dimension_component(self, arg):
        if isinstance(arg, int):
            value = arg
        elif isinstance(arg, str) and arg.isdigit():
            value = int(arg)
        else:
            return None
        return ("x", "y", "z")[value] if value in {0, 1, 2} else None

    def convert_hip_type_to_crossgl(self, hip_type):
        if hip_type is None:
            return "void"
        if not isinstance(hip_type, str):
            hip_type = str(hip_type)
        if self.is_opencl_block_type(hip_type):
            return "i32"
        normalized = self.strip_type_qualifiers(hip_type)
        normalized = self.resolve_opencl_type_alias_chain(normalized)
        if self.is_opencl_block_type(normalized):
            return "i32"
        pipe_type = self.convert_opencl_pipe_type_to_crossgl(normalized)
        if pipe_type is not None:
            return pipe_type
        if normalized in self.OPENCL_VECTOR_TYPE_MAPPING:
            return self.OPENCL_VECTOR_TYPE_MAPPING[normalized]
        if normalized in self.OPENCL_SCALAR_TYPE_MAPPING:
            return self.OPENCL_SCALAR_TYPE_MAPPING[normalized]
        return super().convert_hip_type_to_crossgl(normalized)

    def convert_opencl_pipe_type_to_crossgl(self, type_name):
        parts = str(type_name).split()
        if not self.is_opencl_pipe_type(type_name):
            return None

        pipe_index = parts.index("pipe")
        element_type = " ".join(parts[pipe_index + 1 :]).strip()
        if not element_type:
            return "pipe"

        mapped_element_type = self.convert_hip_type_to_crossgl(element_type)
        return f"pipe_{self.sanitize_opencl_type_identifier(mapped_element_type)}"

    def is_opencl_pipe_type(self, type_name):
        return "pipe" in str(type_name).split()

    def convert_hip_pointer_element_type(self, hip_type):
        pointer_array_element = self.convert_opencl_pointer_to_array_element_type(
            hip_type
        )
        if pointer_array_element is not None:
            return pointer_array_element
        return super().convert_hip_pointer_element_type(hip_type)

    def convert_hip_variable_type_to_crossgl(self, hip_type, name):
        specialized_params = self.opencl_workgroup_pointer_helper_params.get(
            self.opencl_current_function_name,
            set(),
        )
        if name in specialized_params and self.is_opencl_local_pointer_type(hip_type):
            element_type = self.convert_hip_pointer_element_type(hip_type)
            return f"array<{element_type}, {_DYNAMIC_LOCAL_MEMORY_FALLBACK_SIZE}>"
        return self.opencl_resource_parameter_type(
            hip_type, name, super().convert_hip_variable_type_to_crossgl(hip_type, name)
        )

    def convert_opencl_pointer_to_array_element_type(self, hip_type):
        if hip_type is None:
            return None

        normalized = self.strip_type_qualifiers(str(hip_type))
        match = re.match(
            r"^(?P<base>.+?)\s*"
            r"\(\s*\*\s*(?:const|volatile|__restrict__|restrict)?\s*\)"
            r"\s*(?P<dimensions>(?:\[[^\]]*\]\s*)+)$",
            normalized,
        )
        if not match:
            return None

        array_type = f"{match.group('base').strip()}{match.group('dimensions')}"
        return self.convert_hip_type_to_crossgl(array_type)

    def resolve_opencl_type_alias_chain(self, type_name):
        seen = set()
        resolved = self.strip_type_qualifiers(type_name)

        while resolved not in seen:
            seen.add(resolved)
            next_resolved = self.resolve_type_alias(resolved)
            if next_resolved == resolved:
                break
            resolved = self.strip_type_qualifiers(next_resolved)

        return resolved

    def strip_type_qualifiers(self, type_name):
        qualifiers = {
            "const",
            "volatile",
            "__restrict__",
            "restrict",
            "&",
            "&&",
            *self.OPENCL_ADDRESS_SPACE_QUALIFIERS,
        }
        stripped = " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )
        return self.strip_opencl_elaborated_type_keyword(stripped)

    def strip_opencl_elaborated_type_keyword(self, type_name):
        for keyword in ("struct", "enum"):
            prefix = f"{keyword} "
            if type_name.startswith(prefix):
                tag_name = type_name[len(prefix) :].strip()
                if tag_name and not tag_name.startswith("<anonymous>"):
                    return tag_name
        return type_name


def generate_opencl_crossgl(ast_node):
    """Generate CrossGL source from an OpenCL backend AST."""
    return OpenCLToCrossGLConverter().generate(ast_node)
