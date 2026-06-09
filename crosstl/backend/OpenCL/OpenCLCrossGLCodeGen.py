"""OpenCL to CrossGL code generator."""

import re

from crosstl.backend.HIP.HipAst import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionNode,
    KernelNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter

from .OpenCLAst import OpenCLBlockLiteralNode, OpenCLProgramNode


class OpenCLToCrossGLConverter(HipToCrossGLConverter):
    """Serialize OpenCL backend AST nodes into CrossGL source."""

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
        "image1d_buffer_t": "image1D",
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
    OPENCL_ATOMIC_MAP = {
        "atomic_add": "atomicAdd",
        "atomic_fetch_add": "atomicAdd",
        "atomic_fetch_add_explicit": "atomicAdd",
        "atomic_sub": "atomicSub",
        "atomic_fetch_sub": "atomicSub",
        "atomic_fetch_sub_explicit": "atomicSub",
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
        "atomic_dec": "atomicSub",
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

    def generate(self, ast_node):
        self.opencl_sizeof_symbols = {}
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

    def visit_OpenCLProgramNode(self, node):
        """Render an OpenCL program AST as a CrossGL shader block."""
        self.emit("// OpenCL to CrossGL conversion")

        for stmt in node.statements:
            if isinstance(stmt, KernelNode) or (
                isinstance(stmt, FunctionNode)
                and "__global__" in getattr(stmt, "qualifiers", [])
            ):
                self.emit(f"// Kernel: {stmt.name}")
                self.visit_kernel_as_compute_shader(stmt)
                self.emit("")
            elif isinstance(stmt, FunctionNode):
                self.emit(f"// Function: {stmt.name}")
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, (StructNode, VariableNode, TypeAliasNode)):
                self.visit(stmt)
                self.emit("")
            else:
                self.visit(stmt)

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
        original_name = getattr(node, "name", None)
        if original_name:
            node.name = self.sanitize_opencl_type_identifier(original_name)
        try:
            return super().visit_StructNode(node)
        finally:
            node.name = original_name

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
        workgroup_size = self.opencl_workgroup_size(kernel)
        self.emit("@compute")
        self.emit(f"@workgroup_size({workgroup_size})")

        params = []
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            resource_binding = 0
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
            self.emit("let thread_id = gl_GlobalInvocationID;")
            self.emit("let block_id = gl_WorkGroupID;")
            self.emit("let thread_local_id = gl_LocalInvocationID;")
            self.emit("let block_dim = gl_WorkGroupSize;")
            self.emit("")

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
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

    def opencl_workgroup_size(self, kernel):
        for attribute in getattr(kernel, "attributes", []) or []:
            match = re.search(r"reqd_work_group_size\((.*?)\)", str(attribute))
            if match:
                return re.sub(r"\s*,\s*", ", ", match.group(1).strip())
        return "1, 1, 1"

    def format_opencl_kernel_pointer_parameter(
        self, raw_type, output_name, element_type, binding
    ):
        if self.is_opencl_local_pointer_type(raw_type):
            return f"var<workgroup> {output_name}: array<{element_type}>"

        return (
            "@group(0) "
            f"@binding({binding}) "
            "var<storage, read_write> "
            f"{output_name}: array<{element_type}>"
        )

    def is_opencl_local_pointer_type(self, raw_type):
        qualifiers = set(str(raw_type).split())
        return bool(qualifiers & {"__shared__", "__local", "__local__", "local"})

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
            self.emit("workgroupBarrier();")
        elif node.sync_type == "mem_fence":
            args = ", ".join(self.visit(arg) for arg in node.args)
            self.emit(f"// OpenCL mem_fence({args})")
        else:
            super().visit_SyncNode(node)

    def visit_VariableNode(self, node):
        if self.is_opencl_block_declaration(node):
            self.emit(f"// unsupported OpenCL block declaration: {node.name}")
            return
        if self.is_host_embedded_source_string(node):
            self.emit(f"// skipped host OpenCL source string: {node.name}")
            return
        self.register_opencl_sizeof_symbol(node)
        super().visit_VariableNode(node)

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
            if func_name == "sizeof":
                sizeof_value = self.format_opencl_sizeof_call(raw_args)
                if sizeof_value is not None:
                    return str(sizeof_value)
                return self.format_unsupported_opencl_sizeof_expression(raw_args)
            builtin = self.format_opencl_builtin_call(func_name, raw_args)
            if builtin is not None:
                return builtin
        return super().visit_FunctionCallNode(node)

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

        rendered = [self.visit(operand) for operand in operands]
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

    def visit_AtomicOperationNode(self, node):
        args = [self.format_atomic_argument(arg, i) for i, arg in enumerate(node.args)]
        operation = self.OPENCL_ATOMIC_MAP.get(node.operation, node.operation)
        if node.operation == "atomic_inc" and len(args) == 1:
            args.append("1")
        elif node.operation == "atomic_dec" and len(args) == 1:
            args.append("1")
        return f"{operation}({', '.join(args)})"

    def format_opencl_builtin_call(self, func_name, args):
        if func_name == "get_global_size" and len(args) == 1:
            component = self.opencl_dimension_component(args[0])
            if component is not None:
                return f"(gl_NumWorkGroups.{component} * gl_WorkGroupSize.{component})"
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
