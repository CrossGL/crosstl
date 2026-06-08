"""OpenCL to CrossGL code generator."""

import re

from crosstl.backend.HIP.HipAst import (
    FunctionNode,
    KernelNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter

from .OpenCLAst import OpenCLProgramNode


class OpenCLToCrossGLConverter(HipToCrossGLConverter):
    """Serialize OpenCL backend AST nodes into CrossGL source."""

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
            for param in getattr(kernel, "params", []) or []:
                if isinstance(param, dict):
                    raw_type = param.get("type", "int")
                    param_name = param.get("name", "param")
                else:
                    raw_type = getattr(param, "vtype", "int")
                    param_name = getattr(param, "name", "param")

                if "*" in raw_type:
                    element_type = self.convert_hip_pointer_element_type(raw_type)
                    output_name = self.register_identifier_name(param_name)
                    self.register_variable_type(param_name, f"array<{element_type}>")
                    params.append(
                        "@group(0) "
                        f"@binding({len(params)}) "
                        "var<storage, read_write> "
                        f"{output_name}: array<{element_type}>"
                    )
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
        if self.is_host_embedded_source_string(node):
            self.emit(f"// skipped host OpenCL source string: {node.name}")
            return
        super().visit_VariableNode(node)

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
            builtin = self.format_opencl_builtin_call(func_name, raw_args)
            if builtin is not None:
                return builtin
        return super().visit_FunctionCallNode(node)

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
        normalized = self.strip_type_qualifiers(hip_type)
        normalized = self.resolve_opencl_type_alias_chain(normalized)
        if normalized in self.OPENCL_VECTOR_TYPE_MAPPING:
            return self.OPENCL_VECTOR_TYPE_MAPPING[normalized]
        if normalized in self.OPENCL_SCALAR_TYPE_MAPPING:
            return self.OPENCL_SCALAR_TYPE_MAPPING[normalized]
        return super().convert_hip_type_to_crossgl(normalized)

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
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )


def generate_opencl_crossgl(ast_node):
    """Generate CrossGL source from an OpenCL backend AST."""
    return OpenCLToCrossGLConverter().generate(ast_node)
