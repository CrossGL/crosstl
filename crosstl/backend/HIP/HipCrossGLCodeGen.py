"""HIP to CrossGL Code Generator"""

from .HipAst import (
    ASTNode,
    ShaderNode,
    FunctionNode,
    KernelNode,
    StructNode,
    VariableNode,
    AssignmentNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionCallNode,
    AtomicOperationNode,
    CaseNode,
    CastNode,
    DoWhileNode,
    InitializerListNode,
    SyncNode,
    MemberAccessNode,
    ArrayAccessNode,
    IfNode,
    ForNode,
    WhileNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    PreprocessorNode,
    SwitchNode,
    TernaryOpNode,
    HipBuiltinNode,
)


class HipToCrossGLConverter:
    """Converts HIP AST to CrossGL format"""

    VECTOR_TYPE_MAPPING = {
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
    VECTOR_CONSTRUCTOR_MAPPING = {
        **VECTOR_TYPE_MAPPING,
        **{f"make_{name}": mapped for name, mapped in VECTOR_TYPE_MAPPING.items()},
    }

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast_node):
        self.output = []
        self.indent_level = 0
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def emit_statement(self, stmt):
        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, VariableNode):
            var_type = self.convert_hip_type_to_crossgl(getattr(stmt, "vtype", "int"))
            if hasattr(stmt, "value") and stmt.value:
                value = self.visit(stmt.value)
                return f"var {stmt.name}: {var_type} = {value}"
            return f"var {stmt.name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit(stmt.left)
            right = self.visit(stmt.right)
            operator = getattr(stmt, "operator", "=")
            return f"{left} {operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""

    def visit_HipProgramNode(self, node):
        self.emit("// HIP to CrossGL conversion")

        for stmt in node.statements:
            if isinstance(stmt, FunctionNode):
                if hasattr(stmt, "qualifiers") and "__global__" in getattr(
                    stmt, "qualifiers", []
                ):
                    self.emit(f"// Kernel: {stmt.name}")
                    self.visit_kernel_as_compute_shader(stmt)
                else:
                    self.emit(f"// Function: {stmt.name}")
                    self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, StructNode):
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, VariableNode):
                self.visit(stmt)
                self.emit("")
            else:
                self.visit(stmt)

    def visit_FunctionNode(self, node):
        # Skip device functions in CrossGL (they become inline)
        if hasattr(node, "qualifiers") and "__device__" in getattr(
            node, "qualifiers", []
        ):
            return

        return_type = self.convert_hip_type_to_crossgl(
            node.return_type if hasattr(node, "return_type") else "void"
        )
        params = []

        if hasattr(node, "params") and node.params:
            for param in node.params:
                if isinstance(param, dict):
                    param_type = self.convert_hip_type_to_crossgl(
                        param.get("type", "int")
                    )
                    param_name = param.get("name", "param")
                    params.append(f"{param_type} {param_name}")
                else:
                    param_type = self.convert_hip_type_to_crossgl(
                        getattr(param, "vtype", "int")
                    )
                    param_name = getattr(param, "name", "param")
                    params.append(f"{param_type} {param_name}")

        param_str = ", ".join(params)
        self.emit(f"{return_type} {node.name}({param_str}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        params = []
        if hasattr(kernel, "params") and kernel.params:
            for param in kernel.params:
                if isinstance(param, dict):
                    raw_type = param.get("type", "int")
                    param_name = param.get("name", "param")
                    # Add storage buffer qualifiers for pointer parameters
                    if "*" in raw_type:
                        element_type = self.convert_hip_pointer_element_type(raw_type)
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {param_name}: array<{element_type}>"
                        )
                    else:
                        param_type = self.convert_hip_type_to_crossgl(raw_type)
                        params.append(f"{param_type} {param_name}")
                else:
                    raw_type = getattr(param, "vtype", "int")
                    param_name = getattr(param, "name", "param")
                    if "*" in raw_type:
                        element_type = self.convert_hip_pointer_element_type(raw_type)
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {param_name}: array<{element_type}>"
                        )
                    else:
                        param_type = self.convert_hip_type_to_crossgl(raw_type)
                        params.append(f"{param_type} {param_name}")

        self.emit(f"fn {kernel.name}(")
        self.indent_level += 1
        for i, param in enumerate(params):
            if i == len(params) - 1:
                self.emit(f"{param}")
            else:
                self.emit(f"{param},")
        self.indent_level -= 1
        self.emit(") {")

        # Add built-in variable declarations
        self.indent_level += 1
        self.emit("let thread_id = gl_GlobalInvocationID;")
        self.emit("let block_id = gl_WorkGroupID;")
        self.emit("let thread_local_id = gl_LocalInvocationID;")
        self.emit("let block_dim = gl_WorkGroupSize;")
        self.emit("")

        if hasattr(kernel, "body") and kernel.body:
            if isinstance(kernel.body, list):
                for stmt in kernel.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(kernel.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        if hasattr(node, "members") and node.members:
            for member in node.members:
                if isinstance(member, VariableNode):
                    member_type = self.convert_hip_type_to_crossgl(
                        getattr(member, "vtype", "int")
                    )
                    self.emit(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_VariableNode(self, node):
        var_type = self.convert_hip_type_to_crossgl(getattr(node, "vtype", "int"))

        if hasattr(node, "value") and node.value:
            value = self.visit(node.value)
            self.emit(f"var {node.name}: {var_type} = {value};")
        else:
            self.emit(f"var {node.name}: {var_type};")

    def visit_AssignmentNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", "=")
        self.emit(f"{left} {operator} {right};")

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, str) and node.op.endswith("_POST"):
            return f"({operand}{node.op[:-5]})"
        elif hasattr(node, "postfix") and node.postfix:
            return f"({operand}{node.op})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        args = []
        if hasattr(node, "args") and node.args:
            args = [self.visit(arg) for arg in node.args]
        elif hasattr(node, "arguments") and node.arguments:
            args = [self.visit(arg) for arg in node.arguments]

        args_str = ", ".join(args)

        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        return f"{crossgl_func}({args_str})"

    def visit_MemberAccessNode(self, node):
        obj = self.visit(node.object)
        return f"{obj}.{node.member}"

    def visit_ArrayAccessNode(self, node):
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_InitializerListNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_SyncNode(self, node):
        if node.sync_type in {"__syncthreads", "hipDeviceSynchronize"}:
            self.emit("workgroupBarrier();")
        elif node.sync_type == "__syncwarp":
            self.emit("// Warp sync not directly supported in CrossGL")
        else:
            self.emit(f"// {node.sync_type}();")

    def visit_HipBuiltinNode(self, node):
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
        }

        base_name = builtin_map.get(node.builtin_name, node.builtin_name)
        if hasattr(node, "component") and node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

    def visit_ReturnNode(self, node):
        if hasattr(node, "value") and node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_IfNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "if_body") and node.if_body:
            if isinstance(node.if_body, list):
                for stmt in node.if_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.if_body)
        self.indent_level -= 1

        if hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        init = self.format_statement_fragment(
            node.init if hasattr(node, "init") else None
        )
        condition = (
            self.visit(node.condition)
            if hasattr(node, "condition") and node.condition
            else ""
        )
        update = self.format_statement_fragment(
            node.update if hasattr(node, "update") else None
        )

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_WhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_DoWhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit("do {")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit(f"}} while ({condition});")

    def visit_SwitchNode(self, node):
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)

        if getattr(node, "default_case", None):
            self.emit("default:")
            self.indent_level += 1
            for stmt in node.default_case:
                self.emit_statement(stmt)
            self.indent_level -= 1

        self.indent_level -= 1
        self.emit("}")

    def visit_CaseNode(self, node):
        value = self.visit(node.value)
        self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "body", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_TernaryOpNode(self, node):
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_CastNode(self, node):
        target_type = self.convert_hip_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_hip_type_to_crossgl(self, hip_type):
        if hip_type is None:
            return "void"

        if not isinstance(hip_type, str):
            hip_type = str(hip_type)

        hip_type = self.strip_type_qualifiers(hip_type)

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "char": "i8",
            "unsigned char": "u8",
            "short": "i16",
            "unsigned short": "u16",
            "int": "i32",
            "unsigned int": "u32",
            "long": "i64",
            "unsigned long": "u64",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            # HIP vector types
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }

        # Handle arrays
        if "[" in hip_type and "]" in hip_type:
            return self.convert_hip_array_type(hip_type, type_mapping)

        # Handle pointers
        if "*" in hip_type:
            return f"ptr<{self.convert_hip_pointer_element_type(hip_type)}>"

        return type_mapping.get(hip_type, hip_type)

    def convert_hip_pointer_element_type(self, hip_type):
        base_type = hip_type.replace("*", "").strip()
        return self.convert_hip_type_to_crossgl(base_type)

    def strip_type_qualifiers(self, type_name):
        qualifiers = {"const", "volatile", "__restrict__", "restrict"}
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def convert_hip_array_type(self, hip_type, type_mapping):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :]

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :]

        mapped_type = type_mapping.get(base_type, base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = f"array<{mapped_type}, {size}>"
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_hip_builtin_function(self, func_name):
        function_mapping = {
            # Math functions
            "sqrtf": "sqrt",
            "powf": "pow",
            "sinf": "sin",
            "cosf": "cos",
            "tanf": "tan",
            "logf": "log",
            "expf": "exp",
            "fabsf": "abs",
            "fminf": "min",
            "fmaxf": "max",
            "floorf": "floor",
            "ceilf": "ceil",
            # Double precision variants
            "sqrt": "sqrt",
            "pow": "pow",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "exp": "exp",
            "fabs": "abs",
            "fmin": "min",
            "fmax": "max",
            "floor": "floor",
            "ceil": "ceil",
            # Vector functions
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
            # Sync functions
            "__syncthreads": "workgroupBarrier",
            "__threadfence": "memoryBarrier",
        }

        return function_mapping.get(func_name, func_name)

    def visit_EnumNode(self, node):
        self.emit(f"enum {node.name} {{")
        self.indent_level += 1

        if hasattr(node, "variants") and node.variants:
            for i, variant in enumerate(node.variants):
                if hasattr(variant, "value") and variant.value:
                    value = self.visit(variant.value)
                    self.emit(f"{variant.name} = {value},")
                else:
                    self.emit(f"{variant.name},")

        self.indent_level -= 1
        self.emit("}")

    # Legacy method for backwards compatibility
    def convert(self, node):
        """Legacy convert method for compatibility"""
        return self.generate(node)


def hip_to_crossgl(hip_ast) -> str:
    """Convert HIP AST to CrossGL code string"""
    converter = HipToCrossGLConverter()
    return converter.generate(hip_ast)
