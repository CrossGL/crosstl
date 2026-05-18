"""CUDA to CrossGL Code Generator"""

from .CudaAst import AssignmentNode, VariableNode


class CudaToCrossGLConverter:
    """Converts CUDA AST to CrossGL format"""

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
            var_type = self.convert_cuda_type_to_crossgl(stmt.vtype)
            if stmt.value:
                value = self.visit(stmt.value)
                return f"var {stmt.name}: {var_type} = {value}"
            return f"var {stmt.name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit(stmt.left)
            right = self.visit(stmt.right)
            return f"{left} {stmt.operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""

    def visit_ShaderNode(self, node):
        self.emit("// CUDA to CrossGL conversion")

        if hasattr(node, "includes") and node.includes:
            for include in node.includes:
                self.visit(include)
            self.emit("")

        if hasattr(node, "structs") and node.structs:
            for struct in node.structs:
                self.visit(struct)
                self.emit("")

        if hasattr(node, "global_variables") and node.global_variables:
            for var in node.global_variables:
                self.visit(var)
                self.emit("")

        if hasattr(node, "functions") and node.functions:
            for func in node.functions:
                self.emit(f"// Function: {func.name}")
                self.visit(func)
                self.emit("")

        if hasattr(node, "kernels") and node.kernels:
            for kernel in node.kernels:
                self.emit(f"// Kernel: {kernel.name}")
                self.visit_kernel_as_compute_shader(kernel)
                self.emit("")

    def visit_PreprocessorNode(self, node):
        if node.directive == "include":
            if "cuda_runtime.h" in node.content:
                self.emit("// CUDA runtime functionality built-in")
            elif "device_launch_parameters.h" in node.content:
                self.emit("// Device parameters built-in")
            else:
                self.emit(f"// {node.directive} {node.content}")
        else:
            self.emit(f"// {node.directive} {node.content}")

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        for member in node.members:
            member_type = self.convert_cuda_type_to_crossgl(member.vtype)
            self.emit(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_FunctionNode(self, node):
        # Skip device functions in CrossGL (they become inline)
        if "__device__" in node.qualifiers:
            return

        return_type = self.convert_cuda_type_to_crossgl(node.return_type)
        params = []

        for param in node.params:
            param_type = self.convert_cuda_type_to_crossgl(param.vtype)
            params.append(f"{param_type} {param.name}")

        param_str = ", ".join(params)
        self.emit(f"{return_type} {node.name}({param_str}) {{")

        self.indent_level += 1
        for stmt in node.body:
            self.emit_statement(stmt)
        self.indent_level -= 1

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Convert CUDA kernel to CrossGL compute shader"""
        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        params = []
        for param in kernel.params:
            # Add storage buffer qualifiers for pointer parameters
            if "*" in param.vtype:
                element_type = self.convert_cuda_pointer_element_type(param.vtype)
                params.append(
                    f"@group(0) @binding({len(params)}) var<storage, read_write> {param.name}: array<{element_type}>"
                )
            else:
                param_type = self.convert_cuda_type_to_crossgl(param.vtype)
                params.append(f"{param_type} {param.name}")

        self.emit(f"fn {kernel.name}(")
        self.indent_level += 1
        for i, param in enumerate(params):
            if i == len(params) - 1:
                self.emit(f"{param}")
            else:
                self.emit(f"{param},")
        self.indent_level -= 1
        self.emit(") {")

        self.indent_level += 1
        self.emit("let thread_id = gl_GlobalInvocationID;")
        self.emit("let block_id = gl_WorkGroupID;")
        self.emit("let thread_local_id = gl_LocalInvocationID;")
        self.emit("let block_dim = gl_WorkGroupSize;")
        self.emit("")

        for stmt in kernel.body:
            self.emit_statement(stmt)

        self.indent_level -= 1
        self.emit("}")

    def visit_KernelLaunchNode(self, node):
        self.emit(
            f"// Kernel launch: {node.kernel_name}<<<{self.visit(node.blocks)}, {self.visit(node.threads)}>>>()"
        )
        if node.args:
            args_str = ", ".join([self.visit(arg) for arg in node.args])
            self.emit(f"// Arguments: {args_str}")

    def visit_VariableNode(self, node):
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)

        if node.value:
            value = self.visit(node.value)
            self.emit(f"var {node.name}: {var_type} = {value};")
        else:
            self.emit(f"var {node.name}: {var_type};")

    def visit_SharedMemoryNode(self, node):
        # Convert to workgroup memory in CrossGL
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)
        if node.size:
            size = self.visit(node.size)
            self.emit(f"var<workgroup> {node.name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {node.name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        # Convert to uniform buffer in CrossGL
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)
        if node.value:
            value = self.visit(node.value)
            self.emit(
                f"@group(0) @binding(0) var<uniform> {node.name}: {var_type} = {value};"
            )
        else:
            self.emit(f"@group(0) @binding(0) var<uniform> {node.name}: {var_type};")

    def visit_AssignmentNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        self.emit(f"{left} {node.operator} {right};")

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        if node.op.startswith("post"):
            return f"({operand}{node.op[4:]})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)
        func_name = self.convert_cuda_builtin_function(node.name)
        return f"{func_name}({args_str})"

    def visit_AtomicOperationNode(self, node):
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)

        atomic_map = {
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
        }

        crossgl_func = atomic_map.get(node.operation, node.operation)
        return f"{crossgl_func}({args_str})"

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            self.emit("workgroupBarrier();")
        elif node.sync_type == "__syncwarp":
            self.emit("// Warp sync not directly supported in CrossGL")
        else:
            self.emit(f"// {node.sync_type}();")

    def visit_CudaBuiltinNode(self, node):
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "32",  # Constant in CrossGL
        }

        base_name = builtin_map.get(node.builtin_name, node.builtin_name)
        if node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

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

    def visit_IfNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.if_body, list):
            for stmt in node.if_body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.if_body)
        self.indent_level -= 1

        if node.else_body:
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
        init = self.format_statement_fragment(node.init)
        condition = self.visit(node.condition) if node.condition else ""
        update = self.format_statement_fragment(node.update)

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
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
        for case in node.cases:
            self.visit(case)

        if node.default_case:
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
        for stmt in node.body:
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_TernaryOpNode(self, node):
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_CastNode(self, node):
        target_type = self.convert_cuda_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_cuda_type_to_crossgl(self, cuda_type):
        """Convert CUDA types to CrossGL equivalents"""
        cuda_type = self.strip_type_qualifiers(cuda_type)

        type_mapping = {
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
        }

        # Handle arrays
        if "[" in cuda_type and "]" in cuda_type:
            return self.convert_cuda_array_type(cuda_type, type_mapping)

        # Handle pointers
        if "*" in cuda_type:
            return f"ptr<{self.convert_cuda_pointer_element_type(cuda_type)}>"

        return type_mapping.get(cuda_type, cuda_type)

    def convert_cuda_pointer_element_type(self, cuda_type):
        base_type = cuda_type.replace("*", "").strip()
        return self.convert_cuda_type_to_crossgl(base_type)

    def strip_type_qualifiers(self, type_name):
        qualifiers = {"const", "volatile", "__restrict__", "restrict"}
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def convert_cuda_array_type(self, cuda_type, type_mapping):
        base_type = cuda_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = cuda_type[len(base_type) :]

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

    def convert_cuda_builtin_function(self, func_name):
        """Convert CUDA built-in functions to CrossGL equivalents"""
        function_mapping = {
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
            "make_float2": "vec2<f32>",
            "make_float3": "vec3<f32>",
            "make_float4": "vec4<f32>",
            "make_int2": "vec2<i32>",
            "make_int3": "vec3<i32>",
            "make_int4": "vec4<i32>",
        }

        return function_mapping.get(func_name, func_name)
