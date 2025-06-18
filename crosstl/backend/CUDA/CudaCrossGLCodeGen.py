"""CUDA to CrossGL Code Generator"""

from .CudaAst import *


class CudaToCrossGLConverter:
    """Converts CUDA AST to CrossGL format"""

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast_node):
        """Generate CrossGL code from CUDA AST"""
        self.output = []
        self.indent_level = 0
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        """Visit a node and generate code"""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Default visitor for unknown nodes"""
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        """Emit code with proper indentation"""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def visit_ShaderNode(self, node):
        """Visit shader node (main program)"""
        # Always emit the comment first
        self.emit("// CUDA to CrossGL conversion")

        # Process includes
        if hasattr(node, "includes") and node.includes:
            for include in node.includes:
                self.visit(include)
            self.emit("")

        # Process structs
        if hasattr(node, "structs") and node.structs:
            for struct in node.structs:
                self.visit(struct)
                self.emit("")

        # Process global variables
        if hasattr(node, "global_variables") and node.global_variables:
            for var in node.global_variables:
                self.visit(var)
                self.emit("")

        # Process functions (convert kernels to compute shaders)
        if hasattr(node, "functions") and node.functions:
            for func in node.functions:
                self.emit(f"// Function: {func.name}")
                self.visit(func)
                self.emit("")

        # Process kernels (convert to compute shaders)
        if hasattr(node, "kernels") and node.kernels:
            for kernel in node.kernels:
                self.emit(f"// Kernel: {kernel.name}")
                self.visit_kernel_as_compute_shader(kernel)
                self.emit("")

    def visit_PreprocessorNode(self, node):
        """Visit preprocessor directive"""
        if node.directive == "include":
            # Convert CUDA includes to CrossGL equivalents
            if "cuda_runtime.h" in node.content:
                self.emit("// CUDA runtime functionality built-in")
            elif "device_launch_parameters.h" in node.content:
                self.emit("// Device parameters built-in")
            else:
                self.emit(f"// {node.directive} {node.content}")
        else:
            self.emit(f"// {node.directive} {node.content}")

    def visit_StructNode(self, node):
        """Visit struct declaration"""
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        for member in node.members:
            member_type = self.convert_cuda_type_to_crossgl(member.vtype)
            self.emit(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_FunctionNode(self, node):
        """Visit function declaration"""
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
            self.visit(stmt)
        self.indent_level -= 1

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Convert CUDA kernel to CrossGL compute shader"""
        # Generate compute shader layout
        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        # Generate function signature
        params = []
        for param in kernel.params:
            param_type = self.convert_cuda_type_to_crossgl(param.vtype)
            # Add storage buffer qualifiers for pointer parameters
            if "*" in param.vtype:
                params.append(
                    f"@group(0) @binding({len(params)}) var<storage, read_write> {param.name}: array<{param_type.replace('*', '').strip()}>"
                )
            else:
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

        # Add built-in variable declarations
        self.indent_level += 1
        self.emit("let thread_id = gl_GlobalInvocationID;")
        self.emit("let block_id = gl_WorkGroupID;")
        self.emit("let thread_local_id = gl_LocalInvocationID;")
        self.emit("let block_dim = gl_WorkGroupSize;")
        self.emit("")

        # Process kernel body
        for stmt in kernel.body:
            self.visit(stmt)

        self.indent_level -= 1
        self.emit("}")

    def visit_KernelLaunchNode(self, node):
        """Visit kernel launch (convert to dispatch call comment)"""
        self.emit(
            f"// Kernel launch: {node.kernel_name}<<<{self.visit(node.blocks)}, {self.visit(node.threads)}>>>()"
        )
        if node.args:
            args_str = ", ".join([self.visit(arg) for arg in node.args])
            self.emit(f"// Arguments: {args_str}")

    def visit_VariableNode(self, node):
        """Visit variable declaration"""
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)

        if node.value:
            value = self.visit(node.value)
            self.emit(f"var {node.name}: {var_type} = {value};")
        else:
            self.emit(f"var {node.name}: {var_type};")

    def visit_SharedMemoryNode(self, node):
        """Visit shared memory declaration"""
        # Convert to workgroup memory in CrossGL
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)
        if node.size:
            size = self.visit(node.size)
            self.emit(f"var<workgroup> {node.name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {node.name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        """Visit constant memory declaration"""
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
        """Visit assignment"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        self.emit(f"{left} {node.operator} {right};")

    def visit_BinaryOpNode(self, node):
        """Visit binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        """Visit unary operation"""
        operand = self.visit(node.operand)
        if node.op.startswith("post"):
            return f"({operand}{node.op[4:]})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)

        # Convert CUDA built-in functions
        func_name = self.convert_cuda_builtin_function(node.name)
        return f"{func_name}({args_str})"

    def visit_AtomicOperationNode(self, node):
        """Visit atomic operation"""
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)

        # Convert CUDA atomic operations to CrossGL equivalents
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
        """Visit synchronization"""
        if node.sync_type == "__syncthreads":
            self.emit("workgroupBarrier();")
        elif node.sync_type == "__syncwarp":
            self.emit("// Warp sync not directly supported in CrossGL")
        else:
            self.emit(f"// {node.sync_type}();")

    def visit_CudaBuiltinNode(self, node):
        """Visit CUDA built-in variables"""
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
        """Visit member access"""
        obj = self.visit(node.object)
        if node.is_pointer:
            # Convert pointer access to direct access in CrossGL
            return f"{obj}.{node.member}"
        else:
            return f"{obj}.{node.member}"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.if_body, list):
            for stmt in node.if_body:
                self.visit(stmt)
        else:
            self.visit(node.if_body)
        self.indent_level -= 1

        if node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.visit(stmt)
            else:
                self.visit(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.visit(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.visit(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_ContinueNode(self, node):
        """Visit continue statement"""
        self.emit("continue;")

    def convert_cuda_type_to_crossgl(self, cuda_type):
        """Convert CUDA types to CrossGL equivalents"""
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
            # CUDA vector types
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

        # Handle pointers
        if "*" in cuda_type:
            base_type = cuda_type.replace("*", "").strip()
            mapped_base = type_mapping.get(base_type, base_type)
            return f"ptr<{mapped_base}>"

        # Handle arrays
        if "[" in cuda_type and "]" in cuda_type:
            parts = cuda_type.split("[")
            base_type = parts[0].strip()
            size = parts[1].split("]")[0]
            mapped_base = type_mapping.get(base_type, base_type)
            return f"array<{mapped_base}, {size}>"

        return type_mapping.get(cuda_type, cuda_type)

    def convert_cuda_builtin_function(self, func_name):
        """Convert CUDA built-in functions to CrossGL equivalents"""
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
            "make_float2": "vec2<f32>",
            "make_float3": "vec3<f32>",
            "make_float4": "vec4<f32>",
            "make_int2": "vec2<i32>",
            "make_int3": "vec3<i32>",
            "make_int4": "vec4<i32>",
        }

        return function_mapping.get(func_name, func_name)
