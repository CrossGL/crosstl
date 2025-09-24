"""
Modernized CUDA to CrossGL Code Generator using base infrastructure.
"""

from ..base_codegen import CrossGLToCrossGLConverter, TypeMapping
from .CudaAst import *
from ...common.type_system import UniversalTypeMapper, FunctionMapper


class CudaToCrossGLConverter(CrossGLToCrossGLConverter):
    """Converts CUDA AST to CrossGL format using unified infrastructure."""

    def __init__(self):
        super().__init__("cuda")
        
        # CUDA-specific type mappings
        self.cuda_type_mappings = {
            # CUDA vector types to universal
            "float2": "vec2",
            "float3": "vec3", 
            "float4": "vec4",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            
            # CUDA-specific types
            "size_t": "uint32",
            "half": "float16",
        }
        
        # CUDA built-in variable mappings
        self.builtin_mappings = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID", 
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "32",  # Constant
        }
        
        # CUDA function mappings
        self.function_mappings = {
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
            "make_float2": "vec2",
            "make_float3": "vec3",
            "make_float4": "vec4",
            "make_int2": "ivec2",
            "make_int3": "ivec3",
            "make_int4": "ivec4",
        }

    def generate(self, ast_node) -> str:
        """Generate CrossGL IR from CUDA AST."""
        self.clear_output()
        self.emit("// CUDA to CrossGL conversion")
        self.emit()
        
        return self.visit(ast_node)

    def visit_shader_node(self, node) -> str:
        """Visit CUDA shader node."""
        # Handle both old and new node types
        if hasattr(node, 'kernels'):
            # Process kernels as compute shaders
            for kernel in node.kernels:
                self.emit(f"// Kernel: {kernel.name}")
                self.visit_kernel_as_compute_shader(kernel)
                self.emit()
        
        # Process regular functions
        if hasattr(node, 'functions'):
            for func in node.functions:
                # Skip device functions in CrossGL
                if hasattr(func, 'qualifiers') and "__device__" in func.qualifiers:
                    continue
                self.emit(f"// Function: {func.name}")
                self.emit(self.visit_function_node(func))
                self.emit()
        
        # Process structs
        if hasattr(node, 'structs'):
            for struct in node.structs:
                self.emit(self.visit_struct_node(struct))
                self.emit()
        
        # Process global variables
        if hasattr(node, 'global_variables'):
            for var in node.global_variables:
                self.emit(self.visit_variable_node(var))
        
        return self.get_output()

    def visit_kernel_as_compute_shader(self, kernel):
        """Convert CUDA kernel to CrossGL compute shader."""
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
        self.increase_indent()
        for i, param in enumerate(params):
            if i == len(params) - 1:
                self.emit(f"{param}")
            else:
                self.emit(f"{param},")
        self.decrease_indent()
        self.emit(") {")

        # Add built-in variable declarations
        self.increase_indent()
        self.emit("let thread_id = gl_GlobalInvocationID;")
        self.emit("let block_id = gl_WorkGroupID;") 
        self.emit("let thread_local_id = gl_LocalInvocationID;")
        self.emit("let block_dim = gl_WorkGroupSize;")
        self.emit()

        # Process kernel body
        for stmt in kernel.body:
            self.emit(self.visit(stmt))

        self.decrease_indent()
        self.emit("}")

    def visit_function_node(self, node) -> str:
        """Visit CUDA function node."""
        return_type = self.convert_cuda_type_to_crossgl(node.return_type)
        params = []

        for param in node.params:
            param_type = self.convert_cuda_type_to_crossgl(param.vtype)
            params.append(f"{param_type} {param.name}")

        params_str = ", ".join(params)
        result = f"{return_type} {node.name}({params_str}) {{\n"

        self.increase_indent()
        for stmt in node.body:
            stmt_code = self.visit(stmt)
            if stmt_code:
                result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
        self.decrease_indent()

        result += "}"
        return result

    def visit_variable_node(self, node) -> str:
        """Visit CUDA variable node."""
        var_type = self.convert_cuda_type_to_crossgl(node.vtype)

        if hasattr(node, 'value') and node.value:
            value = self.visit(node.value)
            return f"var {node.name}: {var_type} = {value};"
        else:
            return f"var {node.name}: {var_type};"

    def visit_atomic_operation_node(self, node) -> str:
        """Visit CUDA atomic operation."""
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

    def visit_sync_node(self, node) -> str:
        """Visit CUDA synchronization."""
        if node.sync_type == "__syncthreads":
            return "workgroupBarrier();"
        elif node.sync_type == "__syncwarp":
            return "// Warp sync not directly supported in CrossGL"
        else:
            return f"// {node.sync_type}();"

    def visit_builtin_variable_node(self, node) -> str:
        """Visit CUDA built-in variables."""
        base_name = self.builtin_mappings.get(node.builtin_name, node.builtin_name)
        if hasattr(node, 'component') and node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

    def visit_kernel_launch_node(self, node) -> str:
        """Visit CUDA kernel launch."""
        blocks = self.visit(node.blocks)
        threads = self.visit(node.threads) 
        args_str = ", ".join([self.visit(arg) for arg in node.args]) if node.args else ""
        
        result = f"// Kernel launch: {node.kernel_name}<<<{blocks}, {threads}>>>()\n"
        if args_str:
            result += f"// Arguments: {args_str}"
        return result

    def convert_cuda_type_to_crossgl(self, cuda_type: str) -> str:
        """Convert CUDA types to CrossGL equivalents."""
        if not cuda_type:
            return "void"
            
        # Handle pointers
        if "*" in cuda_type:
            base_type = cuda_type.replace("*", "").strip()
            mapped_base = self.cuda_type_mappings.get(base_type, base_type)
            return f"ptr<{mapped_base}>"

        # Handle arrays
        if "[" in cuda_type and "]" in cuda_type:
            parts = cuda_type.split("[")
            base_type = parts[0].strip()
            size = parts[1].split("]")[0]
            mapped_base = self.cuda_type_mappings.get(base_type, base_type)
            return f"array<{mapped_base}, {size}>"

        # Direct mapping
        return self.cuda_type_mappings.get(cuda_type, cuda_type)

    def convert_cuda_builtin_function(self, func_name: str) -> str:
        """Convert CUDA built-in functions to CrossGL equivalents."""
        return self.function_mappings.get(func_name, func_name)

    # Override base visitor methods for CUDA-specific nodes
    
    def visit(self, node) -> str:
        """Visit any node with CUDA-specific handling."""
        if node is None:
            return ""
        
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        
        # Handle CUDA-specific node types
        if hasattr(node, 'node_type'):
            if node.node_type == NodeType.KERNEL:
                return self.visit_kernel_as_compute_shader(node)
            elif node.node_type == NodeType.KERNEL_LAUNCH:
                return self.visit_kernel_launch_node(node)
            elif node.node_type == NodeType.ATOMIC_OP:
                return self.visit_atomic_operation_node(node)
            elif node.node_type == NodeType.SYNC:
                return self.visit_sync_node(node)
            elif node.node_type == NodeType.BUILTIN_VARIABLE:
                return self.visit_builtin_variable_node(node)
        
        # Handle legacy node types by class name
        node_type_name = type(node).__name__
        if "Kernel" in node_type_name and "Launch" not in node_type_name:
            return self.visit_kernel_as_compute_shader(node)
        elif "KernelLaunch" in node_type_name:
            return self.visit_kernel_launch_node(node)
        elif "Atomic" in node_type_name:
            return self.visit_atomic_operation_node(node)
        elif "Sync" in node_type_name:
            return self.visit_sync_node(node)
        elif "Builtin" in node_type_name:
            return self.visit_builtin_variable_node(node)
        
        # Fall back to base implementation
        return super().visit(node)
