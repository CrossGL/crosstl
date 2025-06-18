"""CUDA Code Generator"""

from ..ast import *
from ..lexer import Lexer
from ..parser import Parser


class CudaCodeGen:
    """Generates CUDA code from CrossGL AST"""
    
    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast_node):
        """Generate CUDA code from CrossGL AST"""
        self.output = []
        self.indent_level = 0
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        """Visit an AST node and generate code"""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Generic visitor for unknown nodes"""
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
        """Visit the main shader node"""
        # Add CUDA headers
        self.emit("#include <cuda_runtime.h>")
        self.emit("#include <device_launch_parameters.h>")
        self.emit("")
        
        # Process global variables
        for var in node.global_variables:
            self.visit(var)
        
        if node.global_variables:
            self.emit("")
        
        # Process structs
        for struct in node.structs:
            self.visit(struct)
            self.emit("")
        
        # Process functions
        for func in node.functions:
            self.visit(func)
            self.emit("")

    def visit_StructNode(self, node):
        """Visit struct declaration"""
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1
        
        for member in node.members:
            member_type = self.convert_crossgl_type_to_cuda(member.vtype)
            self.emit(f"{member_type} {member.name};")
        
        self.indent_level -= 1
        self.emit("};")

    def visit_FunctionNode(self, node):
        """Visit function declaration"""
        # Determine function qualifiers
        qualifiers = []
        
        # Check if this is a compute shader (convert to kernel)
        if hasattr(node, 'attributes') and any('@compute' in str(attr) for attr in node.attributes):
            qualifiers.append("__global__")
        else:
            qualifiers.append("__device__")
        
        return_type = self.convert_crossgl_type_to_cuda(node.return_type)
        qualifier_str = " ".join(qualifiers)
        
        # Generate parameters
        params = []
        for param in node.params:
            param_type = self.convert_crossgl_type_to_cuda(param.vtype)
            params.append(f"{param_type} {param.name}")
        
        param_str = ", ".join(params)
        self.emit(f"{qualifier_str} {return_type} {node.name}({param_str}) {{")
        
        self.indent_level += 1
        
        # Add built-in variable mappings for kernels
        if "__global__" in qualifiers:
            self.emit("// CUDA built-in variables")
            self.emit("int3 threadIdx = {threadIdx.x, threadIdx.y, threadIdx.z};")
            self.emit("int3 blockIdx = {blockIdx.x, blockIdx.y, blockIdx.z};")
            self.emit("int3 blockDim = {blockDim.x, blockDim.y, blockDim.z};")
            self.emit("int3 gridDim = {gridDim.x, gridDim.y, gridDim.z};")
            self.emit("")
        
        # Process function body
        for stmt in node.body:
            self.visit(stmt)
        
        self.indent_level -= 1
        self.emit("}")

    def visit_VariableNode(self, node):
        """Visit variable declaration"""
        var_type = self.convert_crossgl_type_to_cuda(node.vtype)
        
        # Check for special memory qualifiers
        qualifiers = []
        if hasattr(node, 'qualifiers'):
            if 'workgroup' in str(node.qualifiers):
                qualifiers.append("__shared__")
            elif 'uniform' in str(node.qualifiers):
                qualifiers.append("__constant__")
        
        qualifier_str = " ".join(qualifiers)
        if qualifier_str:
            qualifier_str += " "
        
        if node.value:
            value = self.visit(node.value)
            self.emit(f"{qualifier_str}{var_type} {node.name} = {value};")
        else:
            self.emit(f"{qualifier_str}{var_type} {node.name};")

    def visit_AssignmentNode(self, node):
        """Visit assignment statement"""
        target = self.visit(node.target)
        value = self.visit(node.value)
        self.emit(f"{target} = {value};")

    def visit_BinaryOpNode(self, node):
        """Visit binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        """Visit unary operation"""
        operand = self.visit(node.operand)
        return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)
        
        # Convert built-in functions
        func_name = self.convert_builtin_function(node.name)
        return f"{func_name}({args_str})"

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        obj = self.visit(node.object)
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
        for stmt in node.if_body:
            self.visit(stmt)
        self.indent_level -= 1
        
        if node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            for stmt in node.else_body:
                self.visit(stmt)
            self.indent_level -= 1
        
        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""
        
        self.emit(f"for ({init}; {condition}; {update}) {{")
        
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        
        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")
        
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        
        self.emit("}")

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def convert_crossgl_type_to_cuda(self, crossgl_type):
        """Convert CrossGL types to CUDA equivalents"""
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
            
            # Vector types
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
        }
        
        # Handle arrays
        if crossgl_type.startswith("array<") and crossgl_type.endswith(">"):
            # Extract element type and size
            inner = crossgl_type[6:-1]  # Remove "array<" and ">"
            if "," in inner:
                parts = inner.split(",")
                element_type = parts[0].strip()
                size = parts[1].strip()
                cuda_element_type = type_mapping.get(element_type, element_type)
                return f"{cuda_element_type}[{size}]"
            else:
                cuda_element_type = type_mapping.get(inner, inner)
                return f"{cuda_element_type}*"
        
        # Handle pointers
        if crossgl_type.startswith("ptr<") and crossgl_type.endswith(">"):
            element_type = crossgl_type[4:-1]  # Remove "ptr<" and ">"
            cuda_element_type = type_mapping.get(element_type, element_type)
            return f"{cuda_element_type}*"
        
        return type_mapping.get(crossgl_type, crossgl_type)

    def convert_builtin_function(self, func_name):
        """Convert CrossGL built-in functions to CUDA equivalents"""
        function_mapping = {
            # Math functions
            "sqrt": "sqrtf",
            "pow": "powf",
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "log": "logf",
            "exp": "expf",
            "abs": "fabsf",
            "min": "fminf",
            "max": "fmaxf",
            "floor": "floorf",
            "ceil": "ceilf",
            
            # Vector constructors
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            
            # Synchronization
            "workgroupBarrier": "__syncthreads",
        }
        
        return function_mapping.get(func_name, func_name) 