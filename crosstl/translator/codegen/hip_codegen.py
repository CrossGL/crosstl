"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API 
for GPU programming.
"""

from typing import Dict, List, Any, Optional
from ..ast import *


class HipCodeGen:
    """Generates HIP code from CrossGL AST"""
    
    def __init__(self):
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.variable_counter = 0
        
        # CrossGL to HIP type mapping
        self.type_map = {
            # Basic types
            'int': 'int',
            'float': 'float',
            'double': 'double',
            'bool': 'bool',
            'void': 'void',
            'uint': 'unsigned int',
            
            # Vector types
            'vec2': 'float2',
            'vec3': 'float3',
            'vec4': 'float4',
            'ivec2': 'int2',
            'ivec3': 'int3',
            'ivec4': 'int4',
            'uvec2': 'uint2',
            'uvec3': 'uint3',
            'uvec4': 'uint4',
            'dvec2': 'double2',
            'dvec3': 'double3',
            'dvec4': 'double4',
            
            # Matrix types
            'mat2': 'float2x2',
            'mat3': 'float3x3',
            'mat4': 'float4x4',
            'dmat2': 'double2x2',
            'dmat3': 'double3x3',
            'dmat4': 'double4x4',
            
            # Texture types
            'sampler2D': 'texture<float4, 2>',
            'sampler3D': 'texture<float4, 3>',
            'samplerCube': 'textureCube<float4>',
            'image2D': 'surface<void, 2>',
            'buffer': 'hipDeviceptr_t',
        }
        
        # CrossGL to HIP function mapping
        self.function_map = {
            # Math functions
            'sin': 'sinf',
            'cos': 'cosf',
            'tan': 'tanf',
            'asin': 'asinf',
            'acos': 'acosf',
            'atan': 'atanf',
            'atan2': 'atan2f',
            'sinh': 'sinhf',
            'cosh': 'coshf',
            'tanh': 'tanhf',
            'exp': 'expf',
            'exp2': 'exp2f',
            'log': 'logf',
            'log2': 'log2f',
            'sqrt': 'sqrtf',
            'inversesqrt': 'rsqrtf',
            'pow': 'powf',
            'abs': 'fabsf',
            'floor': 'floorf',
            'ceil': 'ceilf',
            'round': 'roundf',
            'trunc': 'truncf',
            'fract': 'fracf',
            'mod': 'fmodf',
            'min': 'fminf',
            'max': 'fmaxf',
            'clamp': 'fmaxf(fminf',  # Special handling needed
            'mix': 'lerp',
            'step': 'step',
            'smoothstep': 'smoothstep',
            
            # Vector functions
            'length': 'length',
            'distance': 'distance',
            'dot': 'dot',
            'cross': 'cross',
            'normalize': 'normalize',
            'reflect': 'reflect',
            'refract': 'refract',
            
            # Geometric functions
            'faceforward': 'faceforward',
            
            # Vector constructors
            'vec2': 'make_float2',
            'vec3': 'make_float3',
            'vec4': 'make_float4',
            'ivec2': 'make_int2',
            'ivec3': 'make_int3',
            'ivec4': 'make_int4',
            'uvec2': 'make_uint2',
            'uvec3': 'make_uint3',
            'uvec4': 'make_uint4',
            
            # Texture functions
            'texture': 'tex2D',
            'textureLod': 'tex2DLod',
            'textureGrad': 'tex2DGrad',
        }
        
        # Built-in variable mappings
        self.builtin_map = {
            'gl_LocalInvocationID.x': 'threadIdx.x',
            'gl_LocalInvocationID.y': 'threadIdx.y',
            'gl_LocalInvocationID.z': 'threadIdx.z',
            'gl_WorkGroupID.x': 'blockIdx.x',
            'gl_WorkGroupID.y': 'blockIdx.y',
            'gl_WorkGroupID.z': 'blockIdx.z',
            'gl_WorkGroupSize.x': 'blockDim.x',
            'gl_WorkGroupSize.y': 'blockDim.y',
            'gl_WorkGroupSize.z': 'blockDim.z',
            'gl_NumWorkGroups.x': 'gridDim.x',
            'gl_NumWorkGroups.y': 'gridDim.y',
            'gl_NumWorkGroups.z': 'gridDim.z',
            'gl_GlobalInvocationID.x': '(blockIdx.x * blockDim.x + threadIdx.x)',
            'gl_GlobalInvocationID.y': '(blockIdx.y * blockDim.y + threadIdx.y)',
            'gl_GlobalInvocationID.z': '(blockIdx.z * blockDim.z + threadIdx.z)',
        }
    
    def generate(self, node: ASTNode) -> str:
        """Generate HIP code from CrossGL AST"""
        self.code_lines = []
        self.indent_level = 0
        
        # Add necessary includes
        self.add_includes()
        
        # Generate code
        self.visit(node)
        
        return '\n'.join(self.code_lines)
    
    def add_includes(self):
        """Add necessary HIP includes"""
        self.code_lines.extend([
            '#include <hip/hip_runtime.h>',
            '#include <hip/hip_runtime_api.h>',
            '#include <hip/math_functions.h>',
            '#include <hip/device_functions.h>',
            '',
        ])
    
    def indent(self) -> str:
        """Return current indentation string"""
        return '    ' * self.indent_level
    
    def add_line(self, line: str = ''):
        """Add a line with current indentation"""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append('')
    
    def visit(self, node: ASTNode) -> str:
        """Visit a node and generate code"""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: ASTNode) -> str:
        """Generic visitor for unsupported nodes"""
        raise NotImplementedError(f"Code generation not implemented for {type(node).__name__}")
    
    def visit_ProgramNode(self, node: ProgramNode) -> str:
        """Visit program node"""
        for stmt in node.statements:
            self.visit(stmt)
        return ''
    
    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Visit function node"""
        self.current_function = node.name
        
        # Determine function qualifiers
        qualifiers = []
        if hasattr(node, 'is_kernel') and node.is_kernel:
            qualifiers.append('__global__')
        elif hasattr(node, 'is_device') and node.is_device:
            qualifiers.append('__device__')
        else:
            qualifiers.append('__device__')  # Default for HIP
        
        # Function signature
        return_type = self.map_type(node.return_type)
        params = ', '.join(self.visit_parameter(param) for param in node.parameters)
        
        qualifier_str = ' '.join(qualifiers)
        signature = f"{qualifier_str} {return_type} {node.name}({params})"
        
        self.add_line(signature)
        
        if node.body:
            self.add_line('{')
            self.indent_level += 1
            self.visit(node.body)
            self.indent_level -= 1
            self.add_line('}')
        else:
            self.add_line(';')
        
        self.add_line()
        self.current_function = None
        return ''
    
    def visit_parameter(self, param: ParameterNode) -> str:
        """Visit parameter node"""
        param_type = self.map_type(param.param_type)
        return f"{param_type} {param.name}"
    
    def visit_StructNode(self, node: StructNode) -> str:
        """Visit struct node"""
        self.add_line(f"struct {node.name}")
        self.add_line('{')
        self.indent_level += 1
        
        for member in node.members:
            if isinstance(member, VariableNode):
                member_type = self.map_type(member.var_type)
                self.add_line(f"{member_type} {member.name};")
        
        self.indent_level -= 1
        self.add_line('};')
        self.add_line()
        return ''
    
    def visit_VariableNode(self, node: VariableNode) -> str:
        """Visit variable node"""
        var_type = self.map_type(node.var_type)
        
        if node.value:
            value = self.visit(node.value)
            self.add_line(f"{var_type} {node.name} = {value};")
        else:
            self.add_line(f"{var_type} {node.name};")
        
        return ''
    
    def visit_BlockNode(self, node: BlockNode) -> str:
        """Visit block node"""
        for stmt in node.statements:
            self.visit(stmt)
        return ''
    
    def visit_IfNode(self, node: IfNode) -> str:
        """Visit if node"""
        condition = self.visit(node.condition)
        self.add_line(f"if ({condition})")
        self.add_line('{')
        self.indent_level += 1
        self.visit(node.then_stmt)
        self.indent_level -= 1
        self.add_line('}')
        
        if node.else_stmt:
            self.add_line('else')
            self.add_line('{')
            self.indent_level += 1
            self.visit(node.else_stmt)
            self.indent_level -= 1
            self.add_line('}')
        
        return ''
    
    def visit_ForNode(self, node: ForNode) -> str:
        """Visit for loop node"""
        init = self.visit(node.init) if node.init else ''
        condition = self.visit(node.condition) if node.condition else ''
        update = self.visit(node.update) if node.update else ''
        
        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line('{')
        self.indent_level += 1
        self.visit(node.body)
        self.indent_level -= 1
        self.add_line('}')
        
        return ''
    
    def visit_WhileNode(self, node: WhileNode) -> str:
        """Visit while loop node"""
        condition = self.visit(node.condition)
        self.add_line(f"while ({condition})")
        self.add_line('{')
        self.indent_level += 1
        self.visit(node.body)
        self.indent_level -= 1
        self.add_line('}')
        
        return ''
    
    def visit_ReturnNode(self, node: ReturnNode) -> str:
        """Visit return node"""
        if node.value:
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ''
    
    def visit_ExpressionStatementNode(self, node: ExpressionStatementNode) -> str:
        """Visit expression statement node"""
        expr = self.visit(node.expression)
        self.add_line(f"{expr};")
        return ''
    
    def visit_AssignmentNode(self, node: AssignmentNode) -> str:
        """Visit assignment node"""
        target = self.visit(node.target)
        value = self.visit(node.value)
        return f"{target} = {value}"
    
    def visit_BinaryOpNode(self, node: BinaryOpNode) -> str:
        """Visit binary operation node"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Handle special operators
        if node.operator == 'and':
            return f"({left} && {right})"
        elif node.operator == 'or':
            return f"({left} || {right})"
        else:
            return f"({left} {node.operator} {right})"
    
    def visit_UnaryOpNode(self, node: UnaryOpNode) -> str:
        """Visit unary operation node"""
        operand = self.visit(node.operand)
        
        if node.operator == 'not':
            return f"!{operand}"
        elif node.operator in ['++', '--']:
            if hasattr(node, 'postfix') and node.postfix:
                return f"{operand}{node.operator}"
            else:
                return f"{node.operator}{operand}"
        else:
            return f"{node.operator}{operand}"
    
    def visit_FunctionCallNode(self, node: FunctionCallNode) -> str:
        """Visit function call node"""
        func_name = self.visit(node.function)
        args = [self.visit(arg) for arg in node.arguments]
        
        # Map function name
        mapped_name = self.function_map.get(func_name, func_name)
        
        # Handle special functions
        if func_name == 'clamp':
            if len(args) == 3:
                return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        elif func_name in ['texture', 'tex2D']:
            # Handle texture sampling
            if len(args) >= 2:
                return f"tex2D({args[0]}, {args[1]})"
        elif func_name == 'barrier':
            return '__syncthreads()'
        elif func_name == 'memoryBarrier':
            return '__threadfence()'
        
        args_str = ', '.join(args)
        return f"{mapped_name}({args_str})"
    
    def visit_IdentifierNode(self, node: IdentifierNode) -> str:
        """Visit identifier node"""
        # Map built-in variables
        mapped_name = self.builtin_map.get(node.name, node.name)
        return mapped_name
    
    def visit_LiteralNode(self, node: LiteralNode) -> str:
        """Visit literal node"""
        if node.type == 'float':
            # Ensure float literals have 'f' suffix
            value = str(node.value)
            if '.' not in value and 'e' not in value.lower():
                value += '.0'
            if not value.endswith('f'):
                value += 'f'
            return value
        elif node.type == 'int':
            return str(node.value)
        elif node.type == 'bool':
            return 'true' if node.value else 'false'
        elif node.type == 'string':
            return f'"{node.value}"'
        else:
            return str(node.value)
    
    def visit_ArrayAccessNode(self, node: ArrayAccessNode) -> str:
        """Visit array access node"""
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"
    
    def visit_MemberAccessNode(self, node: MemberAccessNode) -> str:
        """Visit member access node"""
        object_expr = self.visit(node.object)
        
        # Handle vector swizzling
        if node.member in ['x', 'y', 'z', 'w', 'r', 'g', 'b', 'a']:
            return f"{object_expr}.{node.member}"
        elif len(node.member) > 1 and all(c in 'xyzw' for c in node.member):
            # Multi-component swizzle - might need special handling
            return f"{object_expr}.{node.member}"
        else:
            return f"{object_expr}.{node.member}"
    
    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        """Visit array literal node"""
        elements = [self.visit(elem) for elem in node.elements]
        return f"{{{', '.join(elements)}}}"
    
    def visit_TernaryOpNode(self, node: TernaryOpNode) -> str:
        """Visit ternary operation node"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"
    
    def visit_CastNode(self, node: CastNode) -> str:
        """Visit cast node"""
        target_type = self.map_type(node.target_type)
        expr = self.visit(node.expression)
        return f"({target_type})({expr})"
    
    def map_type(self, type_name: str) -> str:
        """Map CrossGL type to HIP type"""
        # Handle array types
        if '[' in type_name and ']' in type_name:
            base_type = type_name.split('[')[0]
            array_part = type_name[type_name.find('['):]
            mapped_base = self.type_map.get(base_type, base_type)
            return f"{mapped_base}{array_part}"
        
        return self.type_map.get(type_name, type_name)
    
    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate host-side kernel launch wrapper"""
        wrapper_lines = []
        
        # Generate wrapper function
        wrapper_name = f"launch_{kernel_node.name}"
        params = []
        args = []
        
        for param in kernel_node.parameters:
            param_type = self.map_type(param.param_type)
            params.append(f"{param_type} {param.name}")
            args.append(param.name)
        
        # Add grid and block size parameters
        params.extend([
            "dim3 gridSize",
            "dim3 blockSize",
            "hipStream_t stream = 0"
        ])
        
        wrapper_lines.extend([
            f"void {wrapper_name}({', '.join(params)})",
            "{",
            f"    hipLaunchKernelGGL({kernel_node.name}, gridSize, blockSize, 0, stream, {', '.join(args)});",
            "}"
        ])
        
        return '\n'.join(wrapper_lines)


def generate_hip_code(ast: ProgramNode) -> str:
    """
    Generate HIP code from CrossGL AST
    
    Args:
        ast: CrossGL program AST
        
    Returns:
        Generated HIP source code
    """
    generator = HipCodeGen()
    return generator.generate(ast) 