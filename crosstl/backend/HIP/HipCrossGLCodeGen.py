"""
HIP to CrossGL Code Generator

This module provides code generation functionality to convert HIP (HIP Is a Portable GPU Runtime)
AST nodes to CrossGL intermediate representation.
"""

from typing import Any
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
    HipBuiltinNode,
)


class HipToCrossGLConverter:
    """Converts HIP AST to CrossGL format"""

    def __init__(self):
        self.indent_level = 0
        self.output = []

    def generate(self, ast_node):
        """Generate CrossGL code from HIP AST"""
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

    def visit_HipProgramNode(self, node):
        """Visit HIP program node (main program)"""
        # Always emit the comment first
        self.emit("// HIP to CrossGL conversion")

        # Process all statements
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
        """Visit function declaration"""
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
                    self.visit(stmt)
            else:
                self.visit(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Convert HIP kernel to CrossGL compute shader"""
        # Generate compute shader layout
        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        # Generate function signature
        params = []
        if hasattr(kernel, "params") and kernel.params:
            for param in kernel.params:
                if isinstance(param, dict):
                    param_type = self.convert_hip_type_to_crossgl(
                        param.get("type", "int")
                    )
                    param_name = param.get("name", "param")
                    # Add storage buffer qualifiers for pointer parameters
                    if "*" in param.get("type", ""):
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {param_name}: array<{param_type.replace('*', '').strip()}>"
                        )
                    else:
                        params.append(f"{param_type} {param_name}")
                else:
                    param_type = self.convert_hip_type_to_crossgl(
                        getattr(param, "vtype", "int")
                    )
                    param_name = getattr(param, "name", "param")
                    if "*" in getattr(param, "vtype", ""):
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {param_name}: array<{param_type.replace('*', '').strip()}>"
                        )
                    else:
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

        # Process kernel body
        if hasattr(kernel, "body") and kernel.body:
            if isinstance(kernel.body, list):
                for stmt in kernel.body:
                    self.visit(stmt)
            else:
                self.visit(kernel.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_StructNode(self, node):
        """Visit struct declaration"""
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
        """Visit variable declaration"""
        var_type = self.convert_hip_type_to_crossgl(getattr(node, "vtype", "int"))

        if hasattr(node, "value") and node.value:
            value = self.visit(node.value)
            self.emit(f"var {node.name}: {var_type} = {value};")
        else:
            self.emit(f"var {node.name}: {var_type};")

    def visit_AssignmentNode(self, node):
        """Visit assignment"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", "=")
        self.emit(f"{left} {operator} {right};")

    def visit_BinaryOpNode(self, node):
        """Visit binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        """Visit unary operation"""
        operand = self.visit(node.operand)
        if hasattr(node, "postfix") and node.postfix:
            return f"({operand}{node.op})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        args = []
        if hasattr(node, "args") and node.args:
            args = [self.visit(arg) for arg in node.args]
        elif hasattr(node, "arguments") and node.arguments:
            args = [self.visit(arg) for arg in node.arguments]

        args_str = ", ".join(args)

        # Get function name
        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        return f"{crossgl_func}({args_str})"

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        obj = self.visit(node.object)
        if hasattr(node, "is_pointer") and node.is_pointer:
            # Convert pointer access to direct access in CrossGL
            return f"{obj}.{node.member}"
        else:
            return f"{obj}.{node.member}"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_HipBuiltinNode(self, node):
        """Visit HIP built-in variables"""
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
        """Visit return statement"""
        if hasattr(node, "value") and node.value:
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

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "if_body") and node.if_body:
            if isinstance(node.if_body, list):
                for stmt in node.if_body:
                    self.visit(stmt)
            else:
                self.visit(node.if_body)
        self.indent_level -= 1

        if hasattr(node, "else_body") and node.else_body:
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
        init = self.visit(node.init) if hasattr(node, "init") and node.init else ""
        condition = (
            self.visit(node.condition)
            if hasattr(node, "condition") and node.condition
            else ""
        )
        update = (
            self.visit(node.update) if hasattr(node, "update") and node.update else ""
        )

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.visit(stmt)
            else:
                self.visit(node.body)
        self.indent_level -= 1

        self.emit("}")

    def convert_hip_type_to_crossgl(self, hip_type):
        """Convert HIP types to CrossGL equivalents"""
        if hip_type is None:
            return "void"

        if not isinstance(hip_type, str):
            hip_type = str(hip_type)

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
        if "*" in hip_type:
            base_type = hip_type.replace("*", "").strip()
            mapped_base = type_mapping.get(base_type, base_type)
            return f"ptr<{mapped_base}>"

        # Handle arrays
        if "[" in hip_type and "]" in hip_type:
            parts = hip_type.split("[")
            base_type = parts[0].strip()
            size = parts[1].split("]")[0]
            mapped_base = type_mapping.get(base_type, base_type)
            return f"array<{mapped_base}, {size}>"

        return type_mapping.get(hip_type, hip_type)

    def convert_hip_builtin_function(self, func_name):
        """Convert HIP built-in functions to CrossGL equivalents"""
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
            # Sync functions
            "__syncthreads": "workgroupBarrier",
            "__threadfence": "memoryBarrier",
        }

        return function_mapping.get(func_name, func_name)

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_EnumNode(self, node):
        """Visit enum declaration"""
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


def hip_to_crossgl(hip_ast: Any) -> str:
    """Convert HIP AST to CrossGL code string"""
    converter = HipToCrossGLConverter()
    return converter.generate(hip_ast)
