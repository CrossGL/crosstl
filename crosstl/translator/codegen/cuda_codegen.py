"""CUDA Code Generator"""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    ArrayAccessNode,
    ArrayNode,
    ShaderNode,
    FunctionNode,
    ExpressionStatementNode,
    IdentifierNode,
    LiteralNode,
    BlockNode,
)


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

        # Process structs
        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)
            self.emit("")

        # Process cbuffers as constant memory
        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit_cbuffer(cbuffer)
            self.emit("")

        # Process global variables
        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)
            self.emit("")

        # Process functions
        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)
            self.emit("")

        # Handle legacy shader structure
        if hasattr(node, "stages") and node.stages:
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    self.visit(stage.entry_point)
                    self.emit("")

    def visit_FunctionNode(self, node):
        """Visit function declaration"""
        # Determine function qualifiers based on CrossGL qualifier
        qualifiers = []

        # Check CrossGL function qualifier (new AST)
        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                if qualifier == "compute":
                    qualifiers.append("__global__")
                elif qualifier in ["vertex", "fragment"]:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        # Check old AST structure
        elif hasattr(node, "qualifier") and node.qualifier:
            if node.qualifier == "compute":
                qualifiers.append("__global__")
            elif node.qualifier in ["vertex", "fragment"]:
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            # Default to device function
            qualifiers.append("__device__")

        # Handle return type - support both old and new AST structures
        if hasattr(node, "return_type"):
            if hasattr(node.return_type, "name"):
                return_type = self.convert_crossgl_type_to_cuda(node.return_type.name)
            else:
                return_type = self.convert_crossgl_type_to_cuda(str(node.return_type))
        else:
            return_type = "void"

        qualifier_str = " ".join(qualifiers)

        # Generate parameters - support both old and new AST structures
        params = []
        param_list = getattr(node, "parameters", getattr(node, "params", []))

        for param in param_list:
            if hasattr(param, "param_type"):
                # New AST structure
                if hasattr(param.param_type, "name"):
                    param_type = self.convert_crossgl_type_to_cuda(
                        param.param_type.name
                    )
                else:
                    param_type = self.convert_crossgl_type_to_cuda(
                        str(param.param_type)
                    )
            elif hasattr(param, "vtype"):
                # Old AST structure
                param_type = self.convert_crossgl_type_to_cuda(param.vtype)
            else:
                param_type = "void"

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

        # Process function body - support both old and new AST structures
        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            # New AST BlockNode structure
            for stmt in body.statements:
                self.visit(stmt)
        elif isinstance(body, list):
            # Old AST structure or list of statements
            for stmt in body:
                self.visit(stmt)

        self.indent_level -= 1
        self.emit("}")

    def visit_StructNode(self, node):
        """Visit struct declaration"""
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                # New AST structure - convert TypeNode properly
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.convert_crossgl_type_to_cuda(member_type_str)
            elif hasattr(member, "vtype"):
                # Old AST structure
                member_type = self.convert_crossgl_type_to_cuda(member.vtype)
            else:
                member_type = "float"

            self.emit(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_VariableNode(self, node):
        """Visit variable declaration"""
        # Handle both declaration and usage cases
        var_type = None

        # New AST structure
        if hasattr(node, "var_type"):
            if hasattr(node.var_type, "name"):
                var_type = self.convert_crossgl_type_to_cuda(node.var_type.name)
            else:
                var_type = self.convert_crossgl_type_to_cuda(str(node.var_type))
        # Old AST structure
        elif hasattr(node, "vtype"):
            var_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if var_type:
            # Check for special memory qualifiers
            qualifiers = []
            if hasattr(node, "qualifiers"):
                for qualifier in node.qualifiers:
                    if "workgroup" in str(qualifier) or "shared" in str(qualifier):
                        qualifiers.append("__shared__")
                    elif "uniform" in str(qualifier):
                        qualifiers.append("__constant__")

            qualifier_str = " ".join(qualifiers)
            if qualifier_str:
                qualifier_str += " "

            self.emit(f"{qualifier_str}{var_type} {node.name};")
        else:
            # This might be a variable reference, just return the name
            return node.name

    def visit_ExpressionStatementNode(self, node):
        """Visit expression statement"""
        expr = self.visit(node.expression)
        if expr and expr.strip():
            self.emit(f"{expr};")

    def visit_IdentifierNode(self, node):
        """Visit identifier"""
        return node.name

    def visit_LiteralNode(self, node):
        """Visit literal value"""
        return str(node.value)

    def visit_AssignmentNode(self, node):
        """Visit assignment statement"""
        target = self.visit(node.target)
        value = self.visit(node.value)
        operator = getattr(node, "operator", "=")
        self.emit(f"{target} {operator} {value};")

    def visit_BinaryOpNode(self, node):
        """Visit binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node):
        """Visit unary operation"""
        operand = self.visit(node.operand)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        return f"({operator}{operand})"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        if hasattr(node, "function"):
            func_name = self.visit(node.function)
        else:
            func_name = getattr(node, "name", "unknown")

        args = []
        if hasattr(node, "arguments"):
            args = [self.visit(arg) for arg in node.arguments]
        elif hasattr(node, "args"):
            args = [self.visit(arg) for arg in node.args]

        args_str = ", ".join(args)

        # Convert built-in functions
        func_name = self.convert_builtin_function(func_name)
        return f"{func_name}({args_str})"

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        if hasattr(node, "object_expr"):
            obj = self.visit(node.object_expr)
        else:
            obj = self.visit(node.object)
        return f"{obj}.{node.member}"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        if hasattr(node, "array_expr"):
            array = self.visit(node.array_expr)
        else:
            array = self.visit(node.array)

        if hasattr(node, "index_expr"):
            index = self.visit(node.index_expr)
        else:
            index = self.visit(node.index)

        return f"{array}[{index}]"

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1

        # Handle then branch
        if hasattr(node, "then_branch"):
            if hasattr(node.then_branch, "statements"):
                for stmt in node.then_branch.statements:
                    self.visit(stmt)
            else:
                self.visit(node.then_branch)
        elif hasattr(node, "if_body"):
            for stmt in node.if_body:
                self.visit(stmt)

        self.indent_level -= 1

        # Handle else branch
        if hasattr(node, "else_branch") and node.else_branch:
            self.emit("} else {")
            self.indent_level += 1

            if hasattr(node.else_branch, "statements"):
                for stmt in node.else_branch.statements:
                    self.visit(stmt)
            else:
                self.visit(node.else_branch)

            self.indent_level -= 1
        elif hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            for stmt in node.else_body:
                self.visit(stmt)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init_str = ""
        if node.init:
            if hasattr(node.init, "expression"):
                init_str = self.visit(node.init.expression)
            else:
                init_str = self.visit(node.init)

        condition_str = ""
        if node.condition:
            condition_str = self.visit(node.condition)

        update_str = ""
        if node.update:
            update_str = self.visit(node.update)

        self.emit(f"for ({init_str}; {condition_str}; {update_str}) {{")

        self.indent_level += 1

        # Handle body
        if hasattr(node, "body"):
            if hasattr(node.body, "statements"):
                for stmt in node.body.statements:
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

    def visit_BlockNode(self, node):
        """Visit block statement"""
        for stmt in node.statements:
            self.visit(stmt)

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
            "int": "int",
            "float": "float",
            "double": "double",
            # Vector types (with generics)
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
            # Vector types (without generics - for compatibility)
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
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

    def visit_cbuffer(self, cbuffer):
        """Visit constant buffer (convert to CUDA constant memory)"""
        self.emit(f"// Constant buffer: {cbuffer.name}")
        for member in cbuffer.members:
            if hasattr(member, "member_type"):
                member_type = self.convert_crossgl_type_to_cuda(str(member.member_type))
            else:
                member_type = self.convert_crossgl_type_to_cuda(member.vtype)
            self.emit(f"__constant__ {member_type} {member.name};")

    def visit_ArrayNode(self, node):
        """Visit array declaration"""
        if hasattr(node, "element_type"):
            element_type = self.convert_crossgl_type_to_cuda(str(node.element_type))
        else:
            element_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if node.size:
            self.emit(f"{element_type} {node.name}[{node.size}];")
        else:
            # Dynamic array - use pointer in CUDA
            self.emit(f"{element_type}* {node.name};")

    def visit_TernaryOpNode(self, node):
        """Visit ternary conditional operator"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_list(self, node_list):
        """Visit a list of nodes"""
        results = []
        for node in node_list:
            result = self.visit(node)
            if result:
                results.append(result)
        return results

    def visit_str(self, node):
        """Visit string literals"""
        return node

    def visit_int(self, node):
        """Visit integer literals"""
        return str(node)

    def visit_float(self, node):
        """Visit float literals"""
        return str(node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # VectorType or ArrayType
            if hasattr(type_node, "rows"):
                # MatrixType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                return f"mat{type_node.rows}x{type_node.cols}"
            elif str(type(type_node)).find("ArrayType") != -1:
                # ArrayType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{type_node.size}]"
                else:
                    return f"{element_type}[]"
            else:
                # VectorType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)
