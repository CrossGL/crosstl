"""HIP to CrossGL Code Generator"""

from .HipAst import (
    ASTNode,
    ShaderNode,
    FunctionNode,
    KernelNode,
    KernelLaunchNode,
    StructNode,
    VariableNode,
    AssignmentNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionCallNode,
    AtomicOperationNode,
    CaseNode,
    CastNode,
    DesignatedInitializerNode,
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
    TypeAliasNode,
    HipBuiltinNode,
)


class HipToCrossGLConverter:
    """Serialize HIP backend AST nodes back into CrossGL source."""

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
    HIP_TEXTURE_TYPE_MAPPING = {
        "1": "sampler1D",
        "2": "sampler2D",
        "3": "sampler3D",
        "hipTextureType1D": "sampler1D",
        "hipTextureType1DLayered": "sampler1DArray",
        "hipTextureType2D": "sampler2D",
        "hipTextureType2DLayered": "sampler2DArray",
        "hipTextureType3D": "sampler3D",
        "hipTextureTypeCubemap": "samplerCube",
        "hipTextureTypeCubemapLayered": "samplerCubeArray",
        "cudaTextureType1D": "sampler1D",
        "cudaTextureType1DLayered": "sampler1DArray",
        "cudaTextureType2D": "sampler2D",
        "cudaTextureType2DLayered": "sampler2DArray",
        "cudaTextureType3D": "sampler3D",
        "cudaTextureTypeCubemap": "samplerCube",
        "cudaTextureTypeCubemapLayered": "samplerCubeArray",
    }
    HIP_SURFACE_TYPE_MAPPING = {
        "2": "image2D",
        "3": "image3D",
        "hipSurfaceType2D": "image2D",
        "hipSurfaceType2DLayered": "image2DArray",
        "hipSurfaceType3D": "image3D",
        "hipSurfaceTypeCubemap": "imageCube",
        "cudaSurfaceType2D": "image2D",
        "cudaSurfaceType2DLayered": "image2DArray",
        "cudaSurfaceType3D": "image3D",
        "cudaSurfaceTypeCubemap": "imageCube",
    }
    HIP_TEXTURE_CALL_TYPE_HINTS = {
        "tex1D": "sampler1D",
        "tex1DLod": "sampler1D",
        "tex1DGrad": "sampler1D",
        "tex2D": "sampler2D",
        "tex2DLod": "sampler2D",
        "tex2DGrad": "sampler2D",
        "tex3D": "sampler3D",
        "tex3DLod": "sampler3D",
        "tex3DGrad": "sampler3D",
        "texCubemap": "samplerCube",
        "texCubemapLod": "samplerCube",
        "texCubemapGrad": "samplerCube",
        "tex1DLayered": "sampler1DArray",
        "tex1DLayeredLod": "sampler1DArray",
        "tex1DLayeredGrad": "sampler1DArray",
        "tex2DLayered": "sampler2DArray",
        "tex2DLayeredLod": "sampler2DArray",
        "tex2DLayeredGrad": "sampler2DArray",
        "texCubemapLayered": "samplerCubeArray",
        "texCubemapLayeredLod": "samplerCubeArray",
        "texCubemapLayeredGrad": "samplerCubeArray",
    }
    HIP_SURFACE_CALL_TYPE_HINTS = {
        "surf2Dread": "image2D",
        "surf2Dwrite": "image2D",
        "surf3Dread": "image3D",
        "surf3Dwrite": "image3D",
        "surf2DLayeredread": "image2DArray",
        "surf2DLayeredwrite": "image2DArray",
        "surfCubemapread": "imageCube",
        "surfCubemapwrite": "imageCube",
    }

    def __init__(self):
        """Initialize HIP-to-CrossGL visitor state."""
        self.indent_level = 0
        self.output = []
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.user_function_names = set()
        self.global_resource_object_type_hints = {}
        self.resource_object_hint_scopes = []

    def generate(self, ast_node):
        """Generate complete CrossGL source from a parsed HIP AST."""
        self.output = []
        self.indent_level = 0
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.user_function_names = self.collect_user_function_names(ast_node)
        self.global_resource_object_type_hints = (
            self.collect_global_resource_object_type_hints(ast_node)
        )
        self.resource_object_hint_scopes = []
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        """Dispatch a HIP backend AST node to its converter method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback converter for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        """Append a line of CrossGL output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            name = getattr(current, "name", None)
            body = getattr(current, "body", None)
            if name is not None and body is not None:
                names.add(name)

            for stmt in getattr(current, "statements", []):
                collect(stmt)
            for function in getattr(current, "functions", []):
                collect(function)
            for kernel in getattr(current, "kernels", []):
                collect(kernel)

        collect(node)
        names.discard(None)
        return names

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def add_resource_object_type_hint(self, hints, name, resource_type):
        if not name or not resource_type:
            return
        if name in hints and hints[name] != resource_type:
            hints[name] = None
            return
        hints[name] = resource_type

    def collect_resource_object_type_hints(self, node, declared_names=None):
        hints = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        self.add_resource_object_type_hint(
                            hints,
                            self.get_resource_object_expression_name(
                                current.args[arg_index]
                            ),
                            resource_type,
                        )

            for value in vars(current).values():
                collect(value)

        collect(node)
        for name in declared_names or []:
            hints.setdefault(name, None)
        return hints

    def collect_global_resource_object_type_hints(self, node):
        hints = {}
        global_names = self.collect_global_declared_variable_names(node)
        for stmt in getattr(node, "statements", []):
            if isinstance(stmt, FunctionNode):
                self.collect_global_resource_object_type_hints_from_function(
                    stmt, global_names, hints
                )
        return hints

    def collect_global_resource_object_type_hints_from_function(
        self, node, global_names, hints
    ):
        local_names = self.collect_declared_variable_names(node)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        name = self.get_resource_object_expression_name(
                            current.args[arg_index]
                        )
                        if name in global_names and name not in local_names:
                            self.add_resource_object_type_hint(
                                hints, name, resource_type
                            )

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))

    def collect_global_declared_variable_names(self, node):
        return {
            stmt.name
            for stmt in getattr(node, "statements", [])
            if isinstance(stmt, VariableNode)
        }

    def collect_declared_variable_names(self, node):
        names = set()
        for param in getattr(node, "params", []) or []:
            if isinstance(param, dict):
                name = param.get("name")
            else:
                name = getattr(param, "name", None)
            if name:
                names.add(name)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, VariableNode):
                names.add(current.name)

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))
        return names

    def get_resource_object_call_hint(self, function_name):
        base_name, _ = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None
        if base_name in self.HIP_TEXTURE_CALL_TYPE_HINTS:
            return 0, self.HIP_TEXTURE_CALL_TYPE_HINTS[base_name]
        if base_name in self.HIP_SURFACE_CALL_TYPE_HINTS:
            if base_name.endswith("write"):
                return 1, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
            return 0, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
        return None

    def get_resource_object_expression_name(self, expression):
        if isinstance(expression, str):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.get_resource_object_expression_name(expression.array)
        if isinstance(expression, CastNode):
            return self.get_resource_object_expression_name(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.get_resource_object_expression_name(expression.operand)
        return None

    def push_resource_object_hint_scope(self, hints):
        self.resource_object_hint_scopes.append(hints)

    def pop_resource_object_hint_scope(self):
        if self.resource_object_hint_scopes:
            self.resource_object_hint_scopes.pop()

    def lookup_resource_object_type_hint(self, name):
        for scope in reversed(self.resource_object_hint_scopes):
            if name in scope:
                return scope[name]
        return self.global_resource_object_type_hints.get(name)

    def emit_statement(self, stmt):
        """Render and append one converted statement."""
        if isinstance(stmt, list):
            for item in stmt:
                self.emit_statement(item)
            return

        if self.emit_hip_runtime_call_statement(stmt):
            return

        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_hip_runtime_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False
        if self.is_user_defined_function(stmt.name):
            return False

        comments = self.format_hip_runtime_call(stmt)
        if comments is None:
            return False

        for comment in comments:
            self.emit(comment)
        return True

    def format_hip_runtime_status_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        comments = self.format_hip_runtime_call(value)
        if comments is None:
            return None

        return comments, "hipSuccess"

    def format_hip_runtime_call(self, node):
        args = [self.visit(arg) for arg in node.args]
        name = node.name

        if name in {"hipMalloc", "hipMallocManaged", "hipHostMalloc"}:
            if len(node.args) >= 2:
                target = self.format_runtime_pointer_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// HIP memory allocate: {target}, bytes: {size}"]
        elif name in {"hipFree", "hipHostFree"}:
            if args:
                return [f"// HIP memory free: {args[0]}"]
        elif name in {"hipMemcpy", "hipMemcpyAsync"}:
            if len(args) >= 4:
                comment = (
                    f"// HIP memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemset", "hipMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipStreamSynchronize"}:
            if args:
                return [f"// HIP synchronize: {args[0]}"]
        elif name == "hipDeviceSynchronize":
            return ["// HIP device synchronize"]
        elif name in {
            "hipStreamCreate",
            "hipStreamCreateWithFlags",
            "hipStreamCreateWithPriority",
            "hipStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "hipStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// HIP stream {action}: {stream}"
                if (
                    name
                    in {
                        "hipStreamCreateWithFlags",
                        "hipStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "hipStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"hipEventCreate", "hipEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "hipEventRecord":
            if args:
                comment = f"// HIP event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipEventSynchronize":
            if args:
                return [f"// HIP event synchronize: {args[0]}"]
        elif name == "hipEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "hipEventDestroy":
            if args:
                return [f"// HIP event destroy: {args[0]}"]
        elif name == "hipEventQuery":
            if args:
                return [f"// HIP event query: {args[0]}"]
        elif name == "hipStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// HIP stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipGetLastError":
            return ["// HIP get last error"]
        elif name == "hipPeekAtLastError":
            return ["// HIP peek at last error"]

        return None

    def format_runtime_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_runtime_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, list):
            return ", ".join(self.format_statement_fragment(item) for item in stmt)
        if isinstance(stmt, VariableNode):
            var_type = self.convert_hip_variable_type_to_crossgl(
                getattr(stmt, "vtype", "int"), stmt.name
            )
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
        """Render a HIP program AST as a CrossGL shader block."""
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
            elif isinstance(stmt, TypeAliasNode):
                self.visit(stmt)
                self.emit("")
            else:
                self.visit(stmt)

    def format_preprocessor_content(self, content):
        text = str(content).strip()
        compact = text.replace(" ", "")
        if compact.startswith("<") and compact.endswith(">"):
            return compact
        return text

    def visit_PreprocessorNode(self, node):
        content = self.format_preprocessor_content(node.content)
        if node.directive == "include":
            if "hip_runtime.h" in content:
                self.emit("// HIP runtime functionality built-in")
            elif "hip/hip_runtime_api.h" in content:
                self.emit("// HIP runtime API functionality built-in")
            else:
                self.emit(f"// include {content}".strip())
        elif content:
            self.emit(f"// {node.directive} {content}")
        else:
            self.emit(f"// {node.directive}")

    def visit_FunctionNode(self, node):
        """Render a HIP function node as a CrossGL function."""
        return_type = self.convert_hip_type_to_crossgl(
            node.return_type if hasattr(node, "return_type") else "void"
        )

        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                node, self.collect_declared_variable_names(node)
            )
        )
        try:
            params = []

            if hasattr(node, "params") and node.params:
                for param in node.params:
                    param_name = (
                        param.get("name", "param")
                        if isinstance(param, dict)
                        else getattr(param, "name", "param")
                    )
                    raw_type = (
                        param.get("type", "int")
                        if isinstance(param, dict)
                        else getattr(param, "vtype", "int")
                    )
                    param_type = self.convert_hip_variable_type_to_crossgl(
                        raw_type, param_name
                    )
                    params.append(f"{param_type} {param_name}")

            param_str = ", ".join(params)
            self.emit(f"{return_type} {node.name}({param_str}) {{")

            self.indent_level += 1
            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            if hasattr(node, "params") and node.params:
                for param in node.params:
                    self.register_unique_ptr_parameter(param)
            if hasattr(node, "body") and node.body:
                try:
                    if isinstance(node.body, list):
                        for stmt in node.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(node.body)
                finally:
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()
                    self.indent_level -= 1
            else:
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Render a HIP kernel as a CrossGL compute shader block."""
        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        params = []
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        try:
            if hasattr(kernel, "params") and kernel.params:
                for param in kernel.params:
                    if isinstance(param, dict):
                        raw_type = param.get("type", "int")
                        param_name = param.get("name", "param")
                    else:
                        raw_type = getattr(param, "vtype", "int")
                        param_name = getattr(param, "name", "param")

                    if "*" in raw_type:
                        element_type = self.convert_hip_pointer_element_type(raw_type)
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {param_name}: array<{element_type}>"
                        )
                    else:
                        param_type = self.convert_hip_variable_type_to_crossgl(
                            raw_type, param_name
                        )
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
                self.push_packed_argument_scope()
                self.push_type_alias_scope()
                self.push_unique_ptr_scope()
                if hasattr(kernel, "params") and kernel.params:
                    for param in kernel.params:
                        self.register_unique_ptr_parameter(param)
                try:
                    if isinstance(kernel.body, list):
                        for stmt in kernel.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(kernel.body)
                finally:
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()

            self.indent_level -= 1
            self.emit("}")
        finally:
            self.pop_resource_object_hint_scope()

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
        var_type = self.convert_hip_variable_type_to_crossgl(
            getattr(node, "vtype", "int"), node.name
        )
        qualifiers = set(getattr(node, "qualifiers", []) or [])

        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(node.name, getattr(node, "vtype", "int"))
        if "__shared__" in qualifiers:
            self.emit(f"var<workgroup> {node.name}: {var_type};")
            return

        if "__constant__" in qualifiers:
            if hasattr(node, "value") and node.value:
                value = self.visit(node.value)
                self.emit(
                    f"@group(0) @binding(0) var<uniform> {node.name}: "
                    f"{var_type} = {value};"
                )
            else:
                self.emit(
                    f"@group(0) @binding(0) var<uniform> {node.name}: {var_type};"
                )
            return

        if "__managed__" in qualifiers:
            self.emit(f"// HIP managed memory: {node.name}")

        if hasattr(node, "value") and node.value:
            runtime_status = self.format_hip_runtime_status_expression(node.value)
            if runtime_status is not None:
                comments, value = runtime_status
                for comment in comments:
                    self.emit(comment)
                self.emit(f"var {node.name}: {var_type} = {value};")
                return

            value = self.visit(node.value)
            self.emit(f"var {node.name}: {var_type} = {value};")
        else:
            self.emit(f"var {node.name}: {var_type};")

    def visit_KernelLaunchNode(self, node):
        kernel_name = self.visit(node.kernel_name)
        config = [self.visit(node.blocks), self.visit(node.threads)]
        if node.shared_mem is not None:
            config.append(self.visit(node.shared_mem))
        if node.stream is not None:
            config.append(self.visit(node.stream))

        self.emit(f"// Kernel launch: {kernel_name}<<<{', '.join(config)}>>>()")
        if node.args:
            args = self.resolve_packed_launch_args(node.args)
            args_str = ", ".join([self.format_kernel_launch_arg(arg) for arg in args])
            self.emit(f"// Arguments: {args_str}")

    def push_packed_argument_scope(self):
        self.packed_argument_scopes.append({})

    def pop_packed_argument_scope(self):
        if self.packed_argument_scopes:
            self.packed_argument_scopes.pop()

    def push_unique_ptr_scope(self):
        self.unique_ptr_scopes.append(set())

    def pop_unique_ptr_scope(self):
        if len(self.unique_ptr_scopes) > 1:
            self.unique_ptr_scopes.pop()

    def push_type_alias_scope(self):
        self.type_alias_scopes.append({})

    def pop_type_alias_scope(self):
        if len(self.type_alias_scopes) > 1:
            self.type_alias_scopes.pop()

    def register_type_alias(self, name, alias_type):
        self.type_alias_scopes[-1][name] = alias_type

    def resolve_type_alias(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        for scope in reversed(self.type_alias_scopes):
            if type_name in scope:
                return scope[type_name]
        return type_name

    def register_unique_ptr_parameter(self, param):
        if isinstance(param, dict):
            self.register_unique_ptr_name(param.get("name", ""), param.get("type", ""))
        else:
            self.register_unique_ptr_name(
                getattr(param, "name", ""), getattr(param, "vtype", "")
            )

    def register_unique_ptr_name(self, name, type_name):
        if self.is_unique_ptr_type_name(type_name):
            self.unique_ptr_scopes[-1].add(name)

    def is_unique_ptr_expression(self, expr):
        if not isinstance(expr, str):
            return False
        return any(expr in scope for scope in reversed(self.unique_ptr_scopes))

    def register_packed_argument_list(self, node):
        if not self.packed_argument_scopes:
            return
        if self.is_packed_argument_list(node):
            self.packed_argument_scopes[-1][node.name] = (
                self.get_initializer_list_elements(node.value)
            )

    def is_packed_argument_list(self, node):
        if self.get_initializer_list_elements(getattr(node, "value", None)) is None:
            return False

        compact_type = getattr(node, "vtype", "").replace(" ", "")
        return compact_type in {"void*[]", "void**"}

    def get_initializer_list_elements(self, value):
        if isinstance(value, InitializerListNode):
            return value.elements
        if isinstance(value, CastNode) and isinstance(
            value.expression, InitializerListNode
        ):
            return value.expression.elements
        return None

    def resolve_packed_launch_args(self, args):
        if len(args) != 1:
            return args

        compound_elements = self.get_packed_compound_literal_elements(args[0])
        if compound_elements is not None:
            return compound_elements

        packed_arg_name = self.get_packed_argument_name(args[0])
        if packed_arg_name is None:
            return args

        for scope in reversed(self.packed_argument_scopes):
            if packed_arg_name in scope:
                return scope[packed_arg_name]

        return args

    def get_packed_argument_name(self, arg):
        if isinstance(arg, str):
            return arg
        if isinstance(arg, CastNode):
            return self.get_packed_argument_name(arg.expression)
        return None

    def get_packed_compound_literal_elements(self, arg):
        if not isinstance(arg, CastNode):
            return None

        compact_type = arg.target_type.replace(" ", "")
        if compact_type not in {"void*[]", "void**"}:
            return None

        return self.get_initializer_list_elements(arg.expression)

    def format_kernel_launch_arg(self, arg):
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def visit_AssignmentNode(self, node):
        left = self.visit(node.left)
        operator = getattr(node, "operator", "=")
        runtime_status = (
            self.format_hip_runtime_status_expression(node.right)
            if operator == "="
            else None
        )
        if runtime_status is not None:
            comments, value = runtime_status
            for comment in comments:
                self.emit(comment)
            self.emit(f"{left} = {value};")
            return

        right = self.visit(node.right)
        return f"{left} {operator} {right}"

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
        if self.is_get_method_call(node):
            return self.visit(node.name.object)

        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        if not isinstance(func_name, str):
            func_name = self.visit(func_name)

        if func_name == "lambda":
            return self.format_lambda_call(getattr(node, "args", []))

        args = []
        if hasattr(node, "args") and node.args:
            args = [self.visit(arg) for arg in node.args]
        elif hasattr(node, "arguments") and node.arguments:
            args = [self.visit(arg) for arg in node.arguments]

        args_str = ", ".join(args)

        make_unique = self.format_make_unique_call(func_name, args)
        if make_unique is not None:
            return make_unique

        unique_ptr_init = self.format_unique_ptr_constructor_call(func_name, args)
        if unique_ptr_init is not None:
            return unique_ptr_init

        if self.is_user_defined_function(func_name):
            return f"{func_name}({args_str})"

        resource_call = self.format_hip_resource_call(func_name, args)
        if resource_call is not None:
            return resource_call

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        return f"{crossgl_func}({args_str})"

    def format_hip_resource_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None

        value_type = template_args[0] if template_args else None
        if base_name in {"tex1D", "tex1DLod", "tex1DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec1", 1)
        if base_name in {"tex2D", "tex2DLod", "tex2DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec2", 2)
        if base_name in {"tex3D", "tex3DLod", "tex3DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3)
        if base_name in {"texCubemap", "texCubemapLod", "texCubemapGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3)
        if base_name in {"tex1DLayered", "tex1DLayeredLod", "tex1DLayeredGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec2", 2)
        if base_name in {"tex2DLayered", "tex2DLayeredLod", "tex2DLayeredGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3)
        if base_name in {
            "texCubemapLayered",
            "texCubemapLayeredLod",
            "texCubemapLayeredGrad",
        }:
            return self.format_hip_texture_call(base_name, args, "vec4", 4)

        if base_name in {
            "surf2Dread",
            "surf3Dread",
            "surf2DLayeredread",
            "surfCubemapread",
            "surfCubemapLayeredread",
        }:
            dimensions = {
                "surf2Dread": 2,
                "surf3Dread": 3,
                "surf2DLayeredread": 3,
                "surfCubemapread": 3,
                "surfCubemapLayeredread": 3,
            }[base_name]
            return self.format_hip_surface_read(args, dimensions, value_type)

        if base_name in {
            "surf2Dwrite",
            "surf3Dwrite",
            "surf2DLayeredwrite",
            "surfCubemapwrite",
            "surfCubemapLayeredwrite",
        }:
            dimensions = {
                "surf2Dwrite": 2,
                "surf3Dwrite": 3,
                "surf2DLayeredwrite": 3,
                "surfCubemapwrite": 3,
                "surfCubemapLayeredwrite": 3,
            }[base_name]
            return self.format_hip_surface_write(args, dimensions, value_type)

        return None

    def format_hip_texture_call(self, function_name, args, vector_name, dimensions):
        if len(args) < 2:
            return None

        extra_count = (
            2 if "Grad" in function_name else 1 if "Lod" in function_name else 0
        )
        coordinate_count = len(args) - 1 - extra_count
        if coordinate_count <= 0:
            return None

        texture_name = args[0]
        coordinate_args = args[1 : 1 + coordinate_count]
        if coordinate_count == 1:
            coordinate = coordinate_args[0]
            consumed = 2
        elif coordinate_count == dimensions:
            coordinate = self.format_vector_constructor(vector_name, coordinate_args)
            consumed = 1 + dimensions
        else:
            return None

        remaining = args[consumed:]
        if "Grad" in function_name:
            if len(remaining) < 2:
                return None
            return (
                f"textureGrad({texture_name}, {coordinate}, "
                f"{remaining[0]}, {remaining[1]})"
            )
        if "Lod" in function_name:
            if not remaining:
                return None
            return f"textureLod({texture_name}, {coordinate}, {remaining[0]})"
        return f"texture({texture_name}, {coordinate})"

    def format_hip_surface_read(self, args, dimensions, value_type):
        if len(args) < dimensions + 1:
            return None
        surface_name = args[0]
        coord_args = [self.strip_surface_byte_offset(args[1], value_type)]
        coord_args.extend(args[2 : dimensions + 1])
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        return f"imageLoad({surface_name}, {coord})"

    def format_hip_surface_write(self, args, dimensions, value_type):
        if len(args) < dimensions + 2:
            return None
        value = args[0]
        surface_name = args[1]
        coord_args = [self.strip_surface_byte_offset(args[2], value_type)]
        coord_args.extend(args[3 : dimensions + 2])
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        return f"imageStore({surface_name}, {coord}, {value})"

    def strip_surface_byte_offset(self, expression, value_type):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        if value_type:
            suffix = f" * sizeof({value_type})"
            if text.endswith(suffix):
                return text[: -len(suffix)].strip()

        marker = " * sizeof("
        if marker in text and text.endswith(")"):
            return text.split(marker, 1)[0].strip()

        return expression

    def format_vector_constructor(self, vector_name, args, element_type="f32"):
        if vector_name == "vec1":
            return args[0]
        return f"{vector_name}<{element_type}>({', '.join(args)})"

    def format_lambda_call(self, args):
        if not args:
            return "lambda()"

        rendered_args = [self.format_lambda_parameter(arg) for arg in args[:-1]] + [
            self.format_lambda_body(args[-1])
        ]
        return f"lambda({', '.join(rendered_args)})"

    def format_lambda_parameter(self, arg):
        if isinstance(arg, VariableNode):
            if arg.vtype:
                param_type = self.convert_hip_type_to_crossgl(arg.vtype)
                return f"{param_type} {arg.name}"
            return arg.name
        return self.format_lambda_body(arg)

    def format_lambda_body(self, arg):
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def visit_AtomicOperationNode(self, node):
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)
        crossgl_func = self.convert_hip_builtin_function(node.operation)
        return f"{crossgl_func}({args_str})"

    def is_get_method_call(self, node):
        return (
            isinstance(getattr(node, "name", None), MemberAccessNode)
            and node.name.member == "get"
            and not getattr(node, "args", [])
            and self.is_unique_ptr_expression(node.name.object)
        )

    def format_make_unique_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if not self.is_std_make_unique_base_name(base_name) or not template_args:
            return None

        target_type, is_array = self.unwrap_array_template_type(template_args[0])
        target_type = self.convert_hip_type_to_crossgl(target_type)
        args_str = ", ".join(args)
        if is_array:
            return f"new_array<{target_type}>({args_str})"
        return f"new<{target_type}>({args_str})"

    def format_unique_ptr_constructor_call(self, function_name, args):
        base_name, _ = self.parse_cpp_template(function_name)
        if len(args) != 1:
            return None
        if not self.is_std_unique_ptr_base_name(
            base_name
        ) and not self.is_unique_ptr_type_name(function_name):
            return None

        return args[0]

    def visit_NewNode(self, node):
        target_type = self.convert_hip_type_to_crossgl(node.target_type)
        if node.is_array:
            size = self.visit(node.size) if node.size is not None else ""
            return f"new_array<{target_type}>({size})"

        args = ", ".join(self.visit(arg) for arg in node.args)
        return f"new<{target_type}>({args})"

    def visit_DeleteNode(self, node):
        target = self.visit(node.expression)
        if node.is_array:
            self.emit(f"// delete array: {target}")
        else:
            self.emit(f"// delete: {target}")

    def visit_TypeAliasNode(self, node):
        self.register_type_alias(node.name, node.alias_type)
        alias_type = self.convert_hip_type_to_crossgl(node.alias_type)
        self.emit(f"typedef {alias_type} {node.name};")

    def visit_MemberAccessNode(self, node):
        obj = self.visit(node.object)
        operator = "->" if getattr(node, "is_pointer", False) else "."
        return f"{obj}{operator}{node.member}"

    def visit_ArrayAccessNode(self, node):
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_InitializerListNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_DesignatedInitializerNode(self, node):
        designators = []
        for kind, target in node.designators:
            if kind == "index":
                designators.append(f"[{self.visit(target)}]")
            else:
                designators.append(f".{target}")

        value = self.visit(node.value)
        return f"{''.join(designators)} = {value}"

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            self.emit("workgroupBarrier();")
        elif node.sync_type == "hipDeviceSynchronize":
            self.emit("// HIP device synchronize")
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
            "warpSize": "32",
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
        init_node = node.init if hasattr(node, "init") else None
        scoped_init = isinstance(init_node, list)
        if scoped_init:
            self.emit("{")
            self.indent_level += 1
            for stmt in init_node:
                self.emit_statement(stmt)
            init = ""
        else:
            init = self.format_statement_fragment(init_node)
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
        if scoped_init:
            self.indent_level -= 1
            self.emit("}")

    def visit_RangeForNode(self, node):
        iterable = self.visit(node.iterable)
        self.emit(f"for {node.name} in {iterable} {{")

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
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    self.emit("default:")
                    self.indent_level += 1
                    for stmt in getattr(case, "body", []):
                        self.emit_statement(stmt)
                    self.indent_level -= 1
                else:
                    self.visit(case)
        else:
            for case in getattr(node, "cases", []):
                self.visit(case)

            if getattr(node, "default_case", None) is not None:
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

    def convert_hip_variable_type_to_crossgl(self, hip_type, name):
        """Convert HIP variable types, using call-site hints for resource handles."""
        resource_type = self.convert_hip_resource_object_type(hip_type, name)
        if resource_type is not None:
            return resource_type
        return self.convert_hip_type_to_crossgl(hip_type)

    def convert_hip_resource_object_type(self, hip_type, name):
        hint = self.lookup_resource_object_type_hint(name)
        if hint is None:
            return None
        return self.convert_hip_resource_object_type_with_hint(hip_type, hint)

    def convert_hip_resource_object_type_with_hint(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)

        if self.has_array_suffix(hip_type):
            base_type = hip_type.split("[", 1)[0].strip()
            mapped_type = self.convert_hip_resource_object_type_with_hint(
                base_type, hint
            )
            if mapped_type is None:
                return None
            return self.wrap_mapped_hip_array_type(hip_type, mapped_type)

        if "*" in hip_type:
            pointer_depth = hip_type.count("*")
            base_type = hip_type.replace("*", "").strip()
            mapped_type = self.convert_hip_resource_object_base_type(base_type, hint)
            if mapped_type is None:
                return None
            for _ in range(pointer_depth):
                mapped_type = f"ptr<{mapped_type}>"
            return mapped_type

        return self.convert_hip_resource_object_base_type(hip_type, hint)

    def convert_hip_resource_object_base_type(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)
        if hip_type == "hipTextureObject_t" and hint.startswith("sampler"):
            return hint
        if hip_type == "hipSurfaceObject_t" and "image" in hint:
            return hint
        return None

    def wrap_mapped_hip_array_type(self, hip_type, mapped_type):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        for size in reversed(dimensions):
            if size:
                mapped_type = f"array<{mapped_type}, {size}>"
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_hip_type_to_crossgl(self, hip_type):
        """Map a HIP type name to the closest CrossGL type name."""
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
            "signed int": "i32",
            "unsigned int": "u32",
            "long": "i64",
            "signed long": "i64",
            "unsigned long": "u64",
            "long long": "i64",
            "signed long long": "i64",
            "unsigned long long": "u64",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            "hipTextureObject_t": "sampler",
            "hipSurfaceObject_t": "image2D",
            # HIP vector types
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }

        unique_ptr_type = self.convert_unique_ptr_type(hip_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        resource_type = self.convert_hip_resource_type(hip_type)
        if resource_type is not None:
            return resource_type

        # Handle arrays
        if self.has_array_suffix(hip_type):
            return self.convert_hip_array_type(hip_type, type_mapping)

        # Handle pointers
        if "*" in hip_type:
            return self.convert_hip_pointer_type(hip_type)

        return type_mapping.get(hip_type, hip_type)

    def convert_hip_resource_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if base_name == "texture" and len(template_args) >= 2:
            return self.HIP_TEXTURE_TYPE_MAPPING.get(template_args[1])
        if base_name == "surface" and len(template_args) >= 2:
            return self.HIP_SURFACE_TYPE_MAPPING.get(template_args[1])
        return None

    def convert_unique_ptr_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if not self.is_unique_ptr_base_name(base_name) or not template_args:
            return None

        target_type, _ = self.unwrap_array_template_type(template_args[0])
        return f"ptr<{self.convert_hip_type_to_crossgl(target_type)}>"

    def is_unique_ptr_type_name(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        type_name = self.resolve_type_alias(type_name)
        base_name, template_args = self.parse_cpp_template(type_name)
        return self.is_unique_ptr_base_name(base_name) and bool(template_args)

    def is_unique_ptr_base_name(self, base_name):
        return self.is_std_unique_ptr_base_name(base_name)

    def is_std_unique_ptr_base_name(self, base_name):
        return base_name in {"unique_ptr", "std::unique_ptr"}

    def is_std_make_unique_base_name(self, base_name):
        return base_name in {"make_unique", "std::make_unique"}

    def has_array_suffix(self, type_name):
        depth = 0
        for char in str(type_name):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "[" and depth == 0:
                return True
        return False

    def unwrap_array_template_type(self, type_name):
        type_name = type_name.strip()
        if type_name.endswith("[]"):
            return type_name[:-2].strip(), True
        return type_name, False

    def parse_cpp_template(self, text):
        if not isinstance(text, str):
            return str(text), []

        start = text.find("<")
        if start == -1 or not text.endswith(">"):
            return text, []

        base_name = text[:start].strip()
        args = self.split_cpp_template_args(text[start + 1 : -1])
        return base_name, args

    def split_cpp_template_args(self, args_text):
        args = []
        depth = 0
        start = 0

        for index, char in enumerate(args_text):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(args_text[start:index].strip())
                start = index + 1

        tail = args_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    def convert_hip_pointer_type(self, hip_type):
        """Convert a HIP pointer type into nested CrossGL pointer syntax."""
        pointer_depth = hip_type.count("*")
        base_type = hip_type.replace("*", "").strip()
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(pointer_depth):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def convert_hip_pointer_element_type(self, hip_type):
        pointer_depth = hip_type.count("*")
        base_type = hip_type.replace("*", "").strip()
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(max(0, pointer_depth - 1)):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def strip_type_qualifiers(self, type_name):
        qualifiers = {"const", "volatile", "__restrict__", "restrict", "&", "&&"}
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def convert_hip_array_type(self, hip_type, type_mapping):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        mapped_type = type_mapping.get(base_type)
        if mapped_type is None:
            mapped_type = self.convert_hip_type_to_crossgl(base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = f"array<{mapped_type}, {size}>"
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_hip_builtin_function(self, func_name):
        """Convert HIP built-in functions to CrossGL equivalents."""
        function_mapping = {
            # Math functions
            "sqrtf": "sqrt",
            "powf": "pow",
            "sinf": "sin",
            "cosf": "cos",
            "tanf": "tan",
            "sinhf": "sinh",
            "coshf": "cosh",
            "tanhf": "tanh",
            "asinhf": "asinh",
            "acoshf": "acosh",
            "atanhf": "atanh",
            "asinf": "asin",
            "acosf": "acos",
            "atanf": "atan",
            "logf": "log",
            "log2f": "log2",
            "expf": "exp",
            "exp2f": "exp2",
            "fabsf": "abs",
            "rsqrtf": "inversesqrt",
            "roundf": "round",
            "truncf": "trunc",
            "atan2f": "atan2",
            "fmodf": "mod",
            "fminf": "min",
            "fmaxf": "max",
            "lerp": "mix",
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
            "rsqrt": "inversesqrt",
            "fmod": "mod",
            "fmin": "min",
            "fmax": "max",
            "floor": "floor",
            "ceil": "ceil",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "int": "i32",
            "long": "i64",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            # Vector functions
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
            # Sync functions
            "__syncthreads": "workgroupBarrier",
            "__threadfence": "memoryBarrier",
            "__threadfence_block": "memoryBarrier",
            "__threadfence_system": "memoryBarrier",
            # Atomic functions
            "atomicAdd": "atomicAdd",
            "hipAtomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "hipAtomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "hipAtomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "hipAtomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "hipAtomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
            "hipAtomicCAS": "atomicCompareExchange",
            "atomicAnd": "atomicAnd",
            "hipAtomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "hipAtomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "hipAtomicXor": "atomicXor",
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
