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

    def __init__(self):
        """Initialize HIP-to-CrossGL visitor state."""
        self.indent_level = 0
        self.output = []
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.user_function_names = set()

    def generate(self, ast_node):
        """Generate complete CrossGL source from a parsed HIP AST."""
        self.output = []
        self.indent_level = 0
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.user_function_names = self.collect_user_function_names(ast_node)
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

    def visit_FunctionNode(self, node):
        """Render a HIP function node as a CrossGL function."""
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

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Render a HIP kernel as a CrossGL compute shader block."""
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

        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(node.name, getattr(node, "vtype", "int"))
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

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        return f"{crossgl_func}({args_str})"

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

        unique_ptr_type = self.convert_unique_ptr_type(hip_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        # Handle arrays
        if self.has_array_suffix(hip_type):
            return self.convert_hip_array_type(hip_type, type_mapping)

        # Handle pointers
        if "*" in hip_type:
            return self.convert_hip_pointer_type(hip_type)

        return type_mapping.get(hip_type, hip_type)

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
