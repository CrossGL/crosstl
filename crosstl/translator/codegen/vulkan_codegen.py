import logging
from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    VariableNode,
)


class VulkanSPIRVCodeGen:
    def __init__(self):
        self.id_counter = 1
        self.type_ids = {}
        self.variable_ids = {}
        self.function_ids = {}
        self.function_types = {}
        self.current_shader = None
        self.shader_inputs = []
        self.shader_outputs = []

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            self.current_shader = ast
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        # Extract struct information to determine inputs and outputs
        self.shader_inputs = []
        self.shader_outputs = []

        # Analyze structures to identify inputs and outputs
        for struct in node.structs:
            if struct.name.startswith("VS"):
                for member in struct.members:
                    if hasattr(member, "semantic") and member.semantic:
                        self.shader_inputs.append((member.vtype, member.name))
            elif struct.name.endswith("Output"):
                for member in struct.members:
                    if hasattr(member, "semantic") and member.semantic:
                        self.shader_outputs.append((member.vtype, member.name))

        self.id_counter = 1
        code = "; SPIR-V\n"
        code += "; Version: 1.0\n"
        code += "; Generator: Custom Vulkan SPIR-V CodeGen\n"
        code += f"; Bound: {self.id_counter + 100}\n"
        code += "; Schema: 0\n"
        code += "OpCapability Shader\n"
        code += '%1 = OpExtInstImport "GLSL.std.450"\n'
        code += "OpMemoryModel Logical GLSL450\n"

        # EntryPoint
        entry_point_args = " ".join(
            f"%{name}" for _, name in self.shader_inputs + self.shader_outputs
        )
        code += f'OpEntryPoint Fragment %main "main" {entry_point_args}\n'

        code += "OpExecutionMode %main OriginUpperLeft\n"
        code += "OpSource GLSL 450\n"

        # Names and decorations
        code += 'OpName %main "main"\n'
        for _, name in self.shader_inputs + self.shader_outputs:
            code += f'OpName %{name} "{name}"\n'

        for i, (_, name) in enumerate(self.shader_inputs):
            code += f"OpDecorate %{name} Location {i}\n"
        for i, (_, name) in enumerate(self.shader_outputs):
            code += f"OpDecorate %{name} Location {i}\n"

        # Type declarations
        code += self.declare_types()

        # Global variable declarations
        code += self.declare_global_variables()

        # Constants
        code += self.declare_constants()

        # Function declarations
        for function in node.functions:
            code += self.declare_function(function)

        # Function definitions
        for function in node.functions:
            code += self.generate_function(function)

        return code

    def declare_types(self):
        code = "%void = OpTypeVoid\n"
        code += "%bool = OpTypeBool\n"
        code += "%float = OpTypeFloat 32\n"
        code += "%int = OpTypeInt 32 1\n"
        code += "%uint = OpTypeInt 32 0\n"

        for vtype in set(
            vtype for vtype, _ in self.shader_inputs + self.shader_outputs
        ):
            components = self.map_type(vtype)
            self.type_ids[vtype] = self.get_id()
            code += f"%{vtype} = OpTypeVector %float {components}\n"

        return code

    def declare_global_variables(self):
        code = ""
        for vtype, name in self.shader_outputs:
            self.get_id()
            code += f"%_ptr_Output_{vtype} = OpTypePointer Output %{vtype}\n"
            self.variable_ids[name] = self.get_id()
            code += f"%{name} = OpVariable %_ptr_Output_{vtype} Output\n"

        for vtype, name in self.shader_inputs:
            self.get_id()
            code += f"%_ptr_Input_{vtype} = OpTypePointer Input %{vtype}\n"
            self.variable_ids[name] = self.get_id()
            code += f"%{name} = OpVariable %_ptr_Input_{vtype} Input\n"

        return code

    def declare_constants(self):
        code = "%float_0 = OpConstant %float 0\n"
        code += "%float_1 = OpConstant %float 1\n"
        code += "%int_0 = OpConstant %int 0\n"
        code += "%int_1 = OpConstant %int 1\n"
        code += "%int_3 = OpConstant %int 3\n"
        return code

    def declare_function(self, node):
        return_type = self.map_type(node.return_type)
        param_types = [self.map_type(param.vtype) for param in node.params]

        # Handle parameters which are VariableNode objects
        param_types = []
        if node.params:
            if isinstance(node.params[0], VariableNode):
                # Handle list of VariableNode objects
                param_types = [self.map_type(param.vtype) for param in node.params]
            else:
                # Handle tuples of (type, name)
                param_types = [self.map_type(param[0]) for param in node.params]

        function_type_id = self.get_id()
        self.function_types[node.name] = function_type_id

        code = f"%{function_type_id} = OpTypeFunction %{return_type}"
        if param_types:
            code += " " + " ".join(f"%{ptype}" for ptype in param_types)
        code += "\n"

        return code

    def generate_function(self, node):
        self.function_ids[node.name] = self.get_id()
        return_type = self.map_type(node.return_type)

        code = f"%{self.function_ids[node.name]} = OpFunction %{return_type} None %{self.function_types[node.name]}\n"

        # Generate parameters
        if node.params:
            if isinstance(node.params[0], VariableNode):
                # Handle list of VariableNode objects
                for param in node.params:
                    param_id = self.get_id()
                    code += f"%{param_id} = OpFunctionParameter %{self.map_type(param.vtype)}\n"
                    self.variable_ids[param.name] = param_id
            else:
                # Handle tuples of (type, name)
                for param_type, param_name in node.params:
                    param_id = self.get_id()
                    code += f"%{param_id} = OpFunctionParameter %{self.map_type(param_type)}\n"
                    self.variable_ids[param_name] = param_id

        code += f"%{self.get_id()} = OpLabel\n"

        for stmt in node.body:
            code += self.generate_statement(stmt)

        if node.return_type == "void":
            code += "OpReturn\n"
        else:
            code += "OpUnreachable\n"  # Add proper return handling if needed
        code += "OpFunctionEnd\n"

        return code

    def generate_statement(self, stmt):
        if isinstance(stmt, AssignmentNode):
            return self.generate_assignment(stmt)
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt)
        elif isinstance(stmt, ReturnNode):
            return f"OpReturnValue {self.generate_expression(stmt.value)}\n"
        else:
            return f"{self.generate_expression(stmt)}\n"

    def generate_assignment(self, node):
        if node.name in [var[1] for var in self.shader_outputs]:
            value_id = self.generate_expression(node.value)
            return f"OpStore %{node.name} {value_id}\n"
        else:
            var_id = self.get_id()
            value_id = self.generate_expression(node.value)
            var_type = self.get_variable_type(node.name)
            return f"%{var_id} = OpVariable %_ptr_Function_{var_type} Function\nOpStore %{var_id} {value_id}\n"

    def generate_if(self, node):
        condition_id = self.generate_expression(node.condition)
        then_label = self.get_id()
        else_label = self.get_id()
        merge_label = self.get_id()

        code = f"OpSelectionMerge %{merge_label} None\n"
        code += f"OpBranchConditional {condition_id} %{then_label} %{else_label}\n"

        code += f"%{then_label} = OpLabel\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt)
        code += f"OpBranch %{merge_label}\n"

        code += f"%{else_label} = OpLabel\n"
        if node.else_body:
            for stmt in node.else_body:
                code += self.generate_statement(stmt)
        code += f"OpBranch %{merge_label}\n"

        code += f"%{merge_label} = OpLabel\n"
        return code

    def generate_for(self, node):
        init_code = self.generate_statement(node.init)
        condition_id = self.generate_expression(node.condition)
        update_code = self.generate_statement(node.update)

        header_label = self.get_id()
        body_label = self.get_id()
        continue_label = self.get_id()
        merge_label = self.get_id()

        code = init_code
        code += f"OpBranch %{header_label}\n"
        code += f"%{header_label} = OpLabel\n"
        code += f"OpLoopMerge %{merge_label} %{continue_label} None\n"
        code += f"OpBranchConditional {condition_id} %{body_label} %{merge_label}\n"

        code += f"%{body_label} = OpLabel\n"
        for stmt in node.body:
            code += self.generate_statement(stmt)
        code += f"OpBranch %{continue_label}\n"

        code += f"%{continue_label} = OpLabel\n"
        code += update_code
        code += f"OpBranch %{header_label}\n"

        code += f"%{merge_label} = OpLabel\n"
        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return self.translate_expression(expr)
        elif isinstance(expr, VariableNode):
            return self.translate_expression(expr.name)
        elif isinstance(expr, BinaryOpNode):
            left_id = self.generate_expression(expr.left)
            right_id = self.generate_expression(expr.right)
            result_id = self.get_id()
            op = self.map_operator(expr.op)
            # Determine result type - default to float for simplicity
            result_type = "float"
            if hasattr(expr, "type"):
                logging.debug(f"Skipping unexpected token {self.current_token[0]}")

            return f"%{result_id} = {op} %{result_type} {left_id} {right_id}\n"
        elif isinstance(expr, FunctionCallNode):
            if expr.name in ["vec2", "vec3", "vec4"]:
                # Handle vector constructors
                args = [self.generate_expression(arg) for arg in expr.args]
                result_id = self.get_id()
                components = " ".join(args)
                return (
                    f"%{result_id} = OpCompositeConstruct %{expr.name} {components}\n"
                )
            else:
                result_id = self.get_id()
                args = [self.generate_expression(arg) for arg in expr.args]
                arg_list = " ".join(args)
                return f"%{result_id} = OpFunctionCall %{self.function_ids[expr.name]} {arg_list}\n"
        elif isinstance(expr, MemberAccessNode):
            return self.translate_expression(expr)
        else:
            return f"; Unhandled expression: {expr}\n"

    def translate_expression(self, expr):
        if isinstance(expr, MemberAccessNode):
            # Handle member access expressions
            obj = self.generate_expression(expr.object)
            member = expr.member
            result_id = self.get_id()
            return f"%{result_id} = OpAccessChain %{obj} %{member}\n"
        elif isinstance(expr, str):
            # Handle string expressions
            if expr in self.variable_ids:
                return f"%{self.variable_ids[expr]}"
            elif expr.startswith("vec"):
                # Vector constructor
                components = int(expr[3:])
                result_id = self.get_id()
                init_values = " ".join(
                    [f"%float_0" for _ in range(components - 1)] + ["%float_1"]
                )
                return f"%{result_id} = OpCompositeConstruct %{expr} {init_values}\n"
            else:
                return expr
        else:
            return str(expr)

    def map_type(self, vtype):
        type_mapping = {
            "void": "void",
            "bool": "bool",
            "int": "int",
            "float": "float",
            "vec2": "v2float",
            "vec3": "v3float",
            "vec4": "v4float",
            "mat2": "mat2v2float",
            "mat3": "mat3v3float",
            "mat4": "mat4v4float",
        }
        return type_mapping.get(vtype, vtype)

    def map_operator(self, op):
        op_map = {
            "PLUS": "OpFAdd",
            "MINUS": "OpFSub",
            "MULTIPLY": "OpFMul",
            "DIVIDE": "OpFDiv",
            "LESS_THAN": "OpFOrdLessThan",
            "GREATER_THAN": "OpFOrdGreaterThan",
            "LESS_EQUAL": "OpFOrdLessThanEqual",
            "GREATER_EQUAL": "OpFOrdGreaterThanEqual",
            "EQUAL": "OpFOrdEqual",
            "NOT_EQUAL": "OpFOrdNotEqual",
            "AND": "OpLogicalAnd",
            "OR": "OpLogicalOr",
        }
        return op_map.get(op, op)

    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return id

    def get_function_return_type(self, function_name):
        # This should be implemented based on how you store function information
        # For now, we'll return a default type
        return "float"

    def get_variable_type(self, variable_name):
        # This should be implemented based on how you store variable information
        # For now, we'll return a default type
        return "float"
