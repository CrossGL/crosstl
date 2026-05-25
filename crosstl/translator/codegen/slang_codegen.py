"""CrossGL-to-Slang code generator."""

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    FunctionNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    DoWhileNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import (
    format_c_style_array_declaration,
    get_array_size_from_node,
    split_array_type_suffix,
)


class SlangCodeGen:
    """Emit Slang shader source from the shared CrossGL AST."""

    BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}

    def __init__(self):
        """Initialize Slang generation state and helper caches."""
        self.indent_level = 0
        self.indent_str = "    "
        self.variable_types = {}
        self.image_resource_types = {}
        self.helper_functions = {}
        self.helper_name_aliases = {}
        self.user_symbol_names = set()
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.user_function_names = set()
        self.user_function_return_types = {}
        self._generating = False
        self.function_map = {
            "mix": "lerp",
            "mod": "fmod",
            "fract": "frac",
            "inversesqrt": "rsqrt",
            "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
        }

    def indent(self):
        """Return whitespace for the current indentation level."""
        return self.indent_str * self.indent_level

    def generate(self, ast):
        """Generate Slang source for a CrossGL AST or AST fragment."""
        outermost = not self._generating
        if outermost:
            self._generating = True
            self.variable_types = {}
            self.image_resource_types = {}
            self.helper_functions = {}
            self.helper_name_aliases = {}
            self.user_symbol_names = self.collect_user_symbol_names(ast)
            self.current_function_return_type = None
            self.current_expression_expected_type = None
            self.user_function_names = self.collect_user_function_names(ast)
            self.user_function_return_types = self.collect_user_function_return_types(
                ast
            )

        if isinstance(ast, list):
            result = ""
            for node in ast:
                result += self.generate(node) + "\n"
            return self.finish_generation(result, outermost)
        elif isinstance(ast, ShaderNode):
            return self.finish_generation(self.generate_shader(ast), outermost)
        elif isinstance(ast, StructNode):
            return self.finish_generation(self.generate_struct(ast), outermost)
        else:
            # Handle new AST structure
            result = ""

            structs = getattr(ast, "structs", [])
            for struct in structs:
                result += self.generate_struct(struct) + "\n\n"

            global_vars = getattr(ast, "global_variables", [])
            for node in global_vars:
                result += self.generate_global_variable(node)

            cbuffers = getattr(ast, "cbuffers", [])
            for node in cbuffers:
                if isinstance(node, StructNode):
                    result += (
                        "cbuffer " + self.generate_struct_definition(node) + "\n\n"
                    )
                elif hasattr(node, "name") and hasattr(node, "members"):
                    result += f"cbuffer {node.name} {{\n"
                    for member in node.members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        result += (
                            f"    {self.convert_type(member_type)} {member.name};\n"
                        )
                    result += "};\n\n"

            functions = getattr(ast, "functions", [])
            for function in functions:
                # Handle both old and new AST function structures
                if hasattr(function, "qualifiers") and function.qualifiers:
                    qualifier = function.qualifiers[0] if function.qualifiers else None
                else:
                    qualifier = getattr(function, "qualifier", None)

                if qualifier == "vertex":
                    result += "// Vertex Shader\n"
                    result += self.generate_function(function) + "\n\n"
                elif qualifier == "fragment":
                    result += "// Fragment Shader\n"
                    result += self.generate_function(function) + "\n\n"
                else:
                    result += self.generate_function(function) + "\n\n"

            # Handle shader stages (new AST structure)
            if hasattr(ast, "stages") and ast.stages:
                for stage_type, stage in ast.stages.items():
                    result += self.generate_stage(stage_type, stage)

            return self.finish_generation(result, outermost)

    def finish_generation(self, result, outermost):
        if not outermost:
            return result

        helpers = self.emit_helper_functions()
        self._generating = False
        if helpers:
            return helpers + result
        return result

    def emit_helper_functions(self):
        if not self.helper_functions:
            return ""
        return "\n\n".join(self.helper_functions.values()) + "\n\n"

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_symbol_names(self, node):
        names = set()

        def add_name(current):
            name = getattr(current, "name", None)
            if name:
                names.add(name)

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, (FunctionNode, StructNode)):
                add_name(current)
            for attr in ("global_variables", "cbuffers"):
                for declaration in getattr(current, attr, []) or []:
                    add_name(declaration)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    for declaration in getattr(stage, "local_variables", []) or []:
                        add_name(declaration)
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                return_types[current.name] = (
                    self.convert_type_node_to_string(return_type)
                    if return_type is not None
                    else "void"
                )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return_types.pop(None, None)
        return return_types

    def generate_shader(self, node):
        """Render a full CrossGL shader AST as a Slang translation unit."""
        result = ""

        structs = getattr(node, "structs", [])
        for struct in structs:
            result += self.generate_struct(struct) + "\n\n"

        global_vars = getattr(node, "global_variables", [])
        for global_var in global_vars:
            result += self.generate_global_variable(global_var)

        functions = getattr(node, "functions", [])
        for function in functions:
            stage_name = self.get_function_stage(function)
            if stage_name:
                result += f"// {stage_name.title()} Shader\n"
                result += self.generate_function(function, shader_type=stage_name)
                result += "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        stages = getattr(node, "stages", {})
        for stage_type, stage in stages.items():
            result += self.generate_stage(stage_type, stage)

        return result

    def get_stage_name(self, stage_type):
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def get_function_stage(self, function):
        if hasattr(function, "qualifiers") and function.qualifiers:
            qualifier = function.qualifiers[0]
        else:
            qualifier = getattr(function, "qualifier", None)

        if qualifier in {"vertex", "fragment", "compute"}:
            return qualifier
        return None

    def generate_stage(self, stage_type, stage):
        """Render one staged entry point and its local functions."""
        stage_name = self.get_stage_name(stage_type)
        result = f"// {stage_name.title()} Shader\n"

        local_variables = getattr(stage, "local_variables", [])
        for local_var in local_variables:
            result += self.generate_global_variable(local_var)

        for func in getattr(stage, "local_functions", []):
            result += self.generate_function(func) + "\n\n"

        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            result += self.generate_function(entry_point, shader_type=stage_name)
            result += "\n\n"

        return result

    def convert_type_node_to_string(self, type_node) -> str:
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        if hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if element_type == "float":
                if type_node.rows == type_node.cols:
                    return f"mat{type_node.rows}"
                return f"mat{type_node.rows}x{type_node.cols}"
            return f"{element_type}{type_node.rows}x{type_node.cols}"
        if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.__class__.__name__ == "ArrayType":
                if type_node.size is None:
                    return f"{element_type}[]"
                size = self.format_array_size_expression(type_node.size)
                return f"{element_type}[{size}]"
            if element_type == "float":
                return f"vec{type_node.size}"
            if element_type == "int":
                return f"ivec{type_node.size}"
            if element_type == "uint":
                return f"uvec{type_node.size}"
            if element_type == "bool":
                return f"bvec{type_node.size}"
            return f"{element_type}{type_node.size}"
        return str(type_node)

    def format_array_size_expression(self, expr):
        if isinstance(expr, int):
            return str(expr)
        if isinstance(expr, BinaryOpNode):
            left = self.format_array_size_expression(expr.left)
            right = self.format_array_size_expression(expr.right)
            return f"({left} {expr.op} {right})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.op}{self.format_array_size_expression(expr.operand)}"
        return self.generate_expression(expr)

    def format_declaration(self, type_name, name, node=None):
        mapped_type = self.map_resource_type_with_format(type_name, node)
        return format_c_style_array_declaration(mapped_type, name)

    def get_variable_type(self, node):
        var_type = getattr(node, "var_type", None)
        if var_type is not None:
            return self.convert_type_node_to_string(var_type)

        vtype = getattr(node, "vtype", None)
        if vtype is not None and vtype != "":
            return vtype

        return None

    def variable_declaration_type(self, node, initial_value=None):
        var_type = self.get_variable_type(node)
        if var_type is not None:
            return var_type
        if initial_value is not None:
            return self.expression_result_type(initial_value) or "auto"
        return "float"

    def initializer_expected_type(self, var_type):
        return None if var_type == "auto" else var_type

    def register_variable_type(self, name, type_name, node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        if self.is_storage_image_type(type_name):
            self.image_resource_types[name] = self.map_resource_type_with_format(
                type_name, node
            )

    def generate_global_variable(self, node):
        if isinstance(node, ArrayNode):
            self.register_variable_type(node.name, node.element_type)
            element_type = self.convert_type(node.element_type)
            size = get_array_size_from_node(node)
            if size is None:
                return f"{element_type} {node.name}[];\n"
            return f"{element_type} {node.name}[{size}];\n"

        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        vtype = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, vtype, node)
        declaration = self.format_declaration(vtype, node.name, node)
        if initial_value is not None:
            initial_expr = self.generate_expression_with_expected(
                initial_value,
                self.initializer_expected_type(vtype),
            )
            return f"{declaration} = {initial_expr};\n"
        return f"{declaration};\n"

    def generate_struct(self, node):
        result = f"struct {node.name}\n{{\n"
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type(
                    self.convert_type_node_to_string(member.member_type)
                )
            elif hasattr(member, "vtype"):
                member_type = self.convert_type(member.vtype)
            else:
                member_type = "float"

            semantic = None
            if hasattr(member, "semantic"):
                semantic = member.semantic
            elif hasattr(member, "attributes"):
                for attr in member.attributes:
                    if hasattr(attr, "name") and attr.name in [
                        "position",
                        "color",
                        "texcoord",
                        "normal",
                    ]:
                        semantic = attr.name
                        break

            semantic_str = f" : {semantic}" if semantic else ""
            declaration = self.format_declaration(member_type, member.name)
            result += f"{self.indent()}{declaration}{semantic_str};\n"

        self.indent_level -= 1
        result += "};"
        return result

    def generate_struct_definition(self, node):
        result = f"{node.name}\n{{\n"

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            result += f"    {self.format_declaration(member_type, member.name)};\n"

        result += "};"
        return result

    def generate_function(self, node, shader_type=None):
        """Render one CrossGL function or shader entry point as Slang code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_types = self.image_resource_types.copy()
        saved_function_return_type = self.current_function_return_type
        if hasattr(node, "return_type"):
            ret_type_name = self.convert_type_node_to_string(node.return_type)
            ret_type = self.convert_type(ret_type_name)
        else:
            ret_type_name = "void"
            ret_type = "void"
        self.current_function_return_type = ret_type_name

        semantic = None
        if hasattr(node, "semantic"):
            semantic = node.semantic
        elif hasattr(node, "attributes"):
            for attr in node.attributes:
                if hasattr(attr, "name"):
                    semantic = attr.name
                    break

        semantic_str = f" : {semantic}" if semantic else ""

        param_list = getattr(node, "parameters", getattr(node, "params", []))
        params_str = ""
        if param_list:
            if param_list and hasattr(param_list[0], "name"):
                params = []
                for param in param_list:
                    if hasattr(param, "param_type"):
                        param_type_name = self.convert_type_node_to_string(
                            param.param_type
                        )
                        self.register_variable_type(param.name, param_type_name, param)
                        param_type = self.map_resource_type_with_format(
                            param_type_name, param
                        )
                    elif hasattr(param, "vtype"):
                        self.register_variable_type(param.name, param.vtype, param)
                        param_type = self.map_resource_type_with_format(
                            param.vtype, param
                        )
                    else:
                        param_type = "float"
                    params.append(
                        format_c_style_array_declaration(param_type, param.name)
                    )
                params_str = ", ".join(params)
            else:
                for param_type, param_name in param_list:
                    self.register_variable_type(param_name, param_type)
                params_str = ", ".join(
                    [
                        f"{self.map_resource_type_with_format(param_type)} {param_name}"
                        for param_type, param_name in param_list
                    ]
                )

        result = ""
        if shader_type:
            result += f'[shader("{shader_type}")]\n'
        result += f"{ret_type} {node.name}({params_str}){semantic_str}\n{{\n"
        self.indent_level += 1

        body = getattr(node, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                result += self.emit_statement(stmt) + "\n"
        elif isinstance(body, list):
            for stmt in body:
                result += self.emit_statement(stmt) + "\n"

        self.indent_level -= 1
        result += "}"
        self.variable_types = saved_variable_types
        self.image_resource_types = saved_image_resource_types
        self.current_function_return_type = saved_function_return_type
        return result

    def emit_statement(self, node):
        statement = self.generate_statement(node)
        lines = statement.splitlines()
        return "\n".join(
            self.indent() + line if line and not line[0].isspace() else line
            for line in lines
        )

    def generate_statement(self, node):
        """Render a single CrossGL statement as Slang code."""
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            return (
                "return "
                f"{self.generate_expression_with_expected(node.value, self.current_function_return_type)};"
            )
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node) + ";"
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression) + ";"
        elif isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            var_type = self.variable_declaration_type(node, initial_value)
            self.register_variable_type(node.name, var_type, node)
            declaration = self.format_declaration(var_type, node.name, node)
            if initial_value is not None:
                initial_expr = self.generate_expression_with_expected(
                    initial_value,
                    self.initializer_expected_type(var_type),
                )
                return f"{declaration} = {initial_expr};"
            return f"{declaration};"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, ForInNode):
            return self.generate_for_in(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, DoWhileNode):
            return self.generate_do_while(node)
        elif isinstance(node, MatchNode):
            return self.generate_match(node)
        elif isinstance(node, SwitchNode):
            return self.generate_switch(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        else:
            return self.generate_expression(node) + ";"

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression_with_expected(
            node.right, self.expression_result_type(node.left)
        )
        if node.operator == "%=" and self.modulo_requires_fmod(node.left, node.right):
            return f"{left} = fmod({left}, {right})"
        return f"{left} {node.operator} {right}"

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def type_name_string(self, type_name):
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def is_scalar_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "float2",
            "float3",
            "float4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def vector_component_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def vector_value_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.convert_type(type_name)
        for component_type in ("double", "float", "uint", "int", "bool"):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if suffix in {"2", "3", "4"}:
                size = int(suffix)
                return {
                    "type": mapped_type,
                    "component_type": component_type,
                    "size": size,
                    "components": ("x", "y", "z", "w")[:size],
                }
        return None

    def generate_bool_mix_call(self, args):
        if len(args) != 3:
            return None

        condition_type = self.expression_result_type(args[2])
        condition_info = self.vector_value_info(condition_type)
        if condition_info is not None:
            if condition_info["component_type"] != "bool":
                return None
            return self.generate_bool_vector_select_expression(
                args[2], args[1], args[0]
            )

        if self.convert_type(condition_type) != "bool":
            return None

        condition = self.generate_expression(args[2])
        true_value = self.generate_expression(args[1])
        false_value = self.generate_expression(args[0])
        return f"({condition} ? {true_value} : {false_value})"

    def generate_bool_vector_select_expression(
        self, condition_expr, true_expr, false_expr
    ):
        condition_info = self.vector_value_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        true_info = self.vector_value_info(self.expression_result_type(true_expr))
        false_info = self.vector_value_info(self.expression_result_type(false_expr))
        if (
            true_info is None
            or false_info is None
            or true_info["type"] != false_info["type"]
            or condition_info["size"] != true_info["size"]
        ):
            return None

        helper_name = self.require_vector_select_helper(true_info, condition_info)
        condition = self.generate_expression(condition_expr)
        true_value = self.generate_expression(true_expr)
        false_value = self.generate_expression(false_expr)
        return f"{helper_name}({condition}, {true_value}, {false_value})"

    def require_vector_select_helper(self, result_info, condition_info):
        result_type = result_info["type"]
        condition_type = condition_info["type"]
        base_name = f"_crossgl_select_{condition_type}_{result_type}"
        helper_name = self.helper_function_name(base_name)
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            f"(mask.{component} ? trueValue.{component} : falseValue.{component})"
            for component in result_info["components"]
        ]
        helper = (
            f"{result_type} {helper_name}("
            f"{condition_type} mask, {result_type} trueValue, "
            f"{result_type} falseValue)\n"
            "{\n"
            f"    return {result_type}({', '.join(components)});\n"
            "}"
        )
        self.register_helper_function(helper_name, helper)
        return helper_name

    def modulo_requires_fmod(self, left_expr, right_expr):
        """Return whether scalar/vector modulo needs Slang fmod lowering."""
        left_component = self.vector_component_type(
            self.expression_result_type(left_expr)
        )
        right_component = self.vector_component_type(
            self.expression_result_type(right_expr)
        )
        return left_component in {"float", "double"} or right_component in {
            "float",
            "double",
        }

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            return self.variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, IdentifierNode):
            return self.variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return "int"
            if isinstance(expr.value, bool):
                return "bool"
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, AssignmentNode):
            return self.expression_result_type(getattr(expr, "left", None))
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            return None
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            if func_name == "imageLoad" and getattr(expr, "args", None):
                return self.image_resource_element_type(
                    self.image_resource_type(expr.args[0])
                )
            if isinstance(func_name, str) and func_name in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "int2",
                "int3",
                "int4",
                "uint2",
                "uint3",
                "uint4",
                "bool2",
                "bool3",
                "bool4",
            }:
                return str(func_name)
            return self.user_function_return_types.get(func_name)
        return None

    def generate_literal(self, node):
        value = node.value
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)

        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def binary_precedence(self, op):
        return self.BINARY_PRECEDENCE.get(op, 0)

    def binary_child_needs_parentheses(self, parent_op, child, is_right_child=False):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_op)
        child_op = getattr(child, "op", getattr(child, "operator", ""))
        child_precedence = self.binary_precedence(child_op)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_op not in self.ASSOCIATIVE_BINARY_OPS or child_op != parent_op
        )

    def generate_binary_expression(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        if self.binary_child_needs_parentheses(node.op, node.left):
            left = f"({left})"
        if self.binary_child_needs_parentheses(node.op, node.right, True):
            right = f"({right})"
        if node.op == "%" and self.modulo_requires_fmod(node.left, node.right):
            return f"fmod({left}, {right})"
        return f"{left} {node.op} {right}"

    def generate_expression(self, node):
        """Render a CrossGL expression as Slang expression syntax."""
        if isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, IdentifierNode):
            return node.name
        elif isinstance(node, LiteralNode):
            return self.generate_literal(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression)
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        elif isinstance(node, ArrayAccessNode):
            array = self.generate_expression(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
            index = self.format_array_access_index(
                getattr(node, "index", getattr(node, "index_expr", None))
            )
            return f"{array}[{index}]"
        elif isinstance(node, ArrayLiteralNode):
            elements = ", ".join(
                self.generate_expression(element) for element in node.elements
            )
            return f"{{{elements}}}"
        elif isinstance(node, MemberAccessNode):
            obj = self.generate_expression(node.object)
            return f"{obj}.{node.member}"
        elif isinstance(node, BinaryOpNode):
            return self.generate_binary_expression(node)
        elif isinstance(node, FunctionCallNode):
            func_expr = getattr(node, "function", None)
            if func_expr is None:
                func_expr = node.name
            if hasattr(func_expr, "name"):
                callee = func_expr.name
            elif isinstance(func_expr, str):
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            if callee not in self.user_function_names:
                resource_call = self.generate_resource_call(callee, node.args)
                if resource_call is not None:
                    return resource_call
            if callee == "mix" and callee not in self.user_function_names:
                bool_mix = self.generate_bool_mix_call(node.args)
                if bool_mix is not None:
                    return bool_mix
            if callee == "lambda":
                lambda_expr = self.generate_lambda_expression(node.args)
                if lambda_expr is not None:
                    return lambda_expr
            args = ", ".join([self.generate_expression(arg) for arg in node.args])
            callee = self.convert_type(callee)
            if (
                callee == "saturate"
                and len(node.args) == 1
                and callee not in self.user_function_names
            ):
                return f"clamp({args}, 0.0, 1.0)"
            if callee not in self.user_function_names:
                callee = self.function_map.get(callee, callee)
            return f"{callee}({args})"
        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand)
            if isinstance(node.operand, BinaryOpNode):
                operand = f"({operand})"
            if getattr(node, "is_postfix", False):
                return f"{operand}{node.op}"
            return f"{node.op}{operand}"
        elif isinstance(node, TernaryOpNode):
            bool_vector_select = self.generate_bool_vector_select_expression(
                node.condition, node.true_expr, node.false_expr
            )
            if bool_vector_select is not None:
                return bool_vector_select
            condition = self.generate_expression(node.condition)
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, str):
            return node
        else:
            return str(node)

    def generate_lambda_expression(self, args):
        """Render supported CrossGL pseudo-lambdas as Slang lambda expressions."""
        if not args:
            return None

        params = []
        for arg in args[:-1]:
            param = self.generate_lambda_parameter(arg)
            if param is None:
                return None
            params.append(param)

        body = self.generate_lambda_body(args[-1])
        if body is None:
            return None
        return f"({', '.join(params)}) => {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return None

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        if mapped_type is None:
            return None
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None
        return raw

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        canonical_type = (
            "".join(type_name.split())
            if "<" in type_name or ">" in type_name
            else type_name
        )
        if any(char.isspace() for char in canonical_type):
            return None
        if any(char in canonical_type for char in "{},;[]()"):
            return None

        mapped_type = self.convert_type(canonical_type)
        if "<" in canonical_type or ">" in canonical_type:
            if mapped_type == canonical_type:
                return None

        if any(char in mapped_type for char in "<>{},;[]()"):
            return None
        if not mapped_type:
            return None
        if not (mapped_type[0].isalpha() or mapped_type[0] == "_"):
            return None
        if not all(char.isalnum() or char == "_" for char in mapped_type):
            return None
        return mapped_type

    def format_array_access_index(self, index):
        if isinstance(index, BinaryOpNode):
            return self.format_array_size_expression(index)
        return self.generate_expression(index)

    def generate_if(self, node):
        condition = self.generate_expression(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )
        result = f"if ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(getattr(node, "if_body", [])):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"

        else_body = getattr(node, "else_body", None)
        if else_body:
            result += "\nelse\n{\n"
            self.indent_level += 1
            for stmt in self.get_statements(else_body):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
            result += self.indent() + "}"

        return result

    def generate_for(self, node):
        init = self.generate_statement(node.init).rstrip(";") if node.init else ""
        condition = self.generate_expression(node.condition) if node.condition else ""
        update = self.generate_statement(node.update).rstrip(";") if node.update else ""

        result = f"for ({init}; {condition}; {update})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_for_in(self, node):
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.generate_expression(iterable.start)
            end = self.generate_expression(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.generate_expression(iterable)
            comparator = "<"

        result = (
            f"for (int {pattern} = {start}; "
            f"{pattern} {comparator} {end}; ++{pattern})\n{{\n"
        )

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_while(self, node):
        condition = self.generate_expression(node.condition)
        result = f"while ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.generate_expression(node.condition)
        result = "do\n{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + f"}} while ({condition});"
        return result

    def generate_switch(self, node):
        expression = self.generate_expression(node.expression)
        result = f"switch ({expression})\n{{\n"

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            if not isinstance(case, CaseNode):
                continue

            if case.value is None:
                result += self.indent() + "default:\n"
            else:
                case_value = self.generate_expression(case.value)
                result += self.indent() + f"case {case_value}:\n"

            self.indent_level += 1
            for stmt in self.get_statements(case.statements):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_match(self, node):
        expression = self.generate_expression(getattr(node, "expression", None))
        result = f"switch ({expression})\n{{\n"

        arms = getattr(node, "arms", []) or []
        if not self.validate_match_arms(arms):
            raise ValueError(
                "Unsupported match arm for Slang codegen; only unguarded "
                "literal patterns and a final wildcard can be lowered to switch"
            )

        wildcard_body = None
        self.indent_level += 1
        for arm in arms:
            pattern = getattr(arm, "pattern", None)
            if isinstance(pattern, WildcardPatternNode):
                wildcard_body = getattr(arm, "body", [])
                continue

            result += (
                self.indent() + f"case {self.generate_expression(pattern.literal)}:\n"
            )
            self.indent_level += 1
            body = getattr(arm, "body", [])
            for stmt in self.get_statements(body):
                result += self.emit_statement(stmt) + "\n"
            if not self.statement_body_terminates(body):
                result += self.indent() + "break;\n"
            self.indent_level -= 1

        if wildcard_body is not None:
            result += self.indent() + "default:\n"
            self.indent_level += 1
            for stmt in self.get_statements(wildcard_body):
                result += self.emit_statement(stmt) + "\n"
            if not self.statement_body_terminates(wildcard_body):
                result += self.indent() + "break;\n"
            self.indent_level -= 1

        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def validate_match_arms(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if not self.is_supported_match_arm(arm):
                return False
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                if wildcard_index is not None:
                    return False
                wildcard_index = index
        return wildcard_index is None or wildcard_index == len(arms) - 1

    def statement_body_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def get_statements(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]

    def convert_type(self, type_name):
        """Map a CrossGL type name or type node to a Slang type string."""
        # Map CrossGL types to Slang types
        type_map = {
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
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "void": "void",
            "sampler": "SamplerState",
            "sampler1D": "Sampler1D<float4>",
            "sampler1DArray": "Sampler1DArray<float4>",
            "sampler2D": "Sampler2D<float4>",
            "sampler3D": "Sampler3D<float4>",
            "samplerCube": "SamplerCube<float4>",
            "sampler2DArray": "Sampler2DArray<float4>",
            "samplerCubeArray": "SamplerCubeArray<float4>",
            "sampler2DMS": "Sampler2DMS<float4>",
            "sampler2DMSArray": "Sampler2DMSArray<float4>",
            "sampler2DShadow": "Sampler2DShadow",
            "sampler2DArrayShadow": "Sampler2DArrayShadow",
            "samplerCubeShadow": "SamplerCubeShadow",
            "samplerCubeArrayShadow": "SamplerCubeArrayShadow",
            "iimage1D": "RWTexture1D<int>",
            "iimage1DArray": "RWTexture1DArray<int>",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "uimage1D": "RWTexture1D<uint>",
            "uimage1DArray": "RWTexture1DArray<uint>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "image1D": "RWTexture1D<float4>",
            "image1DArray": "RWTexture1DArray<float4>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
        }

        return type_map.get(type_name, type_name)

    def supported_image_formats(self):
        return {
            "r8",
            "r8_snorm",
            "r8i",
            "r8ui",
            "r16",
            "r16_snorm",
            "r16f",
            "r16i",
            "r16ui",
            "r32f",
            "r32i",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg8i",
            "rg8ui",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg16i",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba8i",
            "rgba8ui",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba16i",
            "rgba16ui",
            "rgba32f",
            "rgba32i",
            "rgba32ui",
        }

    def scalar_image_format_components(self):
        return {
            "r8": "float",
            "r8_snorm": "float",
            "r16": "float",
            "r16_snorm": "float",
            "r16f": "float",
            "r32f": "float",
            "r8i": "int",
            "r16i": "int",
            "r32i": "int",
            "r8ui": "uint",
            "r16ui": "uint",
            "r32ui": "uint",
        }

    def vector_image_format_components(self):
        return {
            "rg8": "float2",
            "rg8_snorm": "float2",
            "rg16": "float2",
            "rg16_snorm": "float2",
            "rg16f": "float2",
            "rg8i": "int2",
            "rg16i": "int2",
            "rg8ui": "uint2",
            "rg16ui": "uint2",
            "rg32f": "float2",
            "rg32i": "int2",
            "rg32ui": "uint2",
            "rgba8": "float4",
            "rgba8_snorm": "float4",
            "rgba16": "float4",
            "rgba16_snorm": "float4",
            "rgba16f": "float4",
            "rgba32f": "float4",
            "rgba8i": "int4",
            "rgba16i": "int4",
            "rgba32i": "int4",
            "rgba8ui": "uint4",
            "rgba16ui": "uint4",
            "rgba32ui": "uint4",
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name"):
            return str(value.name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def explicit_image_format(self, node):
        if not hasattr(node, "attributes"):
            return None
        supported_formats = self.supported_image_formats()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name
            if attr_name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                format_name = self.attribute_value_to_string(arguments[0])
                if format_name is None:
                    continue
                format_name = str(format_name).lower()
                if format_name in supported_formats:
                    return format_name
        return None

    def map_resource_type_with_format(self, type_name, node=None):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return self.convert_type(type_name)

        if "[" in type_name and "]" in type_name:
            base_type, array_suffix = split_array_type_suffix(type_name)
            mapped_base = self.map_image_base_type_with_format(base_type, node)
            return f"{mapped_base}{array_suffix}"
        return self.map_image_base_type_with_format(type_name, node)

    def map_image_base_type_with_format(self, type_name, node=None):
        base_type = self.resource_base_type(type_name)
        explicit_format = self.explicit_image_format(node) if node is not None else None
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "RWTexture1D",
            "iimage1D": "RWTexture1D",
            "uimage1D": "RWTexture1D",
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "image1DArray": "RWTexture1DArray",
            "iimage1DArray": "RWTexture1DArray",
            "uimage1DArray": "RWTexture1DArray",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "iimage2DMS": "RWTexture2DMS",
            "uimage2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "iimage2DMSArray": "RWTexture2DMSArray",
            "uimage2DMSArray": "RWTexture2DMSArray",
        }
        texture_type = texture_types.get(base_type)
        if component_type and texture_type:
            return f"{texture_type}<{component_type}>"
        return self.convert_type(type_name)

    def is_storage_image_type(self, type_name):
        base_type = self.resource_base_type(type_name)
        return isinstance(base_type, str) and base_type in {
            "image1D",
            "iimage1D",
            "uimage1D",
            "image2D",
            "iimage2D",
            "uimage2D",
            "image3D",
            "iimage3D",
            "uimage3D",
            "image1DArray",
            "iimage1DArray",
            "uimage1DArray",
            "image2DArray",
            "iimage2DArray",
            "uimage2DArray",
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
        }

    def image_resource_type(self, image_arg):
        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_types.get(image_name)

    def image_resource_element_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not image_type or "<" not in image_type or not image_type.endswith(">"):
            return None
        return image_type[image_type.find("<") + 1 : -1]

    def vector_size(self, type_name):
        if not isinstance(type_name, str) or not type_name[-1:].isdigit():
            return None
        size = int(type_name[-1])
        return size if size in {2, 3, 4} else None

    def vector_zero_value(self, type_name):
        if isinstance(type_name, str) and type_name.startswith("uint"):
            return "0u"
        if isinstance(type_name, str) and type_name.startswith("int"):
            return "0"
        return "0.0"

    def image_load_expression(self, args):
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if len(args) >= 3:
            sample = self.generate_expression(args[2])
            load_expr = f"{image_name}[{coord}, {sample}]"
        else:
            load_expr = f"{image_name}[{coord}]"

        image_type = self.image_resource_type(args[0])
        element_type = self.image_resource_element_type(image_type)
        if self.vector_size(element_type) and self.is_scalar_value_type(
            self.current_expression_expected_type
        ):
            return f"{load_expr}.x"
        return load_expr

    def image_store_value_expression(self, image_arg, value_arg):
        value = self.generate_expression(value_arg)
        image_type = self.image_resource_type(image_arg)
        element_type = self.image_resource_element_type(image_type)
        if not self.vector_size(element_type):
            return value
        if not self.is_scalar_value_type(self.expression_result_type(value_arg)):
            return value

        if self.vector_size(element_type) == 2:
            return f"{element_type}({value}, {self.vector_zero_value(element_type)})"
        return f"{element_type}({value})"

    def image_store_expression(self, args):
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if len(args) >= 4:
            sample = self.generate_expression(args[2])
            value = self.image_store_value_expression(args[0], args[3])
            return f"{image_name}[{coord}, {sample}] = {value}"

        value = self.image_store_value_expression(args[0], args[2])
        return f"{image_name}[{coord}] = {value}"

    def image_atomic_intrinsic(self, operation):
        return {
            "imageAtomicAdd": "InterlockedAdd",
            "imageAtomicMin": "InterlockedMin",
            "imageAtomicMax": "InterlockedMax",
            "imageAtomicAnd": "InterlockedAnd",
            "imageAtomicOr": "InterlockedOr",
            "imageAtomicXor": "InterlockedXor",
            "imageAtomicExchange": "InterlockedExchange",
            "imageAtomicCompSwap": "InterlockedCompareExchange",
        }.get(operation)

    def image_atomic_helper_suffix(self, image_type):
        return {
            "RWTexture1D<int>": "iimage1D",
            "RWTexture1D<uint>": "uimage1D",
            "RWTexture2D<int>": "iimage2D",
            "RWTexture2D<uint>": "uimage2D",
            "RWTexture3D<int>": "iimage3D",
            "RWTexture3D<uint>": "uimage3D",
            "RWTexture1DArray<int>": "iimage1DArray",
            "RWTexture1DArray<uint>": "uimage1DArray",
            "RWTexture2DArray<int>": "iimage2DArray",
            "RWTexture2DArray<uint>": "uimage2DArray",
        }.get(image_type)

    def image_atomic_return_type(self, image_type):
        element_type = self.image_resource_element_type(image_type)
        if element_type in {"int", "uint"}:
            return element_type
        return None

    def image_atomic_coord_type(self, image_type):
        if image_type in {"RWTexture1D<int>", "RWTexture1D<uint>"}:
            return "int"
        if image_type in {"RWTexture1DArray<int>", "RWTexture1DArray<uint>"}:
            return "int2"
        if image_type in {"RWTexture2D<int>", "RWTexture2D<uint>"}:
            return "int2"
        if image_type in {
            "RWTexture3D<int>",
            "RWTexture3D<uint>",
            "RWTexture2DArray<int>",
            "RWTexture2DArray<uint>",
        }:
            return "int3"
        return None

    def image_atomic_helper_name(self, operation, image_type):
        suffix = self.image_atomic_helper_suffix(image_type)
        if not suffix:
            return None
        return f"cgl_{operation}_{suffix}"

    def image_atomic_zero_value(self, image_type=None):
        element_type = self.image_resource_element_type(image_type)
        if isinstance(element_type, str) and element_type.startswith("uint"):
            return "0u"

        expected_type = self.convert_type(self.current_expression_expected_type)
        if expected_type == "uint":
            return "0u"
        return "0"

    def unsupported_image_atomic_call(self, operation, reason, image_type=None):
        return (
            f"/* unsupported Slang image atomic: {operation} {reason} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_required_args_reason(self, operation):
        if operation == "imageAtomicCompSwap":
            return "requires image, coordinate, compare, and value arguments"
        return "requires image, coordinate, and value arguments"

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None

        required_args = 4 if operation == "imageAtomicCompSwap" else 3
        if len(args) < required_args:
            return self.unsupported_image_atomic_call(
                operation, self.image_atomic_required_args_reason(operation)
            )

        image_type = self.resource_base_type(self.image_resource_type(args[0]))
        base_helper_name = self.image_atomic_helper_name(operation, image_type)
        if not base_helper_name:
            reason = (
                "requires scalar int or uint "
                "image1D/image1DArray/image2D/image3D/image2DArray resource"
            )
            return self.unsupported_image_atomic_call(
                operation,
                reason,
                image_type,
            )
        helper_name = self.helper_function_name(base_helper_name)

        self.register_helper_function(
            helper_name,
            self.build_image_atomic_helper(helper_name, operation, image_type),
        )

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if operation == "imageAtomicCompSwap":
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        value = self.generate_expression(args[2])
        return f"{helper_name}({image_name}, {coord}, {value})"

    def build_image_atomic_helper(self, helper_name, operation, image_type):
        return_type = self.image_atomic_return_type(image_type)
        coord_type = self.image_atomic_coord_type(image_type)
        intrinsic = self.image_atomic_intrinsic(operation)
        if not return_type or not coord_type or not intrinsic:
            return ""

        if operation == "imageAtomicCompSwap":
            return (
                f"{return_type} {helper_name}({image_type} image, "
                f"{coord_type} coord, {return_type} compareValue, "
                f"{return_type} value)\n"
                "{\n"
                f"    {return_type} original;\n"
                "    InterlockedCompareExchange(image[coord], compareValue, value, original);\n"
                "    return original;\n"
                "}"
            )

        return (
            f"{return_type} {helper_name}({image_type} image, "
            f"{coord_type} coord, {return_type} value)\n"
            "{\n"
            f"    {return_type} original;\n"
            f"    {intrinsic}(image[coord], value, original);\n"
            "    return original;\n"
            "}"
        )

    def resource_query_slang_type(self, resource_arg, resource_type):
        if self.is_storage_image_type(resource_type):
            image_type = self.resource_base_type(self.image_resource_type(resource_arg))
            if image_type:
                return image_type
        return self.convert_type(resource_type)

    def resource_query_helper_name(self, func_name, resource_type, resource_slang_type):
        base_name = f"cgl_{func_name}_{resource_type}"
        if resource_slang_type == self.convert_type(resource_type):
            return base_name
        return f"{base_name}_{self.resource_helper_type_suffix(resource_slang_type)}"

    def resource_helper_type_suffix(self, resource_slang_type):
        return "".join(
            char if char.isalnum() else "_"
            for char in str(resource_slang_type).strip("_")
        ).strip("_")

    def generate_resource_call(self, func_name, args):
        if func_name == "imageLoad" and len(args) >= 2:
            return self.image_load_expression(args)

        if func_name == "imageStore" and len(args) >= 3:
            return self.image_store_expression(args)

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            return self.image_atomic_expression(func_name, args)

        if func_name in {"texture", "textureLod", "textureGrad"}:
            sample_args = self.sampled_texture_args(args)
            if sample_args is None:
                return None

            texture_name, coord, extra_args = sample_args
            if func_name == "texture":
                if extra_args:
                    bias = self.generate_expression(extra_args[0])
                    return f"{texture_name}.SampleBias({coord}, {bias})"
                return f"{texture_name}.Sample({coord})"

            if func_name == "textureLod" and extra_args:
                lod = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleLevel({coord}, {lod})"

            if func_name == "textureGrad" and len(extra_args) >= 2:
                ddx = self.generate_expression(extra_args[0])
                ddy = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy})"

            return None

        if func_name in {"textureOffset", "textureLodOffset", "textureGradOffset"}:
            return self.generate_texture_offset(func_name, args)

        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            return self.generate_texture_projected(func_name, args)

        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return self.generate_texture_gather(func_name, args)

        if func_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
        }:
            return self.generate_texture_compare(func_name, args)

        if func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            return self.generate_texture_gather_compare(func_name, args)

        if func_name == "texelFetch":
            fetch_args = self.sampled_texture_args(args)
            if fetch_args is None:
                return None
            texture_name, coord, extra_args = fetch_args
            if not extra_args:
                return None

            lod_or_sample = self.generate_expression(extra_args[0])
            texture_type = self.get_expression_type(args[0])
            if self.is_multisample_sampler_type(texture_type):
                return f"{texture_name}[{coord}, {lod_or_sample}]"
            coord_constructor = self.texel_fetch_coord_constructor(texture_type)
            return f"{texture_name}.Load({coord_constructor}({coord}, {lod_or_sample}))"

        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(args)

        if func_name == "textureQueryLod":
            return self.generate_texture_query_lod(args)

        return None

    def generate_dimension_query(self, func_name, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_dimension_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )

        if spec["mip"]:
            lod = self.generate_expression(args[1]) if len(args) > 1 else "0"
            return f"{helper_name}({resource_name}, {lod})"
        return f"{helper_name}({resource_name})"

    def generate_sample_count_query(self, func_name, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_sample_count_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )
        return f"{helper_name}({resource_name})"

    def sampled_texture_args(self, args):
        coord_index = self.sampled_texture_coord_index(args)
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord, args[coord_index + 1 :]

    def sampled_texture_coord_index(self, args):
        return 2 if self.is_explicit_sampler_argument(args) else 1

    def generate_texture_offset(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_offset_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = sample_args

        if func_name == "textureOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_offset_call(
                    func_name, "requires one offset argument"
                )
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.Sample({coord}, {offset})"

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({coord}, {lod}, {offset})"

        if len(extra_args) != 3:
            return self.unsupported_texture_offset_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy}, {offset})"

    def unsupported_texture_offset_call(self, func_name, reason):
        return (
            f"/* unsupported Slang texture offset: {func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_projected(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        texture_name, coord, extra_args = sample_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        projected_coord = self.projected_texture_coord(args[0], coord_node, coord)
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires sampler1D/2D/3D projection coordinates"
            )

        if func_name == "textureProj":
            if not extra_args:
                return f"{texture_name}.Sample({projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleBias({projected_coord}, {bias})"
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return f"{texture_name}.Sample({projected_coord}, {offset})"
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleBias({projected_coord}, {bias}, {offset})"
            return self.unsupported_texture_projected_call(
                func_name, "requires offset and optional bias arguments"
            )

        if func_name == "textureProjLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_projected_call(
                    func_name, "requires one lod argument"
                )
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod})"

        if func_name == "textureProjLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod}, {offset})"

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy})"

        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy}, {offset})"

    def projected_texture_coord(self, texture_node, coord_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        coord_type = self.resource_base_type(self.get_expression_type(coord_node))
        specs = {
            "sampler1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "sampler2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "sampler3D": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        resource_specs = specs.get(resource_type)
        if resource_specs is None:
            return None
        coord_spec = resource_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        return f"{coord}.{numerator} / {coord}.{divisor}"

    def unsupported_texture_projected_call(self, func_name, reason):
        return (
            f"/* unsupported Slang projected texture: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_gather(self, func_name, args):
        gather_args = self.sampled_texture_args(args)
        if gather_args is None:
            return self.unsupported_texture_gather_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = gather_args
        offset_args = []
        component_arg = None

        if func_name == "textureGather":
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, "accepts at most one component argument"
                )
            if extra_args:
                component_arg = extra_args[0]
        elif func_name == "textureGatherOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, "requires offset and optional component arguments"
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

        method_args = [coord] + [
            self.generate_expression(offset_arg) for offset_arg in offset_args
        ]
        method = self.texture_gather_method(component_arg)
        if method is not None:
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if isinstance(component_arg, LiteralNode):
            return self.unsupported_texture_gather_call(
                func_name, "component literal must be 0, 1, 2, or 3"
            )

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_method(self, component_arg):
        if component_arg is None:
            return "Gather"

        methods = {
            0: "GatherRed",
            1: "GatherGreen",
            2: "GatherBlue",
            3: "GatherAlpha",
        }
        return methods.get(self.literal_int_value(component_arg))

    def texture_gather_component_expression(self, texture_name, method_args, component):
        arg_list = ", ".join(method_args)
        component_calls = [
            f"{texture_name}.{method}({arg_list})"
            for method in (
                "GatherRed",
                "GatherGreen",
                "GatherBlue",
                "GatherAlpha",
            )
        ]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : "
            f"{component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return (
            f"/* unsupported Slang texture gather: {func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_compare_call(
                func_name, "requires a shadow sampler resource"
            )

        if func_name == "textureCompare":
            if extra_args:
                return self.unsupported_texture_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"{texture_name}.SampleCmp({coord}, {compare})"

        if func_name == "textureCompareOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one offset argument"
                )
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmp({coord}, {compare}, {offset})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one lod argument"
                )
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmpLevel({coord}, {compare}, {lod})"

        if len(extra_args) != 2:
            return self.unsupported_texture_compare_call(
                func_name, "requires gradient x and gradient y arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        return f"{texture_name}.SampleCmpGrad({coord}, {compare}, {ddx}, {ddy})"

    def generate_texture_gather_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires a shadow sampler resource"
            )

        if func_name == "textureGatherCompare":
            if extra_args:
                return self.unsupported_texture_gather_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"{texture_name}.GatherCmp({coord}, {compare})"

        if len(extra_args) != 1:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires one offset argument"
            )
        offset = self.generate_expression(extra_args[0])
        return f"{texture_name}.GatherCmp({coord}, {compare}, {offset})"

    def texture_compare_args(self, func_name, args):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        if len(args) <= coord_index + 1:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        compare = self.generate_expression(args[coord_index + 1])
        return texture_name, coord, compare, args[coord_index + 2 :]

    def is_shadow_compare_resource(self, node):
        resource_type = self.resource_base_type(self.get_expression_type(node))
        return resource_type is None or resource_type in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def unsupported_texture_compare_call(self, func_name, reason):
        return f"/* unsupported Slang shadow compare: {func_name} {reason} */ 0.0"

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return (
            f"/* unsupported Slang shadow gather compare: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def literal_int_value(self, node):
        if not isinstance(node, LiteralNode):
            return None
        value = node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                return None
        return None

    def is_array_expression(self, node):
        type_name = self.get_expression_type(node)
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def generate_texture_query_levels(self, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["mip"]:
            return None

        base_helper_name = f"cgl_textureQueryLevels_{resource_type}"
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_texture_query_levels_helper(helper_name, resource_type, spec),
        )
        return f"{helper_name}({resource_name})"

    def generate_texture_query_lod(self, args):
        query_args = self.texture_query_lod_args(args)
        if query_args is None:
            return None

        texture_name, coord = query_args
        unclamped = f"{texture_name}.CalculateLevelOfDetailUnclamped({coord})"
        clamped = f"{texture_name}.CalculateLevelOfDetail({coord})"
        return f"float2({unclamped}, {clamped})"

    def texture_query_lod_args(self, args):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        if len(args) <= coord_index:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord

    def register_helper_function(self, name, source):
        if name not in self.helper_functions:
            self.helper_functions[name] = source

    def helper_function_name(self, base_name):
        if base_name in self.helper_name_aliases:
            return self.helper_name_aliases[base_name]

        candidate = base_name
        suffix = 1
        used_helper_names = set(self.helper_functions) | set(
            self.helper_name_aliases.values()
        )
        while candidate in self.user_symbol_names or candidate in used_helper_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        self.helper_name_aliases[base_name] = candidate
        return candidate

    def build_dimension_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        return_type = self.query_return_type(spec["dimensions"])
        params = f"{resource_slang_type} tex"
        if spec["mip"]:
            params += ", uint mipLevel"

        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        dimensions = ", ".join(spec["dimensions"])
        if len(spec["dimensions"]) == 1:
            return_value = spec["dimensions"][0]
        else:
            return_value = f"{return_type}({dimensions})"

        return (
            f"{return_type} {helper_name}({params})\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            f"    return {return_value};\n"
            "}"
        )

    def build_sample_count_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return samples;\n"
            "}"
        )

    def build_texture_query_levels_helper(self, helper_name, resource_type, spec):
        resource_slang_type = self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.texture_query_levels_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return levels;\n"
            "}"
        )

    def query_return_type(self, dimensions):
        if len(dimensions) == 1:
            return "int"
        return f"int{len(dimensions)}"

    def query_local_declarations(self, spec):
        names = list(spec["dimensions"])
        if spec["samples"]:
            names.append("samples")
        if spec["mip"]:
            names.append("levels")
        return "".join(f"    int {name};\n" for name in names)

    def get_dimensions_args(self, spec):
        args = []
        if spec["mip"]:
            args.append("mipLevel")
        args.extend(spec["dimensions"])
        if spec["samples"]:
            args.append("samples")
        if spec["mip"]:
            args.append("levels")
        return ", ".join(args)

    def texture_query_levels_args(self, spec):
        args = list(spec["dimensions"])
        args.append("levels")
        return ", ".join(args)

    def dimension_query_spec(self, type_name):
        specs = {
            "sampler1D": (("width",), True, False),
            "sampler1DArray": (("width", "elements"), True, False),
            "sampler2D": (("width", "height"), True, False),
            "sampler2DShadow": (("width", "height"), True, False),
            "sampler2DArray": (("width", "height", "elements"), True, False),
            "sampler2DArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler3D": (("width", "height", "depth"), True, False),
            "samplerCube": (("width", "height"), True, False),
            "samplerCubeShadow": (("width", "height"), True, False),
            "samplerCubeArray": (("width", "height", "elements"), True, False),
            "samplerCubeArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler2DMS": (("width", "height"), False, True),
            "sampler2DMSArray": (("width", "height", "elements"), False, True),
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "image2D": (("width", "height"), False, False),
            "iimage2D": (("width", "height"), False, False),
            "uimage2D": (("width", "height"), False, False),
            "image2DArray": (("width", "height", "elements"), False, False),
            "iimage2DArray": (("width", "height", "elements"), False, False),
            "uimage2DArray": (("width", "height", "elements"), False, False),
            "image3D": (("width", "height", "depth"), False, False),
            "iimage3D": (("width", "height", "depth"), False, False),
            "uimage3D": (("width", "height", "depth"), False, False),
            "image2DMS": (("width", "height"), False, True),
            "iimage2DMS": (("width", "height"), False, True),
            "uimage2DMS": (("width", "height"), False, True),
            "image2DMSArray": (("width", "height", "elements"), False, True),
            "iimage2DMSArray": (("width", "height", "elements"), False, True),
            "uimage2DMSArray": (("width", "height", "elements"), False, True),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {
            "dimensions": dimensions,
            "mip": mip,
            "samples": samples,
        }

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.is_sampler_state_type(self.get_expression_type(args[1]))

    def is_sampler_state_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler",
            "SamplerState",
            "SamplerComparisonState",
        }

    def is_lod_query_sampler_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        return (
            isinstance(resource_type, str)
            and resource_type.startswith("sampler")
            and resource_type != "sampler"
            and "MS" not in resource_type
            and "Shadow" not in resource_type
        )

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            return self.get_expression_name(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
        return None

    def resource_base_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        return type_name.split("[", 1)[0]

    def is_multisample_sampler_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def texel_fetch_coord_constructor(self, type_name):
        base_type = self.resource_base_type(type_name)
        if base_type in {"sampler1D", "sampler1DArray"}:
            return "int2" if base_type == "sampler1D" else "int3"
        if base_type in {"sampler3D", "sampler2DArray"}:
            return "int4"
        return "int3"
