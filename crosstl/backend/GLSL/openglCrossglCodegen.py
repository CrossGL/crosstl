"""
OpenGL to CrossGL converter implementation

This module implements a converter from GLSL to CrossGL syntax.
It translates GLSL AST structures into CrossGL code.
"""

from .OpenglAst import (
    ShaderNode,
    VariableNode,
    AssignmentNode,
    FunctionNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    LayoutNode,
    VectorConstructorNode,
    ConstantNode,
    MemberAccessNode,
    TernaryOpNode,
    ArrayAccessNode,
    StructNode,
    UniformNode,
    SwitchNode,
    CaseNode,
)


class GLSLToCrossGLConverter:
    """Convert GLSL shaders to CrossGL format"""

    def __init__(self, shader_type="vertex"):
        self.shader_type = shader_type
        self.indent_level = 0
        self.indent_str = "    "  # 4 spaces for indentation

        # Mapping of GLSL built-in functions to CrossGL equivalents
        self.function_map = {
            "dot": "dot",
            "normalize": "normalize",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "pow": "pow",
            "exp": "exp",
            "log": "log",
            "sqrt": "sqrt",
            "inversesqrt": "inverseSqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "fract": "fract",
            "mod": "mod",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "mix",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "reflect": "reflect",
            "refract": "refract",
            "texture": "sample",
            "texture2D": "sample",
            "textureCube": "sample",
        }

        # Mapping of GLSL types to CrossGL types
        self.type_map = {
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "bvec2": "bvec2",
            "bvec3": "bvec3",
            "bvec4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
            "void": "void",
        }

        # Map of GLSL operators to CrossGL operators
        self.operator_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "GREATER_THAN": ">",
            "LESS_THAN": "<",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "MOD": "%",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "BITWISE_XOR": "^",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_AND": "&=",
            "ASSIGN_OR": "|=",
            "ASSIGN_XOR": "^=",
        }

        # Shader-specific info
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []

    def indent(self):
        """Return the current indentation string"""
        return self.indent_str * self.indent_level

    def increase_indent(self):
        """Increase the indentation level"""
        self.indent_level += 1

    def decrease_indent(self):
        """Decrease the indentation level"""
        self.indent_level -= 1

    def generate(self, ast):
        """Generate CrossGL code from a GLSL AST

        Args:
            ast: The abstract syntax tree representing the GLSL shader

        Returns:
            str: The equivalent CrossGL code
        """
        if ast is None:
            return "// Empty shader"

        if not isinstance(ast, ShaderNode):
            return f"// Unexpected AST node type: {type(ast)}"

        # Process the shader node
        return self.generate_shader(ast)

    def generate_shader(self, node):
        """Generate CrossGL code for a shader

        Args:
            node: ShaderNode representing a GLSL shader

        Returns:
            str: The CrossGL shader code
        """
        # Reset shader-specific info
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []

        # Collect shader inputs and outputs
        for var in node.io_variables:
            if isinstance(var, LayoutNode) or isinstance(var, VariableNode):
                io_type = getattr(var, "io_type", None)
                if io_type and "IN" in io_type:
                    self.inputs.append(var)
                elif io_type and "OUT" in io_type:
                    self.outputs.append(var)

        # Collect uniforms
        for uniform in node.uniforms:
            self.uniform_vars.append(uniform)

        # Start building the shader
        result = "shader main {\n"

        # Generate struct definitions
        for struct in node.structs:
            result += self.indent_str + self.generate_struct(struct) + "\n\n"

        # Generate input struct if needed
        if self.inputs:
            result += (
                self.indent_str + f"struct {self.shader_type.capitalize()}Input {{\n"
            )
            self.increase_indent()
            for input_var in self.inputs:
                var_type = self.convert_type(
                    input_var.dtype if hasattr(input_var, "dtype") else input_var.vtype
                )
                var_name = input_var.name
                semantic = ""
                if hasattr(input_var, "semantic") and input_var.semantic:
                    semantic = f" @ {input_var.semantic}"
                result += self.indent() + f"{var_type} {var_name}{semantic};\n"
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate output struct if needed
        if self.outputs:
            result += (
                self.indent_str + f"struct {self.shader_type.capitalize()}Output {{\n"
            )
            self.increase_indent()
            for output_var in self.outputs:
                var_type = self.convert_type(
                    output_var.dtype
                    if hasattr(output_var, "dtype")
                    else output_var.vtype
                )
                var_name = output_var.name
                semantic = ""
                if hasattr(output_var, "semantic") and output_var.semantic:
                    semantic = f" @ {output_var.semantic}"
                result += self.indent() + f"{var_type} {var_name}{semantic};\n"
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate uniform block if needed
        if self.uniform_vars:
            result += self.indent_str + "cbuffer Uniforms {\n"
            self.increase_indent()
            for uniform in self.uniform_vars:
                var_type = self.convert_type(uniform.vtype)
                var_name = uniform.name
                result += self.indent() + f"{var_type} {var_name};\n"
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate shader function
        result += self.indent_str + f"{self.shader_type} {{\n"

        # Find the main function and other functions
        main_function = None
        other_functions = []

        for function in node.functions:
            if function.name == "main":
                main_function = function
            else:
                other_functions.append(function)

        # Generate auxiliary functions first
        for function in other_functions:
            self.increase_indent()
            result += self.indent() + self.generate_function(function) + "\n\n"
            self.decrease_indent()

        # Generate the main function if it exists
        if main_function:
            self.increase_indent()

            # Determine function signature based on shader type
            if self.shader_type == "vertex":
                result += (
                    self.indent()
                    + f"{self.shader_type.capitalize()}Output main({self.shader_type.capitalize()}Input input)"
                )
            elif self.shader_type == "fragment":
                if self.outputs:
                    output_type = self.convert_type(
                        self.outputs[0].dtype
                        if hasattr(self.outputs[0], "dtype")
                        else self.outputs[0].vtype
                    )
                    output_name = self.outputs[0].name
                    result += (
                        self.indent()
                        + f"{output_type} main({self.shader_type.capitalize()}Input input) @ {output_name}"
                    )
                else:
                    result += (
                        self.indent()
                        + f"vec4 main({self.shader_type.capitalize()}Input input) @ gl_FragColor"
                    )

            result += " {\n"

            # Generate function body
            self.increase_indent()

            # For vertex shaders, create the output struct
            if self.shader_type == "vertex":
                result += (
                    self.indent() + f"{self.shader_type.capitalize()}Output output;\n"
                )

            # Generate statements for the main function
            for statement in main_function.body:
                result += self.indent() + self.generate_statement(statement) + "\n"

            # Add implicit return for vertex shaders if not present
            if self.shader_type == "vertex" and not any(
                isinstance(stmt, ReturnNode) for stmt in main_function.body
            ):
                result += self.indent() + "return output;\n"

            self.decrease_indent()
            result += self.indent() + "}\n"
            self.decrease_indent()

        result += self.indent_str + "}\n"

        # Close the shader
        result += "}\n"

        return result

    def generate_struct(self, node):
        """Generate CrossGL code for a struct definition

        Args:
            node: StructNode representing a GLSL struct

        Returns:
            str: The CrossGL struct definition
        """
        result = f"struct {node.name} {{\n"

        self.increase_indent()
        for field in node.fields:
            var_type = self.convert_type(field["type"])
            var_name = field["name"]
            semantic = ""
            result += self.indent() + f"{var_type} {var_name}{semantic};\n"
        self.decrease_indent()

        result += self.indent() + "};"
        return result

    def generate_function(self, node):
        """Generate CrossGL code for a function definition

        Args:
            node: FunctionNode representing a GLSL function

        Returns:
            str: The CrossGL function definition
        """
        return_type = self.convert_type(node.return_type)
        name = node.name

        # Process parameters
        params = []
        for param in node.params:
            if isinstance(param, tuple):  # (type, name)
                param_type, param_name = param
                params.append(f"{self.convert_type(param_type)} {param_name}")
            elif isinstance(param, VariableNode):
                params.append(f"{self.convert_type(param.vtype)} {param.name}")

        params_str = ", ".join(params)

        result = f"{return_type} {name}({params_str}) {{\n"

        # Generate function body
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()

        result += self.indent() + "}"
        return result

    def generate_statement(self, node):
        """Generate CrossGL code for a statement

        Args:
            node: The AST node representing a statement

        Returns:
            str: The CrossGL statement
        """
        if isinstance(node, AssignmentNode):
            # Check if this is a variable declaration and initialization
            if isinstance(node.left, VariableNode) and node.left.vtype:
                var_type = self.convert_type(node.left.vtype)
                var_name = node.left.name
                value = self.generate_expression(node.right)
                return f"{var_type} {var_name} = {value};"
            else:
                return self.generate_assignment(node) + ";"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, ReturnNode):
            return self.generate_return(node) + ";"
        elif isinstance(node, VariableNode):
            # Variable declaration
            var_type = self.convert_type(node.vtype)
            var_name = node.name
            return f"{var_type} {var_name};"
        elif isinstance(node, FunctionCallNode):
            # Function call as a statement
            return self.generate_function_call(node) + ";"
        elif isinstance(node, SwitchNode):
            return self.generate_switch_statement(node)
        else:
            # Generic expression statement
            return self.generate_expression(node) + ";"

    def generate_assignment(self, node):
        """Generate CrossGL code for an assignment

        Args:
            node: AssignmentNode representing a GLSL assignment

        Returns:
            str: The CrossGL assignment
        """
        # Handle the old AssignmentNode structure
        if hasattr(node, "name") and hasattr(node, "value"):
            left = ""
            if isinstance(node.name, VariableNode):
                # Variable declaration with initialization
                left = f"{self.convert_type(node.name.vtype)} {node.name.name}"
                self.local_vars.append(node.name.name)
            elif isinstance(node.name, MemberAccessNode):
                # Member access (e.g., struct.field)
                left = self.generate_member_access(node.name)
            elif isinstance(node.name, ArrayAccessNode):
                # Array access (e.g., array[index])
                left = self.generate_array_access(node.name)
            else:
                # Simple variable
                left = str(node.name)

            operator = self.operator_map.get(node.operator, node.operator)
            right = self.generate_expression(node.value)

            return f"{left} {operator} {right}"

        # Handle the BinaryOpNode-based assignments
        elif hasattr(node, "left") and hasattr(node, "right"):
            lhs = node.left
            op = self.operator_map.get(node.op, node.op)
            rhs = node.right

            # If lhs is a VariableNode with a type, it's a variable declaration
            if isinstance(lhs, VariableNode) and lhs.vtype:
                var_type = self.convert_type(lhs.vtype)
                var_name = lhs.name
                value = self.generate_expression(rhs)
                return f"{var_type} {var_name} = {value}"
            else:
                left_expr = self.generate_expression(lhs)
                right_expr = self.generate_expression(rhs)
                return f"({left_expr} {op} {right_expr})"

        # Default fallback
        return f"{self.generate_expression(node)}"

    def generate_if(self, node):
        """Generate CrossGL code for an if statement

        Args:
            node: IfNode representing a GLSL if statement

        Returns:
            str: The CrossGL if statement
        """
        condition = self.generate_expression(node.if_condition)
        result = f"if ({condition}) {{\n"

        # Generate if body
        self.increase_indent()
        for statement in node.if_body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()

        result += self.indent() + "}"

        # Generate else-if blocks
        for i, elif_condition in enumerate(node.else_if_conditions):
            elif_cond = self.generate_expression(elif_condition)
            result += f" else if ({elif_cond}) {{\n"

            self.increase_indent()
            for statement in node.else_if_bodies[i]:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

            result += self.indent() + "}"

        # Generate else block if present
        if node.else_body:
            result += " else {\n"

            self.increase_indent()
            for statement in node.else_body:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

            result += self.indent() + "}"

        return result

    def generate_for(self, node):
        """Generate CrossGL code for a for loop

        Args:
            node: ForNode representing a GLSL for loop

        Returns:
            str: The CrossGL for loop
        """
        init = self.generate_statement(node.init).rstrip(";")
        condition = self.generate_expression(node.condition)
        iteration = self.generate_statement(node.iteration).rstrip(";")

        result = f"for ({init}; {condition}; {iteration}) {{\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_return(self, node):
        """Generate CrossGL code for a return statement

        Args:
            node: ReturnNode representing a GLSL return statement

        Returns:
            str: The CrossGL return statement
        """
        if node.value is None:
            return "return"
        return f"return {self.generate_expression(node.value)}"

    def generate_expression(self, node):
        """Generate CrossGL code for an expression

        Args:
            node: The AST node representing a GLSL expression

        Returns:
            str: The CrossGL expression
        """
        if node is None:
            return ""

        if isinstance(node, str):
            # Literal string value
            return node
        elif isinstance(node, (int, float)):
            # Numeric literal
            return str(node)
        elif isinstance(node, VariableNode):
            # Variable reference
            if self.shader_type == "vertex" and any(
                var.name == node.name for var in self.inputs
            ):
                return f"input.{node.name}"
            elif self.shader_type == "vertex" and any(
                var.name == node.name for var in self.outputs
            ):
                return f"output.{node.name}"
            else:
                return node.name
        elif isinstance(node, BinaryOpNode):
            # Binary operation
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            operator = self.operator_map.get(node.op, node.op)
            return f"({left} {operator} {right})"
        elif isinstance(node, UnaryOpNode):
            # Unary operation
            operand = self.generate_expression(node.operand)
            operator = self.operator_map.get(node.op, node.op)
            return f"({operator}{operand})"
        elif isinstance(node, FunctionCallNode):
            # Function call
            return self.generate_function_call(node)
        elif isinstance(node, MemberAccessNode):
            # Member access (e.g., struct.field)
            return self.generate_member_access(node)
        elif isinstance(node, ArrayAccessNode):
            # Array access (e.g., array[index])
            return self.generate_array_access(node)
        elif isinstance(node, TernaryOpNode):
            # Ternary conditional (cond ? true_expr : false_expr)
            condition = self.generate_expression(node.condition)
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, VectorConstructorNode):
            # Vector constructor (e.g., vec3(1.0, 2.0, 3.0))
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{self.convert_type(node.type_name)}({args})"
        else:
            # Generic expression
            return str(node)

    def generate_function_call(self, node):
        """Generate CrossGL code for a function call

        Args:
            node: FunctionCallNode representing a GLSL function call

        Returns:
            str: The CrossGL function call
        """
        # Check if this is a vector constructor
        if node.name in [
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "bvec2",
            "bvec3",
            "bvec4",
        ]:
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{self.convert_type(node.name)}({args})"

        # Check if this is a built-in function that needs to be mapped
        mapped_name = self.function_map.get(node.name, node.name)

        # Generate argument list
        args = ", ".join(self.generate_expression(arg) for arg in node.args)

        return f"{mapped_name}({args})"

    def generate_member_access(self, node):
        """Generate CrossGL code for a member access expression

        Args:
            node: MemberAccessNode representing a GLSL member access

        Returns:
            str: The CrossGL member access expression
        """
        # Special case for vertex shader input/output access
        object_name = ""
        if isinstance(node.object, VariableNode):
            if self.shader_type == "vertex" and any(
                var.name == node.object.name for var in self.inputs
            ):
                object_name = f"input.{node.object.name}"
            elif self.shader_type == "vertex" and any(
                var.name == node.object.name for var in self.outputs
            ):
                object_name = f"output.{node.object.name}"
            else:
                object_name = node.object.name
        else:
            object_name = self.generate_expression(node.object)

        return f"{object_name}.{node.member}"

    def generate_array_access(self, node):
        """Generate CrossGL code for an array access expression

        Args:
            node: ArrayAccessNode representing a GLSL array access

        Returns:
            str: The CrossGL array access expression
        """
        array = self.generate_expression(node.array)
        index = self.generate_expression(node.index)
        return f"{array}[{index}]"

    def convert_type(self, type_name):
        """Convert a GLSL type to its CrossGL equivalent

        Args:
            type_name: The GLSL type name

        Returns:
            str: The equivalent CrossGL type name
        """
        return self.type_map.get(type_name, type_name)

    def generate_variable_declaration(self, node):
        """Generate CrossGL code for a variable declaration

        Args:
            node: VariableNode representing a GLSL variable declaration

        Returns:
            str: The CrossGL variable declaration
        """
        var_type = self.convert_type(node.vtype)
        var_name = node.name
        array_suffix = ""
        if node.array_size is not None:
            array_suffix = f"[{node.array_size}]"
        return f"{var_type} {var_name}{array_suffix}"

    def generate_switch_statement(self, node):
        """Generate CrossGL code for a switch statement

        Args:
            node: SwitchNode representing a GLSL switch statement

        Returns:
            str: The CrossGL switch statement
        """
        expression = self.generate_expression(node.expression)
        result = f"switch ({expression}) {{\n"

        # Generate case statements
        for case in node.cases:
            case_value = self.generate_expression(case.value)
            result += self.indent() + f"case {case_value}:\n"

            self.increase_indent()
            for statement in case.statements:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        # Generate default case if present
        if node.default:
            result += self.indent() + "default:\n"

            self.increase_indent()
            for statement in node.default:
                result += self.indent() + self.generate_statement(statement) + "\n"
            self.decrease_indent()

        result += self.indent() + "}"
        return result
