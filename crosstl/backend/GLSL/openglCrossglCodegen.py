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
    WhileNode,
    DoWhileNode,
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
    BlockNode,
    NumberNode,
    PostfixOpNode,
    BreakNode,
    ContinueNode,
    DiscardNode,
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
            "sampler1D": "Texture1D",
            "sampler2D": "Texture2D",
            "sampler3D": "Texture3D",
            "samplerCube": "TextureCube",
            "sampler1DArray": "Texture1DArray",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "sampler2DShadow": "Texture2DShadow",
            "sampler1DShadow": "Texture1DShadow",
            "sampler1DArrayShadow": "Texture1DArrayShadow",
            "sampler2DArrayShadow": "Texture2DArrayShadow",
            "samplerCubeShadow": "TextureCubeShadow",
            "samplerCubeArrayShadow": "TextureCubeArrayShadow",
            "sampler2DRect": "Texture2DRect",
            "sampler2DRectShadow": "Texture2DRectShadow",
            "samplerBuffer": "TextureBuffer",
            "sampler2DMS": "Texture2DMS",
            "sampler2DMSArray": "Texture2DMSArray",
            "isampler1D": "Texture1DInt",
            "isampler2D": "Texture2DInt",
            "isampler3D": "Texture3DInt",
            "isamplerCube": "TextureCubeInt",
            "isampler1DArray": "Texture1DArrayInt",
            "isampler2DArray": "Texture2DArrayInt",
            "isamplerCubeArray": "TextureCubeArrayInt",
            "isampler2DRect": "Texture2DRectInt",
            "isamplerBuffer": "TextureBufferInt",
            "isampler2DMS": "Texture2DMSInt",
            "isampler2DMSArray": "Texture2DMSArrayInt",
            "usampler1D": "Texture1DUint",
            "usampler2D": "Texture2DUint",
            "usampler3D": "Texture3DUint",
            "usamplerCube": "TextureCubeUint",
            "usampler1DArray": "Texture1DArrayUint",
            "usampler2DArray": "Texture2DArrayUint",
            "usamplerCubeArray": "TextureCubeArrayUint",
            "usampler2DRect": "Texture2DRectUint",
            "usamplerBuffer": "TextureBufferUint",
            "usampler2DMS": "Texture2DMSUint",
            "usampler2DMSArray": "Texture2DMSArrayUint",
            "image1D": "Image1D",
            "image2D": "Image2D",
            "image3D": "Image3D",
            "imageCube": "ImageCube",
            "image1DArray": "Image1DArray",
            "image2DArray": "Image2DArray",
            "imageCubeArray": "ImageCubeArray",
            "image2DRect": "Image2DRect",
            "imageBuffer": "ImageBuffer",
            "image2DMS": "Image2DMS",
            "image2DMSArray": "Image2DMSArray",
            "iimage1D": "Image1DInt",
            "iimage2D": "Image2DInt",
            "iimage3D": "Image3DInt",
            "iimageCube": "ImageCubeInt",
            "iimage1DArray": "Image1DArrayInt",
            "iimage2DArray": "Image2DArrayInt",
            "iimageCubeArray": "ImageCubeArrayInt",
            "iimage2DRect": "Image2DRectInt",
            "iimageBuffer": "ImageBufferInt",
            "iimage2DMS": "Image2DMSInt",
            "iimage2DMSArray": "Image2DMSArrayInt",
            "uimage1D": "Image1DUint",
            "uimage2D": "Image2DUint",
            "uimage3D": "Image3DUint",
            "uimageCube": "ImageCubeUint",
            "uimage1DArray": "Image1DArrayUint",
            "uimage2DArray": "Image2DArrayUint",
            "uimageCubeArray": "ImageCubeArrayUint",
            "uimage2DRect": "Image2DRectUint",
            "uimageBuffer": "ImageBufferUint",
            "uimage2DMS": "Image2DMSUint",
            "uimage2DMSArray": "Image2DMSArrayUint",
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

    def stage_struct_name(self):
        return "".join(part.capitalize() for part in self.shader_type.split("_"))

    def _qualifier_set(self, var):
        qualifiers = getattr(var, "qualifiers", None) or []
        return {str(q).lower() for q in qualifiers}

    def _is_input_var(self, var):
        io_type = str(getattr(var, "io_type", "") or "").upper()
        qualifiers = self._qualifier_set(var)
        return io_type == "IN" or "in" in qualifiers or "inout" in qualifiers

    def _is_output_var(self, var):
        io_type = str(getattr(var, "io_type", "") or "").upper()
        qualifiers = self._qualifier_set(var)
        return io_type == "OUT" or "out" in qualifiers or "inout" in qualifiers

    def _is_resource_type(self, type_name):
        if not type_name:
            return False
        name = str(type_name)
        return name.startswith(
            ("sampler", "isampler", "usampler", "image", "iimage", "uimage")
        )

    def format_layout(self, layout_entry):
        layout = (
            layout_entry.get("layout", {}) if isinstance(layout_entry, dict) else {}
        )
        qualifiers = (
            layout_entry.get("qualifiers", []) if isinstance(layout_entry, dict) else []
        )
        parts = []
        for key, value in layout.items():
            if value is None:
                parts.append(str(key))
            else:
                parts.append(f"{key} = {value}")
        layout_str = f"layout({', '.join(parts)})" if parts else "layout()"
        if qualifiers:
            layout_str += " " + " ".join(qualifiers)
        return layout_str.strip()

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
            if isinstance(var, (LayoutNode, VariableNode)):
                if self._is_input_var(var):
                    self.inputs.append(var)
                if self._is_output_var(var):
                    self.outputs.append(var)

        # Ensure vertex-like stages include gl_Position
        if self.shader_type in (
            "vertex",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        ):
            has_position = any(
                isinstance(var, VariableNode) and var.name == "gl_Position"
                for var in self.outputs
            )
            if not has_position:
                builtin = VariableNode(
                    "vec4", "gl_Position", qualifiers=["out"], semantic="gl_Position"
                )
                self.outputs.append(builtin)

        # Ensure fragment outputs include gl_FragColor if no outputs declared
        if self.shader_type == "fragment" and not self.outputs:
            builtin = VariableNode(
                "vec4", "gl_FragColor", qualifiers=["out"], semantic="gl_FragColor"
            )
            self.outputs.append(builtin)

        # Collect uniforms (split resources vs constant data)
        for uniform in node.uniforms:
            self.uniform_vars.append(uniform)

        # Start building the shader
        result = ""
        preprocessor = getattr(node, "preprocessor", []) or []
        if preprocessor:
            for line in preprocessor:
                result += f"{line}\n"
            result += "\n"
        result += "shader main {\n"

        layouts = getattr(node, "layouts", []) or []
        if layouts:
            for layout in layouts:
                result += self.indent_str + f"// {self.format_layout(layout)}\n"
            result += "\n"

        # Generate struct definitions
        for struct in node.structs:
            result += self.indent_str + self.generate_struct(struct) + "\n\n"

        # Generate input struct if needed
        if self.inputs and self.shader_type in (
            "vertex",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        ):
            result += self.indent_str + f"struct {self.stage_struct_name()}Input {{\n"
            self.increase_indent()
            for input_var in self.inputs:
                var_type = self.convert_type(input_var.vtype)
                var_name = input_var.name
                semantic = ""
                if getattr(input_var, "semantic", None):
                    semantic = f" @ {input_var.semantic}"
                result += self.indent() + f"{var_type} {var_name}{semantic};\n"
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate output struct for vertex-like stages
        if self.outputs and self.shader_type in (
            "vertex",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        ):
            result += self.indent_str + f"struct {self.stage_struct_name()}Output {{\n"
            self.increase_indent()
            for output_var in self.outputs:
                var_type = self.convert_type(output_var.vtype)
                var_name = output_var.name
                semantic = ""
                if getattr(output_var, "semantic", None):
                    semantic = f" @ {output_var.semantic}"
                result += self.indent() + f"{var_type} {var_name}{semantic};\n"
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate uniforms: split resource uniforms from constant data
        if self.uniform_vars:
            resource_uniforms = [
                u for u in self.uniform_vars if self._is_resource_type(u.vtype)
            ]
            data_uniforms = [
                u for u in self.uniform_vars if not self._is_resource_type(u.vtype)
            ]

            for uniform in resource_uniforms:
                var_type = self.convert_type(uniform.vtype)
                var_name = uniform.name
                array_suffix = ""
                if getattr(uniform, "array_size", None) is not None:
                    array_suffix = f"[{self.generate_expression(uniform.array_size)}]"
                result += self.indent_str + f"{var_type} {var_name}{array_suffix};\n"

            if data_uniforms:
                result += self.indent_str + "cbuffer Uniforms {\n"
                self.increase_indent()
                for uniform in data_uniforms:
                    var_type = self.convert_type(uniform.vtype)
                    var_name = uniform.name
                    array_suffix = ""
                    if getattr(uniform, "array_size", None) is not None:
                        array_suffix = (
                            f"[{self.generate_expression(uniform.array_size)}]"
                        )
                    result += self.indent() + f"{var_type} {var_name}{array_suffix};\n"
                self.decrease_indent()
                result += self.indent_str + "};\n"

            result += "\n"

        # Generate global constants
        for const_var in getattr(node, "constant", []) or []:
            result += (
                self.indent_str + self.generate_variable_declaration(const_var) + ";\n"
            )
        if getattr(node, "constant", []):
            result += "\n"

        # Generate global variables
        for global_var in getattr(node, "global_variables", []) or []:
            result += (
                self.indent_str + self.generate_variable_declaration(global_var) + ";\n"
            )
        if getattr(node, "global_variables", []):
            result += "\n"

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
                    + f"{self.stage_struct_name()}Output main({self.stage_struct_name()}Input input)"
                )
            elif self.shader_type == "fragment":
                output_type = "vec4"
                output_name = "gl_FragColor"
                if self.outputs:
                    output_type = self.convert_type(self.outputs[0].vtype)
                    output_name = self.outputs[0].name
                result += (
                    self.indent()
                    + f"{output_type} main({self.stage_struct_name()}Input input) @ {output_name}"
                )
            elif self.shader_type == "compute":
                result += self.indent() + "void main()"
            else:
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({self.stage_struct_name()}Input input)"
                )

            result += " {\n"

            # Generate function body
            self.increase_indent()

            # For vertex shaders, create the output struct
            if self.shader_type == "vertex":
                result += self.indent() + f"{self.stage_struct_name()}Output output;\n"

            # For fragment shaders, declare a local output if assignments are used
            if self.shader_type == "fragment" and self.outputs:
                output_type = self.convert_type(self.outputs[0].vtype)
                output_name = self.outputs[0].name
                result += self.indent() + f"{output_type} {output_name};\n"

            # Generate statements for the main function
            for statement in main_function.body:
                result += self.indent() + self.generate_statement(statement) + "\n"

            # Add implicit return for stages with output struct if not present
            if self.shader_type in (
                "vertex",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and not any(isinstance(stmt, ReturnNode) for stmt in main_function.body):
                result += self.indent() + "return output;\n"

            # Add implicit return for fragment shaders if not present
            if self.shader_type == "fragment" and not any(
                isinstance(stmt, ReturnNode) for stmt in main_function.body
            ):
                output_name = self.outputs[0].name if self.outputs else "gl_FragColor"
                result += self.indent() + f"return {output_name};\n"

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
        members = getattr(node, "members", None) or getattr(node, "fields", [])
        for field in members:
            if isinstance(field, dict):
                var_type = self.convert_type(field.get("type"))
                var_name = field.get("name")
                semantic = ""
            else:
                var_type = self.convert_type(getattr(field, "vtype", ""))
                var_name = getattr(field, "name", "")
                semantic = ""
                if getattr(field, "semantic", None):
                    semantic = f" @ {field.semantic}"
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
            return self.generate_assignment(node) + ";"
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, DoWhileNode):
            return self.generate_do_while(node)
        elif isinstance(node, ReturnNode):
            return self.generate_return(node) + ";"
        elif isinstance(node, VariableNode):
            return self.generate_variable_declaration(node) + ";"
        elif isinstance(node, FunctionCallNode):
            # Function call as a statement
            return self.generate_function_call(node) + ";"
        elif isinstance(node, SwitchNode):
            return self.generate_switch_statement(node)
        elif isinstance(node, BlockNode):
            return self.generate_block(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        elif isinstance(node, DiscardNode):
            return "discard;"
        elif isinstance(node, PostfixOpNode):
            return self.generate_expression(node) + ";"
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
        if hasattr(node, "left") and hasattr(node, "right"):
            lhs = node.left
            rhs = node.right
            op = getattr(node, "operator", "=")

            if isinstance(lhs, VariableNode) and lhs.vtype:
                var_type = self.convert_type(lhs.vtype)
                var_name = lhs.name
                value = self.generate_expression(rhs)
                return f"{var_type} {var_name} {op} {value}"

            left_expr = self.generate_expression(lhs)
            right_expr = self.generate_expression(rhs)
            return f"{left_expr} {op} {right_expr}"

        return self.generate_expression(node)

    def generate_if(self, node):
        """Generate CrossGL code for an if statement

        Args:
            node: IfNode representing a GLSL if statement

        Returns:
            str: The CrossGL if statement
        """
        condition_node = getattr(node, "condition", None)
        if condition_node is None:
            condition_node = getattr(node, "if_condition", None)
        condition = self.generate_expression(condition_node)
        result = f"if ({condition}) {{\n"

        # Generate if body
        self.increase_indent()
        for statement in getattr(node, "if_body", []) or []:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()

        result += self.indent() + "}"

        # Generate else-if blocks (support multiple representations)
        else_if_chain = []
        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            else_if_chain = list(zip(node.else_if_conditions, node.else_if_bodies))
        elif hasattr(node, "else_if_chain"):
            else_if_chain = node.else_if_chain

        for elif_condition, elif_body in else_if_chain:
            elif_cond = self.generate_expression(elif_condition)
            result += f" else if ({elif_cond}) {{\n"

            self.increase_indent()
            for statement in elif_body:
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
        init = self.generate_statement(node.init).rstrip(";") if node.init else ""
        condition = self.generate_expression(node.condition) if node.condition else ""
        update_node = getattr(node, "update", None) or getattr(node, "iteration", None)
        iteration = (
            self.generate_statement(update_node).rstrip(";") if update_node else ""
        )

        result = f"for ({init}; {condition}; {iteration}) {{\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_while(self, node):
        condition = self.generate_expression(node.condition)
        result = f"while ({condition}) {{\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.generate_expression(node.condition)
        result = "while (true) {\n"
        self.increase_indent()
        for statement in node.body:
            result += self.indent() + self.generate_statement(statement) + "\n"
        result += self.indent() + f"if (!({condition})) {{\n"
        self.increase_indent()
        result += self.indent() + "break;\n"
        self.decrease_indent()
        result += self.indent() + "}\n"
        self.decrease_indent()
        result += self.indent() + "}"
        return result

    def generate_block(self, node):
        result = "{\n"
        self.increase_indent()
        for statement in node.statements:
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
        elif isinstance(node, NumberNode):
            return str(node.value)
        elif isinstance(node, (int, float)):
            # Numeric literal
            return str(node)
        elif isinstance(node, VariableNode):
            # Variable reference
            if self.shader_type in (
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and any(var.name == node.name for var in self.inputs):
                return f"input.{node.name}"
            if self.shader_type in (
                "vertex",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and any(var.name == node.name for var in self.outputs):
                return f"output.{node.name}"
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
        elif isinstance(node, PostfixOpNode):
            operand = self.generate_expression(node.operand)
            return f"({operand}{node.op})"
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
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
        name = node.name
        if isinstance(name, MemberAccessNode):
            name = self.generate_member_access(name)
        elif isinstance(name, VariableNode):
            name = name.name

        if name in [
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
        ]:
            args = ", ".join(self.generate_expression(arg) for arg in node.args)
            return f"{self.convert_type(name)}({args})"

        # Check if this is a built-in function that needs to be mapped
        mapped_name = self.function_map.get(name, name)

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
            if self.shader_type in (
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and any(var.name == node.object.name for var in self.inputs):
                object_name = f"input.{node.object.name}"
            elif self.shader_type in (
                "vertex",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            ) and any(var.name == node.object.name for var in self.outputs):
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
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        prefix = (
            "const "
            if getattr(node, "is_const", False) or "const" in qualifiers
            else ""
        )
        array_suffix = ""
        if node.array_size is not None:
            array_size = self.generate_expression(node.array_size)
            array_suffix = f"[{array_size}]"

        if getattr(node, "value", None) is not None:
            value = self.generate_expression(node.value)
            return f"{prefix}{var_type} {var_name}{array_suffix} = {value}"

        return f"{prefix}{var_type} {var_name}{array_suffix}"

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
