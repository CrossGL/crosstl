"""Reverse code generator that emits CrossGL from Mojo AST nodes."""

import re

from .MojoAst import *
from .MojoLexer import *
from .MojoParser import *


class MojoToCrossGLConverter:
    """Serialize Mojo backend AST nodes back into CrossGL source."""

    VERTEX_ATTRIBUTES = {"vertex", "vertex_main", "vertex_shader"}
    FRAGMENT_ATTRIBUTES = {"fragment", "fragment_main", "fragment_shader"}
    COMPUTE_ATTRIBUTES = {"compute", "compute_main", "compute_shader"}
    SHADER_STAGE_ATTRIBUTES = (
        VERTEX_ATTRIBUTES | FRAGMENT_ATTRIBUTES | COMPUTE_ATTRIBUTES
    )
    MATRIX_TYPE_PATTERN = re.compile(r"^Matrix\[(DType\.\w+),\s*(\d+),\s*(\d+)\]$")
    MATRIX_DTYPE_PREFIXES = {
        "DType.float16": "half",
        "DType.float32": "mat",
        "DType.float64": "dmat",
    }

    def __init__(self):
        self.user_function_names = set()
        self.user_function_arities = {}
        self.scoped_value_names = []
        self.type_map = {
            # Scalar Types
            "void": "void",
            "None": "void",
            "Int": "int",
            "Int8": "int8_t",
            "Int16": "int16_t",
            "Int32": "int",
            "Int64": "int64_t",
            "UInt": "uint",
            "UInt8": "uint8_t",
            "UInt16": "uint16_t",
            "UInt32": "uint",
            "UInt64": "uint64_t",
            "Float": "float",
            "Float16": "half",
            "Float32": "float",
            "Float64": "double",
            "Bool": "bool",
            # Vector Types (Mojo SIMD types)
            "SIMD[DType.float32, 2]": "vec2",
            "SIMD[DType.float32, 3]": "vec3",
            "SIMD[DType.float32, 4]": "vec4",
            "SIMD[DType.int32, 2]": "ivec2",
            "SIMD[DType.int32, 3]": "ivec3",
            "SIMD[DType.int32, 4]": "ivec4",
            "SIMD[DType.uint32, 2]": "uvec2",
            "SIMD[DType.uint32, 3]": "uvec3",
            "SIMD[DType.uint32, 4]": "uvec4",
            "SIMD[DType.bool, 2]": "bvec2",
            "SIMD[DType.bool, 3]": "bvec3",
            "SIMD[DType.bool, 4]": "bvec4",
            # Matrix Types
            "Matrix[DType.float32, 2, 2]": "mat2",
            "Matrix[DType.float32, 3, 3]": "mat3",
            "Matrix[DType.float32, 4, 4]": "mat4",
            "Matrix[DType.float16, 2, 2]": "half2x2",
            "Matrix[DType.float16, 3, 3]": "half3x3",
            "Matrix[DType.float16, 4, 4]": "half4x4",
            # Texture Types (hypothetical Mojo texture types)
            "Texture2D": "sampler2D",
            "TextureCube": "samplerCube",
            "Texture3D": "sampler3D",
            # Common simplified aliases
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
        }

        self.semantic_map = {
            # Vertex attributes
            "position": "Position",
            "normal": "Normal",
            "tangent": "Tangent",
            "binormal": "Binormal",
            "texcoord": "TexCoord",
            "texcoord0": "TexCoord0",
            "texcoord1": "TexCoord1",
            "texcoord2": "TexCoord2",
            "texcoord3": "TexCoord3",
            "color": "Color",
            "color0": "Color0",
            "color1": "Color1",
            "color2": "Color2",
            "color3": "Color3",
            # Built-in variables
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceID",
            "gl_position": "gl_Position",
            "gl_fragcolor": "gl_FragColor",
            "gl_fragdepth": "gl_FragDepth",
            "gl_frontfacing": "gl_IsFrontFace",
            "gl_pointcoord": "gl_PointCoord",
            "gl_primitiveId": "gl_PrimitiveID",
        }

        self.function_map = {
            # Math functions
            "math.sqrt": "sqrt",
            "math.pow": "pow",
            "math.sin": "sin",
            "math.cos": "cos",
            "math.tan": "tan",
            "math.abs": "abs",
            "math.floor": "floor",
            "math.ceil": "ceil",
            "math.exp": "exp",
            "math.fract": "fract",
            "math.log": "log",
            "math.fmod": "mod",
            "fmod": "mod",
            "power": "pow",
            "rsqrt": "inversesqrt",
            "math.rsqrt": "inversesqrt",
            "lerp": "mix",
            "math.lerp": "mix",
            "math.min": "min",
            "math.max": "max",
            "math.clamp": "clamp",
            # SIMD functions
            "dot_product": "dot",
            "cross_product": "cross",
            "magnitude": "length",
            "simd.dot": "dot",
            "simd.cross": "cross",
            "simd.normalize": "normalize",
            "simd.length": "length",
            "simd.reflect": "reflect",
            "simd.refract": "refract",
        }

    def generate(self, ast):
        self.user_function_arities = self.collect_user_function_arities(ast)
        self.user_function_names = set(self.user_function_arities)
        self.scoped_value_names = []
        global_value_names = [
            getattr(variable, "name", None)
            for variable in getattr(ast, "global_variables", []) or []
            if isinstance(variable, (VariableDeclarationNode, VariableNode))
        ]
        self.push_value_scope(global_value_names)
        try:
            return self.generate_shader(ast)
        finally:
            self.pop_value_scope()

    def generate_shader(self, ast):
        code = "shader main {\n"

        if hasattr(ast, "functions") and ast.functions:
            imports = [f for f in ast.functions if isinstance(f, ImportNode)]
            if imports:
                code += "    // Imports\n"
                for imp in imports:
                    code += f"    {self.generate_import_comment(imp)}\n"
                code += "\n"

        if hasattr(ast, "functions"):
            structs = [f for f in ast.functions if isinstance(f, StructNode)]
            for struct_node in structs:
                code += self.generate_struct_like_declaration(
                    struct_node.name,
                    struct_node.members,
                )

            classes = [f for f in ast.functions if isinstance(f, ClassNode)]
            for class_node in classes:
                code += self.generate_struct_like_declaration(
                    class_node.name,
                    class_node.members,
                )

        if hasattr(ast, "functions"):
            cbuffers = [f for f in ast.functions if isinstance(f, ConstantBufferNode)]
            if cbuffers:
                code += "    // Constant Buffers\n"
                for cbuffer in cbuffers:
                    code += self.generate_constant_buffer(cbuffer)

        globals_code = self.generate_global_variables(ast)
        if globals_code:
            code += globals_code

        if hasattr(ast, "functions"):
            functions = [f for f in ast.functions if isinstance(f, FunctionNode)]
            for struct_node in [f for f in ast.functions if isinstance(f, StructNode)]:
                functions.extend(getattr(struct_node, "methods", []))
            for class_node in [f for f in ast.functions if isinstance(f, ClassNode)]:
                functions.extend(getattr(class_node, "methods", []))
            for func in functions:
                if (
                    func.qualifier == "vertex"
                    or self.has_vertex_attribute(func)
                    or "vertex" in func.name
                ):
                    code += "    // Vertex Shader\n"
                    code += "    vertex {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                elif (
                    func.qualifier == "fragment"
                    or self.has_fragment_attribute(func)
                    or "fragment" in func.name
                ):
                    code += "    // Fragment Shader\n"
                    code += "    fragment {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                elif (
                    func.qualifier == "compute"
                    or self.has_compute_attribute(func)
                    or "compute" in func.name
                ):
                    code += "    // Compute Shader\n"
                    code += "    compute {\n"
                    code += self.generate_function(func, indent=2)
                    code += "    }\n\n"
                else:
                    code += self.generate_function(func, indent=1)

        code += "}\n"
        return code

    def collect_user_function_arities(self, ast):
        arities = {}

        def record(function):
            if not isinstance(function, FunctionNode) or function.name is None:
                return
            arities.setdefault(function.name, set()).add(
                len(getattr(function, "params", []) or [])
            )

        for node in getattr(ast, "functions", []):
            if isinstance(node, FunctionNode):
                record(node)
            elif isinstance(node, StructNode):
                for method in getattr(node, "methods", []):
                    record(method)
            elif isinstance(node, ClassNode):
                for method in getattr(node, "methods", []):
                    record(method)
        return arities

    def is_user_defined_function(self, func_name, arg_count=None):
        if not isinstance(func_name, str):
            return False
        if arg_count is None:
            return func_name in self.user_function_arities
        return arg_count in self.user_function_arities.get(func_name, set())

    def push_value_scope(self, names=None):
        scope = set()
        if names:
            scope.update(name for name in names if name)
        self.scoped_value_names.append(scope)

    def pop_value_scope(self):
        if self.scoped_value_names:
            self.scoped_value_names.pop()

    def add_scoped_value_name(self, name):
        if name and self.scoped_value_names:
            self.scoped_value_names[-1].add(name)

    def is_scoped_value_name(self, name):
        return any(name in scope for scope in reversed(self.scoped_value_names))

    def generate_global_variables(self, ast):
        global_variables = getattr(ast, "global_variables", []) or []
        if not global_variables:
            return ""

        code = "    // Global Variables\n"
        for variable in global_variables:
            if not isinstance(variable, VariableDeclarationNode):
                continue
            code += f"    {self.generate_variable_declaration(variable)};\n"
        return code + "\n"

    def generate_struct_like_declaration(self, name, members):
        code = f"    struct {name} {{\n"
        for member in members:
            if not isinstance(member, (VariableDeclarationNode, VariableNode)):
                continue
            if getattr(member, "is_comptime", False) or getattr(
                member, "is_alias", False
            ):
                continue

            semantic = (
                self.map_attributes(member.attributes)
                if hasattr(member, "attributes")
                else ""
            )
            semantic_suffix = f" {semantic}" if semantic else ""
            code += (
                f"        {self.map_type(member.vtype)} "
                f"{member.name}{semantic_suffix};\n"
            )
        code += "    }\n\n"
        return code

    def generate_constant_buffer(self, cbuffer):
        attributes = self.map_attributes(getattr(cbuffer, "attributes", []))
        attributes_suffix = f" {attributes}" if attributes else ""
        code = f"    cbuffer {cbuffer.name}{attributes_suffix} {{\n"
        for member in cbuffer.members:
            member_attributes = self.map_attributes(getattr(member, "attributes", []))
            member_suffix = f" {member_attributes}" if member_attributes else ""
            code += (
                f"        {self.map_type(member.vtype)} "
                f"{member.name}{member_suffix};\n"
            )
        code += "    }\n\n"
        return code

    def has_vertex_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.VERTEX_ATTRIBUTES:
                return True
        return False

    def has_fragment_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.FRAGMENT_ATTRIBUTES:
                return True
        return False

    def has_compute_attribute(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return False
        for attr in func.attributes:
            if hasattr(attr, "name") and attr.name in self.COMPUTE_ATTRIBUTES:
                return True
        return False

    def generate_function(self, func, indent=1):
        """Render one Mojo function node as a CrossGL function."""
        code = ""
        indent_str = "    " * indent

        params = []
        param_names = []
        if hasattr(func, "params") and func.params:
            for p in func.params:
                if not p.vtype:
                    continue
                param_str = f"{self.map_type(p.vtype)} {p.name}"
                if getattr(p, "default_value", None) is not None:
                    default_value = self.generate_expression(p.default_value)
                    param_str += f" = {default_value}"
                if hasattr(p, "attributes") and p.attributes:
                    semantic = self.map_semantic(p.attributes)
                    if semantic:
                        param_str += f" {semantic}"
                params.append(param_str)
                param_names.append(p.name)

        params_str = ", ".join(params) if params else ""
        return_type = self.map_type(func.return_type) if func.return_type else "void"

        func_attributes = self.map_function_attributes(func)

        code += (
            f"{indent_str}{return_type} {func.name}({params_str}){func_attributes} {{\n"
        )

        self.push_value_scope(param_names)
        try:
            if hasattr(func, "body") and func.body:
                code += self.generate_function_body(func.body, indent + 1)
        finally:
            self.pop_value_scope()

        code += f"{indent_str}}}\n\n"
        return code

    def generate_function_body(self, body, indent=0):
        code = ""
        indent_str = "    " * indent

        self.push_value_scope()
        try:
            for stmt in body:
                if isinstance(stmt, PassNode):
                    continue

                code += indent_str
                if isinstance(stmt, VariableDeclarationNode):
                    code += self.generate_variable_declaration(stmt) + ";\n"
                    self.add_scoped_declaration_names(stmt.name)
                elif isinstance(stmt, VariableNode):
                    if hasattr(stmt, "vtype") and stmt.vtype:
                        code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
                    else:
                        code += f"{stmt.name};\n"
                    self.add_scoped_value_name(stmt.name)
                elif isinstance(stmt, AssignmentNode):
                    code += self.generate_assignment(stmt) + ";\n"
                elif isinstance(stmt, ReturnNode):
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += f"return {self.generate_expression(stmt.value)};\n"
                elif isinstance(stmt, BreakNode):
                    code += "break;\n"
                elif isinstance(stmt, ContinueNode):
                    code += "continue;\n"
                elif isinstance(stmt, BinaryOpNode):
                    code += f"{self.generate_expression(stmt)};\n"
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent)
                elif isinstance(stmt, RangeForNode):
                    code += self.generate_range_for_loop(stmt, indent)
                elif isinstance(stmt, WhileNode):
                    code += self.generate_while_loop(stmt, indent)
                elif isinstance(stmt, WithNode):
                    code += self.generate_with_block(stmt, indent)
                elif isinstance(stmt, TryExceptNode):
                    code += self.generate_try_except_block(stmt, indent)
                elif isinstance(stmt, ImportNode):
                    code += f"{self.generate_import_comment(stmt)}\n"
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent)
                elif isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent)
                elif isinstance(stmt, (FunctionCallNode, MethodCallNode, CallNode)):
                    code += f"{self.generate_expression(stmt)};\n"
                elif isinstance(stmt, str):
                    code += f"{stmt};\n"
                else:
                    # For any unhandled statement type
                    code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        finally:
            self.pop_value_scope()

        return code

    def generate_import_comment(self, node):
        if getattr(node, "items", None):
            items = ", ".join(node.items)
            return f"// from {node.module_name} import {items}"
        comment = f"// import {node.module_name}"
        if node.alias:
            comment += f" as {node.alias}"
        return comment

    def generate_variable_declaration(self, node):
        name = self.generate_declaration_name(node.name)
        if node.vtype:
            declaration = f"{self.map_type(node.vtype)} {name}"
        else:
            var_type = "var" if node.var_type == "var" else "let"
            declaration = f"{var_type} {name}"

        attributes = self.map_attributes(getattr(node, "attributes", []))
        if attributes:
            declaration += f" {attributes}"

        if hasattr(node, "initial_value") and node.initial_value is not None:
            value = self.generate_expression(node.initial_value)
            declaration += f" = {value}"
        return declaration

    def generate_declaration_name(self, name):
        if isinstance(name, TupleNode):
            return self.generate_expression(name)
        return name

    def add_scoped_declaration_names(self, name):
        if isinstance(name, TupleNode):
            for element in name.elements:
                if isinstance(element, VariableNode):
                    self.add_scoped_value_name(element.name)
            return
        self.add_scoped_value_name(name)

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        op = node.operator if hasattr(node, "operator") else "="
        return f"{left} {op} {right}"

    def generate_nested_expression(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, AssignmentNode):
            return f"({rendered})"
        return rendered

    def generate_for_loop(self, node, indent):
        indent_str = "    " * indent
        init = self.generate_for_initializer(node.init) if node.init else ""
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )
        update = self.generate_expression(node.update) if node.update else ""

        code = f"for ({init}; {condition}; {update}) {{\n"
        if hasattr(node, "body") and node.body:
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_with_block(self, node, indent):
        indent_str = "    " * indent
        context = self.generate_expression(node.context_expr)
        alias = f" as {node.alias}" if node.alias else ""
        code = f"{{ // with {context}{alias}\n"
        if getattr(node, "body", None):
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_try_except_block(self, node, indent):
        indent_str = "    " * indent
        code = "try {\n"
        if getattr(node, "try_body", None):
            code += self.generate_function_body(node.try_body, indent + 1)
        code += indent_str + "}"
        if getattr(node, "except_body", None):
            exception_name = f" ({node.exception_name})" if node.exception_name else ""
            code += f" catch{exception_name} {{\n"
            code += self.generate_function_body(node.except_body, indent + 1)
            code += indent_str + "}"
        code += "\n"
        return code

    def generate_for_initializer(self, node):
        if isinstance(node, VariableDeclarationNode):
            if node.vtype:
                declaration = f"{self.map_type(node.vtype)} {node.name}"
            else:
                declaration = f"{node.var_type} {node.name}"
            if getattr(node, "initial_value", None) is not None:
                value = self.generate_expression(node.initial_value)
                declaration += f" = {value}"
            return declaration
        return self.generate_expression(node)

    def generate_range_for_loop(self, node, indent):
        indent_str = "    " * indent

        if self.is_range_call(node.iterable):
            code = self.generate_range_call_for_loop(node)
        else:
            iterable = self.generate_expression(node.iterable)
            code = f"for {node.name} in {iterable} {{\n"

        if hasattr(node, "body") and node.body:
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def is_range_call(self, node):
        return isinstance(node, FunctionCallNode) and node.name == "range"

    def generate_range_call_for_loop(self, node):
        args = getattr(node.iterable, "args", [])
        if len(args) == 1:
            start = "0"
            stop = self.generate_expression(args[0])
            step = "1"
        elif len(args) == 2:
            start = self.generate_expression(args[0])
            stop = self.generate_expression(args[1])
            step = "1"
        elif len(args) == 3:
            start = self.generate_expression(args[0])
            stop = self.generate_expression(args[1])
            step = self.generate_expression(args[2])
        else:
            iterable = self.generate_expression(node.iterable)
            return f"for {node.name} in {iterable} {{\n"

        condition = self.generate_range_condition(node.name, stop, args, step)
        update = f"{node.name}++" if step == "1" else f"{node.name} += {step}"
        return f"for (int {node.name} = {start}; {condition}; {update}) {{\n"

    def generate_range_condition(self, name, stop, args, generated_step):
        if self.is_negative_range_step(args, generated_step):
            return f"{name} > {stop}"
        if self.has_dynamic_range_step(args, generated_step):
            return (
                f"(({generated_step} > 0) ? " f"({name} < {stop}) : ({name} > {stop}))"
            )
        return f"{name} < {stop}"

    def has_dynamic_range_step(self, args, generated_step):
        return len(args) == 3 and not self.is_numeric_range_step(args, generated_step)

    def is_negative_range_step(self, args, generated_step):
        if len(args) != 3:
            return False

        step = args[2]
        if isinstance(step, UnaryOpNode) and step.op == "-":
            return self.is_numeric_literal(step.operand)
        return self.is_negative_numeric_literal(generated_step)

    def is_numeric_range_step(self, args, generated_step):
        if len(args) != 3:
            return True

        step = args[2]
        if isinstance(step, UnaryOpNode) and step.op == "-":
            return self.is_numeric_literal(step.operand)
        return self.is_numeric_literal(
            generated_step
        ) or self.is_negative_numeric_literal(generated_step)

    def is_numeric_literal(self, value):
        if isinstance(value, (int, float)):
            return True
        if not isinstance(value, str):
            return False
        try:
            float(value.rstrip("fF"))
        except ValueError:
            return False
        return True

    def is_negative_numeric_literal(self, value):
        if isinstance(value, (int, float)):
            return value < 0
        if not isinstance(value, str):
            return False
        normalized = value.strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        if not normalized.startswith("-"):
            return False
        return self.is_numeric_literal(normalized[1:])

    def generate_while_loop(self, node, indent):
        indent_str = "    " * indent
        condition = (
            self.generate_expression(node.condition) if node.condition else "true"
        )

        code = f"while ({condition}) {{\n"
        if hasattr(node, "body") and node.body:
            code += self.generate_function_body(node.body, indent + 1)
        code += indent_str + "}\n"
        return code

    def generate_if_statement(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)

        code = f"if ({condition}) {{\n"
        if hasattr(node, "if_body") and node.if_body:
            code += self.generate_function_body(node.if_body, indent + 1)
        code += indent_str + "}"

        if hasattr(node, "else_body") and node.else_body:
            if isinstance(node.else_body, list):
                if len(node.else_body) == 1 and isinstance(node.else_body[0], IfNode):
                    code += " else "
                    code += self.generate_if_statement(node.else_body[0], indent)
                else:
                    code += " else {\n"
                    code += self.generate_function_body(node.else_body, indent + 1)
                    code += indent_str + "}"
            elif isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent)
            else:
                code += " else {\n"
                code += self.generate_function_body([node.else_body], indent + 1)
                code += indent_str + "}"

        code += "\n"
        return code

    def generate_switch_statement(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(node.expression)

        code = f"switch ({expression}) {{\n"

        if hasattr(node, "cases") and node.cases:
            for case in node.cases:
                if hasattr(case, "condition") and case.condition is not None:
                    case_value = self.generate_expression(case.condition)
                    code += indent_str + f"    case {case_value}:\n"
                else:
                    code += indent_str + "    default:\n"

                if hasattr(case, "body") and case.body:
                    code += self.generate_function_body(case.body, indent + 2)

        code += indent_str + "}\n"
        return code

    def generate_expression(self, expr):
        """Render a Mojo backend expression node as CrossGL syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            return str(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype:
                return f"{self.map_type(expr.vtype)} {expr.name}"
            else:
                return expr.name
        elif isinstance(expr, VariableDeclarationNode):
            return self.generate_variable_declaration(expr)
        elif isinstance(expr, TupleNode):
            elements = ", ".join(self.generate_expression(e) for e in expr.elements)
            return f"({elements})"
        elif isinstance(expr, ListLiteralNode):
            elements = ", ".join(self.generate_expression(e) for e in expr.elements)
            return f"[{elements}]"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_nested_expression(expr.left)
            right = self.generate_nested_expression(expr.right)
            op = expr.op if hasattr(expr, "op") else "+"
            return f"({left} {op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_nested_expression(expr.operand)
            op = expr.op if hasattr(expr, "op") else "+"
            return f"({op}{operand})"
        elif isinstance(expr, CastNode):
            target_type = self.map_type(expr.target_type)
            value = self.generate_expression(expr.expression)
            return f"{target_type}({value})"
        elif isinstance(expr, FunctionCallNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            func_name = self.map_function(expr.name, len(args))
            return f"{func_name}({args_str})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object)
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            full_name = f"{obj}.{expr.method}"
            if self.is_scoped_value_name(obj):
                return f"{obj}.{expr.method}({args_str})"
            func_name = self.map_function(full_name, len(args))
            if func_name != full_name:
                return f"{func_name}({args_str})"
            return f"{obj}.{expr.method}({args_str})"
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee)
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            return f"{callee}({args_str})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_nested_expression(expr.condition)
            true_expr = self.generate_nested_expression(expr.true_expr)
            false_expr = self.generate_nested_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, VectorConstructorNode):
            args = []
            if hasattr(expr, "args") and expr.args:
                args = [self.generate_expression(arg) for arg in expr.args]
            args_str = ", ".join(args)
            type_name = self.map_type(expr.type_name)
            return f"{type_name}({args_str})"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_array_index(expr.index)
            return f"{array}[{index}]"
        else:
            # For any unhandled expression type
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def generate_array_index(self, index):
        if isinstance(index, TupleNode):
            return ", ".join(
                self.generate_expression(element) for element in index.elements
            )
        return self.generate_expression(index)

    def map_type(self, mojo_type):
        """Map a Mojo type name to the closest CrossGL type name."""
        if mojo_type is None:
            return "void"
        mapped_type = self.type_map.get(mojo_type)
        if mapped_type:
            return mapped_type
        matrix_type = self.map_matrix_type(mojo_type)
        if matrix_type:
            return matrix_type
        return mojo_type

    def map_matrix_type(self, mojo_type):
        if not isinstance(mojo_type, str):
            return None

        match = self.MATRIX_TYPE_PATTERN.match(mojo_type)
        if not match:
            return None

        dtype, columns, rows = match.groups()
        prefix = self.MATRIX_DTYPE_PREFIXES.get(dtype)
        if prefix is None:
            return None

        if columns == rows and prefix in {"mat", "dmat"}:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def map_semantic(self, attributes):
        """Map Mojo decorators or attributes to CrossGL semantic annotations."""
        return self.map_attributes(attributes)

    def map_attributes(self, attributes):
        """Map Mojo decorators or attributes to CrossGL annotation syntax."""
        if not attributes:
            return ""

        mapped_attributes = []
        for attr in attributes:
            if hasattr(attr, "name"):
                mapped_attributes.append(self.map_attribute(attr))
        return " ".join(mapped_attributes)

    def map_attribute(self, attr):
        name = self.semantic_map.get(attr.name, attr.name)
        args = getattr(attr, "args", getattr(attr, "arguments", [])) or []
        if not args:
            return f"@ {name}"

        args_str = ", ".join(self.generate_attribute_argument(arg) for arg in args)
        return f"@ {name}({args_str})"

    def generate_attribute_argument(self, arg):
        if isinstance(arg, str):
            return arg.strip()
        return self.generate_expression(arg)

    def map_function_attributes(self, func):
        if not hasattr(func, "attributes") or not func.attributes:
            return ""

        non_stage_attributes = [
            attr
            for attr in func.attributes
            if not (hasattr(attr, "name") and attr.name in self.SHADER_STAGE_ATTRIBUTES)
        ]
        semantic = self.map_semantic(non_stage_attributes)
        return f" {semantic}" if semantic else ""

    def map_function(self, func_name, arg_count=None):
        if self.is_user_defined_function(func_name, arg_count):
            return func_name
        return self.function_map.get(func_name, func_name)
