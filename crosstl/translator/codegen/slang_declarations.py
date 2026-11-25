"""Declarations in the Slang generator: constants, enum support, type spellings, qualifier prefixes and declared variable types."""

from ..ast import (
    BinaryOpNode,
    LiteralNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
)
from ..matrix_shape import matrix_type_spelling
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from .array_utils import (
    format_c_style_array_declaration,
    prepend_array_dimension,
    split_array_type_suffix,
)
from .enum_utils import (
    generate_enum_constants,
    generate_enum_constructor_functions,
    generate_enum_structs,
    generate_generic_enum_constants,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_structs,
)
from .slang_declaration_modifiers import slang_precise_prefix
from .slang_rasterizer_order import slang_rasterizer_ordered_type
from .slang_specialization_constants import plan_slang_specialization_constants


class SlangDeclarationsMixin:
    """Declaration members of ``SlangCodeGen``."""

    def generate_enum_support_code(self):
        code = ""
        code += generate_enum_constants(
            self, self.plain_enums + self.struct_payload_enums
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
        )
        code += generate_enum_structs(self, self.struct_payload_enums)
        code += generate_generic_enum_structs(self, self.generic_enum_specializations)
        code += generate_enum_constructor_functions(self, self.struct_payload_enums)
        code += generate_generic_enum_constructor_functions(
            self,
            self.generic_enum_specializations,
        )
        return code

    def generate_constants(self, ast, constants=None):
        """Emit CrossGL compile-time constants as Slang static constants.

        Specialization constants keep their id, and the constants that follow them are planned once per program by
        :func:`plan_slang_specialization_constants`.
        """
        plan = self.slang_specialization_plan(ast)
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type_name = self.convert_type_node_to_string(
                getattr(node, "const_type", getattr(node, "vtype", "float"))
            )
            self.register_variable_type(name, const_type_name, node)
            value = getattr(node, "value", None)
            value_code = (
                None if value is None else self.generate_constant_expression(value)
            )
            declaration = format_c_style_array_declaration(
                self.convert_type(const_type_name), name
            )
            specialized = plan.declaration(name, declaration, value_code)
            if specialized is not None:
                code += f"{specialized}\n"
                continue
            code += f"static const {declaration} = {value_code};\n"

        return f"{code}\n" if code else ""

    def slang_interface_array_type(self, member):
        """A stage-interface member's array type sized by its default, when a specialization constant sizes it."""
        cached = getattr(self, "_slang_specialization_plan", None)
        if cached is None:
            return None
        return cached[1].interface_array_type(getattr(member, "member_type", None))

    def slang_specialization_plan(self, ast):
        """The specialization plan of ``ast``, computed once for the program being generated."""
        cached = getattr(self, "_slang_specialization_plan", None)
        if cached is not None and cached[0] is ast:
            return cached[1]
        plan = plan_slang_specialization_constants(
            ast,
            lambda type_node: self.convert_type(
                self.convert_type_node_to_string(type_node)
            ),
        )
        self._slang_specialization_plan = (ast, plan)
        return plan

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def convert_type_node_to_string(self, type_node) -> str:
        if isinstance(type_node, LiteralNode):
            return self.generate_literal(type_node)
        if isinstance(type_node, PointerType):
            return f"{self.convert_type_node_to_string(type_node.pointee_type)}*"
        if isinstance(type_node, ReferenceType):
            return f"{self.convert_type_node_to_string(type_node.referenced_type)}&"
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
            if type(type_node).__name__ == "MatrixType":
                # CrossGL matCxR spelling; convert_type maps to Slang/HLSL floatRxC.
                scalar = "double" if element_type == "double" else "float"
                return matrix_type_spelling(type_node, scalar)
            return f"{element_type}{type_node.rows}x{type_node.cols}"
        if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.__class__.__name__ == "ArrayType":
                if type_node.size is None:
                    return prepend_array_dimension(element_type)
                size = self.format_array_size_expression(type_node.size)
                return prepend_array_dimension(element_type, size)
            if element_type == "float":
                return f"vec{type_node.size}"
            if element_type == "f16":
                return f"vec{type_node.size}<f16>"
            if element_type == "int":
                return f"ivec{type_node.size}"
            if element_type == "uint":
                return f"uvec{type_node.size}"
            if element_type == "bool":
                return f"bvec{type_node.size}"
            if element_type in {"i8", "u8", "i16", "u16", "i64", "u64"}:
                return f"vec{type_node.size}<{element_type}>"
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
        # Only the declaration names the rasterizer-ordered view; its accesses are those of the RW type.
        declared_type = slang_rasterizer_ordered_type(self, mapped_type, node)
        if self.slang_needs_fixed_unsized_value_array(mapped_type) and (
            getattr(node, "initial_value", None) is None
        ):
            # An initializer sizes the array (HLSL and Slang follow C); nothing else can.
            raise UnsizedArrayUnsupportedError("slang", name, "is not a resource array")
        return format_c_style_array_declaration(declared_type, name)

    def slang_needs_fixed_unsized_value_array(self, mapped_type):
        if not isinstance(mapped_type, str) or "[]" not in mapped_type:
            return False
        base_type = self.resource_base_type(mapped_type)
        if not isinstance(base_type, str):
            return False
        if base_type in self.user_struct_names:
            return False
        return self.slang_register_prefix_for_type(mapped_type) is None

    def map_type(self, type_name):
        return self.convert_type(type_name)

    def slang_declaration_qualifier_prefix(self, node):
        qualifiers = []
        seen = set()
        for qualifier in getattr(node, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"groupshared", "shared", "threadgroup", "workgroup"}:
                mapped = "groupshared"
            elif normalized == "uniform":
                mapped = "uniform"
            elif self.slang_interpolation_qualifier(normalized):
                mapped = self.slang_interpolation_qualifier(normalized)
            else:
                mapped = None
            if mapped and mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)
        for mapped in self.slang_semantic_implied_interpolation_qualifiers(node):
            if mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)

        if slang_precise_prefix(node):
            qualifiers.append("precise")

        if not qualifiers:
            return ""
        return " ".join(qualifiers) + " "

    def slang_parameter_qualifier_prefix(self, node):
        qualifiers = []
        seen = set()
        for qualifier in getattr(node, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"const", "in", "out", "inout"}:
                mapped = normalized
            elif self.slang_interpolation_qualifier(normalized):
                mapped = self.slang_interpolation_qualifier(normalized)
            else:
                mapped = None
            if mapped and mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)
        for mapped in self.slang_semantic_implied_interpolation_qualifiers(node):
            if mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)

        if not qualifiers:
            return ""
        return " ".join(qualifiers) + " "

    def slang_interpolation_qualifier(self, qualifier):
        return self.SLANG_INTERPOLATION_QUALIFIER_MAP.get(str(qualifier).lower())

    def slang_semantic_implied_interpolation_qualifiers(self, node):
        semantic = self.semantic_from_node(node)
        if semantic is None:
            return []
        if str(semantic).lower() == "gl_barycoordnoperspext":
            return ["noperspective"]
        return []

    def slang_interpolation_attribute_prefix(self, node):
        qualifiers = []
        seen = set()
        for attr in getattr(node, "attributes", []) or []:
            mapped = self.slang_interpolation_qualifier(getattr(attr, "name", attr))
            if mapped and mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)
        if not qualifiers:
            return ""
        return " ".join(qualifiers) + " "

    def slang_stage_parameter_qualifier_prefix(self, node, shader_type):
        prefix = self.slang_parameter_qualifier_prefix(node)
        if self.slang_shader_stage_name(shader_type) != "geometry":
            return prefix

        primitive = self.slang_geometry_input_primitive_qualifier(node)
        if primitive is None:
            return prefix
        return f"{primitive} {prefix}"

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

    def array_literal_element_expected_type(self, expected_type):
        expected_type = self.type_name_string(expected_type)
        if not expected_type or "[" not in expected_type:
            return None

        return self.array_element_type(expected_type)

    def array_element_type(self, array_type):
        array_type = self.type_name_string(array_type)
        if not array_type or "[" not in array_type:
            return None

        base_type, array_suffix = split_array_type_suffix(array_type)
        if not array_suffix.startswith("["):
            return None

        closing_bracket = array_suffix.find("]")
        if closing_bracket < 0:
            return None

        remaining_suffix = array_suffix[closing_bracket + 1 :]
        return f"{base_type}{remaining_suffix}" if remaining_suffix else base_type

    def register_variable_type(self, name, type_name, node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        self.local_variable_types[name] = type_name
        mapped_type = self.map_resource_type_with_format(type_name, node)
        if self.is_storage_image_type(type_name):
            self.image_resource_types[name] = mapped_type
            access = self.explicit_resource_access(node)
            if access is not None:
                self.image_resource_accesses[name] = access
        if self.is_buffer_resource_type(mapped_type):
            self.buffer_resource_types[name] = self.resource_base_type(mapped_type)
            access = self.explicit_resource_access(node)
            if access is not None:
                self.buffer_resource_accesses[name] = access
