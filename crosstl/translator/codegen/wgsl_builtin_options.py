"""Built-in option types in the WGSL generator: specialization, declarations, payload types and none values."""

from ..ast import (
    ArrayType,
    GenericType,
    IdentifierNode,
    MatrixType,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    VectorType,
)
from ..refusals import WGSLRefusal
from .array_utils import prepend_array_dimension
from .enum_utils import generic_type_parts


class WGSLBuiltinOptionMixin:
    """Builtin option members of ``WGSLCodeGen``."""

    def builtin_option_specialization_member_types(self, specializations):
        member_types = {}
        for type_text, specialization in (specializations or {}).items():
            fields = {
                "variant": PrimitiveType("int"),
                "Some_0": specialization["payload_type"],
            }
            member_types[type_text] = dict(fields)
            member_types[specialization["struct_name"]] = dict(fields)
        return member_types

    def generate_builtin_option_declarations(self):
        if not self._builtin_option_specializations:
            return ""

        sections = ["const Option_Some: i32 = 0;\nconst Option_None: i32 = 1;"]
        for specialization in self._builtin_option_specializations.values():
            struct_name = specialization["struct_name"]
            payload_type = self.type_name_string(specialization["payload_type"])
            sections.append(
                f"struct {struct_name} {{\n"
                "    variant: i32,\n"
                f"    Some_0: {payload_type},\n"
                "};"
            )
            sections.append(
                f"fn {struct_name}_Some_make(payload0: {payload_type}) -> {struct_name} {{\n"
                f"    var result: {struct_name};\n"
                "    result.variant = Option_Some;\n"
                "    result.Some_0 = payload0;\n"
                "    return result;\n"
                "}"
            )
            sections.append(
                f"fn {struct_name}_None_make() -> {struct_name} {{\n"
                f"    var result: {struct_name};\n"
                "    result.variant = Option_None;\n"
                "    result.Some_0 = "
                f"{self.default_value_expression_for_type(specialization['payload_type'])};\n"
                "    return result;\n"
                "}"
            )
        return "\n\n".join(sections)

    def default_value_expression_for_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name == "bool":
            return "false"
        if type_name == "f32":
            return "0.0"
        if type_name in {"i32", "u32"}:
            return f"{type_name}(0)"
        return f"{type_name}()"

    def raw_type_name_string(self, vtype):
        if vtype is None:
            return None
        if isinstance(vtype, PrimitiveType):
            return vtype.name
        if isinstance(vtype, VectorType):
            return (
                f"vec{vtype.size}<" f"{self.raw_type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.raw_type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            element = self.raw_type_name_string(vtype.element_type)
            return prepend_array_dimension(element, vtype.size)
        if isinstance(vtype, NamedType):
            if not vtype.generic_args:
                return str(vtype.name)
            generic_args = ", ".join(
                self.raw_type_name_string(arg) for arg in vtype.generic_args
            )
            return f"{vtype.name}<{generic_args}>"
        if isinstance(vtype, GenericType):
            return str(vtype.name)
        if isinstance(vtype, str):
            return vtype
        if isinstance(vtype, ReferenceType):
            return self.raw_type_name_string(vtype.referenced_type)
        if isinstance(vtype, PointerType):
            return f"{self.raw_type_name_string(vtype.pointee_type)}*"
        return str(vtype)

    def builtin_option_payload_type(self, vtype):
        if isinstance(vtype, NamedType):
            base_name = str(vtype.name).rsplit("::", 1)[-1]
            if base_name == "Option" and len(vtype.generic_args) == 1:
                return vtype.generic_args[0]
            return None
        if isinstance(vtype, str):
            base_name, generic_args = generic_type_parts(vtype)
            if base_name.rsplit("::", 1)[-1] == "Option" and len(generic_args) == 1:
                return generic_args[0]
        return None

    def builtin_option_specialization(self, vtype):
        if self.builtin_option_payload_type(vtype) is None:
            return None
        return self._builtin_option_specializations.get(
            self.raw_type_name_string(vtype)
        )

    def builtin_option_specialized_type_name(self, vtype):
        specialization = self.builtin_option_specialization(vtype)
        if specialization is None:
            return None
        return specialization["struct_name"]

    def expected_builtin_option_specialization(self):
        return self.builtin_option_specialization(
            self._current_expression_expected_type
        ) or self.builtin_option_specialization(self._current_function_return_type)

    def is_builtin_option_none_expression(self, expr):
        return (
            isinstance(expr, IdentifierNode)
            and str(expr.name).rsplit("::", 1)[-1] == "None"
        )

    def generate_builtin_option_none_value(self):
        specialization = self.expected_builtin_option_specialization()
        if specialization is None:
            return None
        return f"{specialization['struct_name']}_None_make()"

    def generate_builtin_option_call(self, node, function_name):
        base_name = str(function_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            specialization = self.expected_builtin_option_specialization()
            if specialization is None:
                return None
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                raise WGSLRefusal(
                    f"WGSL target expects Some() to receive 1 argument, got {len(args)}"
                )
            payload = self.generate_expression_for_target(
                args[0], specialization["payload_type"]
            )
            return f"{specialization['struct_name']}_Some_make({payload})"

        if base_name in {"is_Some", "is_None", "unwrap_Some"}:
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                raise WGSLRefusal(
                    "WGSL target expects "
                    f"{function_name}() to receive 1 argument, got {len(args)}"
                )
            subject = self.generate_expression(args[0])
            if base_name == "is_Some":
                return f"({subject}.variant == Option_Some)"
            if base_name == "is_None":
                return f"({subject}.variant == Option_None)"
            return f"{subject}.Some_0"

        return None

    def builtin_option_call_type(self, node, function_name):
        base_name = str(function_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            specialization = self.expected_builtin_option_specialization()
            return None if specialization is None else specialization["type_name"]
        if base_name in {"is_Some", "is_None"}:
            return PrimitiveType("bool")
        if base_name == "unwrap_Some":
            args = list(getattr(node, "arguments", []) or [])
            if len(args) != 1:
                return None
            option_type = self.expression_type(args[0])
            return self.builtin_option_payload_type(option_type)
        return None

    def generate_builtin_option_none_comparison(self, left, operator, right):
        if operator not in {"==", "!="}:
            return None
        left_is_none = self.is_builtin_option_none_expression(left)
        right_is_none = self.is_builtin_option_none_expression(right)
        if left_is_none == right_is_none:
            return None
        subject = right if left_is_none else left
        subject_expr = self.generate_expression(subject)
        comparison = "==" if operator == "==" else "!="
        return f"({subject_expr}.variant {comparison} Option_None)"
