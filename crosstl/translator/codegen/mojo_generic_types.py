"""Generic enums and structs in the Mojo generator: specialization metadata, mapped types and CrossGL type-node spelling."""

import re

from ..ast import StructNode
from ..matrix_shape import matrix_type_spelling
from .enum_utils import (
    build_generic_enum_specialization,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_parameter_names,
    enum_constant_name,
    enum_variant_constructor_name,
    generic_enum_specialized_fields,
    generic_enum_specialized_variant_fields,
    generic_type_parts,
)
from .generic_function_utils import (
    collect_unresolved_generic_function_call_names,
    iter_function_nodes,
    raise_unresolved_generic_function_call,
)
from .mojo_tables import (
    MOJO_DTYPE_INFO,
    MOJO_SCALAR_DTYPES,
)


class MojoGenericTypeMixin:
    """Generic-type members of ``MojoCodeGen``."""

    def prepare_generic_enum_metadata(self, ast, structs):
        self.generic_parameter_names = collect_generic_parameter_names(ast)
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        for node in structs:
            if isinstance(node, StructNode):
                self.validate_user_defined_type_name(node.name, "struct")
                if node.name not in self.generic_enum_struct_definitions:
                    self.register_generic_struct_type_metadata(node)
            if (
                isinstance(node, StructNode)
                and node.name not in self.generic_enum_struct_definitions
            ):
                self.register_struct_type_metadata(node)
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )

        for name, definition in self.generic_enum_struct_definitions.items():
            self.enum_variant_names[name] = [
                variant.name
                for variant in getattr(definition["enum"], "variants", []) or []
            ]
            for variant_name in self.enum_variant_names[name]:
                alias_name = self.enum_variant_alias_name(name, variant_name)
                self.register_enum_variant_metadata(
                    name,
                    variant_name,
                    alias_name,
                    None,
                )

        for specialization in self.generic_enum_specializations.values():
            struct_name = specialization["struct_name"]
            self.struct_types[struct_name] = {"variant": "int"}
            for field_name, field_type in generic_enum_specialized_fields(
                self, specialization
            ):
                self.struct_types[struct_name][field_name] = self.type_name(field_type)

    def reject_unsupported_generic_functions(self, ast):
        functions = list(iter_function_nodes(ast)) + list(
            (self.generic_function_specializations or {}).values()
        )
        unresolved = collect_unresolved_generic_function_call_names(self, functions)
        if unresolved:
            raise_unresolved_generic_function_call(self, unresolved[0], "Mojo")

    def maybe_register_generic_enum_type(self, type_value, specializations):
        type_text = self.type_name(type_value)
        base_name, generic_args = generic_type_parts(type_text)
        definition = self.generic_enum_struct_definitions.get(base_name)
        if definition is None:
            return None
        if len(generic_args) != len(definition["generic_params"]):
            return None
        if self.generic_type_contains_parameter(type_text):
            return None
        specialization = build_generic_enum_specialization(type_text, definition)
        specializations[type_text] = specialization
        return specialization

    def generic_enum_specialization_for_type(self, type_value, expected_base=None):
        if type_value is None:
            return None
        type_text = self.type_name(type_value)
        base_name, generic_args = generic_type_parts(type_text)
        if expected_base is not None and base_name != expected_base:
            return None
        definition = self.generic_enum_struct_definitions.get(base_name)
        if definition is None:
            return None
        if len(generic_args) != len(definition["generic_params"]):
            return None
        if self.generic_type_contains_parameter(type_text):
            return None
        specialization = self.generic_enum_specializations.get(type_text)
        if specialization is None:
            specialization = build_generic_enum_specialization(type_text, definition)
            self.generic_enum_specializations[type_text] = specialization
        return specialization

    def generic_enum_pattern_specialization_for_type(
        self, type_value, expected_base=None
    ):
        if type_value is None:
            return None
        type_text = self.type_name(type_value)
        base_name, generic_args = generic_type_parts(type_text)
        if expected_base is not None and base_name != expected_base:
            return None
        definition = self.generic_enum_struct_definitions.get(base_name)
        if definition is None:
            return None
        if len(generic_args) != len(definition["generic_params"]):
            return None
        return self.generic_enum_specializations.get(type_text) or (
            build_generic_enum_specialization(type_text, definition)
        )

    def generic_type_contains_parameter(self, type_text):
        base_name, generic_args = generic_type_parts(type_text)
        if base_name in self.generic_parameter_names and not generic_args:
            return True
        return any(self.generic_type_contains_parameter(arg) for arg in generic_args)

    def register_generic_struct_type_metadata(self, node):
        param_names = [
            param.name
            for param in getattr(node, "generic_params", []) or []
            if getattr(param, "name", None)
        ]
        if param_names:
            self.generic_struct_type_params[node.name] = param_names
        else:
            self.generic_struct_type_params.pop(node.name, None)

    def generic_struct_specialization(self, type_value):
        type_text = self.type_name(type_value)
        generic = self.parse_generic_type_name(type_text)
        if generic is None:
            return None

        base_name, generic_args = generic
        param_names = self.generic_struct_type_params.get(base_name)
        if param_names is None or len(param_names) != len(generic_args):
            return None
        if base_name not in self.struct_types:
            return None
        return base_name, dict(zip(param_names, generic_args))

    def substitute_generic_struct_type_params(self, type_value, substitutions):
        type_text = self.type_name(type_value)
        if self.is_array_type_name(type_text):
            element_type, size = self.parse_array_type_name(type_text)
            return self.array_type_name(
                self.substitute_generic_struct_type_params(element_type, substitutions),
                size,
            )

        generic = self.parse_generic_type_name(type_text)
        if generic is not None:
            base_name, generic_args = generic
            mapped_args = [
                self.substitute_generic_struct_type_params(arg, substitutions)
                for arg in generic_args
            ]
            return f"{base_name}<{', '.join(mapped_args)}>"

        if type_text in substitutions:
            return substitutions[type_text]
        # Not one of the structure's type parameters: a concrete type, which the substitution leaves as it is.
        return type_text

    def generic_struct_fields_for_type(self, type_value):
        specialization = self.generic_struct_specialization(type_value)
        if specialization is None:
            return None

        base_name, substitutions = specialization
        return {
            field_name: self.substitute_generic_struct_type_params(
                field_type, substitutions
            )
            for field_name, field_type in self.struct_types.get(base_name, {}).items()
        }

    def struct_type_name_and_fields(self, type_value):
        type_text = self.type_name(type_value)
        generic_fields = self.generic_struct_fields_for_type(type_text)
        if generic_fields is not None:
            base_name, _ = self.generic_struct_specialization(type_text)
            return base_name, generic_fields

        if type_text in self.struct_types:
            return type_text, self.struct_types[type_text]

        generic_enum_type = self.generic_enum_mapped_type(type_text)
        if generic_enum_type in self.struct_types:
            return generic_enum_type, self.struct_types[generic_enum_type]

        return None, {}

    def generic_struct_mapped_type(self, type_value):
        specialization = self.generic_struct_specialization(type_value)
        if specialization is None:
            return None

        base_name, substitutions = specialization
        type_args = [
            self.map_type(substitutions[param_name])
            for param_name in self.generic_struct_type_params[base_name]
        ]
        return f"{base_name}[{', '.join(type_args)}]"

    def constructor_node_struct_type_and_fields(self, expr, target_type=None):
        constructor_type = self.convert_type_node_to_string(expr.constructor_type)
        target_name = self.type_name(target_type)
        target_base, _ = self.resource_base_type_and_count(target_name)
        for candidate in (target_name, target_base, constructor_type):
            struct_type, fields = self.struct_type_name_and_fields(candidate)
            if struct_type is not None:
                return struct_type, fields
        return None, {}

    def generic_enum_mapped_type(self, type_value):
        specialization = self.generic_enum_specialization_for_type(type_value)
        if specialization is None:
            return None
        struct_name = specialization["struct_name"]
        if struct_name not in self.struct_types:
            self.struct_types[struct_name] = {"variant": "int"}
            for field_name, field_type in generic_enum_specialized_fields(
                self, specialization
            ):
                self.struct_types[struct_name][field_name] = self.type_name(field_type)
        return struct_name

    def generate_generic_payload_enum_constants(self):
        code = ""
        for name, definition in sorted(self.generic_enum_struct_definitions.items()):
            next_value = 0
            for variant in getattr(definition["enum"], "variants", []) or []:
                value = getattr(variant, "value", None)
                if value is None:
                    value_text = str(next_value)
                    next_value += 1
                else:
                    value_text = self.generate_expression(value)
                    literal_value = self.literal_int_value(value)
                    next_value = literal_value + 1 if literal_value is not None else 0
                code += f"comptime {enum_constant_name(name, variant.name)} = {value_text}\n"
        return code + ("\n" if code else "")

    def generate_generic_payload_enum_structs(self):
        code = ""
        for specialization in self.generic_enum_specializations.values():
            struct_name = specialization["struct_name"]
            code += f"@value\nstruct {struct_name}:\n"
            code += "    var variant: Int32\n"
            for field_name, field_type in generic_enum_specialized_fields(
                self, specialization
            ):
                code += (
                    f"    var {self.mojo_identifier(field_name)}: "
                    f"{self.map_type(field_type)}\n"
                )
            code += "\n"
        return code

    def generate_generic_payload_enum_constructors(self):
        code = ""
        for specialization in self.generic_enum_specializations.values():
            all_fields = generic_enum_specialized_fields(self, specialization)
            all_field_names = {field_name for field_name, _field_type in all_fields}
            definition = specialization["definition"]
            for variant in getattr(definition["enum"], "variants", []) or []:
                variant_fields = generic_enum_specialized_variant_fields(
                    self,
                    specialization,
                    variant.name,
                )
                if variant_fields is None:
                    continue
                constructor_name = enum_variant_constructor_name(
                    specialization["struct_name"], variant.name
                )
                params = ", ".join(
                    f"payload{index}: {self.map_type(field_type)}"
                    for index, (_field_name, field_type) in enumerate(variant_fields)
                )
                code += (
                    f"fn {constructor_name}({params}) -> "
                    f"{specialization['struct_name']}:\n"
                )
                rendered_fields = {
                    "variant": self.enum_variant_alias_name(
                        definition["name"], variant.name
                    )
                }
                active_fields = {field_name for field_name, _ in variant_fields}
                for field_name, field_type in all_fields:
                    if field_name in active_fields:
                        continue
                    rendered_fields[field_name] = self.zero_value_for_type(field_type)
                for index, (field_name, _field_type) in enumerate(variant_fields):
                    if field_name in all_field_names:
                        rendered_fields[field_name] = f"payload{index}"
                field_args = [
                    f"{self.mojo_identifier(field_name)}={rendered_fields[field_name]}"
                    for field_name in self.struct_types[specialization["struct_name"]]
                ]
                code += (
                    f"    return {specialization['struct_name']}"
                    f"({', '.join(field_args)})\n\n"
                )
        return code

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            return self.convert_type_node_to_string(type_node.referenced_type)
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if hasattr(arg, "value") and getattr(arg, "name", None) is None:
                        converted_args.append(str(arg.value))
                    else:
                        converted_args.append(self.convert_type_node_to_string(arg))
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            dtype = MOJO_SCALAR_DTYPES.get(element_type)
            if dtype in MOJO_DTYPE_INFO:
                return self.vector_type_name_for_dtype_width(dtype, size)
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            return matrix_type_spelling(type_node, element_type)
        else:
            return str(type_node)

    def variable_declared_type(self, node):
        """Return the explicit type on a variable declaration, if one exists."""
        var_type = getattr(node, "var_type", None)
        if var_type is not None:
            return self.explicit_image_format_resource_type(
                self.convert_type_node_to_string(var_type), node
            )

        member_type = getattr(node, "member_type", None)
        if member_type is not None:
            return self.explicit_image_format_resource_type(
                self.convert_type_node_to_string(member_type), node
            )

        vtype = getattr(node, "vtype", None)
        if vtype is None or vtype == "":
            return None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.explicit_image_format_resource_type(
                self.convert_type_node_to_string(vtype), node
            )
        return self.explicit_image_format_resource_type(vtype, node)

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        if hasattr(size, "name"):
            return size.name
        return size

    def parse_generic_type_name(self, type_name):
        if not isinstance(type_name, str) or "<" not in type_name:
            return None
        match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*<(.+)>\s*", type_name)
        if match is None:
            return None
        return match.group(1), self.split_generic_arguments(match.group(2))

    def split_generic_arguments(self, args_text):
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
        args.append(args_text[start:].strip())
        return [arg for arg in args if arg]
