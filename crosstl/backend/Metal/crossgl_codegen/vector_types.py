"""Wide and small vector types and their helpers."""

import re

from ..MetalAst import (
    ArrayAccessNode,
    MemberAccessNode,
)
from .support import (
    MetalIndexedComponentTypeResolutionError,
    MetalWideVectorLoweringError,
)


class MetalVectorTypeMixin:
    """Members of ``MetalToCrossGLConverter``: wide and small vector types and their helpers."""

    def generic_type_parts(self, metal_type):
        base = self.normalized_metal_type(metal_type)
        if "<" not in base or not base.endswith(">"):
            return None, []
        base_name, inner = base.split("<", 1)
        return base_name.strip(), self.split_generic_arguments(inner[:-1])

    def concrete_generic_vector_width(self, size):
        width_text = str(size).strip()
        while width_text.startswith("(") and width_text.endswith(")"):
            width_text = width_text[1:-1].strip()
        try:
            width = int(width_text, 0)
        except ValueError:
            return None
        return width if width > 0 else None

    def wide_vector_type_info(self, metal_type, source_location=None):
        if metal_type is None:
            return None

        resolved_type = self.resolve_local_type_aliases(metal_type)
        resolved_type = self.resolve_type_alias(resolved_type)
        candidate = str(resolved_type).strip()
        candidate = re.sub(r"^typename\s+", "", candidate)
        candidate = re.sub(
            r"^(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)+",
            "",
            candidate,
        )
        while candidate.endswith("*") or candidate.endswith("&"):
            candidate = candidate[:-1].strip()

        vector_parts = self.metal_vector_type_parts(candidate)
        if vector_parts is None:
            return None
        element_type, size = vector_parts
        width = self.concrete_generic_vector_width(size)
        if width is None or width <= 4:
            return None

        mapped_element = self.map_type(self.resolve_type_alias(element_type))
        supported_elements = self.wide_vector_supported_scalar_types()
        if mapped_element not in supported_elements:
            raise MetalWideVectorLoweringError(
                candidate,
                f"element type '{element_type}' is not a supported scalar type",
                source_location,
            )

        key = (mapped_element, width)
        info = self.wide_vector_types.get(key)
        if info is not None:
            return info

        element_name = re.sub(r"[^A-Za-z0-9_]", "_", mapped_element).strip("_")
        base_name = f"CrossGLMetalVector_{element_name}_{width}"
        type_name = base_name
        suffix = 1
        while type_name in self.wide_vector_reserved_names or (
            self.wide_vector_generated_name_conflicts(type_name)
        ):
            type_name = f"{base_name}_{suffix}"
            suffix += 1
        self.wide_vector_reserved_names.add(type_name)
        info = {
            "key": key,
            "source_type": candidate,
            "element_type": mapped_element,
            "width": width,
            "type_name": type_name,
        }
        self.wide_vector_types[key] = info
        return info

    def wide_vector_supported_scalar_types(self):
        return {
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int",
            "uint",
            "int64",
            "uint64",
            "float16",
            "bfloat16",
            "float",
            "double",
        }

    def wide_vector_generated_name_conflicts(self, type_name):
        generated_names = {
            type_name,
            f"{type_name}_splat",
            f"{type_name}_make",
        }
        operation_names = {
            operation_name
            for operator in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>")
            for operation_name in [self.wide_vector_binary_operation_name(operator)]
        }
        generated_names.update(
            f"{type_name}_{operation_name}_{left_kind}_{right_kind}"
            for operation_name in operation_names
            for left_kind in ("scalar", "vector")
            for right_kind in ("scalar", "vector")
        )
        generated_names.update(
            f"{type_name}_{operation_name}_assign_{right_kind}"
            for operation_name in operation_names
            for right_kind in ("scalar", "vector")
        )
        return bool(
            generated_names.intersection(
                self.user_function_names | self.wide_vector_reserved_names
            )
        )

    def wide_vector_constructor_argument_is_scalar(self, argument):
        argument_type = self.expression_metal_type(argument)
        if argument_type is None:
            return True
        mapped_type = self.map_type(self.resolve_type_alias(argument_type))
        return mapped_type in self.wide_vector_supported_scalar_types()

    def wide_vector_type_info_from_parts(
        self, element_type, size, source_location=None
    ):
        width = self.concrete_generic_vector_width(size)
        if width is None or width <= 4:
            return None
        return self.wide_vector_type_info(
            f"metal::vec<{element_type},{width}>", source_location
        )

    def wide_vector_default_value(self, info):
        return "false" if info["element_type"] == "bool" else "0"

    def wide_vector_helper_name(self, info, operation):
        return f"{info['type_name']}_{operation}"

    def wide_vector_binary_operation_name(self, operator):
        return {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            "&": "bit_and",
            "|": "bit_or",
            "^": "bit_xor",
            "<<": "shift_left",
            ">>": "shift_right",
        }.get(operator)

    def wide_vector_binary_helper_name(self, info, operator, left_kind, right_kind):
        operation_name = self.wide_vector_binary_operation_name(operator)
        return self.wide_vector_helper_name(
            info, f"{operation_name}_{left_kind}_{right_kind}"
        )

    def wide_vector_compound_helper_name(self, info, operator, right_kind):
        operation_name = self.wide_vector_binary_operation_name(operator)
        return self.wide_vector_helper_name(
            info, f"{operation_name}_assign_{right_kind}"
        )

    def small_vector_index_info(self, expression, *, require_type=False):
        if not isinstance(expression, ArrayAccessNode):
            return None
        # Opaque tables and special aggregate accessors must remain renderable in
        # contexts that do not require a result type. Auto/type-required paths go
        # through expression_metal_type() and emit the structured diagnostic.
        try:
            base_type = self.expression_metal_type(expression.array)
        except MetalIndexedComponentTypeResolutionError:
            if require_type:
                raise
            return None
        if base_type is None:
            if require_type:
                self.metal_indexed_type_selection(expression)
            return None
        selection = self.metal_indexed_type_selection(expression)
        if selection["kind"] == "aggregate" and require_type:
            raise MetalIndexedComponentTypeResolutionError(
                selection["base_type"],
                "aggregate-subscript",
                "the user-defined aggregate subscript result type cannot be inferred",
                getattr(expression, "source_location", None),
                self.metal_index_expression_text(expression.index),
            )
        if selection["kind"] != "vector":
            return None

        width = selection["width"]
        if width is None:
            raise MetalIndexedComponentTypeResolutionError(
                selection["base_type"],
                "vector-subscript",
                f"the vector width '{selection['width_text']}' is not concrete",
                getattr(expression, "source_location", None),
                self.metal_index_expression_text(expression.index),
            )
        if width > 4:
            return None
        if width < 2:
            raise MetalIndexedComponentTypeResolutionError(
                selection["base_type"],
                "vector-subscript",
                f"the vector width {width} is outside the supported range 2..4",
                getattr(expression, "source_location", None),
                self.metal_index_expression_text(expression.index),
            )

        mapped_vector = self.map_generic_vector_type(selection["element_type"], width)
        mapped_element = self.map_type(selection["element_type"])
        key = (mapped_vector, mapped_element, width)
        info = self.small_vector_index_types.get(key)
        if info is not None:
            return info

        mapped_name = re.sub(r"[^A-Za-z0-9_]", "_", mapped_vector).strip("_")
        base_prefix = f"CrossGLMetalVectorIndex_{mapped_name}"
        prefix = base_prefix
        suffix = 1
        operation_names = {
            "get",
            "set",
            "add_assign",
            "sub_assign",
            "mul_assign",
            "div_assign",
            "mod_assign",
            "bit_and_assign",
            "bit_or_assign",
            "bit_xor_assign",
            "shift_left_assign",
            "shift_right_assign",
            "pre_increment",
            "pre_decrement",
            "post_increment",
            "post_decrement",
        }
        while True:
            generated_names = {
                f"{prefix}_{operation_name}" for operation_name in operation_names
            } | {
                f"{prefix}_{operation_name}_value" for operation_name in operation_names
            }
            if not generated_names.intersection(
                self.user_function_names | self.wide_vector_reserved_names
            ):
                break
            prefix = f"{base_prefix}_{suffix}"
            suffix += 1
        self.wide_vector_reserved_names.update(generated_names)
        info = {
            "key": key,
            "vector_type": mapped_vector,
            "element_type": mapped_element,
            "source_element_type": selection["element_type"],
            "width": width,
            "prefix": prefix,
        }
        self.small_vector_index_types[key] = info
        return info

    def small_vector_index_operation_name(self, operator):
        return {
            "+=": "add_assign",
            "-=": "sub_assign",
            "*=": "mul_assign",
            "/=": "div_assign",
            "%=": "mod_assign",
            "&=": "bit_and_assign",
            "|=": "bit_or_assign",
            "^=": "bit_xor_assign",
            "<<=": "shift_left_assign",
            ">>=": "shift_right_assign",
            "++": "increment",
            "--": "decrement",
        }.get(operator)

    def small_vector_index_compound_operator(self, operation):
        return {
            "add_assign": "+",
            "sub_assign": "-",
            "mul_assign": "*",
            "div_assign": "/",
            "mod_assign": "%",
            "bit_and_assign": "&",
            "bit_or_assign": "|",
            "bit_xor_assign": "^",
            "shift_left_assign": "<<",
            "shift_right_assign": ">>",
        }.get(operation)

    def small_vector_compound_operation_types(self, info, operator, right, target):
        operation = self.small_vector_index_operation_name(operator)
        binary_operator = self.small_vector_index_compound_operator(operation)
        right_source_type = self.expression_metal_type(right)
        if right_source_type is None:
            raise MetalIndexedComponentTypeResolutionError(
                info["vector_type"],
                "vector-component-assignment",
                "the compound right operand type could not be inferred",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(target.index),
            )
        computation_source_type = self.metal_scalar_binary_result_type(
            binary_operator,
            info["source_element_type"],
            right_source_type,
        )
        if computation_source_type is None:
            raise MetalIndexedComponentTypeResolutionError(
                info["vector_type"],
                "vector-component-assignment",
                f"operator '{operator}' has no provable scalar conversion contract",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(target.index),
            )

        right_parameter_source_type = computation_source_type
        if binary_operator in {"<<", ">>"}:
            right_type_info = self.metal_scalar_arithmetic_type_info(right_source_type)
            if right_type_info is None or right_type_info[0] != "integer":
                raise MetalIndexedComponentTypeResolutionError(
                    info["vector_type"],
                    "vector-component-assignment",
                    "the shift count does not have an integral source type",
                    getattr(target, "source_location", None),
                    self.metal_index_expression_text(target.index),
                )
            right_parameter_source_type = self.promoted_metal_integer_type(
                right_type_info
            )[0]
        return (
            self.map_type(right_parameter_source_type),
            self.map_type(computation_source_type),
        )

    def small_vector_update_computation_type(self, info, target):
        computation_source_type = self.metal_scalar_binary_result_type(
            "+", info["source_element_type"], "int"
        )
        if computation_source_type is None:
            raise MetalIndexedComponentTypeResolutionError(
                info["vector_type"],
                "vector-component-update",
                "the component type has no provable increment/decrement contract",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(target.index),
            )
        return self.map_type(computation_source_type)

    def small_vector_index_helper_name(
        self,
        info,
        operation,
        *,
        right_type=None,
        computation_type=None,
    ):
        self.small_vector_index_operations.add(
            (info["key"], operation, right_type, computation_type)
        )
        return f"{info['prefix']}_{operation}"

    def small_vector_index_operation_has_right(self, operation):
        return operation not in {
            "get",
            "pre_increment",
            "pre_decrement",
            "post_increment",
            "post_decrement",
        }

    def small_vector_index_branch_statement(
        self, info, operation, member, computation_type=None
    ):
        element_type = info["element_type"]
        selected = f"value.{member}"
        if operation == "get":
            return f"return {selected};"
        if operation == "set":
            return f"{selected} = selected; return selected;"

        compound_operator = self.small_vector_index_compound_operator(operation)
        if compound_operator is not None:
            expression = (
                self.render_profiled_metal_arithmetic(
                    f"{computation_type}(original)",
                    "right",
                    computation_type,
                    operator=compound_operator,
                )
                if compound_operator in {"/", "+", "-", "*"}
                else None
            )
            expression = (
                expression or f"{computation_type}(original) {compound_operator} right"
            )
            return (
                f"{element_type} original = {selected}; "
                f"{computation_type} computed = "
                f"{expression}; "
                f"{element_type} updated = computed; "
                f"{selected} = updated; return updated;"
            )

        update_operator = "+" if operation.endswith("increment") else "-"
        update = (
            f"{element_type} original = {selected}; "
            f"{computation_type} computed = {computation_type}(original) "
            f"{update_operator} {computation_type}(1); "
            f"{element_type} updated = computed; {selected} = updated; "
        )
        if operation.startswith("pre_"):
            return f"{update}return updated;"
        return f"{update}return original;"

    def structured_buffer_access_path(self, expression):
        if self.is_direct_structured_buffer_element_access(expression):
            return expression, []
        if isinstance(expression, MemberAccessNode):
            parent = self.structured_buffer_access_path(expression.object)
            if parent is None:
                return None
            root, path = parent
            return root, [*path, ("member", str(expression.member))]
        if isinstance(expression, ArrayAccessNode):
            parent = self.structured_buffer_access_path(expression.array)
            if parent is None:
                return None
            root, path = parent
            return root, [*path, ("index", expression.index)]
        return None

    def small_vector_resource_index_parameter_type(self, expression, target):
        source_type = self.expression_metal_type(expression)
        type_info = self.metal_scalar_arithmetic_type_info(source_type)
        if type_info is None or type_info[0] != "integer":
            raise MetalIndexedComponentTypeResolutionError(
                self.expression_metal_type(target.array),
                "resource-vector-component",
                "a resource aggregate index does not have an integral source type",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(expression),
            )
        return self.map_type(self.promoted_metal_integer_type(type_info)[0])

    def small_vector_resource_component_access(self, target):
        access_path = self.structured_buffer_access_path(target.array)
        if access_path is None:
            return None
        root, path = access_path
        container_source_type = self.expression_metal_type(root)
        container_type = self.map_type(container_source_type)
        if not container_type or any(char in str(container_type) for char in "[]*&"):
            raise MetalIndexedComponentTypeResolutionError(
                container_source_type,
                "resource-vector-component",
                "the resource element type cannot be materialized as a local value",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(target.index),
            )

        root_index_type = self.small_vector_resource_index_parameter_type(
            root.index, target
        )
        descriptor_path = []
        path_index_expressions = []
        for kind, value in path:
            if kind == "member":
                descriptor_path.append((kind, value))
                continue
            index_type = self.small_vector_resource_index_parameter_type(value, target)
            descriptor_path.append((kind, index_type))
            path_index_expressions.append(value)
        return {
            "root": root,
            "container_type": container_type,
            "root_index_type": root_index_type,
            "path": tuple(descriptor_path),
            "path_index_expressions": path_index_expressions,
        }

    def register_small_vector_resource_index_operation(
        self,
        info,
        operation,
        access,
        *,
        right_type=None,
        computation_type=None,
    ):
        self.small_vector_index_helper_name(
            info,
            operation,
            right_type=right_type,
            computation_type=computation_type,
        )
        key = (
            info["key"],
            operation,
            right_type,
            computation_type,
            access["container_type"],
            access["root_index_type"],
            access["path"],
        )
        existing = self.small_vector_resource_index_operations.get(key)
        if existing is not None:
            return existing

        path_name = "_".join(
            "index" if kind == "index" else self.sanitize_identifier(value)
            for kind, value in access["path"]
        )
        base_name = f"{info['prefix']}_{operation}_resource"
        if path_name:
            base_name = f"{base_name}_{path_name}"
        helper_name = base_name
        suffix = 1
        while helper_name in self.wide_vector_reserved_names:
            helper_name = f"{base_name}_{suffix}"
            suffix += 1
        self.wide_vector_reserved_names.add(helper_name)
        descriptor = {
            "name": helper_name,
            "info_key": info["key"],
            "operation": operation,
            "right_type": right_type,
            "computation_type": computation_type,
            "container_type": access["container_type"],
            "root_index_type": access["root_index_type"],
            "path": access["path"],
        }
        self.small_vector_resource_index_operations[key] = descriptor
        return descriptor

    def generate_small_vector_resource_component_operation(
        self,
        target,
        info,
        operation,
        rendered_value=None,
        *,
        right_type=None,
        computation_type=None,
        is_main=False,
    ):
        access = self.small_vector_resource_component_access(target)
        if access is None:
            return None
        descriptor = self.register_small_vector_resource_index_operation(
            info,
            operation,
            access,
            right_type=right_type,
            computation_type=computation_type,
        )
        root = access["root"]
        arguments = [
            self.generate_without_structured_buffer_index_lowering(root.array, is_main),
            self.generate_expression(root.index, is_main),
        ]
        arguments.extend(
            self.generate_expression(index, is_main)
            for index in access["path_index_expressions"]
        )
        arguments.append(self.generate_expression(target.index, is_main))
        if rendered_value is not None:
            arguments.append(rendered_value)
        return f"{descriptor['name']}({', '.join(arguments)})"

    def generate_small_vector_index_support_code(self, indent=0):
        if not (
            self.small_vector_index_operations
            or self.small_vector_resource_index_operations
        ):
            return ""

        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        branch_pad = "    " * (indent + 2)
        code = f"{pad}// Metal small-vector component indexing\n"
        operation_order = {
            "get": 0,
            "set": 1,
            "add_assign": 2,
            "sub_assign": 3,
            "mul_assign": 4,
            "div_assign": 5,
            "mod_assign": 6,
            "bit_and_assign": 7,
            "bit_or_assign": 8,
            "bit_xor_assign": 9,
            "shift_left_assign": 10,
            "shift_right_assign": 11,
            "pre_increment": 12,
            "pre_decrement": 13,
            "post_increment": 14,
            "post_decrement": 15,
        }
        operations = sorted(
            self.small_vector_index_operations,
            key=lambda item: (
                self.small_vector_index_types[item[0]]["prefix"],
                operation_order[item[1]],
                item[2] or "",
                item[3] or "",
            ),
        )
        for key, operation, right_type, computation_type in operations:
            info = self.small_vector_index_types[key]
            vector_type = info["vector_type"]
            element_type = info["element_type"]
            helper_name = f"{info['prefix']}_{operation}"
            has_right = self.small_vector_index_operation_has_right(operation)
            right_name = "selected" if operation == "set" else "right"
            right_parameter = f", {right_type} {right_name}" if has_right else ""

            result_type = element_type
            value_parameter = (
                f"{vector_type} value"
                if operation == "get"
                else f"inout {vector_type} value"
            )
            code += f"{pad}@metal_static\n"
            code += (
                f"{pad}{result_type} {helper_name}"
                f"({value_parameter}, uint lane{right_parameter}) {{\n"
            )
            members = "xyzw"[: info["width"]]
            for lane, member in enumerate(members):
                condition = "if" if lane == 0 else "else if"
                if lane == len(members) - 1:
                    condition = "else"
                    code += f"{body_pad}{condition} {{\n"
                else:
                    code += f"{body_pad}{condition} (lane == {lane}u) {{\n"
                statement = self.small_vector_index_branch_statement(
                    info, operation, member, computation_type
                )
                code += f"{branch_pad}{statement}\n"
                code += f"{body_pad}}}\n"
            code += f"{pad}}}\n\n"

        for descriptor in sorted(
            self.small_vector_resource_index_operations.values(),
            key=lambda item: item["name"],
        ):
            info = self.small_vector_index_types[descriptor["info_key"]]
            operation = descriptor["operation"]
            has_right = self.small_vector_index_operation_has_right(operation)
            right_name = "selected" if operation == "set" else "right"
            parameters = [
                f"device {descriptor['container_type']}* data",
                f"{descriptor['root_index_type']} element_index",
            ]
            value_path = "value"
            path_index = 0
            for kind, value in descriptor["path"]:
                if kind == "member":
                    value_path += f".{value}"
                    continue
                parameter_name = f"path_index_{path_index}"
                path_index += 1
                parameters.append(f"{value} {parameter_name}")
                value_path += f"[{parameter_name}]"
            parameters.append("uint lane")
            if has_right:
                parameters.append(f"{descriptor['right_type']} {right_name}")

            code += f"{pad}@metal_static\n"
            code += (
                f"{pad}{info['element_type']} {descriptor['name']}"
                f"({', '.join(parameters)}) {{\n"
            )
            code += (
                f"{body_pad}{descriptor['container_type']} value = "
                "buffer_load(data, element_index);\n"
            )
            call_arguments = [value_path, "lane"]
            if has_right:
                call_arguments.append(right_name)
            code += (
                f"{body_pad}{info['element_type']} result = "
                f"{info['prefix']}_{operation}"
                f"({', '.join(call_arguments)});\n"
            )
            code += f"{body_pad}buffer_store(data, element_index, value);\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"
        return code

    def generate_wide_vector_support_code(self, indent=0):
        if not self.wide_vector_types:
            return ""

        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        code = f"{pad}// Concrete Metal vectors wider than four lanes\n"
        for info in sorted(
            self.wide_vector_types.values(), key=lambda item: item["type_name"]
        ):
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            code += f"{pad}struct {type_name} {{\n"
            code += f"{body_pad}{element_type} lanes[{width}];\n"
            code += f"{pad}}};\n\n"

            splat_name = self.wide_vector_helper_name(info, "splat")
            code += f"{pad}@metal_static\n"
            code += f"{pad}{type_name} {splat_name}({element_type} value) {{\n"
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                code += f"{body_pad}result.lanes[{lane}] = value;\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

            make_name = self.wide_vector_helper_name(info, "make")
            parameters = ", ".join(
                f"{element_type} value{lane}" for lane in range(width)
            )
            code += f"{pad}@metal_static\n"
            code += f"{pad}{type_name} {make_name}({parameters}) {{\n"
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                code += f"{body_pad}result.lanes[{lane}] = value{lane};\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

        for key, operator, left_kind, right_kind in sorted(
            self.wide_vector_binary_helpers,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        ):
            info = self.wide_vector_types[key]
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            helper_name = self.wide_vector_binary_helper_name(
                info, operator, left_kind, right_kind
            )
            left_type = type_name if left_kind == "vector" else element_type
            right_type = type_name if right_kind == "vector" else element_type
            code += f"{pad}@metal_static\n"
            code += (
                f"{pad}{type_name} {helper_name}"
                f"({left_type} left, {right_type} right) {{\n"
            )
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                left = f"left.lanes[{lane}]" if left_kind == "vector" else "left"
                right = f"right.lanes[{lane}]" if right_kind == "vector" else "right"
                expression = (
                    self.render_profiled_metal_arithmetic(
                        left, right, element_type, operator=operator
                    )
                    if operator in {"/", "+", "-", "*"}
                    else None
                )
                expression = expression or f"{left} {operator} {right}"
                code += f"{body_pad}result.lanes[{lane}] = {expression};\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

        for key, operator, right_kind in sorted(
            self.wide_vector_compound_helpers,
            key=lambda item: (item[0], item[1], item[2]),
        ):
            info = self.wide_vector_types[key]
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            helper_name = self.wide_vector_compound_helper_name(
                info, operator, right_kind
            )
            right_type = type_name if right_kind == "vector" else element_type
            code += f"{pad}@metal_static\n"
            code += (
                f"{pad}void {helper_name}"
                f"(inout {type_name} value, {right_type} right) {{\n"
            )
            for lane in range(width):
                right = f"right.lanes[{lane}]" if right_kind == "vector" else "right"
                left = f"value.lanes[{lane}]"
                expression = (
                    self.render_profiled_metal_arithmetic(
                        left, right, element_type, operator=operator
                    )
                    if operator in {"/", "+", "-", "*"}
                    else None
                )
                expression = expression or f"{left} {operator} {right}"
                code += f"{body_pad}{left} = {expression};\n"
            code += f"{pad}}}\n\n"
        return code

    def generate_wide_vector_constructor(
        self,
        metal_type,
        arguments,
        is_main=False,
        source_location=None,
        braced=False,
    ):
        info = self.wide_vector_type_info(metal_type, source_location)
        if info is None:
            return None

        arguments = list(arguments or [])
        if braced:
            if len(arguments) > info["width"] or not all(
                self.wide_vector_constructor_argument_is_scalar(argument)
                for argument in arguments
            ):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "braced construction requires at most one scalar value per lane",
                    source_location,
                    operation="constructor",
                )
            arguments.extend(
                self.wide_vector_default_value(info)
                for _lane in range(info["width"] - len(arguments))
            )
            rendered = ", ".join(
                self.generate_expression(argument, is_main) for argument in arguments
            )
            return f"{self.wide_vector_helper_name(info, 'make')}({rendered})"

        if not arguments:
            arguments = [self.wide_vector_default_value(info)]

        if len(arguments) == 1:
            argument = arguments[0]
            argument_info = self.wide_vector_type_info(
                self.expression_metal_type(argument),
                getattr(argument, "source_location", None),
            )
            rendered = self.generate_expression(argument, is_main)
            if argument_info is not None:
                if argument_info["key"] != info["key"]:
                    raise MetalWideVectorLoweringError(
                        info["source_type"],
                        "copy construction requires the same element type and width",
                        source_location,
                    )
                return rendered
            if not self.wide_vector_constructor_argument_is_scalar(argument):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "single-argument construction requires a scalar or one "
                    "matching wide vector",
                    source_location,
                    operation="constructor",
                )
            return f"{self.wide_vector_helper_name(info, 'splat')}({rendered})"

        if len(arguments) == info["width"]:
            if not all(
                self.wide_vector_constructor_argument_is_scalar(argument)
                for argument in arguments
            ):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "full construction requires one scalar value per lane",
                    source_location,
                    operation="constructor",
                )
            rendered = ", ".join(
                self.generate_expression(argument, is_main) for argument in arguments
            )
            return f"{self.wide_vector_helper_name(info, 'make')}({rendered})"

        raise MetalWideVectorLoweringError(
            info["source_type"],
            "constructor arguments must be one scalar, one matching vector, "
            f"or exactly {info['width']} scalar lanes",
            source_location,
            operation="constructor",
        )

    def wide_vector_expression_info(self, expression):
        try:
            expression_type = self.expression_metal_type(expression)
        except MetalIndexedComponentTypeResolutionError:
            return None
        if self.metal_pointer_pointee_type_once(expression_type) is not None:
            return None
        return self.wide_vector_type_info(
            expression_type,
            getattr(expression, "source_location", None),
        )

    def is_portable_atomic_scalar_type(self, mapped_type):
        """Whether an atomic target of ``mapped_type`` has a portable scalar contract.

        A type parameter the enclosing generic function leaves open is accepted: every
        backend monomorphizes generic functions and checks the bound scalar type of each
        specialization's atomic target before it emits the operation.
        """
        mapped_type = str(mapped_type or "").strip()
        if mapped_type in {"int", "uint", "float"}:
            return True
        func = self.current_function
        if func is None or getattr(func, "qualifier", None):
            return False
        bound = self.current_function_materialization_bindings or {}
        return any(
            kind in {"typename", "class"}
            and name
            and name not in bound
            and mapped_type in {name, self.sanitize_identifier(name)}
            for kind, name in getattr(func, "template_parameters", None) or ()
        )

    def metal_pointer_pointee_type_once(self, metal_type):
        if metal_type is None:
            return None
        resolved_type = str(self.resolve_type_alias(metal_type)).strip()
        while resolved_type.endswith("&"):
            resolved_type = resolved_type[:-1].strip()
        if not resolved_type.endswith("*"):
            return None
        return resolved_type[:-1].strip()

    def generate_wide_vector_binary_expression(self, expression, is_main=False):
        left_info = self.wide_vector_expression_info(expression.left)
        right_info = self.wide_vector_expression_info(expression.right)
        if left_info is None and right_info is None:
            return None

        info = left_info or right_info
        if left_info is not None and right_info is not None:
            if left_info["key"] != right_info["key"]:
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "binary operands have different element types or widths",
                    getattr(expression, "source_location", None),
                    operation=expression.op,
                )
        if self.wide_vector_binary_operation_name(expression.op) is None:
            raise MetalWideVectorLoweringError(
                info["source_type"],
                "the operator has no semantics-preserving aggregate lowering",
                getattr(expression, "source_location", None),
                operation=expression.op,
            )
        scalar_operand = expression.right if left_info is not None else expression.left
        if (left_info is None or right_info is None) and not (
            self.wide_vector_constructor_argument_is_scalar(scalar_operand)
        ):
            raise MetalWideVectorLoweringError(
                info["source_type"],
                "the non-vector operand is not scalar",
                getattr(expression, "source_location", None),
                operation=expression.op,
            )

        left_kind = "vector" if left_info is not None else "scalar"
        right_kind = "vector" if right_info is not None else "scalar"
        self.wide_vector_binary_helpers.add(
            (info["key"], expression.op, left_kind, right_kind)
        )
        helper_name = self.wide_vector_binary_helper_name(
            info, expression.op, left_kind, right_kind
        )
        left = self.generate_expression(expression.left, is_main)
        right = self.generate_expression(expression.right, is_main)
        return f"{helper_name}({left}, {right})"

    def reject_unsupported_wide_vector_call(self, expression):
        wide_arguments = [
            info
            for argument in expression.args
            for info in [self.wide_vector_expression_info(argument)]
            if info is not None
        ]
        if not wide_arguments:
            return

        if str(expression.name) in self.user_function_names:
            return

        binding, _function = self.resolve_metal_user_function_overload(
            str(expression.name), expression.args
        )
        if binding == "user":
            return
        result_type = self.metal_constructor_result_type(expression.name)
        if result_type in self.struct_member_types:
            return

        raise MetalWideVectorLoweringError(
            wide_arguments[0]["source_type"],
            "the call has no semantics-preserving aggregate overload",
            getattr(expression, "source_location", None),
            operation=f"call {expression.name}",
        )

    def wide_vector_lane_index(self, member, width):
        direct_members = {
            "x": 0,
            "r": 0,
            "y": 1,
            "g": 1,
            "z": 2,
            "b": 2,
            "w": 3,
            "a": 3,
        }
        if member in direct_members:
            index = direct_members[member]
        else:
            selector = re.fullmatch(r"s([0-9a-fA-F])", str(member))
            if selector is None:
                return None
            index = int(selector.group(1), 16)
        return index if index < width else None
