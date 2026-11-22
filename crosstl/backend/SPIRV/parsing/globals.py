"""Global declarations in the SPIR-V parser: interface, ray-storage, private and workgroup variables, uniform constants, specialization constants, structs, built-in interface blocks and constant expressions."""

import re

from ...common_ast import InitializerListNode
from ..block_arrays import block_array_element
from ..import_exact_lookups import (
    constant_or_literal,
    keyword_safe_identifier,
)
from ..matrix_order_import import ROW_MAJOR_ATTRIBUTE, member_is_row_major
from ..ray_tracing_import import (
    SpirvInstructionUnsupportedError,
    ray_variable_locations,
)
from ..type_declaration_names import declarable_struct_name
from ..VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    LayoutNode,
    MemberAccessNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..workgroup_size_import import workgroup_size_component


class VulkanGlobalMixin:
    """Global members of ``VulkanParser``."""

    def spirv_assembly_interface_variables(
        self,
        variables,
        entry_interface_ids,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        layouts = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if pointer_type.get("kind") != "pointer":
                continue

            if storage_class == "UniformConstant":
                resource_layout = self.spirv_assembly_uniform_constant_layout(
                    variable, pointer_type, names, decorations, types, constants
                )
                if resource_layout is not None:
                    layouts.append(resource_layout)
                continue

            if storage_class in {
                "PushConstant",
                "StorageBuffer",
                "Uniform",
                *self.SPIRV_SHADER_RECORD_STORAGE_CLASSES,
            }:
                resource_block_layout = self.spirv_assembly_resource_block_layout(
                    variable,
                    pointer_type,
                    storage_class,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    constants,
                )
                if resource_block_layout is not None:
                    layouts.append(resource_block_layout)
                continue

            ray_qualifier = self.SPIRV_RAY_STORAGE_QUALIFIERS.get(storage_class)
            if ray_qualifier is not None:
                ray_layout = self.spirv_assembly_ray_storage_layout(
                    variable,
                    pointer_type,
                    storage_class,
                    ray_qualifier,
                    names,
                    decorations,
                    types,
                    constants,
                    ray_variable_locations(variables, decorations),
                )
                if ray_layout is not None:
                    layouts.append(ray_layout)
                continue

            layout_type = self.SPIRV_INTERFACE_STORAGE_CLASSES.get(storage_class)
            if layout_type is None:
                continue
            if entry_interface_ids and variable["id"] not in entry_interface_ids:
                continue

            struct_layouts = self.spirv_assembly_struct_interface_layouts(
                variable,
                pointer_type,
                storage_class,
                names,
                decorations.get(variable["id"], []),
                member_decorations,
                member_names,
                types,
                constants,
            )
            if struct_layouts:
                layouts.extend(struct_layouts)
                continue

            variable_decorations = decorations.get(variable["id"], [])
            qualifiers = self.spirv_layout_qualifiers(variable_decorations)
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                variable_decorations
            )
            if not self.spirv_has_interface_qualifier(
                qualifiers, declaration_qualifiers
            ):
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants
            )
            if data_type is None:
                continue

            variable_name = self.spirv_assembly_value_name(
                variable["id"], names, decorations={}
            )
            variable_name = self.spirv_builtin_variable_name_from_qualifiers(
                qualifiers, variable_name, storage_class=storage_class
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=layout_type,
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=variable["id"],
                    spirv_decorations=variable_decorations,
                    spirv_storage_class=storage_class,
                )
            )

        return layouts

    def spirv_assembly_ray_storage_layout(
        self,
        variable,
        pointer_type,
        storage_class,
        ray_qualifier,
        names,
        decorations,
        types,
        constants,
        ray_locations,
    ):
        """A ray payload, callable data or hit attribute variable, as its GL_EXT_ray_tracing declaration."""
        data_type, array_suffix = self.spirv_type_name_and_suffix(
            pointer_type.get("type_id"), types, constants, names=names
        )
        if data_type is None:
            return None
        variable_decorations = decorations.get(variable["id"], [])
        qualifiers = [
            (name, value)
            for name, value in self.spirv_layout_qualifiers(variable_decorations)
            if name == "location"
        ]
        location = ray_locations.get(variable["id"])
        if not qualifiers and location is not None:
            qualifiers = [("location", str(location))]
        variable_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations={}
        )
        return LayoutNode(
            qualifiers,
            data_type=data_type,
            variable_name=f"{variable_name}{array_suffix}",
            declaration_qualifiers=[ray_qualifier],
            spirv_id=variable["id"],
            spirv_decorations=variable_decorations,
            spirv_storage_class=storage_class,
        )

    def spirv_assembly_private_global_variables(
        self, variables, names, decorations, types, constants
    ):
        declarations = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class != "Private" or pointer_type.get("kind") != "pointer":
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants, names=names
            )
            if data_type is None:
                continue

            # Private storage is the invocation's own, as CrossGL's static module-scope variable is; a plain one is
            # read-only uniform data in HLSL and constant data in Metal (SPIR-V 1.6 section 3.7, Private).
            declaration = VariableNode(
                f"static {data_type}",
                f"{self.spirv_assembly_value_name(variable['id'], names, decorations)}"
                f"{array_suffix}",
                spirv_id=variable["id"],
                spirv_type_id=pointer_type.get("type_id"),
            )
            initializer = variable.get("initializer")
            if initializer is not None:
                declarations.append(
                    AssignmentNode(
                        declaration,
                        self.spirv_assembly_operand_expression(
                            initializer,
                            {},
                            names,
                            decorations,
                            constants,
                        ),
                    )
                )
            else:
                declarations.append(declaration)

        return declarations

    def spirv_assembly_workgroup_global_variables(
        self, variables, names, decorations, types, constants
    ):
        declarations = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            # A task or mesh stage's payload (SPV_EXT_mesh_shader) is GL_EXT_mesh_shader's taskPayloadSharedEXT.
            qualifier = {
                "Workgroup": "groupshared",
                "TaskPayloadWorkgroupEXT": "taskPayloadSharedEXT",
            }.get(storage_class)
            if qualifier is None or pointer_type.get("kind") != "pointer":
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants, names=names
            )
            if data_type is None:
                continue

            declarations.append(
                VariableNode(
                    f"{qualifier} {data_type}",
                    f"{self.spirv_assembly_value_name(variable['id'], names, decorations)}"
                    f"{array_suffix}",
                    spirv_id=variable["id"],
                    spirv_type_id=pointer_type.get("type_id"),
                    spirv_storage_class=storage_class,
                )
            )

        return declarations

    def spirv_assembly_uniform_constant_layout(
        self, variable, pointer_type, names, decorations, types, constants
    ):
        resource_type = types.get(pointer_type.get("type_id"), {})
        resource_element_type = self.spirv_resource_element_type(resource_type, types)
        data_type, array_suffix = self.spirv_type_name_and_suffix(
            pointer_type.get("type_id"), types, constants
        )
        if data_type is None:
            return None

        variable_decorations = decorations.get(variable["id"], [])
        qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
        image_format = self.spirv_image_format_qualifier(
            resource_element_type.get("format")
        )
        if image_format is not None:
            qualifiers.append((image_format, None))
        qualifier_names = {name for name, _value in qualifiers}
        if not {"set", "binding"}.issubset(qualifier_names):
            return None

        declaration_qualifiers = self.spirv_declaration_qualifiers(variable_decorations)
        access_qualifier = self.spirv_image_access_qualifier(
            resource_element_type.get("access_qualifier")
        )
        if access_qualifier and access_qualifier not in declaration_qualifiers:
            declaration_qualifiers.append(access_qualifier)

        variable_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations
        )
        variable_name += array_suffix
        return LayoutNode(
            qualifiers,
            layout_type="UNIFORM",
            data_type=data_type,
            variable_name=variable_name,
            declaration_qualifiers=declaration_qualifiers,
            spirv_id=variable["id"],
            spirv_decorations=variable_decorations,
            spirv_storage_class="UniformConstant",
        )

    def spirv_resource_element_type(self, resource_type, types):
        while resource_type.get("kind") in {"array", "runtime_array"}:
            resource_type = types.get(resource_type.get("element_type"), {})
        return resource_type

    def spirv_assembly_specialization_constants(
        self, spec_constant_ids, names, decorations, types, constants, constant_types
    ):
        layouts = []
        for result_id in spec_constant_ids:
            constant_decorations = decorations.get(result_id, [])
            constant_id = self.spirv_spec_constant_id(constant_decorations)

            qualifiers = []
            if constant_id is not None:
                qualifiers.append(("constant_id", constant_id))
            qualifiers.extend(self.spirv_layout_qualifiers(constant_decorations))

            variable_name = self.spirv_builtin_variable_name_from_qualifiers(qualifiers)
            if variable_name is None:
                variable_name = names.get(result_id)
            if variable_name is None:
                if constant_id is None:
                    continue
                variable_name = self.spirv_fallback_identifier(
                    constant_id, "spec_constant"
                )

            # An OpSpecConstantComposite may build an array (SPIR-V 1.6 section 3.52.7); the importer reads a
            # named one by name, so it is declared as a sized const array.
            data_type, array_suffix = self.spirv_type_name_and_suffix(
                constant_types.get(result_id), types, constants
            )
            if data_type is None:
                raise SpirvInstructionUnsupportedError(
                    "OpSpecConstantComposite",
                    f"declares {variable_name} with a type CrossGL cannot name",
                )
            declaration = AssignmentNode(
                VariableNode(f"const {data_type}", f"{variable_name}{array_suffix}"),
                constants.get(result_id),
            )
            layouts.append(
                LayoutNode(
                    qualifiers,
                    declaration=declaration,
                    layout_type="CONST",
                    spirv_id=result_id,
                    spirv_decorations=constant_decorations,
                )
            )

        return layouts

    def spirv_register_struct_type_names(self, types, names, member_names=None):
        # OpName may give several struct types one name (glslang calls every modf and frexp result ``ResType``).
        # Struct types whose names and members agree are declared once; any other one takes the first free
        # ``name_N``, so no two CrossGL struct declarations conflict.
        member_names = member_names or {}
        taken = {}
        used_names = set(names.values())
        for type_id, type_info in types.items():
            if type_info.get("kind") != "struct":
                continue
            name = declarable_struct_name(
                names.get(type_id) or self.spirv_fallback_identifier(type_id, "struct"),
                used_names,
            )
            signature = (
                tuple(type_info.get("member_types", [])),
                tuple(sorted(member_names.get(type_id, {}).items())),
            )
            if taken.get(name, signature) != signature:
                suffix = 1
                while f"{name}_{suffix}" in taken or f"{name}_{suffix}" in used_names:
                    suffix += 1
                name = f"{name}_{suffix}"
            taken.setdefault(name, signature)
            type_info["name"] = name

    def spirv_assembly_structs(
        self,
        names,
        member_names,
        member_decorations,
        types,
        constants,
        skip_type_ids=None,
        builtin_interface_struct_storage_classes=None,
    ):
        skip_type_ids = set(skip_type_ids or [])
        builtin_interface_struct_storage_classes = (
            builtin_interface_struct_storage_classes or {}
        )
        structs = []
        seen_struct_signatures = set()
        for type_id, type_info in types.items():
            if type_info.get("kind") != "struct":
                continue
            if type_id in skip_type_ids:
                continue

            members = []
            for member_index, member_type_id in enumerate(
                type_info.get("member_types", [])
            ):
                data_type, array_suffix = self.spirv_type_name_and_suffix(
                    member_type_id, types, constants, names=names
                )
                if data_type is None:
                    continue

                member_key = str(member_index)
                member_layout_decorations = self.spirv_member_decorations_for_member(
                    member_decorations, type_id, member_key
                )
                member_name = None
                if type_id in builtin_interface_struct_storage_classes:
                    member_name = self.spirv_builtin_variable_name_from_qualifiers(
                        self.spirv_layout_qualifiers(member_layout_decorations),
                        storage_class=(
                            builtin_interface_struct_storage_classes.get(type_id)
                        ),
                    )
                if member_name is None:
                    member_name = member_names.get(type_id, {}).get(
                        member_key, f"member{member_key}"
                    )
                member_node = VariableNode(data_type, f"{member_name}{array_suffix}")
                if type_id in builtin_interface_struct_storage_classes:
                    member_node.declaration_qualifiers = (
                        self.spirv_declaration_qualifiers(member_layout_decorations)
                    )
                if member_is_row_major(member_decorations, type_id, member_key):
                    member_node.declaration_qualifiers = list(
                        getattr(member_node, "declaration_qualifiers", None) or []
                    ) + [ROW_MAJOR_ATTRIBUTE]
                members.append(member_node)

            if members:
                signature = (
                    type_info["name"],
                    tuple((member.vtype, member.name) for member in members),
                )
                if signature in seen_struct_signatures:
                    continue
                seen_struct_signatures.add(signature)
                struct = StructNode(type_info["name"], members)
                storage_class = builtin_interface_struct_storage_classes.get(type_id)
                if storage_class is not None:
                    struct.spirv_builtin_interface_storage_class = storage_class
                    struct.spirv_builtin_interface_block_name = (
                        self.spirv_builtin_interface_block_name(
                            type_info["name"], type_id, member_decorations
                        )
                    )
                structs.append(struct)

        return structs

    def spirv_builtin_interface_struct_storage_classes(
        self, variables, types, member_decorations
    ):
        storage_classes = {}
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class not in self.SPIRV_INTERFACE_STORAGE_CLASSES:
                continue

            struct_type_id = pointer_type.get("type_id")
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue
            if not self.spirv_struct_has_builtin_member_decorations(
                struct_type_id, member_decorations
            ):
                continue

            storage_classes.setdefault(struct_type_id, storage_class)
        return storage_classes

    def spirv_struct_has_builtin_member_decorations(
        self, struct_type_id, member_decorations
    ):
        return any(
            decoration == "BuiltIn"
            for _member, decoration, _operands in member_decorations.get(
                struct_type_id, []
            )
        )

    def spirv_member_decorations_for_member(
        self, member_decorations, struct_type_id, member_key
    ):
        return [
            (decoration, operands)
            for member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if member == member_key
        ]

    def spirv_builtin_interface_block_name(
        self, fallback_name, struct_type_id, member_decorations
    ):
        builtin_names = {
            operands[0]
            for _member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if decoration == "BuiltIn" and operands
        }
        per_vertex_builtins = {
            "Position",
            "PointSize",
            "ClipDistance",
            "CullDistance",
        }
        if builtin_names and builtin_names.issubset(per_vertex_builtins):
            return "gl_PerVertex"
        return fallback_name

    def spirv_resource_block_struct_type_ids(self, variables, types, decorations):
        resource_block_type_ids = set()
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class not in {"PushConstant", "StorageBuffer", "Uniform"}:
                continue

            struct_type_id = pointer_type.get("type_id")
            array_element = block_array_element(struct_type_id, types)
            if array_element is not None:
                struct_type_id = array_element[0]
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue

            struct_decorations = decorations.get(struct_type_id, [])
            has_block = self.spirv_has_decoration(struct_decorations, "Block")
            has_buffer_block = self.spirv_has_decoration(
                struct_decorations, "BufferBlock"
            )

            if storage_class == "PushConstant" and has_block:
                resource_block_type_ids.add(struct_type_id)
            elif storage_class == "StorageBuffer" and has_block:
                resource_block_type_ids.add(struct_type_id)
            elif storage_class == "Uniform" and (has_block or has_buffer_block):
                resource_block_type_ids.add(struct_type_id)

        return resource_block_type_ids

    def spirv_spec_constant_id(self, decorations):
        for decoration, operands in decorations:
            if decoration == "SpecId" and operands:
                return operands[0]
        return None

    def spirv_constant_composite_expression(
        self, type_id, constituent_ids, names, types, constants
    ):
        type_name = self.spirv_type_name(type_id, types)
        args = [
            self.spirv_constant_operand_expression(constituent_id, names, constants)
            for constituent_id in constituent_ids
        ]
        kind = types.get(type_id, {}).get("kind")
        if kind in {"array", "runtime_array"}:
            # CrossGL writes array values as initializer lists, as OpCompositeConstruct does.
            return InitializerListNode(args)
        if kind == "struct":
            # Struct types are named only once every OpName is read;
            # spirv_name_struct_constants gives the constructor its name then.
            constructor = FunctionCallNode(type_id, args)
            constructor.spirv_struct_type_id = type_id
            return constructor
        if type_name is None:
            raise SpirvInstructionUnsupportedError(
                "OpConstantComposite", f"of type {type_id} has no CrossGL constructor"
            )
        return FunctionCallNode(type_name, args)

    @staticmethod
    def spirv_name_struct_constants(constants, types):
        """Name the struct constructors of constant composites after their now-registered struct types."""
        for value in constants.values():
            type_id = getattr(value, "spirv_struct_type_id", None)
            if isinstance(value, FunctionCallNode) and type_id is not None:
                value.name = types[type_id]["name"]

    def spirv_spec_constant_op_expression(
        self, operation, operands, names, constants, types
    ):
        if (
            operation in {
                "CooperativeMatrixLengthKHR",
                "CooperativeMatrixLengthNV",
            }
            and operands
        ):
            length = self.spirv_cooperative_matrix_length(operands[0], types, constants)
            if length is not None:
                return str(length)
            return "1"

        args = [
            self.spirv_constant_operand_expression(operand, names, constants)
            for operand in operands
        ]
        if operation in self.SPIRV_SPEC_CONSTANT_BINARY_OPS and len(args) == 2:
            return BinaryOpNode(
                args[0], self.SPIRV_SPEC_CONSTANT_BINARY_OPS[operation], args[1]
            )
        if operation in self.SPIRV_SPEC_CONSTANT_UNARY_OPS and len(args) == 1:
            return UnaryOpNode(self.SPIRV_SPEC_CONSTANT_UNARY_OPS[operation], args[0])
        if operation == "Select" and len(args) == 3:
            return TernaryOpNode(args[0], args[1], args[2])
        if operation == "CompositeExtract" and len(args) >= 2:
            expression = args[0]
            for index in args[1:]:
                extracted = workgroup_size_component(
                    expression, index
                ) or self.spirv_static_composite_element(expression, index)
                expression = (
                    extracted
                    if extracted is not None
                    else ArrayAccessNode(expression, index)
                )
            return expression
        return FunctionCallNode(f"spirv_{operation}", args)

    def spirv_static_composite_element(self, expression, index):
        try:
            element_index = int(str(index), 0)
        except (TypeError, ValueError):
            return None

        if isinstance(expression, FunctionCallNode):
            elements = expression.args
        elif isinstance(expression, InitializerListNode):
            elements = expression.elements
        else:
            return None

        if 0 <= element_index < len(elements):
            return elements[element_index]
        return None

    def spirv_constant_operand_expression(self, operand, names, constants):
        value = constants.get(operand)
        if self.spirv_is_generated_constant_expression(value):
            return value
        if names and operand in names:
            return names[operand]
        if value is not None:
            return value
        if isinstance(operand, str) and operand.startswith("%"):
            return operand.lstrip("%")
        return operand

    def spirv_constant_literal_expression(self, type_id, literal, types):
        type_info = types.get(type_id, {})
        if type_info.get("name") == "uint" and str(literal).isdigit():
            # CrossGL reads an unsuffixed literal as int; one above INT_MAX would widen to a 64-bit integer.
            return f"{literal}u" if int(str(literal)) > 0x7FFFFFFF else literal
        if type_info.get("name") not in {"half", "float", "double"}:
            return literal

        text = str(literal)
        stripped = text.lstrip("+-")
        if stripped.isdigit():
            # spirv-dis prints integral floating-point values without a fraction; CrossGL reads those as integers.
            return f"{text}.0"
        if not stripped.lower().startswith("0x") or "p" not in stripped.lower():
            return literal

        try:
            float.fromhex(text)
        except OverflowError:
            return "-1e+309" if text.startswith("-") else "1e+309"
        except ValueError:
            return literal
        return literal

    def spirv_integer_constant_operand(self, operand, constants):
        value = constant_or_literal(constants, operand)
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def spirv_constant_expression_text(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, VariableNode):
            return value.name
        if isinstance(value, BinaryOpNode):
            left = self.spirv_constant_expression_text(value.left)
            right = self.spirv_constant_expression_text(value.right)
            return f"({left} {value.op} {right})"
        if isinstance(value, UnaryOpNode):
            operand = self.spirv_constant_expression_text(value.operand)
            return f"{value.op}{operand}"
        if isinstance(value, TernaryOpNode):
            condition = self.spirv_constant_expression_text(value.condition)
            true_expr = self.spirv_constant_expression_text(value.true_expr)
            false_expr = self.spirv_constant_expression_text(value.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        if isinstance(value, FunctionCallNode):
            args = ", ".join(
                self.spirv_constant_expression_text(arg) for arg in value.args
            )
            return f"{value.name}({args})"
        if isinstance(value, ArrayAccessNode):
            array = self.spirv_constant_expression_text(value.array)
            index = self.spirv_constant_expression_text(value.index)
            return f"{array}[{index}]"
        if isinstance(value, MemberAccessNode):
            return f"{self.spirv_constant_expression_text(value.object)}.{value.member}"
        return str(value)

    def spirv_string_literal(self, value):
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def spirv_fallback_identifier(self, raw_value, prefix):
        identifier = re.sub(r"\W", "_", str(raw_value or "").lstrip("%"))
        if not identifier:
            return keyword_safe_identifier(self.SPIRV_VALUE_FALLBACK_RENAMES, prefix)
        if identifier[0].isdigit():
            identifier = f"{prefix}_{identifier}"
        if prefix == "part" and identifier.endswith("_"):
            identifier = identifier.rstrip("_") or prefix
        return keyword_safe_identifier(self.SPIRV_VALUE_FALLBACK_RENAMES, identifier)

    def spirv_identifier_name(self, raw_name, fallback_value=None, prefix="value"):
        name = str(raw_name or "")
        signature_index = name.find("(")
        if signature_index > 0:
            name = name[:signature_index]

        identifier = re.sub(r"[^0-9A-Za-z_]", "_", name)
        if not identifier:
            identifier = self.spirv_fallback_identifier(fallback_value, prefix)
        if identifier and identifier[0].isdigit():
            identifier = f"{prefix}_{identifier}"
        if identifier in self.CROSSGL_RESERVED_IDENTIFIERS:
            identifier = f"{identifier}_"
        return identifier
