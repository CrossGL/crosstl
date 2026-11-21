"""Resource block and interface layouts, descriptor and location qualifiers, and SPIR-V type names in the SPIR-V parser."""

from ..block_arrays import (
    SpirvBlockArrayUnsupportedError,
    block_array_element,
)
from ..block_layout_import import spirv_block_layout
from ..buffer_memory_qualifiers import storage_block_memory_qualifiers
from ..image_formats import crossgl_image_format
from ..import_exact_lookups import (
    builtin_variable_name,
    constant_or_literal,
)
from ..matrix_order_import import member_is_row_major
from ..VulkanAst import LayoutNode


class VulkanLayoutMixin:
    """Layout members of ``VulkanParser``."""

    def spirv_assembly_resource_block_layout(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        instance_suffix = ""
        array_element = block_array_element(struct_type_id, types)
        if array_element is not None:
            struct_type_id, length_id = array_element
            instance_suffix = "[]"
            if length_id is not None:
                length = self.spirv_constant_expression_text(
                    constants.get(length_id, str(length_id).lstrip("%"))
                )
                instance_suffix = f"[{length}]"
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return None

        struct_decorations = decorations.get(struct_type_id, [])
        variable_id = variable["id"]
        variable_decorations = decorations.get(variable_id, [])
        has_block = self.spirv_has_decoration(struct_decorations, "Block")
        has_buffer_block = self.spirv_has_decoration(struct_decorations, "BufferBlock")
        if storage_class == "PushConstant":
            if not has_block:
                return None
            layout_type = "UNIFORM"
        elif storage_class == "Uniform":
            if has_buffer_block:
                layout_type = "BUFFER"
            elif has_block:
                layout_type = "UNIFORM"
            else:
                return None
        elif storage_class == "StorageBuffer":
            if not has_block:
                return None
            layout_type = "BUFFER"
        elif storage_class in self.SPIRV_SHADER_RECORD_STORAGE_CLASSES:
            if not has_block:
                return None
            layout_type = "SHADER_RECORD"
        else:
            return None
        if instance_suffix and layout_type != "BUFFER":
            raise SpirvBlockArrayUnsupportedError(
                self.spirv_assembly_value_name(variable_id, names, decorations)
                or variable_id,
                storage_class,
            )

        struct_fields = []
        struct_field_indices = []
        row_major_fields = set()
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue
            struct_field_indices.append(member_index)

            member_key = str(member_index)
            field_name = variable.get("flattened_member_aliases", {}).get(
                member_key
            ) or member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            if member_is_row_major(member_decorations, struct_type_id, member_key):
                row_major_fields.add(len(struct_fields))
            struct_fields.append((data_type, f"{field_name}{array_suffix}"))

        if not struct_fields:
            return None

        variable_name = self.spirv_assembly_value_name(variable_id, names, decorations)
        # A storage block is declared as a struct type plus a variable of it, so an unnamed block type takes a name
        # of its own: the variable's name would make each use of the variable also read as a use of the type.
        block_name = (
            names.get(struct_type_id)
            or (None if layout_type == "BUFFER" else variable_name)
            or self.spirv_fallback_identifier(struct_type_id, "block")
        )
        qualifiers = []
        if storage_class in {"StorageBuffer", "Uniform"}:
            qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
            qualifier_names = {name for name, _value in qualifiers}
            if not {"set", "binding"}.issubset(qualifier_names):
                return None
        declaration_qualifiers = self.spirv_declaration_qualifiers(
            struct_decorations + variable_decorations
        )
        if layout_type == "BUFFER":
            memory_qualifiers = storage_block_memory_qualifiers(
                struct_decorations + variable_decorations,
                member_decorations.get(struct_type_id, []),
                len(struct_type.get("member_types", [])),
            )
            declaration_qualifiers = [
                qualifier
                for qualifier in declaration_qualifiers
                if qualifier not in memory_qualifiers
            ] + memory_qualifiers

        node = LayoutNode(
            qualifiers,
            layout_type=layout_type,
            push_constant=storage_class == "PushConstant",
            block_name=block_name,
            variable_name=variable_name,
            struct_fields=struct_fields,
            declaration_qualifiers=declaration_qualifiers,
            spirv_id=variable_id,
            spirv_decorations=struct_decorations + variable_decorations,
            spirv_storage_class=storage_class,
        )
        node.instance_array_suffix = instance_suffix
        node.row_major_fields = frozenset(row_major_fields)
        if storage_class in {"Uniform", "PushConstant"} and layout_type == "UNIFORM":
            block_layout = spirv_block_layout(
                block_name,
                struct_type_id,
                storage_class == "PushConstant",
                types,
                decorations,
                member_decorations,
                lambda length_id: self.spirv_integer_constant_operand(
                    length_id, constants
                ),
            )
            if block_layout is not None:
                node.buffer_layout = block_layout.layout
                if block_layout.member_offsets is not None:
                    # One explicit offset per emitted field, in struct_fields order.
                    node.member_offsets = tuple(
                        block_layout.member_offsets[index]
                        for index in struct_field_indices
                    )
        return node

    def spirv_has_decoration(self, decorations, target_decoration):
        return any(decoration == target_decoration for decoration, _ in decorations)

    def spirv_assign_flattened_resource_block_member_aliases(
        self,
        variables,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        occurrences = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            struct_type_id = pointer_type.get("type_id")
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue
            if not self.spirv_is_flattened_resource_block_access(
                storage_class, struct_type_id, decorations
            ):
                continue
            if storage_class == "Uniform" and not self.spirv_has_descriptor_binding(
                decorations.get(variable["id"], [])
            ):
                continue

            variable_name = self.spirv_assembly_value_name(
                variable["id"], names, decorations
            )
            for member_index, member_type_id in enumerate(
                struct_type.get("member_types", [])
            ):
                data_type, _array_suffix = self.spirv_type_name_and_suffix(
                    member_type_id, types, constants
                )
                if data_type is None:
                    continue

                member_key = str(member_index)
                member_name = member_names.get(struct_type_id, {}).get(
                    member_key, f"member{member_key}"
                )
                occurrences.append(
                    {
                        "variable": variable,
                        "variable_name": variable_name,
                        "member_key": member_key,
                        "member_name": member_name,
                    }
                )

        member_counts = {}
        for occurrence in occurrences:
            member_counts[occurrence["member_name"]] = (
                member_counts.get(occurrence["member_name"], 0) + 1
            )
        duplicate_members = {
            member_name for member_name, count in member_counts.items() if count > 1
        }
        if not duplicate_members:
            return

        used_names = {
            occurrence["member_name"]
            for occurrence in occurrences
            if occurrence["member_name"] not in duplicate_members
        }
        for occurrence in occurrences:
            if occurrence["member_name"] not in duplicate_members:
                continue

            alias_base = self.spirv_identifier_name(
                f"{occurrence['variable_name']}_{occurrence['member_name']}",
                f"{occurrence['variable']['id']}_{occurrence['member_key']}",
                prefix="member",
            )
            alias = self.spirv_unique_identifier(alias_base, used_names)
            used_names.add(alias)
            occurrence["variable"].setdefault("flattened_member_aliases", {})[
                occurrence["member_key"]
            ] = alias

    def spirv_has_descriptor_binding(self, decorations):
        qualifier_names = {
            name for name, _value in self.spirv_descriptor_qualifiers(decorations)
        }
        return {"set", "binding"}.issubset(qualifier_names)

    def spirv_unique_identifier(self, base_identifier, used_identifiers):
        identifier = base_identifier
        suffix = 1
        while identifier in used_identifiers:
            identifier = f"{base_identifier}_{suffix}"
            suffix += 1
        return identifier

    def spirv_descriptor_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            if not operands:
                continue
            if decoration == "DescriptorSet":
                qualifiers.append(("set", operands[0]))
            elif decoration == "Binding":
                qualifiers.append(("binding", operands[0]))
        return qualifiers

    def spirv_assembly_struct_interface_layouts(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        variable_decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return []

        layouts = []
        block_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations={}
        )
        variable_qualifiers = self.spirv_layout_qualifiers(variable_decorations)
        variable_declaration_qualifiers = self.spirv_declaration_qualifiers(
            variable_decorations
        )
        location_cursor = self.spirv_interface_base_location(variable_qualifiers)
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            member_key = str(member_index)
            member_layout_decorations = [
                (decoration, operands)
                for member, decoration, operands in member_decorations.get(
                    struct_type_id, []
                )
                if member == member_key
            ]
            qualifiers = self.spirv_layout_qualifiers(member_layout_decorations)
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                member_layout_decorations
            )
            if not self.spirv_has_interface_qualifier(
                qualifiers, declaration_qualifiers
            ):
                if location_cursor is None:
                    continue
                qualifiers = self.spirv_interface_member_location_qualifiers(
                    variable_qualifiers, location_cursor
                )
                declaration_qualifiers = list(variable_declaration_qualifiers)

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue

            variable_name = self.spirv_struct_member_variable_name(
                struct_type_id,
                member_key,
                block_name,
                qualifiers,
                member_names,
                storage_class=storage_class,
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=self.SPIRV_INTERFACE_STORAGE_CLASSES[storage_class],
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=f"{variable['id']}.{member_key}",
                    spirv_decorations=member_layout_decorations,
                    spirv_storage_class=storage_class,
                )
            )
            if location_cursor is not None:
                location_cursor += self.spirv_interface_location_count(
                    member_type_id, types, constants
                )

        return layouts

    def spirv_interface_base_location(self, qualifiers):
        for name, value in qualifiers:
            if name != "location":
                continue
            try:
                return int(str(value), 0)
            except (TypeError, ValueError):
                return None
        return None

    def spirv_interface_member_location_qualifiers(self, variable_qualifiers, location):
        qualifiers = [
            (name, value) for name, value in variable_qualifiers if name != "location"
        ]
        qualifiers.append(("location", str(location)))
        return qualifiers

    def spirv_interface_location_count(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        kind = type_info.get("kind")

        if kind == "array":
            length = self.spirv_integer_constant_operand(
                type_info.get("length_id"), constants
            )
            if length is None:
                return 1
            return max(
                1,
                length
                * self.spirv_interface_location_count(
                    type_info.get("element_type"), types, constants
                ),
            )

        if kind == "matrix":
            try:
                return max(1, int(str(type_info.get("column_count")), 0))
            except (TypeError, ValueError):
                return 1

        if kind == "struct":
            count = sum(
                self.spirv_interface_location_count(member_type_id, types, constants)
                for member_type_id in type_info.get("member_types", [])
            )
            return max(1, count)

        return 1

    def spirv_layout_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            qualifier_name = self.SPIRV_INTERFACE_DECORATIONS.get(decoration)
            if qualifier_name and operands:
                qualifiers.append((qualifier_name, operands[0]))
        return qualifiers

    def spirv_declaration_qualifiers(self, decorations):
        qualifiers = []
        for decoration, _operands in decorations:
            qualifier = self.SPIRV_DECLARATION_DECORATION_QUALIFIERS.get(decoration)
            if qualifier:
                qualifiers.append(qualifier)
        return qualifiers

    def spirv_image_format_qualifier(self, image_format):
        return crossgl_image_format(image_format)

    def spirv_image_access_qualifier(self, access_qualifier):
        return {"ReadOnly": "readonly", "WriteOnly": "writeonly"}.get(access_qualifier)

    def spirv_has_interface_qualifier(self, qualifiers, declaration_qualifiers=None):
        if any(name in {"builtin", "location"} for name, _value in qualifiers):
            return True
        declaration_qualifiers = set(declaration_qualifiers or [])
        return bool(
            declaration_qualifiers
            & {
                "centroid",
                "flat",
                "invariant",
                "noperspective",
                "patch",
                "perprimitiveEXT",
                "pervertexEXT",
                "sample",
            }
        )

    def spirv_struct_member_variable_name(
        self,
        struct_type_id,
        member_key,
        block_name,
        qualifiers,
        member_names,
        storage_class=None,
    ):
        member_name = member_names.get(struct_type_id, {}).get(member_key)
        if member_name:
            return member_name

        builtin_name = self.spirv_builtin_variable_name_from_qualifiers(
            qualifiers, storage_class=storage_class
        )
        if builtin_name:
            return builtin_name

        if block_name:
            return f"{block_name}_{member_key}"
        return f"member{member_key}"

    def spirv_builtin_variable_name_from_qualifiers(
        self, qualifiers, fallback_name=None, storage_class=None
    ):
        for name, value in qualifiers:
            if name == "builtin":
                storage_builtin_name = self.SPIRV_STORAGE_BUILTIN_VARIABLE_NAMES.get(
                    (value, storage_class)
                )
                if storage_builtin_name:
                    return storage_builtin_name
                if (
                    value == "PrimitiveId"
                    and storage_class == "Input"
                    and getattr(self, "spirv_geometry_module", False)
                ):
                    # A geometry stage reads the incoming primitive's ID as gl_PrimitiveIDIn and writes the one it
                    # passes on as gl_PrimitiveID (GLSL 4.60 section 7.1.4); both are BuiltIn PrimitiveId.
                    return "gl_PrimitiveIDIn"
                return builtin_variable_name(
                    self.SPIRV_BUILTIN_VARIABLE_NAMES, value, storage_class
                )
        return fallback_name

    def spirv_type_name_and_suffix(self, type_id, types, constants, names=None):
        type_info = types.get(type_id)
        if type_info is None:
            return None, ""

        if type_info.get("kind") == "array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants, names=names
            )
            if base_type is None:
                return None, ""
            length_id = type_info.get("length_id")
            length = self.spirv_constant_expression_text(
                constants.get(length_id, str(length_id).lstrip("%"))
            )
            return base_type, f"[{length}]{suffix}"

        if type_info.get("kind") == "runtime_array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants, names=names
            )
            if base_type is None:
                return None, ""
            return base_type, f"[]{suffix}"

        if type_info.get("kind") == "pointer":
            if names and names.get(type_id):
                return names[type_id], ""
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("type_id"), types, constants, names=names
            )
            if base_type is None:
                return self.spirv_fallback_identifier(type_id, "ptr"), ""
            return f"{base_type}*", suffix

        return type_info.get("name"), ""

    def spirv_type_name(self, type_id, types):
        type_info = types.get(type_id)
        if type_info is None:
            return None
        return type_info.get("name")

    def spirv_sampled_image_type_name(self, image_type, types):
        sampled_type = self.spirv_type_name(image_type.get("sampled_type"), types)
        return self.spirv_image_type_name(
            sampled_type,
            image_type.get("dim"),
            image_type.get("depth"),
            image_type.get("arrayed"),
            image_type.get("multisampled"),
            "1",
            combined=True,
        )

    def spirv_image_query_name(
        self, sampled_name, storage_name, image_operand, expression_type_ids, types
    ):
        """GLSL spells a size or sample-count query of a storage image (``Sampled`` 2) ``imageSize``/``imageSamples``.

        ``OpImageQuerySize`` and ``OpImageQuerySamples`` take an image of either kind (SPIR-V 1.6 section 3.49.10);
        ``textureSize`` of an ``image2D`` is not a GLSL overload, and the emitters refuse it.
        """
        image_type = types.get(expression_type_ids.get(image_operand), {})
        if image_type.get("kind") == "image" and str(image_type.get("sampled")) == "2":
            return storage_name
        return sampled_name

    def spirv_image_type_name(
        self, sampled_type, dim, depth, arrayed, multisampled, sampled, combined=False
    ):
        if dim == "SubpassData":
            return "subpassInputMS" if multisampled == "1" else "subpassInput"

        base_name = self.spirv_image_base_type_name(dim, arrayed, multisampled)
        if base_name is None:
            return None

        sampled_family = self.spirv_image_sampled_type_family(sampled_type)
        if sampled == "2":
            prefix = {"int": "i", "uint": "u"}.get(sampled_family, "")
            return f"{prefix}image{base_name}"

        prefix = {"int": "i", "uint": "u"}.get(sampled_family, "")
        if sampled == "1" and not combined:
            # An OpTypeImage read through a separate sampler is Vulkan GLSL's texture2D: only an OpTypeSampledImage
            # is a combined sampler2D. The depth flag belongs to the sampler type the shader samples it as.
            return f"{prefix}texture{base_name}"
        suffix = "Shadow" if depth == "1" and not prefix else ""
        return f"{prefix}sampler{base_name}{suffix}"

    def spirv_image_sampled_type_family(self, sampled_type):
        if sampled_type in {"int", "i8", "i16", "i64"}:
            return "int"
        if sampled_type in {"uint", "u8", "u16", "u64"}:
            return "uint"
        return sampled_type

    def spirv_image_base_type_name(self, dim, arrayed, multisampled):
        if dim == "Buffer":
            return "Buffer"
        if dim == "Cube":
            return "CubeArray" if arrayed == "1" else "Cube"
        if dim == "1D":
            return "1DArray" if arrayed == "1" else "1D"
        if dim == "2D":
            if multisampled == "1":
                return "2DMSArray" if arrayed == "1" else "2DMS"
            return "2DArray" if arrayed == "1" else "2D"
        if dim == "3D":
            return "3D"
        return None

    def spirv_matrix_type_name(self, component_type, row_count, column_count):
        if not component_type or not row_count or not column_count:
            return None

        if component_type == "half":
            return f"half{column_count}x{row_count}"

        prefix = {"double": "dmat", "float": "mat"}.get(component_type)
        if prefix is None:
            return None

        if row_count == column_count:
            return f"{prefix}{column_count}"
        return f"{prefix}{column_count}x{row_count}"

    def spirv_cooperative_matrix_type_name(self, opcode, operands, types, constants):
        component_type = self.spirv_type_name(operands[0], types) or operands[0]
        variant = "khr" if opcode.endswith("KHR") else "nv"
        parts = [
            "spirvCoopMat",
            variant,
            component_type,
            "scope",
            self.spirv_cooperative_matrix_dimension_name(operands[1], constants),
            "rows",
            self.spirv_cooperative_matrix_dimension_name(operands[2], constants),
            "cols",
            self.spirv_cooperative_matrix_dimension_name(operands[3], constants),
        ]
        if len(operands) >= 5:
            parts.extend(
                [
                    "use",
                    self.spirv_cooperative_matrix_dimension_name(
                        operands[4], constants
                    ),
                ]
            )
        return "_".join(self.spirv_fallback_identifier(part, "part") for part in parts)

    def spirv_cooperative_matrix_dimension_name(self, operand, constants):
        value = constant_or_literal(constants, operand)
        text = self.spirv_constant_expression_text(value)
        return self.spirv_fallback_identifier(text, "value")

    def spirv_cooperative_matrix_length(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        if type_info.get("kind") != "cooperative_matrix":
            return None
        rows = self.spirv_cooperative_matrix_static_dimension(
            type_info.get("rows"), constants
        )
        columns = self.spirv_cooperative_matrix_static_dimension(
            type_info.get("columns"), constants
        )
        if rows is None or columns is None:
            return None
        return rows * columns

    def spirv_cooperative_matrix_static_dimension(self, operand, constants):
        value = constant_or_literal(constants, operand)
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def spirv_float_type_name(self, width):
        if width == "16":
            return "half"
        if width == "64":
            return "double"
        return "float"

    def spirv_int_type_name(self, width, signedness):
        if width == "1":
            return "bool"
        if width in {"8", "16", "64"}:
            prefix = "u" if signedness == "0" else "i"
            return f"{prefix}{width}"
        if signedness == "0":
            return "uint"
        return "int"

    def spirv_vector_type_name(self, component_type, component_count):
        if not component_type or not component_count:
            return None

        mapped_type = self.SPIRV_VECTOR_TYPES.get((component_type, component_count))
        if mapped_type:
            return mapped_type

        if component_type in {"i8", "u8", "i16", "u16", "i64", "u64"}:
            return f"vec{component_count}<{component_type}>"

        return f"{component_type}{component_count}"
