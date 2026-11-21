"""Stage I/O structs and compute built-ins of the Metal generator: constant conversion operators, plain structs, stage I/O member lowering, aggregate constructors, struct member declarations and their order, enum constructors, default values, return guarantees, compute built-in parameters and wave-lane parameters."""

import re

from ...backend.common_ast import ReturnNode as BackendReturnNode
from ..ast import (
    ArrayLiteralNode,
    ArrayNode,
    BlockNode,
    FunctionCallNode,
    IfNode,
    LiteralNode,
    ReturnNode,
    StructNode,
    WaveOpNode,
)
from ..refusals import MetalRefusal
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from .array_utils import (
    format_c_style_array_declaration,
    parse_array_type,
    split_array_type_suffix,
)
from .buffer_member_layout import member_is_packed
from .enum_utils import (
    build_generic_enum_specialization,
    enum_struct_fields,
    enum_variant_constructor_name,
    enum_variant_payload_fields,
    generic_enum_specialized_fields,
    generic_enum_specialized_variant_fields,
    generic_type_parts,
)
from .generic_struct_utils import (
    build_generic_struct_specialization,
    format_struct_constructor_expression,
    generic_struct_specialized_fields,
    normalize_specialization_type_text,
    render_constructor_arguments,
)
from .metal_array_copies import (
    generate_metal_array_initializer,
    render_metal_split_member_constructor_arguments,
)
from .metal_grid_quads import metal_grid_quad_lane_requirements
from .metal_library_structs import is_metal_library_struct
from .metal_member_attributes import attach_member_attributes
from .metal_nested_stage_io import collect_metal_nested_stage_io_member_lowerings
from .metal_render_walks import names_identifier_class
from .metal_varyings import metal_varying_attribute
from .metal_wave_extended import (
    metal_subgroup_lane_mask_expression,
    metal_wave_extension_lane_requirements,
)


class MetalStageIOStructMixin:
    """Stage IO struct members of ``MetalCodeGen``."""

    def collect_metal_constant_conversion_operators(self, functions):
        """Return ``{struct: [(receiver qualifiers, value), ...]}``.

        The Metal frontend moves a struct's ``operator bool()`` into a free
        helper named ``Struct__operator_bool``, with a
        ``__metal_receiver_<cv>_<address space>_<reference>`` suffix when it
        has an overload per receiver, as MLX's masked GEMV's ``_NoMask`` has
        for thread, threadgroup, device and constant values. Metal type-checks
        branches that are statically dead, so a translated ``!mask`` or
        ``bool(mask[i])`` needs the members again. Only helpers whose body is
        a single ``return true;`` or ``return false;`` are restored, as
        ``constexpr`` members with the same receiver qualifiers (a lone
        operator, whose helper has no suffix, becomes a ``const`` member).
        """

        operators = {}
        for function in functions or []:
            match = self.METAL_CONVERSION_RECEIVER_RE.fullmatch(
                str(getattr(function, "name", "") or "")
            )
            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            body = getattr(function, "body", None)
            statements = getattr(body, "statements", body)
            if (
                match is None
                or self.type_name_string(getattr(function, "return_type", None))
                != "bool"
                or len(parameters) != 1
                or self.type_name_string(getattr(parameters[0], "param_type", None))
                != match.group("struct")
                or not isinstance(statements, list)
                or len(statements) != 1
                or not isinstance(statements[0], ReturnNode)
                or not isinstance(statements[0].value, LiteralNode)
                or not isinstance(statements[0].value.value, bool)
            ):
                continue
            if match.group("cv") is None:
                # A lone operator's helper does not record whether the member
                # was const. A literal body never writes its receiver, so a
                # const member accepts every receiver the source's did.
                qualifiers = {
                    str(item).lower()
                    for item in list(
                        getattr(parameters[0], "qualifiers", []) or []
                    ) + list(getattr(parameters[0].param_type, "qualifiers", []) or [])
                }
                receiver = ["const"] + (
                    [
                        space
                        for space in self.METAL_RECEIVER_ADDRESS_SPACES
                        if space in qualifiers
                    ][:1]
                    or ["thread"]
                )
            else:
                spaces = match.group("spaces").split("_")
                if not set(spaces) <= set(self.METAL_RECEIVER_ADDRESS_SPACES):
                    continue
                receiver = {
                    "const_volatile": ["const", "volatile"],
                    "const": ["const"],
                    "volatile": ["volatile"],
                    "mutable": [],
                }[match.group("cv")] + spaces
                receiver.append(
                    {"unqualified": "", "lvalue": "&", "rvalue": "&&"}[
                        match.group("reference")
                    ]
                )
            text = " ".join(item for item in receiver if item)
            entries = operators.setdefault(match.group("struct"), [])
            if text not in {existing for existing, _ in entries}:
                entries.append((text, statements[0].value.value))
        return operators

    def metal_plain_struct_definition(self, node):
        if not isinstance(node, StructNode):
            return None
        if node.name in self.generic_enum_struct_definitions:
            return None
        if node.name in self.generic_struct_definitions:
            return None
        if node.name in self.lowered_glsl_buffer_block_struct_names:
            return None
        if is_metal_library_struct(node):
            # <metal_stdlib> declares it; a second declaration is ambiguous.
            return None
        if node.name in self.glsl_buffer_block_struct_names:
            code = self.glsl_buffer_block_diagnostic("Metal", node.name, None, None)
            code += self.unsupported_glsl_buffer_block_struct_placeholder(
                "Metal", node.name
            )
            return {"name": node.name, "dependencies": set(), "code": code}

        is_union = node.name in self.metal_union_layouts
        layout_plan = self.metal_hlsl_struct_plans.get(node.name)
        if layout_plan is not None and not is_union:
            dependencies = {
                member.nested_struct
                for member in layout_plan.members
                if member.nested_struct and member.nested_struct != node.name
            }
            return {
                "name": node.name,
                "dependencies": dependencies,
                "code": self.metal_hlsl_layout_struct_code(layout_plan),
            }
        if not is_union:
            self.validate_struct_member_semantic_types(node)
        declaration_kind = "union" if is_union else "struct"
        alignment = self.metal_struct_alignment(node)
        alignment_suffix = f" alignas({alignment})" if alignment is not None else ""
        code = f"{declaration_kind}{alignment_suffix} {node.name} {{\n"
        dependencies = set()
        default_member_semantics = self.metal_default_struct_member_semantics(node)
        for member in getattr(node, "members", []) or []:
            member_code, member_dependencies = self.metal_struct_member_declaration(
                member,
                default_member_semantics,
                node.name,
                preserve_native_narrow_storage=(
                    alignment is not None
                    or node.name not in self.metal_stage_io_struct_names
                ),
            )
            code += member_code
            dependencies.update(member_dependencies)
        fallback_position = self.metal_vertex_output_position_fallback_member_name(node)
        if fallback_position is not None:
            code += (
                "    /* CrossGL fallback: Metal vertex entry points require "
                "a position output even when the GLSL source did not write "
                "gl_Position. */\n"
                f"    float4 {fallback_position} [[position]];\n"
            )
        for receiver, value in (
            getattr(self, "metal_constant_conversion_operators", {}) or {}
        ).get(node.name, ()):
            code += (
                f"    constexpr operator bool() {receiver} "
                f"{{ return {'true' if value else 'false'}; }}\n"
            )
        code += "};\n"
        dependencies.discard(node.name)
        return {"name": node.name, "dependencies": dependencies, "code": code}

    def collect_metal_stage_io_member_lowerings(self, structs):
        lowerings = {}
        stage_io_names = set(getattr(self, "metal_stage_io_struct_names", set()))
        if not stage_io_names:
            return lowerings

        for struct in structs or []:
            struct_name = getattr(struct, "name", None)
            if struct_name not in stage_io_names:
                continue
            default_member_semantics = self.metal_default_struct_member_semantics(
                struct
            )
            for member in getattr(struct, "members", []) or []:
                lowering = self.metal_stage_io_member_lowering(
                    member, default_member_semantics, struct_name
                )
                if lowering is not None:
                    lowerings.setdefault(struct_name, {})[member.name] = lowering
        return collect_metal_nested_stage_io_member_lowerings(
            self, structs, stage_io_names, lowerings
        )

    def metal_stage_io_member_lowering(
        self, member, default_member_semantics=None, struct_name=None
    ):
        member_name = getattr(member, "name", None)
        if not member_name:
            return None
        semantic = self.metal_struct_member_effective_semantic(
            member, default_member_semantics or {}, struct_name
        )

        array_info = self.metal_stage_io_member_array_info(member)
        if array_info is not None:
            element_type, size = array_info
            fields = [
                (self.metal_stage_io_array_field_name(member_name, index), element_type)
                for index in range(size)
            ]
            return {
                "kind": "array",
                "element_type": element_type,
                "size": size,
                "fields": fields,
                "field_semantics": self.metal_stage_io_lowered_field_semantics(
                    semantic, len(fields)
                ),
                "interpolation_attr": self.metal_interpolation_attribute_suffix(member),
            }

        raw_type = self.struct_member_raw_type(member)
        mapped_type = self.map_type(raw_type)
        dimensions = self.metal_matrix_dimensions(mapped_type)
        if dimensions is None:
            return None
        component_prefix, columns, rows = dimensions
        column_type = f"{component_prefix}{rows}"
        fields = [
            (self.metal_stage_io_matrix_field_name(member_name, index), column_type)
            for index in range(columns)
        ]
        return {
            "kind": "matrix",
            "mapped_type": mapped_type,
            "columns": columns,
            "rows": rows,
            "fields": fields,
            "field_semantics": self.metal_stage_io_lowered_field_semantics(
                semantic, len(fields)
            ),
            "interpolation_attr": self.metal_interpolation_attribute_suffix(member),
        }

    def metal_stage_io_lowered_field_semantics(self, semantic, field_count):
        attribute_index = self.metal_attribute_index_from_semantic(semantic)
        if attribute_index is None:
            return [None] * field_count
        return [f"attribute({attribute_index + index})" for index in range(field_count)]

    def metal_stage_io_member_array_info(self, member):
        if self.canonical_metal_semantic(self.semantic_from_node(member)) in {
            "clip_distance"
        }:
            return None

        if isinstance(member, ArrayNode):
            resource_array_declaration = self.format_struct_resource_array_member(
                member
            )
            if resource_array_declaration is not None:
                return None
            element_type = self.map_type(
                getattr(member, "element_type", getattr(member, "vtype", "float"))
            )
            size = self.literal_int_value(member.size, self.literal_int_constants)
            return (element_type, size) if size else None

        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if self.is_array_type_node(raw_type):
            element_type = self.map_type(raw_type.element_type)
            size = self.literal_int_value(raw_type.size, self.literal_int_constants)
            return (element_type, size) if size else None

        raw_type_name = self.type_name_string(raw_type)
        if raw_type_name and "[" in raw_type_name and "]" in raw_type_name:
            element_type, array_size = parse_array_type(raw_type_name)
            size = self.literal_int_value(array_size, self.literal_int_constants)
            return (self.map_type(element_type), size) if size else None
        return None

    def metal_matrix_dimensions(self, mapped_type):
        mapped_type = self.map_type(mapped_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return None
        for component_prefix in ("float", "half", "double"):
            if not base_type.startswith(component_prefix):
                continue
            suffix = base_type[len(component_prefix) :]
            if "x" not in suffix:
                return None
            columns, rows = suffix.split("x", 1)
            if columns in {"2", "3", "4"} and rows in {"2", "3", "4"}:
                return component_prefix, int(columns), int(rows)
        return None

    def metal_aggregate_constructor_name(self, function_name):
        """Return a known unshadowed struct type used as a call expression."""
        if not isinstance(function_name, str):
            return None

        candidates = [function_name]
        mapped_name = self.map_type(function_name)
        if mapped_name != function_name:
            candidates.append(mapped_name)
        if any(name in self.user_function_names for name in candidates):
            return None

        return next(
            (name for name in candidates if name in self.struct_member_types),
            None,
        )

    def generate_struct_constructor_argument(self, expr, field_type):
        """Convert promoted narrow values back to the aggregate's physical field type."""
        array_initializer = generate_metal_array_initializer(self, expr, field_type)
        if array_initializer is not None:
            return array_initializer
        raw_type = self.resolve_metal_type_alias(self.type_name_string(field_type))
        base_type, array_suffix = split_array_type_suffix(raw_type)
        native = self.metal_native_narrow_bitcast_storage_type(base_type)
        if native is None or not re.fullmatch(r"(?:u?char|u?short)[234]?", native):
            return None
        if expr is None:
            return "{}" if array_suffix else f"{native}(0)"
        if array_suffix:
            if not isinstance(expr, ArrayLiteralNode):
                return None
            element_type = self.metal_for_in_array_element_type(raw_type)
            elements = [
                self.generate_struct_constructor_argument(
                    element,
                    (
                        element_type
                        if isinstance(element, ArrayLiteralNode)
                        else base_type
                    ),
                )
                for element in expr.elements
            ]
            return "{" + ", ".join(elements) + "}"
        if isinstance(expr, ArrayLiteralNode):
            component_type = native[:-1] if native[-1:] in {"2", "3", "4"} else native
            elements = []
            for element in expr.elements:
                width = self.expression_component_count(element)
                storage_type = (
                    f"{component_type}{width}" if width in {2, 3, 4} else component_type
                )
                rendered = self.generate_expression_with_expected(element, None)
                elements.append(f"{storage_type}({rendered})")
            return native + "{" + ", ".join(elements) + "}"
        rendered = self.generate_expression_with_expected(expr, field_type)
        return f"{native}({rendered})"

    def generate_metal_aggregate_constructor_call(self, expr, function_name):
        """Restore Metal brace semantics lost when aggregate IR uses a call."""
        constructor_name = self.metal_aggregate_constructor_name(function_name)
        if constructor_name is None:
            return None

        fields = list(self.struct_member_types[constructor_name].items())
        rendered_args = render_metal_split_member_constructor_arguments(
            self,
            expr,
            fields,
            self.metal_stage_io_member_lowerings.get(constructor_name, {}),
        )
        if rendered_args is None:
            rendered_args = render_constructor_arguments(
                self,
                constructor_name,
                expr,
                fields,
            )
        return format_struct_constructor_expression(
            self,
            self.map_type(constructor_name),
            rendered_args,
        )

    def generate_metal_matrix_resize_constructor(self, target_type, args):
        target_dimensions = self.metal_matrix_dimensions(target_type)
        if target_dimensions is None or len(args or []) != 1:
            return None

        arg_type = self.expression_result_type(args[0])
        arg_dimensions = self.metal_matrix_dimensions(arg_type)
        if arg_dimensions is None:
            return None

        _target_prefix, target_columns, target_rows = target_dimensions
        _arg_prefix, arg_columns, arg_rows = arg_dimensions
        if target_columns == arg_columns and target_rows == arg_rows:
            return None

        # A helper takes the matrix by value, so the argument is evaluated once and indexed as a whole.
        helper = self.metal_matrix_builtins.resize(target_dimensions, arg_dimensions)
        rendered_arg = self.generate_expression_with_expected(args[0], None)
        return f"{helper}({rendered_arg})"

    def vector_swizzle(self, width):
        return {1: "x", 2: "xy", 3: "xyz", 4: "xyzw"}[width]

    def metal_stage_io_matrix_field_name(self, member_name, index):
        return f"{member_name}_{index}"

    def metal_stage_io_array_field_name(self, member_name, index):
        return f"{member_name}_{index}"

    def generate_metal_stage_io_array_helpers(self):
        code = ""
        for struct_name in sorted(self.metal_stage_io_member_lowerings):
            for member_name, lowering in sorted(
                self.metal_stage_io_member_lowerings[struct_name].items()
            ):
                if lowering.get("kind") != "array":
                    continue
                helper_name = self.metal_stage_io_array_getter_name(
                    struct_name, member_name
                )
                element_type = lowering["element_type"]
                code += (
                    f"static inline {element_type} __attribute__((unused)) {helper_name}"
                    f"({struct_name} value, int index) {{\n"
                    "    switch (index) {\n"
                )
                for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                    code += f"    case {index}: return value.{field_name};\n"
                code += (
                    "    default: "
                    f"return {self.metal_default_value_expression(element_type)};\n"
                    "    }\n"
                    "}\n\n"
                )
        return code

    def metal_stage_io_array_getter_name(self, struct_name, member_name):
        return f"__crossgl_stage_io_get_{struct_name}_{member_name}"

    def metal_is_varying_struct(self, struct_name):
        """Whether ``struct_name`` carries values from a vertex entry to a fragment entry (:mod:`.metal_varyings`)."""
        if struct_name in getattr(self, "metal_vertex_entry_input_struct_names", ()):
            return False
        return (
            struct_name in getattr(self, "metal_vertex_entry_output_struct_names", ())
            or struct_name in getattr(self, "metal_mesh_primitive_struct_names", ())
            or struct_name in getattr(
                self, "metal_fragment_entry_input_struct_names", ()
            )
        )

    def metal_varying_member_attribute(
        self, member, struct_name, default_member_semantics, offset=0
    ):
        if not self.metal_is_varying_struct(struct_name):
            return None
        semantic = self.metal_struct_member_effective_semantic(
            member, default_member_semantics, struct_name
        )
        return metal_varying_attribute(self, member, semantic, offset=offset)

    def metal_member_semantic_attribute(
        self, member, semantic, struct_name, default_member_semantics
    ):
        varying = self.metal_varying_member_attribute(
            member, struct_name, default_member_semantics
        )
        if varying is not None:
            return varying
        if semantic and str(semantic).lower() == "location":
            # A location on a member of a struct no entry passes as stage I/O. Metal reads a member index only
            # through [[stage_in]], where a vertex input location is its attribute index (MSL 5.2.3.1), so the
            # struct keeps the index whichever way an entry later reads it.
            semantic = self.metal_vertex_input_location_semantic(member) or semantic
        return self.map_semantic(semantic) if semantic else ""

    def metal_struct_member_declaration(
        self,
        member,
        default_member_semantics,
        struct_name=None,
        preserve_native_narrow_storage=False,
    ):
        static_declaration = self.metal_static_struct_member_declaration(member)
        if static_declaration is not None:
            return static_declaration

        dependencies = set()
        lowering = self.metal_stage_io_member_lowerings.get(struct_name, {}).get(
            getattr(member, "name", None)
        )
        if lowering is not None:
            code = ""
            field_semantics = lowering.get("field_semantics", [])
            interpolation_attr = lowering.get("interpolation_attr", "")
            for index, (field_name, field_type) in enumerate(lowering["fields"]):
                semantic = (
                    field_semantics[index] if index < len(field_semantics) else None
                )
                semantic_attr = self.metal_varying_member_attribute(
                    member, struct_name, default_member_semantics, offset=index
                )
                if semantic_attr is None:
                    semantic_attr = self.map_semantic(semantic) if semantic else ""
                code += (
                    f"    {field_type} {field_name}{semantic_attr}"
                    f"{interpolation_attr};\n"
                )
                dependencies.update(self.metal_struct_type_dependencies(field_type))
            return code, dependencies

        if isinstance(member, ArrayNode):
            semantic = self.metal_struct_member_effective_semantic(
                member, default_member_semantics, struct_name
            )
            resource_array_declaration = self.format_struct_resource_array_member(
                member
            )
            if resource_array_declaration is not None:
                abi_attr = self.format_metal_struct_member_abi_attributes(member)
                semantic_attr = self.metal_member_semantic_attribute(
                    member, semantic, struct_name, default_member_semantics
                )
                interpolation_attr = self.metal_interpolation_attribute_suffix(member)
                declaration = attach_member_attributes(
                    resource_array_declaration, member.name, abi_attr
                )
                return (
                    f"    {declaration}{semantic_attr}{interpolation_attr};\n",
                    set(),
                )

            element_type = getattr(
                member,
                "element_type",
                getattr(member, "vtype", "float"),
            )
            semantic_attr = self.metal_member_semantic_attribute(
                member, semantic, struct_name, default_member_semantics
            )
            interpolation_attr = self.metal_interpolation_attribute_suffix(member)
            mapped_type = self.map_type(element_type)
            if preserve_native_narrow_storage:
                mapped_type = self.metal_struct_storage_type(element_type)
            mapped_type = self.metal_packed_storage_type(member, mapped_type)
            dependencies.update(self.metal_struct_type_dependencies(mapped_type))
            if member.size:
                if self.metal_array_semantic_attribute_precedes_extent(semantic):
                    return (
                        f"    {mapped_type} {member.name}{semantic_attr}"
                        f"{interpolation_attr}[{member.size}];\n",
                        dependencies,
                    )
                return (
                    f"    {mapped_type} {member.name}[{member.size}]"
                    f"{semantic_attr}{interpolation_attr};\n",
                    dependencies,
                )
            raise UnsizedArrayUnsupportedError(
                "metal", f"{struct_name or '?'}.{member.name}", "is a struct member"
            )

        semantic = self.metal_struct_member_effective_semantic(
            member, default_member_semantics, struct_name
        )
        abi_attr = self.format_metal_struct_member_abi_attributes(member)
        semantic_attr = self.metal_member_semantic_attribute(
            member, semantic, struct_name, default_member_semantics
        )
        interpolation_attr = self.metal_interpolation_attribute_suffix(member)

        if hasattr(member, "member_type"):
            if self.is_array_type_node(member.member_type):
                resource_array_declaration = self.format_struct_resource_array_member(
                    member
                )
                if resource_array_declaration is not None:
                    declaration = attach_member_attributes(
                        resource_array_declaration, member.name, abi_attr
                    )
                    return (
                        f"    {declaration}{semantic_attr}{interpolation_attr};\n",
                        set(),
                    )
            address_space_declaration = self.format_address_space_parameter_declaration(
                member.member_type,
                self.map_type(self.convert_type_node_to_string(member.member_type)),
                member.name,
                member,
            )
            if address_space_declaration is not None:
                dependencies.update(
                    self.metal_struct_type_dependencies(
                        self.convert_type_node_to_string(member.member_type)
                    )
                )
                declaration = attach_member_attributes(
                    address_space_declaration, member.name, abi_attr
                )
                return (
                    f"    {declaration}{semantic_attr}{interpolation_attr};\n",
                    dependencies,
                )
            if str(type(member.member_type)).find("ArrayType") != -1:
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.map_type(member_type_str)
                if preserve_native_narrow_storage:
                    member_type = self.metal_struct_storage_type(member_type_str)
                member_type = self.metal_packed_storage_type(member, member_type)
                dependencies.update(self.metal_struct_type_dependencies(member_type))
                declaration = format_c_style_array_declaration(member_type, member.name)
                if self.metal_array_semantic_attribute_precedes_extent(semantic):
                    base_type, array_size = split_array_type_suffix(member_type)
                    if member.member_type.size is None:
                        raise UnsizedArrayUnsupportedError(
                            "metal",
                            f"{struct_name or '?'}.{member.name}",
                            "is a struct member",
                        )
                    if not array_size:
                        array_size = self.safe_expression_to_string(
                            member.member_type.size
                        )
                    array_size = str(array_size).strip()
                    if array_size.startswith("[") and array_size.endswith("]"):
                        array_size = array_size[1:-1]
                    return (
                        f"    {base_type} {member.name}{abi_attr}{semantic_attr}"
                        f"{interpolation_attr}[{array_size}];\n",
                        dependencies,
                    )
                if member.member_type.size is None:
                    raise UnsizedArrayUnsupportedError(
                        "metal",
                        f"{struct_name or '?'}.{member.name}",
                        "is a struct member",
                    )
                # [[id(n)]] after the extent would appertain to the array type (metal_member_attributes).
                declaration = attach_member_attributes(
                    declaration, member.name, abi_attr
                )
                return (
                    f"    {declaration}{semantic_attr}{interpolation_attr};\n",
                    dependencies,
                )

            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_resource_type_with_format(member_type_str, member)
        elif hasattr(member, "vtype"):
            member_type = self.map_resource_type_with_format(member.vtype, member)
        else:
            member_type = "float"

        if preserve_native_narrow_storage:
            raw_type = self.struct_member_raw_type(member)
            if self.metal_native_narrow_bitcast_storage_type(raw_type) is not None:
                member_type = self.metal_struct_storage_type(raw_type)
        member_type = self.metal_packed_storage_type(member, member_type)
        if member.name in self.metal_signed_builtin_members.get(struct_name, ()):
            member_type = "uint"
        dependencies.update(self.metal_struct_type_dependencies(member_type))
        return (
            f"    {member_type} {member.name}{abi_attr}{semantic_attr}"
            f"{interpolation_attr};\n",
            dependencies,
        )

    METAL_PACKED_VECTOR_RE = re.compile(
        r"(?:metal::)?(?P<vector>(?:float|half|int|uint|short|ushort|char|uchar|long|ulong|bfloat)[234])"
    )

    def metal_packed_storage_type(self, member, type_text):
        """``type_text`` in MSL's packed vector form (``packed_float3``) for a ``@ packed`` member.

        A packed vector is aligned to its scalar and takes exactly its components (MSL 3.2 section 2.2.3), the
        layout an MSL source declared with ``packed_float3``; the value is the ordinary vector.
        """
        if not member_is_packed(member) or str(type_text).startswith("packed_"):
            return type_text
        template = self.metal_array_template_parts(str(type_text).strip())
        if template is not None:
            # metal::array<T, N> keeps its extent; its element takes the packed form.
            prefix, element, extent = template
            element = self.metal_packed_storage_type(member, element)
            return f"{prefix}{element}, {extent}>"
        base, suffix = split_array_type_suffix(str(type_text))
        match = self.METAL_PACKED_VECTOR_RE.fullmatch(base.strip())
        if match is None:
            raise MetalRefusal(
                f"Metal struct member '{getattr(member, 'name', '?')}' is marked packed, but {type_text} has no "
                "packed vector form"
            )
        return f"packed_{match.group('vector')}{suffix}"

    @staticmethod
    def metal_array_template_parts(text):
        """``(prefix, element, extent)`` of a ``metal::array<element, extent>`` spelling, or ``None``."""
        for prefix in ("metal::array<", "array<"):
            if not (text.startswith(prefix) and text.endswith(">")):
                continue
            inner, depth = text[len(prefix) : -1], 0
            for index in range(len(inner) - 1, -1, -1):
                character = inner[index]
                if character == ">":
                    depth += 1
                elif character == "<":
                    depth -= 1
                elif character == "," and depth == 0:
                    return prefix, inner[:index].strip(), inner[index + 1 :].strip()
        return None

    def metal_struct_type_dependencies(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return set()
        mapped_type = self.map_type(type_name)
        base_type, _ = split_array_type_suffix(str(mapped_type))
        base_type = base_type.strip()
        if base_type.startswith("array<"):
            base_type = base_type[len("array<") :].split(",", 1)[0].rstrip(">")
        base_type = base_type.replace("const ", "").strip()
        for separator in ("&", "*"):
            if separator in base_type:
                base_type = base_type.split(separator, 1)[0].strip()
        if "<" in base_type:
            base_type = base_type.split("<", 1)[0].strip()
        return {base_type}

    def order_metal_struct_definitions(self, definitions):
        generated_names = {
            definition["name"] for definition in definitions if definition.get("name")
        }
        remaining = [
            {
                **definition,
                "dependencies": (
                    set(definition.get("dependencies", set())) & generated_names
                ),
            }
            for definition in definitions
        ]
        ordered = []
        emitted = set()

        while remaining:
            progressed = False
            for definition in list(remaining):
                if definition["dependencies"] <= emitted:
                    ordered.append(definition)
                    emitted.add(definition["name"])
                    remaining.remove(definition)
                    progressed = True
            if not progressed:
                ordered.extend(remaining)
                break

        return ordered

    def add_metal_generic_specializations_from_type_text(self, type_text, visited=None):
        type_text = normalize_specialization_type_text(type_text)
        if not type_text:
            return
        visited = set(visited or set())
        if type_text in visited:
            return
        visited.add(type_text)

        base_name, generic_args = generic_type_parts(type_text)
        for arg in generic_args:
            self.add_metal_generic_specializations_from_type_text(arg, visited)

        enum_definition = self.generic_enum_struct_definitions.get(base_name)
        if enum_definition is not None and len(generic_args) == len(
            enum_definition["generic_params"]
        ):
            specialization = self.generic_enum_specializations.get(type_text)
            if specialization is None:
                specialization = build_generic_enum_specialization(
                    type_text,
                    enum_definition,
                )
                self.generic_enum_specializations[type_text] = specialization
            for _field_name, field_type in generic_enum_specialized_fields(
                self,
                specialization,
            ):
                self.add_metal_generic_specializations_from_type_text(
                    field_type, visited
                )
            return

        struct_definition = self.generic_struct_definitions.get(base_name)
        if struct_definition is not None and len(generic_args) == len(
            struct_definition["generic_params"]
        ):
            specialization = self.generic_struct_specializations.get(type_text)
            if specialization is None:
                specialization = build_generic_struct_specialization(
                    type_text,
                    struct_definition,
                )
                self.generic_struct_specializations[type_text] = specialization
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                self.add_metal_generic_specializations_from_type_text(
                    field_type, visited
                )

    def generate_metal_enum_constructor_functions(self, enums):
        code = ""
        for enum in enums or []:
            all_fields = enum_struct_fields(enum) or []
            for variant in enum.variants or []:
                variant_fields = enum_variant_payload_fields(variant) or []
                params = ", ".join(
                    f"{self.map_type(field_type)} payload{index}"
                    for index, (_field_name, field_type) in enumerate(variant_fields)
                )
                code += (
                    f"{enum.name} "
                    f"{enum_variant_constructor_name(enum.name, variant.name)}"
                    f"({params}) {{\n"
                )
                code += f"    {enum.name} result;\n"
                code += (
                    f"    result.variant = "
                    f"{enum_variant_constructor_name(enum.name, variant.name).rsplit('_make', 1)[0]};\n"
                )
                active_fields = {
                    field_name for field_name, _field_type in variant_fields
                }
                for field_name, field_type in all_fields:
                    if field_name in active_fields:
                        continue
                    code += (
                        f"    result.{field_name} = "
                        f"{self.metal_default_value_expression(field_type)};\n"
                    )
                for index, (field_name, _field_type) in enumerate(variant_fields):
                    code += f"    result.{field_name} = payload{index};\n"
                code += "    return result;\n"
                code += "}\n\n"
        return code

    def generate_metal_generic_enum_constructor_functions(self, specializations):
        code = ""
        for specialization in (specializations or {}).values():
            all_fields = generic_enum_specialized_fields(self, specialization)
            definition = specialization["definition"]
            for variant in definition["enum"].variants or []:
                variant_fields = (
                    generic_enum_specialized_variant_fields(
                        self,
                        specialization,
                        variant.name,
                    )
                    or []
                )
                params = ", ".join(
                    f"{self.map_type(field_type)} payload{index}"
                    for index, (_field_name, field_type) in enumerate(variant_fields)
                )
                code += (
                    f"{specialization['struct_name']} "
                    f"{enum_variant_constructor_name(specialization['struct_name'], variant.name)}"
                    f"({params}) {{\n"
                )
                code += f"    {specialization['struct_name']} result;\n"
                code += (
                    "    result.variant = "
                    f"{enum_variant_constructor_name(definition['name'], variant.name).rsplit('_make', 1)[0]};\n"
                )
                active_fields = {
                    field_name for field_name, _field_type in variant_fields
                }
                for field_name, field_type in all_fields:
                    if field_name in active_fields:
                        continue
                    code += (
                        f"    result.{field_name} = "
                        f"{self.metal_default_value_expression(field_type)};\n"
                    )
                for index, (field_name, _field_type) in enumerate(variant_fields):
                    code += f"    result.{field_name} = payload{index};\n"
                code += "    return result;\n"
                code += "}\n\n"
        return code

    def metal_default_value_expression(self, field_type):
        mapped_type = self.map_type(field_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return f"{base_type}()"
        if (
            base_type in self.metal_generated_struct_names
            or base_type in self.struct_member_types
            or base_type in self.structs_by_name
        ):
            return f"{base_type}()"
        if base_type == "bool":
            return "false"
        if base_type.startswith("bool") or base_type.startswith("bvec"):
            return f"{base_type}(false)"
        return f"{base_type}(0)"

    def is_input_attachment_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        return bool(re.fullmatch(r"[iu]?subpassInput(?:MS)?", str(type_name or "")))

    def unsupported_input_attachment_call(self, func_name):
        if func_name != "subpassLoad":
            return None
        fallback_type = self.current_expression_expected_type or "vec4"
        fallback = self.metal_default_value_expression(fallback_type)
        return self.refuse_lowering(
            "input attachment load: subpassLoad requires Vulkan subpass input "
            "lowering",
            fallback,
        )

    def metal_needs_fallback_return(self, body, return_type):
        if return_type == "void":
            return False
        return not self.metal_statement_sequence_guarantees_return(body)

    def metal_statement_sequence_guarantees_return(self, body):
        if hasattr(body, "statements"):
            statements = list(getattr(body, "statements", []) or [])
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]
        return any(self.metal_statement_guarantees_return(stmt) for stmt in statements)

    def metal_statement_guarantees_return(self, statement):
        if isinstance(statement, (ReturnNode, BackendReturnNode)):
            return True
        if isinstance(statement, BlockNode):
            return self.metal_statement_sequence_guarantees_return(statement)
        if not isinstance(statement, IfNode):
            return False
        then_branch = getattr(
            statement, "then_branch", getattr(statement, "if_body", None)
        )
        else_branch = getattr(statement, "else_branch", None)
        return else_branch is not None and (
            self.metal_statement_sequence_guarantees_return(then_branch)
            and self.metal_statement_sequence_guarantees_return(else_branch)
        )

    def compute_builtin_parameter_specs(self):
        return [
            ("gl_GlobalInvocationID", "uint3", "thread_position_in_grid"),
            ("gl_LocalInvocationID", "uint3", "thread_position_in_threadgroup"),
            ("gl_WorkGroupID", "uint3", "threadgroup_position_in_grid"),
            ("gl_LocalInvocationIndex", "uint", "thread_index_in_threadgroup"),
            ("gl_QuadGroupID", "uint", "quadgroup_index_in_threadgroup"),
            ("gl_QuadGroupInvocationID", "uint", "thread_index_in_quadgroup"),
            ("gl_WorkGroupSize", "uint3", "threads_per_threadgroup"),
            ("gl_NumWorkGroups", "uint3", "threadgroups_per_grid"),
            ("gl_SubgroupInvocationID", "uint", "thread_index_in_simdgroup"),
            ("gl_SubgroupSize", "uint", "threads_per_simdgroup"),
            ("gl_SubgroupID", "uint", "simdgroup_index_in_threadgroup"),
            ("gl_NumSubgroups", "uint", "simdgroups_per_threadgroup"),
            # A ray generation kernel's launch size (semantics.metal_ray_dispatch).
            ("gl_LaunchSizeEXT", "uint3", "threads_per_grid"),
        ]

    def compute_builtin_name_for_metal_attribute(self, attribute):
        for (
            builtin_name,
            _param_type,
            metal_attribute,
        ) in self.compute_builtin_parameter_specs():
            if metal_attribute == attribute:
                return builtin_name
        return None

    def metal_compute_builtin_expression_name(self, name):
        """The compute entry parameter that carries built-in ``name``, or None when the entry has none."""
        if name is None:
            return None
        return self.current_metal_compute_builtin_parameter_names.get(name)

    def metal_glsl_subgroup_lane_builtin_diagnostic(self, name):
        if name in self.local_variable_types:
            return None
        builtin = self.METAL_GLSL_SUBGROUP_LANE_BUILTINS.get(name)
        if builtin is None:
            return None
        attribute, result_type = builtin
        fallback = self.diagnostic_zero_value_for_type(result_type)
        return self.refuse_lowering(
            f"GLSL subgroup builtin: {name} requires compute-stage {attribute} value",
            fallback,
        )

    def metal_graphics_builtin_expression_name(self, name):
        """The graphics entry parameter that carries built-in ``name``, or None when the entry has none."""
        if name is None or name in self.local_variable_types:
            return None
        parameter = self.current_metal_graphics_builtin_parameter_names.get(name)
        if parameter is None:
            return None
        if name == "gl_ViewIndex":
            # GLSL's view index is an int; Metal's amplification_id is a uint.
            return f"int({parameter})"
        return parameter

    def metal_builtin_expression_name(self, name):
        graphics_name = self.metal_graphics_builtin_expression_name(name)
        if graphics_name is not None and graphics_name != name:
            return graphics_name
        compute_name = self.metal_compute_builtin_expression_name(name)
        if compute_name is not None and compute_name != name:
            return compute_name
        lane_mask = metal_subgroup_lane_mask_expression(self, name)
        if lane_mask is not None:
            return lane_mask
        diagnostic = self.metal_glsl_subgroup_lane_builtin_diagnostic(name)
        if diagnostic is not None:
            return diagnostic
        return name

    def metal_graphics_builtin_result_type(self, name):
        if (
            name is None
            or name in self.local_variable_types
            or name not in self.current_metal_graphics_builtin_parameter_names
        ):
            return None
        if name in ("gl_VertexIndex", "gl_InstanceIndex"):
            return "uint"
        if name == "gl_ViewIndex":
            return "int"
        if name == "gl_FragCoord":
            return "float4"
        if name == "gl_PointCoord":
            return "float2"
        return None

    def metal_compute_builtin_result_type(self, name):
        if name is None:
            return None
        if name in self.current_metal_compute_builtin_parameter_names:
            for (
                builtin_name,
                param_type,
                _attribute,
            ) in self.compute_builtin_parameter_specs():
                if builtin_name == name:
                    return param_type
        return None

    def metal_compute_builtin_parameter_base_name(self, builtin_name, attribute):
        return {
            "gl_GlobalInvocationID": "thread_position_in_grid",
            "gl_LocalInvocationID": "thread_position_in_threadgroup",
            "gl_WorkGroupID": "threadgroup_position_in_grid",
            "gl_LocalInvocationIndex": "thread_index_in_threadgroup",
            "gl_QuadGroupID": "quadgroup_index_in_threadgroup",
            "gl_QuadGroupInvocationID": "thread_index_in_quadgroup",
            "gl_WorkGroupSize": "threads_per_threadgroup",
            "gl_NumWorkGroups": "threadgroups_per_grid",
            "gl_SubgroupInvocationID": "thread_index_in_simdgroup",
            "gl_SubgroupSize": "threads_per_simdgroup",
            "gl_SubgroupID": "simdgroup_index_in_threadgroup",
            "gl_NumSubgroups": "simdgroups_per_threadgroup",
            "gl_LaunchSizeEXT": "threads_per_grid",
        }.get(builtin_name, attribute)

    def required_compute_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_compute_builtin_names(getattr(func, "body", []))
        used_wave_operations = self.used_wave_stage_builtin_operations(
            getattr(func, "body", [])
        )
        func_name = getattr(func, "name", None)
        if func_name:
            used_wave_operations.update(
                self.function_metal_wave_lane_dependencies.get(func_name, set())
            )
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.compute_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            name = self.unique_metal_generated_name(
                self.metal_compute_builtin_parameter_base_name(builtin_name, attribute),
                reserved_names,
            )
            reserved_names.add(name)
            required_parameters.append((name, param_type, attribute))
        required_attributes = {
            attribute for _name, _param_type, attribute in required_parameters
        }
        wave_builtin_parameters = [
            (
                "WaveGetLaneIndex",
                "crossglWaveLaneIndex",
                "uint",
                "thread_index_in_simdgroup",
            ),
            (
                "WaveGetLaneCount",
                "crossglWaveLaneCount",
                "uint",
                "threads_per_simdgroup",
            ),
        ]
        for operation, base_name, param_type, attribute in wave_builtin_parameters:
            if operation not in used_wave_operations:
                continue
            if attribute in explicit_stage_builtins or attribute in required_attributes:
                continue
            name = self.unique_metal_generated_name(base_name, reserved_names)
            reserved_names.add(name)
            required_parameters.append((name, param_type, attribute))
            required_attributes.add(attribute)
        return required_parameters

    def used_compute_builtin_names(self, body):
        builtin_names = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_QuadGroupID",
            "gl_QuadGroupInvocationID",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
            "gl_SubgroupID",
            "gl_NumSubgroups",
            "gl_LaunchSizeEXT",
        }
        used_names = set()
        for node in self.metal_render_walks.nodes_with_class(
            body, names_identifier_class
        ):
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def used_wave_stage_builtin_operations(self, body):
        operations = set()
        for node in self.iter_ast_nodes(body):
            operations.update(metal_wave_extension_lane_requirements(node))
            if isinstance(node, WaveOpNode):
                operation = getattr(node, "operation", None)
            elif isinstance(node, FunctionCallNode):
                operation = self.function_call_name(node)
            else:
                continue
            if operation == "WaveMatch":
                operations.add("WaveGetLaneCount")
            elif operation in self.METAL_WAVE_MULTI_PREFIX_INTRINSICS:
                operations.add("WaveGetLaneIndex")
                operations.add("WaveGetLaneCount")
            elif operation in {"WaveGetLaneIndex", "WaveGetLaneCount"}:
                operations.add(operation)
        return operations

    def explicit_compute_stage_builtin_parameters(self, parameters):
        stage_parameters = {}
        builtin_attributes = {
            attribute
            for _builtin_name, _param_type, attribute in (
                self.compute_builtin_parameter_specs()
            )
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            if metal_semantic in builtin_attributes or metal_semantic in {
                "thread_index_in_simdgroup",
                "threads_per_simdgroup",
            }:
                stage_parameters[metal_semantic] = getattr(parameter, "name", None)
        return stage_parameters

    def validate_compute_builtin_parameter_types(self, parameters):
        # Metal allows the positional / dimension compute builtins to be declared
        # as a scalar `uint`, `uint2`, or `uint3` (the driver fills the requested
        # component count). See Apple's "Creating threads and threadgroups" guide,
        # which uses the uint2 form. Restricting these to uint3 rejected valid MLX
        # reduction kernels (softmax/rms_norm/layer_norm/... use scalar `uint`).
        positional_builtin_types = ("uint", "uint2", "uint3")
        expected_types = {
            "thread_position_in_grid": positional_builtin_types,
            "thread_position_in_threadgroup": positional_builtin_types,
            "threadgroup_position_in_grid": positional_builtin_types,
            "thread_index_in_threadgroup": "uint",
            "quadgroup_index_in_threadgroup": "uint",
            "thread_index_in_quadgroup": "uint",
            "threads_per_threadgroup": positional_builtin_types,
            "threadgroups_per_grid": positional_builtin_types,
            "thread_index_in_simdgroup": "uint",
            "threads_per_simdgroup": "uint",
            "simdgroup_index_in_threadgroup": "uint",
            "simdgroups_per_threadgroup": "uint",
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            expected_values = (
                expected_type if isinstance(expected_type, tuple) else (expected_type,)
            )
            if actual_type not in expected_values:
                name = getattr(parameter, "name", "<anonymous>")
                expected_label = (
                    expected_values[0]
                    if len(expected_values) == 1
                    else "one of " + ", ".join(expected_values)
                )
                raise MetalRefusal(
                    f"Metal compute semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_label}, got {actual_type}"
                )

    def collect_function_metal_wave_lane_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.used_wave_stage_builtin_operations(
                getattr(func, "body", [])
            ) | metal_grid_quad_lane_requirements(self, func)
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                current_dependencies = dependencies.setdefault(func_name, set())
                before = set(current_dependencies)
                for called_name in calls:
                    current_dependencies.update(dependencies.get(called_name, set()))
                if current_dependencies != before:
                    changed = True
        return dependencies

    def collect_function_metal_wave_lane_parameter_names(self, functions):
        parameter_names = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            dependencies = self.function_metal_wave_lane_dependencies.get(
                func_name, set()
            )
            if not func_name or not dependencies:
                continue
            reserved_names = {
                getattr(parameter, "name", None)
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                )
                if getattr(parameter, "name", None)
            }
            reserved_names.update(self.metal_function_local_variable_names(func))
            names = {}
            for operation, base_name in (
                ("WaveGetLaneIndex", "crossglWaveLaneIndex"),
                ("WaveGetLaneCount", "crossglWaveLaneCount"),
            ):
                if operation not in dependencies:
                    continue
                name = self.unique_metal_generated_name(base_name, reserved_names)
                reserved_names.add(name)
                names[operation] = name
            parameter_names[func_name] = names
        return parameter_names

    def append_required_metal_wave_lane_parameters(self, params_str, func_name):
        parameter_names = self.function_metal_wave_lane_parameter_names.get(
            func_name, {}
        )
        lane_index_name = parameter_names.get("WaveGetLaneIndex")
        if lane_index_name is not None:
            params_str = self.append_parameter_declaration(
                params_str, f"uint {lane_index_name}"
            )
            self.current_metal_wave_lane_index_parameter = lane_index_name
        lane_count_name = parameter_names.get("WaveGetLaneCount")
        if lane_count_name is not None:
            params_str = self.append_parameter_declaration(
                params_str, f"uint {lane_count_name}"
            )
            self.current_metal_wave_lane_count_parameter = lane_count_name
        return params_str
