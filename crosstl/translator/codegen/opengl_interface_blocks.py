"""Interface blocks in the GLSL generator: typed memory qualifiers, struct members, access metadata and expression statements."""

from ..ast import (
    ArrayNode,
    ArrayType,
    PointerType,
    ReferenceType,
    ResourceMemoryQualifierNode,
)
from ..refusals import OpenGLRefusal
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from ..vulkan_memory_scope import glsl_memory_scope_qualifiers
from .array_utils import (
    format_c_style_array_declaration,
    get_array_size_from_node,
    prepend_array_dimension,
)
from .crossgl_vector_names import crossgl_named_vector_type
from .enum_utils import enum_struct_fields
from .glsl_draw_parameters import (
    draw_parameter_spelling,
    vertex_builtin_spelling,
)
from .glsl_exact_lookups import (
    glsl_unmapped_identifier,
    glsl_unmapped_semantic_name,
)
from .glsl_interface_block_io import stage_block_member_layout
from .image_access_contracts import (
    is_glsl_storage_image_type,
    normalized_image_access,
)
from .opengl_values import OpenGLResourceMemoryQualifierError
from .operator_spellings import map_c_family_operator
from .stage_utils import normalize_stage_name


class OpenGLInterfaceBlockMixin:
    """Interface block members of ``GLSLCodeGen``."""

    def typed_resource_memory_qualifiers(self, node=None, raw_type=None):
        """Return typed resource-memory contracts without flattening their scope."""
        values = []
        contract_type = (
            raw_type if raw_type is not None else self.resource_node_type(node)
        )
        while isinstance(contract_type, ArrayType):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            values.extend(getattr(contract_type, "resource_qualifiers", []) or [])
        values.extend(getattr(node, "resource_qualifiers", []) or [])

        qualifiers = []
        seen = set()
        for value in values:
            if not isinstance(value, ResourceMemoryQualifierNode):
                continue
            contract = (value.kind, value.scope)
            if contract in seen:
                continue
            seen.add(contract)
            qualifiers.append(value)
        return qualifiers

    def opengl_resource_memory_qualifier_error(self, node, qualifier, reason):
        qualifier_kind = str(getattr(qualifier, "kind", "")).lower() or None
        requested_scope = getattr(qualifier, "scope", None)
        if requested_scope is not None:
            requested_scope = str(requested_scope).lower()
        requested_contract = (
            f"{qualifier_kind}({requested_scope})"
            if qualifier_kind and requested_scope
            else qualifier_kind or str(qualifier)
        )
        resource_name = self.resource_node_name(node, "<unnamed>")
        target_qualifier = (
            qualifier_kind if qualifier_kind in {"coherent", "volatile"} else None
        )
        target_scope = "device" if qualifier_kind == "coherent" else None

        if reason == "unsupported-system-coherence-scope":
            detail = (
                "GLSL coherent memory only covers accesses by shader invocations "
                "on the GPU and cannot provide system/host visibility"
            )
            action = (
                "retain the Metal target, or request device scope only when that "
                "weaker contract is valid"
            )
        elif reason == "unsupported-coherence-scope":
            detail = (
                "GLSL 4.60 has no resource qualifier for that exact coherence scope"
            )
            action = (
                "retain the source target, or request device scope only when its "
                "broader visibility is valid"
            )
        elif reason == "unsupported-volatile-scope":
            detail = "GLSL volatile does not encode a separate visibility scope"
            action = "remove the scope or retain the source target"
        elif reason == "unsupported-resource-type":
            detail = (
                "GLSL memory qualifiers are representable only on shader-storage "
                "buffers and storage images"
            )
            action = "use a storage resource or retain the source target"
        else:
            detail = "GLSL has no equivalent typed resource-memory qualifier"
            action = "remove the qualifier or retain the source target"

        return OpenGLResourceMemoryQualifierError(
            "OpenGL GLSL 4.60 cannot preserve typed resource memory qualifier "
            f"'{requested_contract}' for resource '{resource_name}': {detail}; "
            f"{action}",
            resource_name=resource_name,
            qualifier_kind=qualifier_kind,
            requested_scope=requested_scope,
            target_qualifier=target_qualifier,
            target_scope=target_scope,
            reason=reason,
            source_location=(
                getattr(qualifier, "source_location", None)
                or getattr(node, "source_location", None)
            ),
        )

    def glsl_resource_supports_typed_memory_qualifiers(self, node):
        vtype = self.resource_node_type(node)
        contract_type = vtype
        while isinstance(contract_type, ArrayType):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            address_space = str(
                getattr(contract_type, "address_space", "") or ""
            ).lower()
            # ``buffer T* p``: a storage pointer the emitter declares as a shader-storage block.
            if not address_space and "buffer" in {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }:
                return True
            return address_space in {"device", "global", "storage"}
        if self.is_structured_buffer_type(vtype):
            return True
        if self.is_glsl_buffer_block_variable(node, vtype):
            return True
        mapped_type = self.map_resource_type_with_format(vtype, node)
        return self.is_opaque_resource_type(mapped_type) and is_glsl_storage_image_type(
            mapped_type
        )

    def glsl_typed_resource_memory_qualifier(
        self,
        node,
        qualifier,
        *,
        require_storage_resource=False,
    ):
        kind = str(getattr(qualifier, "kind", "")).lower()
        scope = getattr(qualifier, "scope", None)
        scope = str(scope).lower() if scope is not None else None

        if kind == "volatile":
            if scope is not None:
                raise self.opengl_resource_memory_qualifier_error(
                    node, qualifier, "unsupported-volatile-scope"
                )
            mapped_qualifier = "volatile"
        elif kind == "coherent":
            if scope in {"system", "host"}:
                raise self.opengl_resource_memory_qualifier_error(
                    node, qualifier, "unsupported-system-coherence-scope"
                )
            if scope not in {None, "device"}:
                raise self.opengl_resource_memory_qualifier_error(
                    node, qualifier, "unsupported-coherence-scope"
                )
            mapped_qualifier = "coherent"
        else:
            raise self.opengl_resource_memory_qualifier_error(
                node, qualifier, "unsupported-qualifier-kind"
            )

        if require_storage_resource and not (
            self.glsl_resource_supports_typed_memory_qualifiers(node)
        ):
            raise self.opengl_resource_memory_qualifier_error(
                node, qualifier, "unsupported-resource-type"
            )
        return mapped_qualifier

    def validate_typed_resource_memory_qualifier_contracts(
        self, ast, target_stage=None
    ):
        validated = set()
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                for qualifier in self.typed_resource_memory_qualifiers(node):
                    qualifier_id = id(qualifier)
                    if qualifier_id in validated:
                        continue
                    validated.add(qualifier_id)
                    self.glsl_typed_resource_memory_qualifier(node, qualifier)

    def resource_memory_qualifiers(self, node):
        supported = {
            "coherent",
            "globallycoherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
        }
        qualifiers = set()
        access_choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, access_choices)

        for qualifier in self.typed_resource_memory_qualifiers(node):
            qualifiers.add(
                self.glsl_typed_resource_memory_qualifier(
                    node,
                    qualifier,
                    require_storage_resource=True,
                )
            )

        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier = str(qualifier).lower()
            if qualifier in supported:
                qualifiers.add(qualifier)

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported:
                qualifiers.add(attr_name)

        for _, access in access_choices:
            if access == "read":
                qualifiers.add("readonly")
            elif access == "write":
                qualifiers.add("writeonly")

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        scoped = glsl_memory_scope_qualifiers(node, self.glsl_vulkan_memory_model)
        order = ("coherent", "volatile", "restrict", "readonly", "writeonly")
        names = [*scoped, *(name for name in order if name in qualifiers)]
        return " ".join(dict.fromkeys(names))

    def resource_access_metadata_choices(self, node):
        choices = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            source = str(qualifier).lower()
            access = normalized_image_access(source)
            if access is not None:
                choices.append((source, access))

        for qualifier in getattr(node, "resource_qualifiers", []) or []:
            source = str(qualifier).lower()
            access = normalized_image_access(source)
            if access is not None:
                choices.append((source, access))

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                source = (
                    self.attribute_value_to_string(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if len(arguments) != 1:
                    raise OpenGLRefusal(
                        self.invalid_resource_access_message(node, source)
                    )
                source = self.attribute_value_to_string(arguments[0])
                access = normalized_image_access(source)
                if access is None:
                    raise OpenGLRefusal(
                        self.invalid_resource_access_message(node, source)
                    )
                choices.append((f"access({source})", access))
                continue

            access = normalized_image_access(attr_name)
            if access is not None:
                choices.append((attr_name, access))
        return choices

    def validate_resource_access_metadata_operands(self, node):
        self.resource_access_metadata_choices(node)

    def validate_resource_access_metadata_consistency(self, node, choices):
        if not choices:
            return
        first_source, first_access = choices[0]
        node_name = self.resource_node_name(node, "<unnamed>")
        seen_sources = {first_source}
        for source, access in choices[1:]:
            if source in seen_sources:
                raise OpenGLRefusal(
                    "Duplicate OpenGL resource access metadata for "
                    f"'{node_name}': {self.resource_access_metadata_label(source)}"
                )
            seen_sources.add(source)
            if access != first_access:
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource access metadata for "
                    f"'{node_name}': {first_source} differs from {source}"
                )

    def resource_access_metadata_label(self, source):
        return f"@{source}"

    def map_operator(self, op):
        """Spell a CrossGL operator; an operator CrossGL does not define is refused."""
        return map_c_family_operator(op, self.LOWERING_REFUSAL_TYPE.label)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the corresponding GLSL builtin or layout."""
        if semantic is None:
            return ""
        mapped = self.semantic_map.get(semantic)
        if mapped is None:
            mapped = glsl_unmapped_semantic_name(self, semantic)
        return f"{mapped}"

    def map_stage_input_semantic(self, semantic, stage_name=None):
        if (
            normalize_stage_name(stage_name) == "compute"
            and semantic == "threads_per_grid"
        ):
            return self.GLSL_THREADS_PER_GRID_EXPRESSION
        mapped = self.map_semantic(semantic)
        if normalize_stage_name(stage_name) == "fragment":
            if mapped == "gl_Position":
                return "gl_FragCoord"
            semantic_text = str(semantic).lower() if semantic is not None else ""
            if semantic_text in {
                "sv_coverage",
                "sample_mask",
                "gl_samplemask",
                "gl_samplemaskin",
                "sample_mask_in",
            }:
                return "gl_SampleMaskIn"
        return mapped

    def stage_input_builtin_alias(self, semantic, stage_name=None):
        mapped = self.map_stage_input_semantic(semantic, stage_name)
        if normalize_stage_name(stage_name) == "vertex":
            mapped = vertex_builtin_spelling(self, mapped)
        else:
            mapped = draw_parameter_spelling(self, mapped)
        if normalize_stage_name(stage_name) == "compute":
            quadgroup_alias = {
                "gl_QuadGroupID": "(gl_LocalInvocationIndex / 4u)",
                "gl_QuadGroupInvocationID": "(gl_LocalInvocationIndex % 4u)",
            }.get(mapped)
            if quadgroup_alias is not None:
                return quadgroup_alias
        if self.software_subgroup_width is not None:
            subgroup_count = self.glsl_software_subgroup_count or 1
            if subgroup_count == 1:
                subgroup_id = "0u"
                subgroup_lane = "gl_LocalInvocationIndex"
            else:
                subgroup_id = (
                    "(gl_LocalInvocationIndex / "
                    f"{self.GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO})"
                )
                subgroup_lane = (
                    "(gl_LocalInvocationIndex % "
                    f"{self.GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO})"
                )
            software_alias = {
                "gl_NumSubgroups": f"{subgroup_count}u",
                "gl_SubgroupID": subgroup_id,
                "gl_SubgroupSize": self.GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO,
                "gl_SubgroupInvocationID": subgroup_lane,
            }.get(mapped)
            if software_alias is not None:
                return software_alias
        if (
            normalize_stage_name(stage_name) == "fragment"
            and mapped == "gl_SampleMaskIn"
        ):
            return "gl_SampleMaskIn[0]"
        return mapped

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, PointerType):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if isinstance(type_node, ReferenceType):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"{referenced_type}&"
        generic_args = getattr(type_node, "generic_args", [])
        type_name = getattr(type_node, "name", None)
        if type_name is None and hasattr(type_node, "value"):
            return str(type_node.value)
        if type_name and generic_args:
            converted_args = []
            for arg in generic_args:
                converted = self.convert_type_node_to_string(arg)
                if converted is None:
                    if hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                converted_args.append(converted)
            args = ", ".join(converted_args)
            return f"{type_name}<{args}>"
        if type_name:
            return type_name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "d" if element_type == "double" else ""
            if type_node.rows == type_node.cols:
                return f"{prefix}mat{type_node.rows}"
            return f"{prefix}mat{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return prepend_array_dimension(element_type, type_node.size)
                    else:
                        size_str = self.expression_to_string(type_node.size)
                        return prepend_array_dimension(element_type, size_str)
                else:
                    return prepend_array_dimension(element_type)
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                named_vector = crossgl_named_vector_type(element_type, size)
                if named_vector is not None:
                    return named_vector
                return f"{element_type}{size}"
        elif hasattr(type_node, "operator") or hasattr(type_node, "identifier"):
            return self.expression_to_string(type_node)
        else:
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_array_declaration(self, stmt, indent):
        indent_str = "    " * indent
        element_type = self.map_type(stmt.element_type)
        size = get_array_size_from_node(stmt)
        if getattr(
            self, "current_glsl_precise_function", False
        ) and self.glsl_precise_value_type(stmt.element_type):
            element_type = f"precise {element_type}"

        if size is None:
            # In GLSL, dynamic sized arrays need special handling
            # For instance in shader storage blocks, but for simple cases:
            return f"{indent_str}{element_type} {stmt.name}[];\n"
        else:
            return f"{indent_str}{element_type} {stmt.name}[{size}];\n"

    def glsl_attribute(self, node, name):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == name:
                return attr
        return None

    def is_glsl_interface_block_struct(self, node):
        return self.glsl_attribute(node, "glsl_interface_block") is not None

    def glsl_attribute_arguments(self, node, name):
        attr = self.glsl_attribute(node, name)
        if attr is None:
            return []
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def glsl_interface_block_qualifiers(self, node):
        return [
            self.attribute_value_to_string(argument)
            for argument in self.glsl_attribute_arguments(node, "glsl_interface_block")
        ]

    def glsl_interface_block_name(self, node):
        block_name_args = self.glsl_attribute_arguments(
            node, "glsl_interface_block_name"
        )
        if block_name_args:
            return self.attribute_value_to_string(block_name_args[0])
        return node.name

    def glsl_interface_block_instance_suffix(self, node):
        instance_args = self.glsl_attribute_arguments(node, "glsl_interface_instance")
        if not instance_args:
            return ""

        source_instance_name = self.attribute_value_to_string(instance_args[0])
        instance_name = self.glsl_global_identifier_name(source_instance_name)
        array_attr = self.glsl_attribute(node, "glsl_interface_array")
        array_suffix = ""
        if array_attr is not None:
            array_args = getattr(
                array_attr, "arguments", getattr(array_attr, "args", [])
            )
            if array_args:
                array_suffix = "".join(
                    f"[{self.attribute_value_to_string(arg)}]" for arg in array_args
                )
            else:
                array_suffix = "[]"
        return f" {instance_name}{array_suffix}"

    def glsl_struct_member_name_map(self, struct_name):
        if not struct_name:
            return {}
        struct_name = self.glsl_source_type_identifier_name(struct_name)
        if struct_name in self.struct_member_name_maps:
            return self.struct_member_name_maps[struct_name]

        struct = self.structs_by_name.get(struct_name)
        if struct is not None:
            member_names = [
                getattr(member, "name", None)
                for member in getattr(struct, "members", []) or []
            ]
        else:
            member_names = list(self.struct_member_types.get(struct_name, {}))
            for enum in self.struct_payload_enums:
                if enum.name == struct_name:
                    member_names = [
                        "variant",
                        *(
                            field_name
                            for field_name, _ in enum_struct_fields(enum) or []
                        ),
                    ]
                    break
        member_names = [name for name in member_names if name]
        member_map = self.glsl_identifier_map(member_names)

        self.struct_member_name_maps[struct_name] = member_map
        return member_map

    def glsl_struct_member_name(self, struct_name, member_name):
        member_map = self.glsl_struct_member_name_map(struct_name)
        if member_name in member_map:
            return member_map[member_name]
        if member_map:
            return self.refuse_lowering(
                f"member '{member_name}' is not declared by structure '{struct_name}'",
                member_name,
            )
        return glsl_unmapped_identifier(self, member_name, "structure member")

    def generate_glsl_interface_block_member_declaration(
        self,
        member,
        indent="    ",
        struct_name=None,
        members_are_global=False,
        builtin_block=False,
    ):
        qualifier = self.glsl_variable_qualifier_prefix(member)
        qualifier = f"{qualifier} " if qualifier else ""
        source_member_name = getattr(member, "name", "")
        if builtin_block and source_member_name.startswith("gl_"):
            member_name = source_member_name
        elif members_are_global:
            member_name = self.glsl_global_identifier_name(source_member_name)
        else:
            member_name = self.glsl_struct_member_name(struct_name, source_member_name)

        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            if member.size:
                return (
                    f"{indent}{qualifier}{self.map_type(element_type)} "
                    f"{member_name}[{member.size}];\n"
                )
            return (
                f"{indent}{qualifier}{self.map_type(element_type)} {member_name}[];\n"
            )

        if hasattr(member, "member_type"):
            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_type(member_type_str)
            if str(type(member.member_type)).find("ArrayType") != -1:
                declaration = format_c_style_array_declaration(member_type, member_name)
                return f"{indent}{qualifier}{declaration};\n"
            return f"{indent}{qualifier}{member_type} {member_name};\n"

        if hasattr(member, "vtype"):
            member_type = self.map_type(member.vtype)
            return f"{indent}{qualifier}{member_type} {member_name};\n"

        return f"{indent}{qualifier}float {member_name};\n"

    def generate_glsl_interface_block_declaration(self, node):
        layout = self.glsl_variable_layout_prefix(node)
        qualifiers = self.glsl_interface_block_qualifiers(node)
        qualifier_prefix = " ".join(qualifiers)
        if qualifier_prefix:
            qualifier_prefix += " "

        source_block_name = self.glsl_interface_block_name(node)
        block_name = (
            source_block_name
            if source_block_name in self.GLSL_BUILTIN_INTERFACE_BLOCK_NAMES
            else self.glsl_module_identifier_name(source_block_name)
        )
        instance_suffix = self.glsl_interface_block_instance_suffix(node)
        code = f"{layout}{qualifier_prefix}{block_name} {{\n"
        for member in getattr(node, "members", []) or []:
            declaration = self.generate_glsl_interface_block_member_declaration(
                member,
                struct_name=node.name,
                members_are_global=not instance_suffix,
                builtin_block=(
                    source_block_name in self.GLSL_BUILTIN_INTERFACE_BLOCK_NAMES
                ),
            )
            member_layout = stage_block_member_layout(self, node, member)
            if member_layout:
                declaration = f"    {member_layout}{declaration[4:]}"
            code += declaration
        code += f"}}{instance_suffix};\n"
        return code

    def generate_struct_member_declaration(
        self,
        member,
        indent="    ",
        struct_name=None,
        preserve_unsized_arrays=False,
    ):
        member_name = self.glsl_struct_member_name(
            struct_name, getattr(member, "name", "")
        )
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            if member.size:
                return f"{indent}{self.map_type(element_type)} {member_name}[{member.size}];\n"
            if preserve_unsized_arrays:
                return f"{indent}{self.map_type(element_type)} {member_name}[];\n"
            raise UnsizedArrayUnsupportedError(
                "opengl", f"{struct_name or '?'}.{member_name}", "is a struct member"
            )

        if hasattr(member, "member_type"):
            if str(type(member.member_type)).find("ArrayType") != -1:
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.map_type(member_type_str)
                if str(member_type).endswith("[]") and not preserve_unsized_arrays:
                    raise UnsizedArrayUnsupportedError(
                        "opengl",
                        f"{struct_name or '?'}.{member_name}",
                        "is a struct member",
                    )
                declaration = format_c_style_array_declaration(member_type, member_name)
                return f"{indent}{declaration};\n"

            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_type(member_type_str)
            return f"{indent}{member_type} {member_name};\n"

        if hasattr(member, "vtype"):
            member_type = self.map_type(member.vtype)
            return f"{indent}{member_type} {member_name};\n"

        return f"{indent}float {member_name};\n"

    def generate_struct(self, node):
        union = self.union_storage.declaration(node)
        if union is not None:
            return union
        code = f"struct {self.glsl_type_identifier_name(node.name)} {{\n"
        members = [
            member
            for member in getattr(node, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if not members:
            code += "    int _crossgl_empty;\n"
        for member in members:
            code += self.generate_struct_member_declaration(
                member, struct_name=node.name
            )
        code += "};\n"
        return code

    def glsl_static_struct_member(self, member):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", attribute)).lower()
            for attribute in getattr(member, "attributes", []) or []
        }
        return "static" in qualifiers or "static" in attributes

    def remove_glsl_static_struct_member_types(self, structs):
        for struct in structs or []:
            members = self.struct_member_types.get(getattr(struct, "name", None))
            if members is None:
                continue
            for member in getattr(struct, "members", []) or []:
                if self.glsl_static_struct_member(member):
                    members.pop(getattr(member, "name", None), None)

    def glsl_instance_generic_struct_definitions(self, definitions):
        filtered = {}
        for name, definition in (definitions or {}).items():
            filtered[name] = {
                **definition,
                "members": [
                    member
                    for member in definition.get("members", [])
                    if not self.glsl_static_struct_member(member)
                ],
            }
        return filtered

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        if hasattr(stmt, "expression"):
            expr = self.generate_discarded_expression(stmt.expression)
            return expr
        else:
            return self.generate_discarded_expression(stmt)
