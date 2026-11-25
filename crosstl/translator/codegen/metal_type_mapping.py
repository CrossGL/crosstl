"""Type mapping in the Metal generator: types, cooperative matrix contracts, geometry streams, tessellation patches, storage pointer reinterprets, and operator and semantic mapping."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CooperativeMatrixType,
    MemberAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
)
from .array_utils import split_array_type_suffix
from .crossgl_vector_names import crossgl_named_vector_spelling
from .enum_utils import (
    generic_enum_specialized_type_name,
    generic_type_parts,
)
from .generic_struct_utils import generic_struct_specialized_type_name
from .metal_exact_lookups import metal_unmapped_semantic_attribute
from .metal_hlsl_matrix import (
    hlsl_emulated_matrix_shape,
    hlsl_emulated_matrix_type_name,
)
from .metal_type_name_memo import (
    memoized_bare_metal_type,
    note_unmemoized_metal_mapping,
)
from .operator_spellings import map_c_family_operator
from .pointer_reinterpret import (
    PointerReinterpretationError,
    metal_local_single_field_reinterpret_contract,
    metal_storage_pointer_reinterpret_contract,
    scalar_storage_layout,
)
from .storage_word_views import reinterpret_source_layout
from .written_type_names import (
    METAL_TYPE_NAMES,
    sized_float_spelling,
    written_type_name,
)


class MetalTypeMappingMixin:
    """Type mapping members of ``MetalCodeGen``."""

    def map_type(self, vtype, *, memoized=True):
        """Map types to Metal equivalents, handling both strings and TypeNode objects.

        ``memoized=False`` skips the bare type-name memo (``metal_type_name_memo``).
        """
        if memoized:
            if vtype.__class__ is str:
                return memoized_bare_metal_type(self, vtype)
            note_unmemoized_metal_mapping(self)
        if vtype is None:
            return "float"

        cooperative_matrix = self.cooperative_matrix_contract(vtype)
        if cooperative_matrix is not None:
            return self.map_cooperative_matrix_type(vtype, cooperative_matrix)
        if isinstance(vtype, PointerType):
            pointee_type = self.metal_native_narrow_bitcast_storage_type(
                vtype.pointee_type
            ) or self.map_type(vtype.pointee_type)
            return f"{pointee_type}*"
        if isinstance(vtype, ReferenceType):
            referent_type = self.metal_native_narrow_bitcast_storage_type(
                vtype.referenced_type
            ) or self.map_type(vtype.referenced_type)
            return f"{referent_type}&"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        resolved_alias = self.resolve_metal_type_alias(vtype_str)
        if resolved_alias != vtype_str:
            return self.map_type(resolved_alias)

        named_vector = crossgl_named_vector_spelling(vtype_str)
        if named_vector != vtype_str:
            return self.map_type(named_vector)

        # Some imported frontends retain a local pointer/reference declarator as
        # text even though the CrossGL parser represented it structurally first.
        # Map the value type before restoring that exact indirection; otherwise
        # aliases such as bfloat16_t* and fixed-width spellings such as int16*
        # leak into native Metal unchanged while equivalent parameter nodes map
        # correctly through PointerType/ReferenceType above.
        indirection = re.fullmatch(
            r"(?P<base>.+?)(?P<suffix>[*&]+)",
            vtype_str.strip(),
        )
        if indirection is not None:
            base = indirection.group("base").strip()
            if base:
                mapped_base = self.metal_native_narrow_bitcast_storage_type(
                    base
                ) or self.map_type(base)
                return f"{mapped_base}{indirection.group('suffix')}"

        tessellation_patch_type = self.metal_tessellation_patch_mapped_type(vtype)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        geometry_stream_type = self.metal_geometry_stream_mapped_type(vtype)
        if geometry_stream_type is not None:
            return geometry_stream_type

        if self.is_metal_ray_query_type_name(vtype_str):
            self.require_metal_ray_query_runtime()
            self.refuse_metal_ray_query()
            return "CglRayQuery"

        standard_array = self.split_metal_array_resource_type(vtype_str)
        if standard_array is not None:
            element, extent = standard_array
            return f"array<{self.metal_standard_array_element_type(element)}, {extent}>"

        if self.requires_metal_builtin_ray_desc(vtype_str):
            self.required_metal_ray_desc_runtime = True
            return "CglRayDesc"

        emulated_matrix = hlsl_emulated_matrix_shape(vtype_str)
        if emulated_matrix is not None:
            self.metal_emulated_matrix_shapes.add(emulated_matrix)
            return hlsl_emulated_matrix_type_name(emulated_matrix)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        if self.is_visible_function_table_type(vtype_str):
            return self.map_visible_function_table_type(vtype_str)

        if self.is_intersection_function_table_type(vtype_str):
            return self.map_intersection_function_table_type(vtype_str)

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        option_payload = self.lowerable_option_payload_type_name(vtype_str)
        if option_payload is not None:
            return self.map_type(option_payload)

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return vtype_str

        fixed_width_vector = re.fullmatch(
            r"(?P<unsigned>u?)int(?P<bits>8|16|32|64)(?:_t)?(?P<lanes>[234])",
            vtype_str,
        )
        if fixed_width_vector is not None:
            bits = fixed_width_vector.group("bits")
            unsigned = bool(fixed_width_vector.group("unsigned"))
            if bits == "64":
                component = "ulong" if unsigned else "long"
            else:
                component = "uint" if unsigned else "int"
            return f"{component}{fixed_width_vector.group('lanes')}"

        fixed_float_vector = re.fullmatch(
            r"(?P<component>(?:float(?:16|32|64)|bfloat16)(?:_t)?)" r"(?P<lanes>[234])",
            vtype_str,
        )
        if fixed_float_vector is not None:
            component_name = fixed_float_vector.group("component")
            if component_name.startswith("bfloat"):
                component = "bfloat"
            elif "16" in component_name:
                component = "half"
            elif "64" in component_name:
                component = "double"
            else:
                component = "float"
            return f"{component}{fixed_float_vector.group('lanes')}"

        if vtype_str not in self.type_mapping:
            if sized_float_spelling(vtype_str) != vtype_str:
                return self.map_type(sized_float_spelling(vtype_str))
            if re.fullmatch(r"double(?:[234](?:x[234])?)?", vtype_str):
                # Refused with the specification's reason by ``refuse_metal_double_values``.
                return vtype_str
            return written_type_name(
                vtype_str,
                METAL_TYPE_NAMES,
                self.program_type_names,
                self.written_foreign_type_names,
            )
        return self.type_mapping[vtype_str]

    def cooperative_matrix_contract(self, matrix_type):
        if isinstance(matrix_type, CooperativeMatrixType):
            return (
                matrix_type.element_type,
                matrix_type.rows,
                matrix_type.cols,
                matrix_type.scope,
                matrix_type.use,
                matrix_type.layout,
                matrix_type.fragment_layout,
                matrix_type.subgroup_size,
                matrix_type.elements_per_lane,
                matrix_type.fragment_provenance,
                matrix_type.fragment_mapping,
                matrix_type.fragment_mapping_provenance,
            )

        type_name = str(matrix_type).strip()
        base_name, generic_args = generic_type_parts(type_name)
        if base_name.rsplit("::", 1)[-1] != "CooperativeMatrix" or not (
            3 <= len(generic_args) <= 12
        ):
            return None
        defaults = (
            "subgroup",
            "unspecified",
            "unspecified",
            "unspecified",
            "unspecified",
            "unspecified",
            "unspecified",
            "unspecified",
            "unspecified",
        )
        generic_args = [*generic_args, *defaults[len(generic_args) - 3 :]]
        return tuple(generic_args)

    def metal_geometry_stream_info(self, vtype):
        if vtype is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if isinstance(vtype, (PointerType, ReferenceType)):
            return None

        name = getattr(vtype, "name", None)
        generic_args = getattr(vtype, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return None

        base_name, generic_arg = type_name.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.map_type(generic_arg)

    def metal_geometry_stream_mapped_type(self, vtype):
        stream_info = self.metal_geometry_stream_info(vtype)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CrossGLMetal{stream_name}<{output_type}>"

    def format_metal_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.metal_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"thread {stream_type}& {name}"

    def split_metal_generic_argument_string(self, arguments):
        split_arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(arguments):
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                split_arguments.append(arguments[start:index].strip())
                start = index + 1
        split_arguments.append(arguments[start:].strip())
        return [argument for argument in split_arguments if argument]

    def metal_type_generic_arguments(self, vtype):
        generic_args = getattr(vtype, "generic_args", None)
        if generic_args:
            return generic_args

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return []
        arguments = type_name.split("<", 1)[1][:-1].strip()
        return self.split_metal_generic_argument_string(arguments)

    def metal_tessellation_patch_type_base(self, vtype):
        if isinstance(vtype, PointerType):
            vtype = vtype.pointee_type
        elif isinstance(vtype, ReferenceType):
            vtype = vtype.referenced_type

        name = getattr(vtype, "name", None)
        if name:
            return name

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        return type_name.split("<", 1)[0].strip()

    def metal_tessellation_patch_argument_text(self, argument):
        if argument is None:
            return None
        if isinstance(argument, str):
            return argument
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return self.type_name_string(argument) or str(argument)

    def metal_tessellation_patch_info(self, vtype):
        if isinstance(vtype, PointerType):
            vtype = vtype.pointee_type
        elif isinstance(vtype, ReferenceType):
            vtype = vtype.referenced_type

        patch_type = self.metal_tessellation_patch_type_base(vtype)
        if patch_type not in {"InputPatch", "OutputPatch"}:
            return None

        generic_args = self.metal_type_generic_arguments(vtype)
        valid = len(generic_args) == 2
        element_type = None
        count_text = None
        count_value = None
        if valid:
            element_type = self.map_type(generic_args[0])
            count_text = self.metal_tessellation_patch_argument_text(generic_args[1])
            count_value = self.literal_int_value(
                generic_args[1], self.literal_int_constants
            )
            if count_value is None:
                count_value = self.literal_int_value(
                    count_text, self.literal_int_constants
                )

        return {
            "kind": patch_type,
            "valid": valid,
            "element_type": element_type,
            "count_text": count_text,
            "count_value": count_value,
        }

    def metal_tessellation_patch_mapped_type(self, vtype):
        info = self.metal_tessellation_patch_info(vtype)
        if info is None or not info["valid"]:
            return None
        if not info["element_type"] or not info["count_text"]:
            return None
        return (
            f"CrossGLMetal{info['kind']}<"
            f"{info['element_type']}, {info['count_text']}>"
        )

    def format_metal_tessellation_patch_parameter(self, raw_param_type, name):
        patch_type = self.metal_tessellation_patch_mapped_type(raw_param_type)
        if patch_type is None:
            return None
        return f"thread const {patch_type}& {name}"

    def metal_storage_pointer_reinterpret_member_node(self, expression):
        if isinstance(expression, UnaryOpNode) and getattr(
            expression, "operator", getattr(expression, "op", None)
        ) in {"&", "*"}:
            return self.metal_storage_pointer_reinterpret_member_node(
                getattr(expression, "operand", None)
            )
        if isinstance(expression, ArrayAccessNode):
            return self.metal_storage_pointer_reinterpret_member_node(
                getattr(expression, "array", getattr(expression, "array_expr", None))
            )
        if isinstance(expression, BinaryOpNode) and getattr(
            expression, "operator", getattr(expression, "op", None)
        ) in {"+", "-"}:
            for operand in (
                getattr(expression, "left", None),
                getattr(expression, "right", None),
            ):
                operand_type = self.expression_result_type(operand)
                if self.metal_type_is_pointer_like(operand_type) or (
                    self.argument_address_space(operand) is not None
                ):
                    member = self.metal_storage_pointer_reinterpret_member_node(operand)
                    if member is not None:
                        return member
            return None
        if not isinstance(expression, MemberAccessNode):
            return None
        object_expr = getattr(
            expression, "object", getattr(expression, "object_expr", None)
        )
        return self.struct_member_named_node(object_expr, str(expression.member))

    def metal_storage_pointer_reinterpret_source_is_readonly(
        self, expression, source_type
    ):
        pointer_operand = self.pointer_arithmetic_operand(expression)
        if pointer_operand is not None:
            # Offsets add no storage; only the pointer operand can be const.
            return self.metal_storage_pointer_reinterpret_source_is_readonly(
                pointer_operand, self.expression_result_type(pointer_operand)
            )
        if isinstance(expression, PointerReinterpretNode):
            # A nested view is lowered with its own constness proof.
            return not getattr(
                getattr(expression, "target_type", None), "is_mutable", True
            )
        member = self.metal_storage_pointer_reinterpret_member_node(expression)
        if member is not None:
            raw_member_type = getattr(
                member,
                "member_type",
                getattr(member, "vtype", getattr(member, "element_type", None)),
            )
            if isinstance(raw_member_type, (PointerType, ReferenceType)):
                return not self.is_mutable_metal_parameter(raw_member_type, member)

        if self.structured_buffer_type_name(source_type) == "StructuredBuffer":
            return True

        root_name = self.assignment_target_root_name(expression)
        return root_name in self.current_readonly_metal_parameters or (
            root_name in self.current_readonly_raw_buffer_parameters
        )

    def metal_storage_pointer_reinterpret_error(
        self, expression, contract, reason, source_type=None
    ):
        source_name = self.type_name_string(source_type)
        if source_name is None and source_type is not None:
            source_name = str(source_type)
        raise PointerReinterpretationError(
            "metal cannot prove a semantics-preserving storage pointer "
            f"reinterpretation ({reason})",
            source_type=source_name,
            target_type=contract.get("targetTypeName"),
            address_space=contract.get("addressSpace"),
            access=(
                "read"
                if contract.get("usage") == "immediate-value-read"
                else "write" if contract.get("isMutable") else "read"
            ),
            target_backend="metal",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def generate_metal_storage_pointer_reinterpret(self, expression):
        contract = metal_storage_pointer_reinterpret_contract(expression)
        if contract is None:
            self.metal_storage_pointer_reinterpret_error(
                expression,
                {
                    "targetTypeName": self.type_name_string(
                        getattr(
                            getattr(expression, "target_type", None),
                            "pointee_type",
                            None,
                        )
                    ),
                    "addressSpace": getattr(
                        getattr(expression, "target_type", None),
                        "address_space",
                        None,
                    ),
                    "isMutable": True,
                },
                "target-lowering-unavailable",
            )

        source = getattr(expression, "expression", None)
        source_type = self.expression_result_type(source)
        source_is_address = (
            isinstance(source, UnaryOpNode)
            and getattr(source, "operator", getattr(source, "op", None)) == "&"
        )
        if not source_is_address and not self.metal_type_is_pointer_like(source_type):
            self.metal_storage_pointer_reinterpret_error(
                expression, contract, "source-not-pointer-storage", source_type
            )

        if self.argument_address_space_conflict(source) is not None:
            self.metal_storage_pointer_reinterpret_error(
                expression, contract, "source-address-space-conflict", source_type
            )
        source_address_space = self.argument_address_space(source)
        target_address_space = contract["addressSpace"]
        if source_address_space is None:
            self.metal_storage_pointer_reinterpret_error(
                expression, contract, "source-address-space-unproven", source_type
            )
        if source_address_space != target_address_space:
            self.metal_storage_pointer_reinterpret_error(
                expression, contract, "address-space-mismatch", source_type
            )

        source_pointee = (
            self.type_name_string(
                self.expression_result_type(getattr(source, "operand", None))
            )
            if source_is_address
            else self.pointer_pointee_type_name(source_type)
        )
        if source_pointee is None and self.is_structured_buffer_type(source_type):
            source_pointee = self.structured_buffer_element_type(source_type)

        if contract.get("aggregateByteView"):
            source_layout = scalar_storage_layout(source_pointee)
            required_alignment = contract.get("alignment")
            if (
                source_layout is None
                or required_alignment is None
                or source_layout.byte_width < required_alignment
                or source_layout.byte_width % required_alignment
            ):
                self.metal_storage_pointer_reinterpret_error(
                    expression, contract, "source-alignment-unproven", source_type
                )

        if contract.get("scalarWrapperView"):
            # The source words must be one 32-bit scalar or such a wrapper.
            source_layout, _ = reinterpret_source_layout(
                source_pointee, self.structs_by_name
            )
            if source_layout is None or source_layout.bit_width != 32:
                self.metal_storage_pointer_reinterpret_error(
                    expression, contract, "source-element-layout-mismatch", source_type
                )

        if contract.get("aggregateHomogeneousScalarArrayRead"):
            source_layout = scalar_storage_layout(source_pointee)
            target_layout = scalar_storage_layout(contract.get("elementTypeName"))
            element_count = contract.get("elementCount")
            target_layout_is_exact = (
                target_layout is not None
                and isinstance(element_count, int)
                and not isinstance(element_count, bool)
                and element_count > 0
                and contract.get("alignment") == target_layout.byte_width
                and contract.get("byteWidth")
                == target_layout.byte_width * element_count
            )
            if not target_layout_is_exact:
                self.metal_storage_pointer_reinterpret_error(
                    expression, contract, "target-layout-unproven", source_type
                )
            if source_layout is None or source_layout != target_layout:
                self.metal_storage_pointer_reinterpret_error(
                    expression, contract, "source-element-layout-mismatch", source_type
                )

        source_is_readonly = self.metal_storage_pointer_reinterpret_source_is_readonly(
            source, source_type
        )
        target_type = self.metal_native_narrow_bitcast_storage_type(
            contract["targetType"]
        ) or self.map_type(contract["targetType"])
        effective_is_mutable = contract["isMutable"]
        if source_is_readonly and contract["isMutable"]:
            if contract.get("readOnlyUse"):
                # The source spelling may remove pointee constness before a
                # value load. Reconstruct the same native pointer identity as
                # a const view because the validated AST cannot write through
                # or escape this pointer.
                effective_is_mutable = False
            else:
                source_storage_type = self.metal_native_narrow_bitcast_storage_type(
                    source_pointee
                ) or self.map_type(source_pointee)
                if (
                    source_pointee is None
                    or source_storage_type != target_type
                    or contract.get("usage") != "call-argument"
                ):
                    self.metal_storage_pointer_reinterpret_error(
                        expression, contract, "const-removal", source_type
                    )
                return self.generate_expression(source)

        const_prefix = "" if effective_is_mutable else "const "
        if target_address_space == "constant":
            const_prefix = ""
        cast_type = f"{const_prefix}{target_address_space} {target_type}*"
        return f"reinterpret_cast<{cast_type}>({self.generate_expression(source)})"

    def generate_metal_local_reinterpret_read(self, expression):
        if getattr(expression, "operator", getattr(expression, "op", None)) != "*":
            return None
        reinterpret = getattr(expression, "operand", None)
        if not isinstance(reinterpret, PointerReinterpretNode):
            return None
        contract = metal_local_single_field_reinterpret_contract(reinterpret)
        if contract is None:
            return None
        address = getattr(reinterpret, "expression", None)
        source = getattr(address, "operand", None)
        if source is None:
            return None
        target_type = self.map_type(contract["targetType"])
        value = self.generate_expression(source)
        native_type = self.metal_native_narrow_bitcast_storage_type(
            contract["sourceType"]
        )
        if native_type is not None:
            value = f"{native_type}({value})"
        return f"{target_type}{{{value}}}"

    def map_operator(self, op):
        """Spell a CrossGL operator; an operator CrossGL does not define is refused."""
        return map_c_family_operator(op, "Metal")

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to Metal attribute syntax."""
        if semantic is not None:
            hlsl_attribute = self.metal_hlsl_attribute_semantic(semantic)
            if hlsl_attribute is not None:
                return f" [[{hlsl_attribute}]]"
            mapped_semantic = self.semantic_map.get(str(semantic))
            if mapped_semantic is not None and mapped_semantic.startswith("attribute("):
                # A classic vertex semantic keeps the location every target gives it.
                return f" [[{mapped_semantic}]]"
            hlsl_color = self.metal_hlsl_color_semantic(semantic)
            if hlsl_color is not None:
                return f" [[user({hlsl_color})]]"
            if (
                self.is_metal_tessellation_helper_semantic(semantic)
                and mapped_semantic != "primitive_id"
            ):
                return ""
            if mapped_semantic is None:
                mapped_semantic = metal_unmapped_semantic_attribute(self, semantic)
                if mapped_semantic is None:
                    return ""
            if mapped_semantic.startswith("[[") and mapped_semantic.endswith("]]"):
                return f" {mapped_semantic}"
            else:
                return f" [[{mapped_semantic}]]"
        else:
            return ""

    def metal_hlsl_attribute_semantic(self, semantic):
        match = re.fullmatch(r"ATTRIB(\d+)", str(semantic), re.IGNORECASE)
        if match:
            return f"attribute({match.group(1)})"
        return None

    def metal_hlsl_color_semantic(self, semantic):
        match = re.fullmatch(r"COLOR(\d+)", str(semantic), re.IGNORECASE)
        if not match:
            return None
        return f"Color{match.group(1)}"

    def metal_array_semantic_attribute_precedes_extent(self, semantic):
        return self.canonical_metal_semantic(semantic) in {"clip_distance"}

    def is_metal_tessellation_helper_semantic(self, semantic):
        if semantic is None:
            return False
        return str(semantic).lower() in {
            "gl_tesslevelinner",
            "gl_tesslevelouter",
            "sv_domainlocation",
            "sv_insidetessfactor",
            "sv_outputcontrolpointid",
            "sv_primitiveid",
            "sv_tessfactor",
        }
