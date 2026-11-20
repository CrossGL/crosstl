"""Struct and union members in the HLSL generator: specialization constants, implicit samplers and texture access, lowered sampler declarations and Metal preprocessor constructs."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    FunctionCallNode,
    MemberAccessNode,
    PointerType,
    PreprocessorNode,
    ReferenceType,
    StructNode,
)
from .apple_sdk_includes import is_apple_sdk_include
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import (
    DirectXResourcePointerArrayError,
    DirectXSpecializationConstantError,
)
from .enum_utils import generic_type_parts
from .hlsl_specialization_macros import render_hlsl_required_specialization_constant
from .hlsl_struct_initializers import hlsl_brace_struct_initializer
from .image_access_contracts import (
    is_projected_texture_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_operation,
    is_texture_query_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
)
from .union_lanes import hlsl_lane_call
from .union_layout import is_lane_kind


class DirectXStructMemberMixin:
    """Struct member members of ``HLSLCodeGen``."""

    def hlsl_union_layout_for_type(self, type_value):
        type_name = self.type_name_string(type_value)
        base_type, _array_suffix = split_array_type_suffix(str(type_name or ""))
        base_type = str(base_type).strip()
        while base_type.startswith(("const ", "volatile ")):
            base_type = base_type.split(" ", 1)[1].strip()
        base_type = base_type.rstrip("&*").strip()
        return self.hlsl_union_layouts.get(base_type)

    def hlsl_union_member_access_info(self, expression):
        if not isinstance(expression, MemberAccessNode):
            return None
        object_expr = getattr(
            expression, "object_expr", getattr(expression, "object", None)
        )
        layout = self.hlsl_union_layout_for_type(
            self.expression_result_type(object_expr)
        )
        if layout is None:
            return None
        member_name = str(getattr(expression, "member", ""))
        member = layout["members"].get(member_name)
        if member is None:
            return None
        return {
            "expression": expression,
            "object": object_expr,
            "layout": layout,
            "member_name": member_name,
            "member": member,
        }

    def hlsl_union_array_access_info(self, expression):
        if not isinstance(expression, ArrayAccessNode):
            return None
        array_expr = getattr(
            expression, "array_expr", getattr(expression, "array", None)
        )
        member_info = self.hlsl_union_member_access_info(array_expr)
        if member_info is None or not member_info["member"]["kind"].endswith("_array"):
            return None
        return {
            **member_info,
            "expression": expression,
            "member_expression": array_expr,
            "index": getattr(
                expression, "index_expr", getattr(expression, "index", None)
            ),
        }

    def hlsl_union_storage_expression(self, info, index=None):
        object_expr = self.generate_expression_with_expected(info["object"], None)
        storage = f"{object_expr}.{info['layout']['storage_field']}"
        if index is not None:
            rendered_index = self.generate_expression_with_expected(index, "uint")
            storage = f"{storage}[{rendered_index}]"
        return storage

    def hlsl_union_usage_error(self, info, expression, reason, detail):
        layout = info["layout"]
        return self.hlsl_union_layout_error(
            layout["node"],
            reason=reason,
            detail=detail,
            member=info["member"].get("node"),
            member_type=info["member"].get("type"),
            size=layout["size"],
            alignment=layout["alignment"],
        )

    def generate_hlsl_union_member_read(self, expression):
        info = self.hlsl_union_member_access_info(expression)
        if info is None:
            return None
        kind = info["member"]["kind"]
        if kind.endswith("_array"):
            raise self.hlsl_union_usage_error(
                info,
                expression,
                "whole-array-member-access-unsupported",
                "HLSL cannot materialize the union's fixed-array member as a value",
            )
        storage = self.hlsl_union_storage_expression(info)
        if kind == "uint32":
            return storage
        if kind == "int32":
            return f"asint({storage})"
        if kind == "float32":
            return f"asfloat({storage})"
        if kind == "u8x4":
            return f"__crossgl_union_unpack_u8x4({storage})"
        if kind == "bool8x4":
            return f"__crossgl_union_unpack_bool4({storage})"
        return None

    def generate_hlsl_union_array_read(self, expression):
        info = self.hlsl_union_array_access_info(expression)
        if info is None:
            return None
        if is_lane_kind(info["member"]["kind"]):
            return hlsl_lane_call(self, info, "get")
        storage = self.hlsl_union_storage_expression(info, info["index"])
        kind = info["member"]["kind"]
        if kind == "uint32_array":
            return storage
        if kind == "int32_array":
            return f"asint({storage})"
        if kind == "float32_array":
            return f"asfloat({storage})"
        if kind == "u8x4_array":
            return f"__crossgl_union_unpack_u8x4({storage})"
        if kind == "bool8x4_array":
            return f"__crossgl_union_unpack_bool4({storage})"
        return None

    def hlsl_union_small_vector_lane_target(self, expression):
        lane = None
        value_expr = None
        if isinstance(expression, ArrayAccessNode):
            value_expr = getattr(
                expression, "array_expr", getattr(expression, "array", None)
            )
            lane = getattr(expression, "index_expr", getattr(expression, "index", None))
        elif isinstance(expression, MemberAccessNode):
            lane_name = str(getattr(expression, "member", ""))
            lane_index = {
                "x": 0,
                "r": 0,
                "y": 1,
                "g": 1,
                "z": 2,
                "b": 2,
                "w": 3,
                "a": 3,
            }.get(lane_name)
            if lane_index is None:
                return None
            value_expr = getattr(
                expression, "object_expr", getattr(expression, "object", None)
            )
            lane = f"{lane_index}u"
        if value_expr is None:
            return None

        array_info = self.hlsl_union_array_access_info(value_expr)
        if array_info is not None and array_info["member"]["kind"] in {
            "u8x4_array",
            "bool8x4_array",
        }:
            return {
                "info": array_info,
                "storage": self.hlsl_union_storage_expression(
                    array_info, array_info["index"]
                ),
                "lane": lane,
            }
        member_info = self.hlsl_union_member_access_info(value_expr)
        if member_info is not None and member_info["member"]["kind"] in {
            "u8x4",
            "bool8x4",
        }:
            return {
                "info": member_info,
                "storage": self.hlsl_union_storage_expression(member_info),
                "lane": lane,
            }
        return None

    def hlsl_union_access_ancestor(self, expression):
        current = expression
        while isinstance(current, (MemberAccessNode, ArrayAccessNode)):
            array_info = self.hlsl_union_array_access_info(current)
            if array_info is not None:
                return array_info
            member_info = self.hlsl_union_member_access_info(current)
            if member_info is not None:
                return member_info
            if isinstance(current, MemberAccessNode):
                current = getattr(
                    current, "object_expr", getattr(current, "object", None)
                )
            else:
                current = getattr(
                    current, "array_expr", getattr(current, "array", None)
                )
        return None

    def generate_hlsl_union_assignment(self, target, value, operator):
        lane_target = self.hlsl_union_small_vector_lane_target(target)
        if lane_target is not None:
            if operator != "=":
                raise self.hlsl_union_usage_error(
                    lane_target["info"],
                    target,
                    "byte-lane-compound-write-unsupported",
                    "compound byte-lane writes require an unimplemented read-modify-write contract",
                )
            lane = lane_target["lane"]
            rendered_lane = (
                lane
                if isinstance(lane, str)
                else self.generate_expression_with_expected(lane, "uint")
            )
            kind = lane_target["info"]["member"]["kind"]
            is_bool = kind in {"bool8x4", "bool8x4_array"}
            selected = self.generate_expression_with_expected(
                value, "bool" if is_bool else "uint"
            )
            helper = (
                "__crossgl_union_set_bool4_lane"
                if is_bool
                else "__crossgl_union_set_u8x4_lane"
            )
            return (
                f"{helper}(" f"{lane_target['storage']}, {rendered_lane}, {selected})"
            )

        array_info = self.hlsl_union_array_access_info(target)
        if array_info is not None:
            kind = array_info["member"]["kind"]
            if is_lane_kind(kind):
                if operator != "=":
                    raise self.hlsl_union_usage_error(
                        array_info,
                        target,
                        "lane-compound-write-unsupported",
                        "compound lane writes need a read-modify-write contract",
                    )
                return hlsl_lane_call(self, array_info, "set", value)
            storage = self.hlsl_union_storage_expression(
                array_info, array_info["index"]
            )
            element_type, _suffix = split_array_type_suffix(
                array_info["member"]["type"]
            )
            rendered = self.generate_expression_with_expected(value, element_type)
            if kind == "uint32_array":
                return f"{storage} {operator} {rendered}"
            if operator != "=":
                raise self.hlsl_union_usage_error(
                    array_info,
                    target,
                    "interpreted-compound-write-unsupported",
                    "compound writes through a reinterpreted array member are not supported",
                )
            if kind in {"int32_array", "float32_array"}:
                rendered = f"asuint({rendered})"
            elif kind == "u8x4_array":
                rendered = f"__crossgl_union_pack_u8x4({rendered})"
            elif kind == "bool8x4_array":
                rendered = f"__crossgl_union_pack_bool4({rendered})"
            return f"{storage} = {rendered}"

        member_info = self.hlsl_union_member_access_info(target)
        if member_info is not None:
            kind = member_info["member"]["kind"]
            if kind.endswith("_array"):
                raise self.hlsl_union_usage_error(
                    member_info,
                    target,
                    "whole-array-member-write-unsupported",
                    "HLSL cannot assign the union's fixed-array member as a value",
                )
            storage = self.hlsl_union_storage_expression(member_info)
            rendered = self.generate_expression_with_expected(
                value, member_info["member"]["type"]
            )
            if kind == "uint32":
                return f"{storage} {operator} {rendered}"
            if operator != "=":
                raise self.hlsl_union_usage_error(
                    member_info,
                    target,
                    "interpreted-compound-write-unsupported",
                    "compound writes through a reinterpreted member are not supported",
                )
            if kind in {"int32", "float32"}:
                rendered = f"asuint({rendered})"
            elif kind == "u8x4":
                rendered = f"__crossgl_union_pack_u8x4({rendered})"
            elif kind == "bool8x4":
                rendered = f"__crossgl_union_pack_bool4({rendered})"
            return f"{storage} = {rendered}"

        ancestor = self.hlsl_union_access_ancestor(target)
        if ancestor is not None and ancestor["member"]["kind"] not in {
            "uint32",
            "uint32_array",
        }:
            raise self.hlsl_union_usage_error(
                ancestor,
                target,
                "interpreted-subcomponent-write-unsupported",
                "this reinterpreted member subcomponent is not a canonical HLSL lvalue",
            )
        return None

    def generate_hlsl_union_small_vector_set_call(self, function_name, arguments):
        contracts = {
            "CrossGLMetalVectorIndex_u8vec4_set": (
                "u8x4",
                "u8x4_array",
                "uint",
                "__crossgl_union_set_u8x4_lane",
            ),
            "CrossGLMetalVectorIndex_bvec4_set": (
                "bool8x4",
                "bool8x4_array",
                "bool",
                "__crossgl_union_set_bool4_lane",
            ),
        }
        contract = contracts.get(function_name)
        if contract is None or len(arguments or []) != 3:
            return None
        member_kind, array_kind, selected_type, helper = contract
        value_expr, lane_expr, selected_expr = arguments
        array_info = self.hlsl_union_array_access_info(value_expr)
        if array_info is not None and array_info["member"]["kind"] == array_kind:
            storage = self.hlsl_union_storage_expression(
                array_info, array_info["index"]
            )
        else:
            member_info = self.hlsl_union_member_access_info(value_expr)
            if member_info is None or member_info["member"]["kind"] != member_kind:
                return None
            storage = self.hlsl_union_storage_expression(member_info)
        lane = self.generate_expression_with_expected(lane_expr, "uint")
        selected = self.generate_expression_with_expected(selected_expr, selected_type)
        return f"{helper}({storage}, {lane}, {selected})"

    def hlsl_static_struct_member(self, member):
        """Return whether a member is compile-time metadata, not instance state."""
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", attribute)).lower()
            for attribute in getattr(member, "attributes", []) or []
        }
        return "static" in qualifiers or "static" in attributes

    def remove_hlsl_static_struct_member_types(self, structs):
        """Keep constructor and member layouts aligned with emitted HLSL structs."""
        for struct in structs or []:
            members = self.struct_member_types.get(getattr(struct, "name", None))
            if members is None:
                continue
            for member in getattr(struct, "members", []) or []:
                if self.hlsl_static_struct_member(member):
                    members.pop(getattr(member, "name", None), None)

    def hlsl_struct_member_raw_type(self, member):
        if isinstance(member, ArrayNode):
            return getattr(member, "element_type", getattr(member, "vtype", None))
        if hasattr(member, "member_type"):
            return member.member_type
        if hasattr(member, "vtype"):
            return member.vtype
        return None

    def hlsl_unresolved_generic_pointer_member(
        self,
        raw_type,
        *,
        array_node=False,
        visited=None,
    ):
        """Find an unresolved generic pointer in an aggregate type graph."""

        type_name = self.type_name_string(raw_type)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        aggregate_type, array_suffix = split_array_type_suffix(type_name)
        array_depth = (1 if array_node else 0) + array_suffix.count("[")
        aggregate_type = aggregate_type.strip()
        while aggregate_type:
            base_name, generic_args = generic_type_parts(aggregate_type)
            if base_name.rsplit("::", 1)[-1] != "array" or not generic_args:
                break
            array_depth += 1
            aggregate_type = str(generic_args[0]).strip()

        base_name, generic_args = generic_type_parts(aggregate_type)
        if base_name.rsplit("::", 1)[-1] == "ptr":
            return {
                "array_depth": array_depth,
                "member_path": (),
                "pointee_type": str(generic_args[-1]).strip() if generic_args else None,
                "type_name": type_name,
            }

        aggregate_names = (
            aggregate_type,
            base_name,
            base_name.rsplit("::", 1)[-1],
        )
        aggregate_name = next(
            (
                candidate
                for candidate in aggregate_names
                if candidate in self.struct_member_types
            ),
            None,
        )
        if aggregate_name is None:
            return None

        visited = set(visited or ())
        if aggregate_name in visited:
            return None
        visited.add(aggregate_name)
        for nested_name, nested_type in sorted(
            self.struct_member_types[aggregate_name].items()
        ):
            nested_info = self.hlsl_unresolved_generic_pointer_member(
                nested_type,
                visited=visited,
            )
            if nested_info is None:
                continue
            return {
                **nested_info,
                "member_path": (
                    nested_name,
                    *nested_info.get("member_path", ()),
                ),
            }
        return None

    def hlsl_reject_struct_pointer_member(
        self,
        struct_name,
        member_name,
        raw_type,
        *,
        array_node=False,
        source_node=None,
    ):
        """Reject source struct state that HLSL cannot represent faithfully."""

        array_depth = 1 if array_node else 0
        element_type = raw_type
        while isinstance(element_type, ArrayType):
            array_depth += 1
            element_type = getattr(element_type, "element_type", None)
        if isinstance(element_type, (PointerType, ReferenceType)):
            address_space = (
                str(getattr(element_type, "address_space", None) or "thread")
                .strip()
                .lower()
            )
            pointer_kind = "pointer-array" if array_depth else "pointer"
            raise DirectXResourcePointerArrayError(
                "DirectX cannot faithfully lower source struct "
                f"'{struct_name}' member '{member_name}' because its "
                f"{pointer_kind} state has no HLSL value representation",
                function_name=self.current_function_name,
                array_name=member_name,
                address_space=address_space,
                reason="struct-pointer-member-unsupported",
                source_location=getattr(source_node, "source_location", None),
            )

        generic_pointer = self.hlsl_unresolved_generic_pointer_member(
            raw_type,
            array_node=array_node,
        )
        if generic_pointer is None:
            return
        member_path = (
            member_name,
            *generic_pointer.get("member_path", ()),
        )
        qualified_member_name = ".".join(
            str(path_component) for path_component in member_path if path_component
        )
        pointer_kind = "pointer array" if generic_pointer["array_depth"] else "pointer"
        raise DirectXResourcePointerArrayError(
            "DirectX cannot faithfully lower source struct "
            f"'{struct_name}' member '{qualified_member_name}' because type "
            f"'{generic_pointer['type_name']}' contains an unresolved generic "
            f"{pointer_kind} with no HLSL value representation",
            function_name=self.current_function_name,
            array_name=qualified_member_name,
            address_space="unresolved",
            reason="struct-pointer-member-unsupported",
            source_location=getattr(source_node, "source_location", None),
        )

    def hlsl_struct_member_is_resource(self, raw_type):
        return (
            self.is_texture_type(raw_type)
            or self.is_image_type(raw_type)
            or self.is_sampler_type(raw_type)
        )

    def hlsl_struct_member_should_lower_resource(self, raw_type):
        return self.is_texture_type(raw_type) or self.is_image_type(raw_type)

    def hlsl_array_element_type_name(self, raw_type):
        if raw_type is None:
            return None
        if hasattr(raw_type, "element_type") and "ArrayType" in str(type(raw_type)):
            return self.type_name_string(raw_type.element_type)
        type_name = self.type_name_string(raw_type)
        base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
        return base_type if array_suffix else None

    def collect_hlsl_lowered_struct_resource_members(self, structs, functions=None):
        resource_members = {}
        for node in structs or []:
            if not isinstance(node, StructNode):
                continue
            for member in getattr(node, "members", []) or []:
                raw_type = self.hlsl_struct_member_raw_type(member)
                if not self.hlsl_struct_member_should_lower_resource(raw_type):
                    continue
                resource_members.setdefault(node.name, {})[member.name] = raw_type

        lowered_struct_names = set()
        for node in structs or []:
            if not isinstance(node, StructNode):
                continue
            for member in getattr(node, "members", []) or []:
                element_type = self.hlsl_array_element_type_name(
                    self.hlsl_struct_member_raw_type(member)
                )
                if element_type in resource_members:
                    lowered_struct_names.add(element_type)

        for parameter in self.collect_function_parameters(functions):
            parameter_type = self.function_parameter_type_name(parameter)
            if parameter_type in resource_members:
                lowered_struct_names.add(parameter_type)

        lowered_members = {}
        member_name_counts = {}
        for struct_name in lowered_struct_names:
            for member_name in resource_members.get(struct_name, {}):
                member_name_counts[member_name] = (
                    member_name_counts.get(member_name, 0) + 1
                )

        used_global_names = set(self.global_variable_types)
        for struct_name in sorted(lowered_struct_names):
            for member_name, raw_type in sorted(
                resource_members.get(struct_name, {}).items()
            ):
                global_name = member_name
                if (
                    member_name_counts.get(member_name, 0) > 1
                    or member_name in used_global_names
                ):
                    global_name = f"{struct_name}_{member_name}"
                suffix = 2
                base_global_name = global_name
                while global_name in used_global_names:
                    global_name = f"{base_global_name}_{suffix}"
                    suffix += 1
                used_global_names.add(global_name)
                lowered_members.setdefault(struct_name, {})[member_name] = {
                    "type": raw_type,
                    "global_name": global_name,
                }
        return lowered_members

    def hlsl_should_lower_struct_resource_member(self, struct_name, member_name):
        return member_name in self.hlsl_lowered_struct_resource_members.get(
            struct_name, {}
        )

    def hlsl_lowered_struct_resource_member_global_name(self, expr):
        if not (
            isinstance(expr, MemberAccessNode)
            or (hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__))
        ):
            return None
        member_name = str(getattr(expr, "member", ""))
        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        object_type = self.type_name_string(self.expression_result_type(obj_expr))
        member_info = self.hlsl_lowered_struct_resource_members.get(
            object_type, {}
        ).get(member_name)
        if member_info is None:
            return None
        return member_info["global_name"]

    def generate_hlsl_lowered_struct_resource_member_globals(
        self,
        used_resource_registers,
        texture_registers,
        sampler_registers,
        uav_registers,
    ):
        code = ""
        emitted_names = set(self.global_variable_types)
        for members in self.hlsl_lowered_struct_resource_members.values():
            for member_info in members.values():
                var_name = member_info["global_name"]
                if var_name in emitted_names:
                    continue
                emitted_names.add(var_name)
                vtype = member_info["type"]
                mapped_type = self.map_resource_type_with_format(vtype)
                declaration_type = self.directx_resource_declaration_type(mapped_type)
                declaration = format_c_style_array_declaration(
                    declaration_type, var_name
                )
                register = ""
                space = None
                if (
                    mapped_type.startswith("Texture")
                    or self.is_multisample_storage_image_resource_type(mapped_type)
                    or self.is_hlsl_acceleration_structure_type(mapped_type)
                ):
                    self.texture_variables.add(var_name)
                    self.texture_variable_types[var_name] = mapped_type
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "t",
                        texture_registers,
                        space,
                        1,
                    )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "t",
                        binding,
                        1,
                        var_name,
                        space,
                    )
                    register = self.resource_register_suffix_for_space(
                        "t", binding, space
                    )
                    self.advance_resource_register(texture_registers, space, binding, 1)
                elif self.is_hlsl_rw_texture_type(
                    mapped_type
                ) or self.is_hlsl_feedback_texture_type(mapped_type):
                    self.texture_variable_types[var_name] = mapped_type
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "u",
                        uav_registers,
                        space,
                        1,
                    )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "u",
                        binding,
                        1,
                        var_name,
                        space,
                    )
                    register = self.resource_register_suffix_for_space(
                        "u", binding, space
                    )
                    self.advance_resource_register(uav_registers, space, binding, 1)
                elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
                    self.sampler_variables.add(var_name)
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "s",
                        sampler_registers,
                        space,
                        1,
                    )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "s",
                        binding,
                        1,
                        var_name,
                        space,
                    )
                    register = self.resource_register_suffix_for_space(
                        "s", binding, space
                    )
                    self.advance_resource_register(sampler_registers, space, binding, 1)
                else:
                    continue
                code += f"{declaration}{register};\n"
        return code

    def hlsl_struct_field_dependencies(self, record_name, field_types, all_names):
        dependencies = set()
        for field_type in field_types or []:
            dependencies.update(
                dependency
                for dependency in self.hlsl_struct_dependency_names(
                    field_type, all_names
                )
                if dependency != record_name
            )
        return dependencies

    def hlsl_struct_dependency_names(self, field_type, all_names):
        dependencies = set()
        pending = [self.type_name_string(field_type)]
        visited = set()

        while pending:
            type_text = str(pending.pop() or "").strip()
            if not type_text or type_text in visited:
                continue
            visited.add(type_text)

            mapped_type = str(self.map_type(type_text) or "").strip()
            for candidate in (type_text, mapped_type):
                base_type, _array_suffix = split_array_type_suffix(candidate)
                base_type = base_type.strip()
                if not base_type:
                    continue
                base_name, generic_args = generic_type_parts(base_type)
                for dependency_name in (
                    base_type,
                    base_name,
                    base_name.rsplit("::", 1)[-1],
                ):
                    if dependency_name in all_names:
                        dependencies.add(dependency_name)
                pending.extend(str(argument).strip() for argument in generic_args)

        return dependencies

    def hlsl_struct_dependency_name(self, field_type, all_names):
        dependencies = self.hlsl_struct_dependency_names(field_type, all_names)
        if not dependencies:
            return None
        return min(dependencies)

    def hlsl_order_struct_declaration_records(self, records):
        remaining = {record["name"]: record for record in records}
        ordered = []
        while remaining:
            remaining_names = set(remaining)
            ready = [
                record
                for record in remaining.values()
                if not (record.get("dependencies", set()) & remaining_names)
            ]
            if not ready:
                ready = [min(remaining.values(), key=lambda record: record["index"])]

            for record in sorted(ready, key=lambda record: record["index"]):
                ordered.append(record)
                remaining.pop(record["name"], None)
        return ordered

    def hlsl_texture_member_name(self, expr):
        while isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        if isinstance(expr, MemberAccessNode) or (
            hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__)
        ):
            member = getattr(expr, "member", None)
            return str(member) if member else None
        return None

    def hlsl_implicit_member_sampler_base_name(self, texture_arg):
        lowered_name = self.hlsl_lowered_struct_resource_member_global_name(texture_arg)
        if lowered_name:
            return lowered_name
        return self.hlsl_texture_member_name(texture_arg)

    def hlsl_implicit_member_sampler_name(self, texture_arg, sampler_info=None):
        member_name = self.hlsl_implicit_member_sampler_base_name(texture_arg)
        if not member_name:
            return None
        global_sampler_info = getattr(self, "implicit_member_texture_samplers", {}).get(
            member_name
        )
        sampler_info = global_sampler_info or sampler_info
        if (
            sampler_info
            and sampler_info.get("regular")
            and sampler_info.get("comparison")
        ):
            return f"{member_name}RegularSampler"
        return f"{member_name}Sampler"

    def hlsl_implicit_member_query_lod_sampler_name(self, texture_arg):
        member_name = self.hlsl_implicit_member_sampler_base_name(texture_arg)
        if not member_name:
            return None
        sampler_info = getattr(self, "implicit_member_texture_samplers", {}).get(
            member_name
        )
        if (
            sampler_info
            and sampler_info.get("query_lod")
            and sampler_info.get("comparison")
        ):
            return f"{member_name}QuerySampler"
        return f"{member_name}Sampler"

    def hlsl_texture_operation_uses_sampler(self, func_name):
        return (
            is_texture_sample_operation(func_name)
            or is_texture_sample_offset_operation(func_name)
            or is_projected_texture_operation(func_name)
            or is_texture_gather_operation(func_name)
            or is_texture_compare_operation(func_name)
            or is_texture_gather_compare_operation(func_name)
            or is_texture_query_lod_operation(func_name)
        )

    def hlsl_member_sampler_info_for_call(self, func_name, args):
        if not self.hlsl_texture_operation_uses_sampler(func_name) or not args:
            return None
        texture_arg = args[0]
        member_name = self.hlsl_texture_member_name(texture_arg)
        if not member_name:
            return None
        if self.texture_call_uses_explicit_sampler(args):
            return None
        texture_type = self.texture_resource_type(texture_arg)
        if self.is_multisample_texture_resource_type(texture_type):
            return None

        comparison = is_texture_compare_operation(
            func_name
        ) or is_texture_gather_compare_operation(func_name)
        regular = (
            is_texture_sample_operation(func_name)
            or is_texture_sample_offset_operation(func_name)
            or is_projected_texture_operation(func_name)
            or is_texture_gather_operation(func_name)
        )
        query_lod = is_texture_query_lod_operation(func_name)
        sampler_name = f"{member_name}Sampler"
        return {
            "sampler_name": sampler_name,
            "comparison": comparison,
            "regular": regular,
            "query_lod": query_lod,
            "synthetic": True,
        }

    def merge_hlsl_member_sampler_info(self, samplers, member_name, sampler_info):
        existing = samplers.setdefault(
            member_name,
            {
                "sampler_name": f"{member_name}Sampler",
                "comparison": False,
                "regular": False,
                "query_lod": False,
                "synthetic": True,
            },
        )
        existing["comparison"] = existing["comparison"] or sampler_info["comparison"]
        existing["regular"] = existing["regular"] or sampler_info["regular"]
        existing["query_lod"] = existing["query_lod"] or sampler_info["query_lod"]

    def collect_implicit_member_texture_samplers_from_function(self, func):
        samplers = {}
        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {
            **previous_local_variable_types,
            **self.function_scope_variable_types(func),
        }
        try:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_name = self.function_call_name(node)
                args = getattr(node, "arguments", getattr(node, "args", []))
                sampler_info = self.hlsl_member_sampler_info_for_call(func_name, args)
                if sampler_info is None:
                    continue
                member_name = self.hlsl_implicit_member_sampler_base_name(args[0])
                self.merge_hlsl_member_sampler_info(samplers, member_name, sampler_info)
        finally:
            self.local_variable_types = previous_local_variable_types
        return samplers

    def collect_implicit_member_texture_samplers(self, ast):
        samplers = {}

        def merge_from_function(func):
            for (
                member_name,
                sampler_info,
            ) in self.collect_implicit_member_texture_samplers_from_function(
                func
            ).items():
                self.merge_hlsl_member_sampler_info(samplers, member_name, sampler_info)

        for func in getattr(ast, "functions", []) or []:
            merge_from_function(func)

        stages = getattr(ast, "stages", {}) or {}
        if isinstance(stages, dict):
            for stage in stages.values():
                for func in getattr(stage, "local_functions", []) or []:
                    merge_from_function(func)
                entry_point = getattr(stage, "entry_point", None)
                if entry_point is not None:
                    merge_from_function(entry_point)

        return samplers

    def generate_implicit_member_texture_samplers(
        self,
        implicit_member_samplers,
        used_resource_registers,
        sampler_registers,
    ):
        code = ""
        for member_name, sampler_info in sorted(
            (implicit_member_samplers or {}).items()
        ):
            sampler_names = []
            sampler_type = (
                "SamplerComparisonState"
                if sampler_info.get("comparison")
                else "SamplerState"
            )
            sampler_names.append((sampler_info["sampler_name"], sampler_type))
            if sampler_info.get("regular") and sampler_info.get("comparison"):
                sampler_names.append((f"{member_name}RegularSampler", "SamplerState"))
            if sampler_info.get("query_lod") and sampler_info.get("comparison"):
                sampler_names.append((f"{member_name}QuerySampler", "SamplerState"))

            for sampler_name, sampler_type in sampler_names:
                if sampler_name in self.sampler_variables:
                    continue
                self.sampler_variables.add(sampler_name)
                space = None
                binding = self.next_available_resource_register(
                    used_resource_registers,
                    "s",
                    sampler_registers,
                    space,
                    1,
                )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "s",
                    binding,
                    1,
                    sampler_name,
                    space,
                )
                sampler_register = self.resource_register_suffix_for_space(
                    "s", binding, space
                )
                code += f"{sampler_type} {sampler_name}{sampler_register};\n"
                self.advance_resource_register(sampler_registers, space, binding, 1)
        return code

    def hlsl_inverse_helper_name(self, mapped_type):
        return f"__crossgl_inverse_{mapped_type.replace('x', '_')}"

    def hlsl_inverse_function_call(self, func_name, args):
        if func_name != "inverse" or len(args) != 1:
            return None

        mapped_type = self.map_type(self.expression_result_type(args[0]) or "")
        if mapped_type not in {"float2x2", "float3x3", "float4x4"}:
            return None

        self.required_hlsl_inverse_helpers.add(mapped_type)
        helper_name = self.hlsl_inverse_helper_name(mapped_type)
        return f"{helper_name}({self.generate_expression(args[0])})"

    def generate_hlsl_inverse_helpers(self):
        helpers = []
        required_helpers = getattr(self, "required_hlsl_inverse_helpers", set())
        if "float2x2" in required_helpers:
            helpers.append("""
float2x2 __crossgl_inverse_float2_2(float2x2 m) {
    float det = (m[0][0] * m[1][1]) - (m[0][1] * m[1][0]);
    if (abs(det) <= 1.0e-8) {
        return float2x2(1.0, 0.0, 0.0, 1.0);
    }
    return float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]) / det;
}
""")
        if "float3x3" in required_helpers:
            helpers.append("""
float3x3 __crossgl_inverse_float3_3(float3x3 m) {
    float det = determinant(m);
    if (abs(det) <= 1.0e-8) {
        return float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }
    float3x3 r;
    r[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    r[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det;
    r[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    r[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / det;
    r[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    r[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det;
    r[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    r[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / det;
    r[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    return r;
}
""")
        if "float4x4" in required_helpers:
            helpers.append("""
float __crossgl_det3_float4_4(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22
) {
    return (a00 * ((a11 * a22) - (a12 * a21)))
         - (a01 * ((a10 * a22) - (a12 * a20)))
         + (a02 * ((a10 * a21) - (a11 * a20)));
}

float __crossgl_cofactor_float4_4(float4x4 m, int row, int column) {
    float values[9];
    int index = 0;
    for (int r = 0; r < 4; ++r) {
        if (r == row) {
            continue;
        }
        for (int c = 0; c < 4; ++c) {
            if (c == column) {
                continue;
            }
            values[index] = m[r][c];
            ++index;
        }
    }
    float minorDet = __crossgl_det3_float4_4(
        values[0], values[1], values[2],
        values[3], values[4], values[5],
        values[6], values[7], values[8]
    );
    return (((row + column) & 1) == 0) ? minorDet : -minorDet;
}

float4x4 __crossgl_inverse_float4_4(float4x4 m) {
    float det = determinant(m);
    if (abs(det) <= 1.0e-8) {
        return float4x4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }

    float4x4 result;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result[row][column] = __crossgl_cofactor_float4_4(m, column, row) / det;
        }
    }
    return result;
}
""")
        if not helpers:
            return ""
        return "".join(helpers) + "\n"

    def generate_preprocessor_directive(self, directive):
        if isinstance(directive, PreprocessorNode):
            directive_name = (directive.directive or "").strip()
            content = str(directive.content or "").strip()
            if directive_name.lower() in {"version", "extension", "precision"}:
                return None
            if self.is_metal_only_preprocessor_directive(directive_name, content):
                return None
            return f"#{directive_name} {content}".strip()

        line = str(directive).strip()
        lowered = line.lower()
        if (
            lowered.startswith("#version")
            or lowered.startswith("#extension")
            or lowered.startswith("precision ")
        ):
            return None
        if self.is_metal_only_preprocessor_line(line):
            return None
        return line

    def is_metal_only_preprocessor_directive(self, directive_name, content):
        directive_key = directive_name.lower()
        if directive_key == "pragma":
            return str(content).strip().lower().startswith("metal")
        if directive_key != "include":
            return False
        raw_content = str(content).strip()
        if not raw_content:
            return False
        is_system_include = raw_content.startswith("<") and raw_content.endswith(">")
        normalized = raw_content.strip("<>\"'").replace("\\", "/").lower()
        return is_system_include and (
            normalized in self.METAL_ONLY_SYSTEM_INCLUDES
            or is_apple_sdk_include(normalized)
        )

    def is_metal_only_preprocessor_line(self, line):
        stripped = str(line).strip()
        if not stripped.startswith("#"):
            return False
        parts = stripped[1:].strip().split(maxsplit=1)
        if not parts:
            return False
        directive_name = parts[0]
        content = parts[1] if len(parts) > 1 else ""
        return self.is_metal_only_preprocessor_directive(directive_name, content)

    def hlsl_should_omit_metal_global_declaration(self, var_name, vtype):
        if not var_name:
            return False
        type_name = self.type_name_string(vtype)
        type_key = str(type_name or "").rsplit("::", 1)[-1]
        if var_name == "float16_t" and type_key in {"f16", "float16", "half"}:
            return True
        if var_name == "bfloat16_t" and type_key in {
            "bfloat",
            "bfloat16",
            "f16",
            "half",
        }:
            return True
        mapped_type = self.map_type(type_name)
        if var_name in self.METAL_TYPE_ALIAS_GLOBALS and (
            mapped_type == self.map_type(var_name)
            or (var_name == "bfloat16_t" and mapped_type == self.map_type("half"))
        ):
            return True
        return type_key == "thread_scope" and str(var_name).startswith("thread_scope")

    @staticmethod
    def directx_is_specialization_constant_attribute(attr):
        name = str(getattr(attr, "name", "")).lower().replace("-", "_")
        if name.startswith("metal_") or name.startswith("msl_"):
            name = name.split("_", 1)[1]
        return name in {"function_constant", "constant_id"}

    def directx_specialization_constant_attributes(self, node):
        return [
            attr
            for attr in getattr(node, "attributes", []) or []
            if self.directx_is_specialization_constant_attribute(attr)
        ]

    def directx_specialization_constant_source_location(self, node, attributes=()):
        for candidate in (node, *attributes):
            source_location = getattr(candidate, "source_location", None)
            if source_location is not None:
                return source_location
        return None

    def directx_specialization_constant_type(self, node):
        return getattr(
            node,
            "const_type",
            getattr(node, "var_type", getattr(node, "vtype", "float")),
        )

    def directx_specialization_constant_value(self, node):
        if hasattr(node, "const_type"):
            return getattr(node, "value", None)
        return getattr(node, "initial_value", getattr(node, "value", None))

    def directx_specialization_constant_id(self, node, attributes=None):
        attributes = list(
            self.directx_specialization_constant_attributes(node)
            if attributes is None
            else attributes
        )
        if not attributes:
            return None

        name = getattr(node, "name", getattr(node, "variable_name", "<unnamed>"))
        type_name = self.type_name_string(
            self.directx_specialization_constant_type(node)
        )
        source_location = self.directx_specialization_constant_source_location(
            node, attributes
        )
        constant_ids = []
        for attr in attributes:
            arguments = list(getattr(attr, "arguments", []) or [])
            raw_id = (
                self.attribute_value_to_string(arguments[0])
                if len(arguments) == 1
                else None
            )
            constant_id = (
                self.binding_index_value(arguments[0]) if len(arguments) == 1 else None
            )
            if constant_id is None:
                rendered_id = raw_id if raw_id is not None else "<missing>"
                raise DirectXSpecializationConstantError(
                    f"DirectX specialization constant '{name}' requires one "
                    f"non-negative integer id; received {rendered_id}",
                    constant_name=name,
                    constant_id=raw_id,
                    constant_type=type_name,
                    reason="invalid-id",
                    source_location=source_location,
                )
            constant_ids.append(constant_id)

        first_id = constant_ids[0]
        for conflicting_id in constant_ids[1:]:
            if conflicting_id == first_id:
                continue
            raise DirectXSpecializationConstantError(
                f"DirectX specialization constant '{name}' carries conflicting "
                f"ids {first_id} and {conflicting_id}",
                constant_name=name,
                constant_id=first_id,
                constant_type=type_name,
                reason="conflicting-id",
                source_location=source_location,
                conflicting_id=conflicting_id,
            )
        return first_id

    def collect_directx_specialization_constants(self, ast):
        global_nodes = list(getattr(ast, "global_variables", []) or [])
        global_node_ids = {id(node) for node in global_nodes}
        declarations = list(getattr(ast, "constants", []) or []) + global_nodes
        records = []
        records_by_id = {}
        records_by_name = {}

        for node in declarations:
            attributes = self.directx_specialization_constant_attributes(node)
            if not attributes:
                continue

            name = getattr(node, "name", getattr(node, "variable_name", "<unnamed>"))
            type_node = self.directx_specialization_constant_type(node)
            source_type = self.type_name_string(type_node)
            mapped_type = self.map_type(type_node)
            source_location = self.directx_specialization_constant_source_location(
                node, attributes
            )
            constant_id = self.directx_specialization_constant_id(node, attributes)

            if mapped_type not in self.HLSL_SPECIALIZATION_CONSTANT_TYPES:
                supported = ", ".join(sorted(self.HLSL_SPECIALIZATION_CONSTANT_TYPES))
                raise DirectXSpecializationConstantError(
                    f"DirectX specialization constant '{name}' (id {constant_id}) "
                    f"cannot use type '{source_type}'; concrete DirectX variants "
                    f"currently support scalar {supported}",
                    constant_name=name,
                    constant_id=constant_id,
                    constant_type=source_type,
                    reason="incompatible-type",
                    source_location=source_location,
                )

            # A constant without a default reads a host macro (hlsl_specialization_macros).
            value = self.directx_specialization_constant_value(node)

            previous = records_by_id.get(constant_id)
            if previous is not None:
                if previous["mapped_type"] != mapped_type:
                    reason = "conflicting-id-type"
                    message = (
                        f"DirectX specialization constant id {constant_id} has "
                        f"incompatible declarations: '{previous['name']}' uses "
                        f"'{previous['source_type']}', while '{name}' uses "
                        f"'{source_type}'"
                    )
                else:
                    reason = "duplicate-id"
                    message = (
                        f"DirectX specialization constant id {constant_id} is "
                        f"assigned to both '{previous['name']}' and '{name}'"
                    )
                raise DirectXSpecializationConstantError(
                    message,
                    constant_name=name,
                    constant_id=constant_id,
                    constant_type=source_type,
                    reason=reason,
                    source_location=source_location,
                    conflicting_name=previous["name"],
                    conflicting_type=previous["source_type"],
                )

            previous = records_by_name.get(name)
            if previous is not None:
                raise DirectXSpecializationConstantError(
                    f"DirectX specialization constant name '{name}' is assigned "
                    f"to both id {previous['id']} and id {constant_id}",
                    constant_name=name,
                    constant_id=constant_id,
                    constant_type=source_type,
                    reason="duplicate-name",
                    source_location=source_location,
                    conflicting_name=previous["name"],
                    conflicting_id=previous["id"],
                    conflicting_type=previous["source_type"],
                )

            record = {
                "node": node,
                "name": name,
                "id": constant_id,
                "type": type_node,
                "source_type": source_type,
                "mapped_type": mapped_type,
                "value": value,
                "is_global": id(node) in global_node_ids,
            }
            records.append(record)
            records_by_id[constant_id] = record
            records_by_name[name] = record

        return records

    def directx_specialization_constant_record(self, node):
        return next(
            (
                record
                for record in self.directx_specialization_constant_records
                if record["node"] is node
            ),
            None,
        )

    def generate_directx_specialization_constant_declaration(self, record):
        if record["value"] is None:
            return render_hlsl_required_specialization_constant(
                record["name"], record["id"], record["mapped_type"]
            )
        declaration = format_c_style_array_declaration(
            record["mapped_type"], record["name"]
        )
        value_code = self.generate_constant_expression(record["value"], record["type"])
        return (
            "/* CrossGL DirectX specialization constant "
            f"id {record['id']} fixed to its default. */\n"
            f"static const {declaration} = {value_code};\n"
        )

    def generate_directx_specialization_constant_globals(self):
        code = "".join(
            self.generate_directx_specialization_constant_declaration(record)
            for record in self.directx_specialization_constant_records
            if record["is_global"]
        )
        return f"{code}\n" if code else ""

    def generate_constants(self, ast, constants=None):
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            name = getattr(node, "name", None)
            if not name:
                continue

            specialization_record = self.directx_specialization_constant_record(node)
            if specialization_record is not None:
                code += self.generate_directx_specialization_constant_declaration(
                    specialization_record
                )
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = hlsl_brace_struct_initializer(self, value, const_type)
            if value_code is None:
                value_code = self.generate_constant_expression(value, const_type)
            declaration = format_c_style_array_declaration(
                self.map_type(const_type), name
            )
            code += f"static const {declaration} = {value_code};\n"

        return f"{code}\n" if code else ""
