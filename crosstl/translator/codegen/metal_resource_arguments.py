"""Resource bindings and arguments in the Metal generator: resource binding reservation, texture samplers, struct resource members, texture and image argument checks, image atomic and function resource arguments, and coordinate, offset, gradient, LOD, compare and bias arguments."""

import ast as py_ast
import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    MemberAccessNode,
    ReferenceType,
)
from ..refusals import MetalRefusal
from ..semantics.sampler_states import is_sampler_constructor
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    expression_debug_name,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gradient_argument_indices,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
)
from .array_utils import parse_array_type
from .image_access_contracts import (
    floating_coordinate_dimension_from_type_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
)
from .image_access_contracts import (
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
)
from .image_access_contracts import (
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_image_atomic_operation,
    is_image_resource_operation,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    metal_storage_image_component_type,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    requires_integer_coordinate,
    resolve_image_atomic_component_kind,
)
from .image_access_contracts import (
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
)
from .image_access_contracts import (
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_type_error,
)
from .stage_utils import (
    normalize_stage_name,
    stage_matches,
)


class MetalResourceArgumentMixin:
    """Resource argument members of ``MetalCodeGen``."""

    def global_resource_binding_metadata(self, node):
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_glsl_buffer_block_variable(node, vtype):
            return None
        elif self.metal_buffer_resource_address_space(node, vtype) is not None:
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
            resource_count = self.metal_buffer_resource_binding_count(node)
        elif self.is_structured_buffer_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_acceleration_structure_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_visible_function_table_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_intersection_function_table_type(vtype):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_texture_or_image_resource_type(vtype):
            namespace = "texture"
            attribute_names = {"binding", "texture"}
            prefixes = ("t", "u")
        elif self.is_sampler_type(vtype):
            namespace = "sampler"
            attribute_names = {"binding", "sampler"}
            prefixes = ("s",)
        else:
            return None

        binding = self.explicit_resource_binding_index(node, attribute_names, prefixes)
        if binding is None:
            return None
        return namespace, binding, resource_count, var_name

    def reserve_explicit_global_resource_bindings(
        self, global_vars, used_bindings, source_bindings
    ):
        metadata_items = []
        for node in global_vars:
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            metadata_items.append((node, metadata))

        for descriptor_set in (None, 0):
            for node, metadata in metadata_items:
                if self.explicit_resource_set_index(node) != descriptor_set:
                    continue
                namespace, binding, resource_count, var_name = metadata
                if descriptor_set is None and self.defer_hlsl_register_binding(
                    node,
                    namespace,
                    resource_count,
                    var_name,
                    self.metal_namespace_register_classes(namespace),
                ):
                    continue
                self.pre_reserve_explicit_resource_binding(
                    used_bindings,
                    source_bindings,
                    namespace,
                    binding,
                    resource_count,
                    var_name,
                    node,
                )
            if descriptor_set is None:
                self.reserve_deferred_hlsl_register_bindings(
                    used_bindings, source_bindings
                )

    def next_available_resource_binding(
        self, used_bindings, namespace, binding_index, count
    ):
        count = max(count or 1, 1)
        ranges = used_bindings.get(namespace, [])
        while True:
            end = binding_index + count - 1
            conflict_end = None
            for used_start, used_end, _ in ranges:
                if binding_index <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding_index
            binding_index = conflict_end + 1

    def lowest_free_resource_binding(self, used_bindings, namespace, count):
        """The lowest index at which ``count`` consecutive slots of ``namespace`` are free.

        A resource the source gives no binding takes the lowest free slots, in declaration order, so an explicit
        binding elsewhere in the program does not move the unbound ones past it: a kernel whose constants block is
        bound at buffer 3 still takes its unbound storage buffers at buffers 0 and 1.
        """
        return self.next_available_resource_binding(used_bindings, namespace, 0, count)

    def collect_stage_local_resource_stage_scopes(
        self, ast, target_stage, stage_local_resource_variables
    ):
        stage_scopes = {}
        stage_local_ids = {id(node) for node in stage_local_resource_variables}
        if not stage_local_ids:
            return stage_scopes

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            for variable in getattr(stage, "local_variables", []) or []:
                if id(variable) in stage_local_ids:
                    stage_scopes[id(variable)] = stage_name
        return stage_scopes

    def metal_source_binding_stage_scope(self, node):
        return self.metal_source_binding_stage_by_id.get(id(node))

    def pre_reserve_cbuffer_bindings(self, used_bindings, source_bindings):
        for cbuffer in self.cbuffer_variables:
            if self.explicit_resource_set_index(cbuffer) is not None:
                continue
            binding = self.explicit_resource_binding_index(
                cbuffer, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                continue
            if self.defer_hlsl_register_binding(
                cbuffer,
                "buffer",
                1,
                getattr(cbuffer, "name", "<anonymous>"),
                ("b",),
            ):
                continue
            self.pre_reserve_explicit_resource_binding(
                used_bindings,
                source_bindings,
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
                cbuffer,
            )

    def reserve_cbuffer_bindings(self, used_bindings, source_bindings):
        for cbuffer in self.cbuffer_variables:
            binding = self.explicit_resource_binding_index(
                cbuffer, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(used_bindings, "buffer", 1)
            binding = self.reserve_or_remap_resource_binding(
                used_bindings,
                source_bindings,
                "buffer",
                binding,
                1,
                getattr(cbuffer, "name", "<anonymous>"),
                cbuffer,
            )
            self.cbuffer_binding_indices[id(cbuffer)] = binding

    def pre_reserve_explicit_resource_binding(
        self, used_bindings, source_bindings, namespace, binding, count, name, node
    ):
        if binding is None:
            return None
        descriptor_set = self.explicit_resource_set_index(node)
        if descriptor_set not in (None, 0):
            return None
        if descriptor_set == 0:
            self.reserve_source_resource_binding_range(
                source_bindings,
                namespace,
                descriptor_set,
                binding,
                1,
                name,
                stage_scope=self.metal_source_binding_stage_scope(node),
            )
            target_binding = self.next_available_resource_binding(
                used_bindings,
                namespace,
                binding,
                count,
            )
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                target_binding,
                count,
                name,
            )
            self.metal_resource_binding_indices_by_id[id(node)] = target_binding
            return target_binding
        self.reserve_resource_binding_range(
            used_bindings,
            "Metal",
            namespace,
            binding,
            count,
            name,
        )
        self.metal_resource_binding_indices_by_id[id(node)] = binding
        return binding

    def reserve_or_remap_resource_binding(
        self,
        used_bindings,
        source_bindings,
        namespace,
        binding,
        count,
        name,
        node,
    ):
        cached = self.metal_resource_binding_indices_by_id.get(id(node))
        if cached is not None:
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                cached,
                count,
                name,
            )
            return cached

        requested_binding = binding
        descriptor_set = self.explicit_resource_set_index(node)
        if binding is None:
            binding = self.lowest_free_resource_binding(used_bindings, namespace, count)

        if descriptor_set is not None:
            if requested_binding is not None:
                self.reserve_source_resource_binding_range(
                    source_bindings,
                    namespace,
                    descriptor_set,
                    requested_binding,
                    1,
                    name,
                    stage_scope=self.metal_source_binding_stage_scope(node),
                )
            target_cursor = 0 if descriptor_set != 0 else binding
            target_binding = self.next_available_resource_binding(
                used_bindings,
                namespace,
                target_cursor,
                count,
            )
        else:
            target_binding = binding

        self.reserve_resource_binding_range(
            used_bindings,
            "Metal",
            namespace,
            target_binding,
            count,
            name,
        )
        self.metal_resource_binding_indices_by_id[id(node)] = target_binding
        return target_binding

    def reserve_source_resource_binding_range(
        self,
        source_bindings,
        namespace,
        descriptor_set,
        start,
        count,
        name,
        stage_scope=None,
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        key = (namespace, descriptor_set, stage_scope)
        ranges = source_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise MetalRefusal(
                    f"Conflicting Metal source resource binding for '{name}': "
                    f"{self.source_resource_binding_range_label(namespace, descriptor_set, start, end, stage_scope)} "
                    f"overlaps '{used_name}' "
                    f"{self.source_resource_binding_range_label(namespace, descriptor_set, used_start, used_end, stage_scope)}"
                )
        ranges.append((start, end, name))

    def source_resource_binding_range_label(
        self, namespace, descriptor_set, start, end, stage_scope=None
    ):
        stage_label = f" {stage_scope}" if stage_scope else ""
        if start == end:
            return f"set {descriptor_set}{stage_label} {namespace}({start})"
        return f"set {descriptor_set}{stage_label} {namespace}({start}-{end})"

    def reserve_resource_binding_range(
        self, used_bindings, target, namespace, start, count, name
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        ranges = used_bindings.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise MetalRefusal(
                    f"Conflicting {target} resource binding for '{name}': "
                    f"{self.resource_binding_range_label(namespace, start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.resource_binding_range_label(namespace, used_start, used_end)}"
                )
        ranges.append((start, end, name))

    def resource_binding_range_label(self, namespace, start, end):
        if start == end:
            return f"{namespace}({start})"
        return f"{namespace}({start}-{end})"

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.expression_name(array_expr)
        return None

    def texture_sampler_expression(self, texture_name, texture_arg=None):
        sampler_name = f"{texture_name}Sampler"
        if sampler_name in self.sampler_variable_names():
            index = self.array_access_index_expression(texture_arg)
            if index is not None and self.sampler_array_size(sampler_name) is not None:
                return f"{sampler_name}[{index}]"
            return sampler_name

        source_arg = self.texture_alias_sampler_source(texture_name)
        if source_arg is not None:
            member_sampler = self.struct_member_paired_sampler_expression(
                source_arg, texture_arg
            )
            if member_sampler is not None:
                return member_sampler
            source_name = self.expression_name(source_arg)
            source_sampler_name = f"{source_name}Sampler" if source_name else None
            if source_sampler_name in self.sampler_variable_names():
                index = self.array_access_index_expression(
                    source_arg
                ) or self.array_access_index_expression(texture_arg)
                if (
                    index is not None
                    and self.sampler_array_size(source_sampler_name) is not None
                ):
                    return f"{source_sampler_name}[{index}]"
                return source_sampler_name
        return self.default_sampler_expression()

    def texture_alias_sampler_source(self, texture_name):
        source_arg = self.current_texture_alias_sources.get(texture_name)
        seen = {texture_name}
        while source_arg is not None:
            source_name = self.expression_name(source_arg)
            if not source_name or source_name in seen:
                return source_arg
            next_source_arg = self.current_texture_alias_sources.get(source_name)
            if next_source_arg is None:
                return source_arg
            seen.add(source_name)
            source_arg = next_source_arg
        return None

    def struct_member_paired_sampler_expression(self, source_arg, texture_arg=None):
        member_expr = self.member_access_source_expression(source_arg)
        if member_expr is None:
            return None

        sampler_member = f"{member_expr.member}Sampler"
        sampler_node = self.struct_member_named_node(member_expr.object, sampler_member)
        if sampler_node is None or not self.struct_member_is_sampler_resource(
            sampler_node
        ):
            return None

        object_expr = self.generate_expression(member_expr.object)
        sampler_expr = f"{object_expr}.{sampler_member}"
        index = self.array_access_index_expression(
            source_arg
        ) or self.array_access_index_expression(texture_arg)
        if (
            index is not None
            and self.struct_member_array_size(sampler_node) is not None
        ):
            return f"{sampler_expr}[{index}]"
        return sampler_expr

    def array_access_index_expression(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        if index_expr is None:
            return None
        return self.generate_expression(index_expr)

    def sampler_array_size(self, sampler_name):
        if sampler_name in self.current_sampler_parameter_array_sizes:
            return self.current_sampler_parameter_array_sizes[sampler_name]
        for sampler_variable, _, array_size in self.sampler_variables:
            if getattr(sampler_variable, "name", None) == sampler_name:
                return array_size
        return None

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.texture_call_uses_explicit_sampler(args)

    # The answers of texture_call_uses_explicit_sampler during one texture call attempt (texture_call_names).
    explicit_sampler_answers = None

    def texture_call_uses_explicit_sampler(self, args):
        answers = self.explicit_sampler_answers
        if answers is not None:
            known = answers.get(id(args))
            if known is not None and known[0] is args:
                return known[1]
        answer = self.texture_call_names_explicit_sampler(args)
        if answers is not None:
            answers[id(args)] = (args, answer)
        return answer

    def texture_call_names_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_names = self.sampler_variable_names()
        sampler_name = self.expression_name(args[1])
        if not sampler_name and (sampler_names or self.program_scope_sampler_constants):
            # Rendered only when some sampler name could match the text.
            sampler_name = self.generate_expression(args[1])
        if sampler_name in sampler_names:
            return True
        if sampler_name in self.program_scope_sampler_constants or (
            is_sampler_constructor(args[1])
        ):
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        sampler_arg = (
            self.generate_expression(args[1])
            if explicit_sampler
            else self.texture_sampler_expression(texture_base_name, args[0])
        )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_arg, coord, extra_args

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return self.struct_member_texture_resource_type(texture_arg)
        texture_type = self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )
        return texture_type or self.struct_member_texture_resource_type(texture_arg)

    def texture_sample_result_type(self, texture_arg):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        if not texture_type:
            return None
        if texture_type.startswith("depth"):
            return "float"
        component_type = "float"
        if "<" in texture_type and ">" in texture_type:
            component_type = texture_type.split("<", 1)[1].split(">", 1)[0]
            component_type = component_type.split(",", 1)[0].strip() or "float"
        return f"{component_type}4"

    def struct_member_texture_resource_type(self, texture_arg):
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if raw_type is None and isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", None)
        if self.is_array_type_node(raw_type):
            raw_type = raw_type.element_type
        elif isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", getattr(member, "vtype", None))
        if raw_type is None:
            return None
        raw_type_name = self.type_name_string(raw_type)
        if not self.is_texture_or_image_resource_type(raw_type_name):
            return None
        return self.map_resource_type_with_format(raw_type, member)

    def struct_member_resource_array_size(self, texture_arg):
        member = self.struct_resource_member_node(texture_arg)
        if member is None:
            return None
        return self.struct_member_array_size(member)

    def struct_resource_member_node(self, texture_arg):
        expr = self.member_access_source_expression(texture_arg)
        if not isinstance(expr, MemberAccessNode):
            return None
        return self.struct_member_named_node(expr.object, str(expr.member))

    def member_access_source_expression(self, expr):
        while isinstance(expr, ArrayAccessNode):
            expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        return expr if isinstance(expr, MemberAccessNode) else None

    def struct_member_named_node(self, object_expr, member_name):
        object_type = self.expression_result_type(object_expr)
        if object_type is None:
            return None
        object_type = self.member_lookup_type_name(object_type)
        struct_node = self.structs_by_name.get(self.type_name_string(object_type))
        if struct_node is None:
            return None

        for member in getattr(struct_node, "members", []) or []:
            if getattr(member, "name", None) == member_name:
                return member
        return None

    def struct_member_array_size(self, member):
        if isinstance(member, ArrayNode):
            return (
                self.safe_expression_to_string(member.size)
                if member.size is not None
                else ""
            )

        raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if self.is_array_type_node(raw_type):
            return (
                self.safe_expression_to_string(raw_type.size)
                if raw_type.size is not None
                else ""
            )

        raw_type_name = self.type_name_string(raw_type)
        if raw_type_name and "[" in raw_type_name and "]" in raw_type_name:
            _, array_size = parse_array_type(raw_type_name)
            return array_size or ""
        return None

    def struct_member_is_sampler_resource(self, member):
        if isinstance(member, ArrayNode):
            raw_type = getattr(member, "element_type", getattr(member, "vtype", None))
        else:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if self.is_array_type_node(raw_type):
                raw_type = raw_type.element_type
        return self.is_sampler_type(self.type_name_string(raw_type))

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_resource_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

    def texture_argument_raw_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name:
            raw_type = self.current_texture_parameter_raw_types.get(
                texture_name, self.texture_variable_raw_types.get(texture_name)
            )
            if raw_type is not None:
                return raw_type

        member = self.struct_resource_member_node(texture_arg)
        if member is not None:
            raw_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if raw_type is None and isinstance(member, ArrayNode):
                raw_type = getattr(member, "element_type", None)
            if self.is_array_type_node(raw_type):
                raw_type = raw_type.element_type
            return self.resource_base_type(self.type_name_string(raw_type))

        return self.expression_result_type(texture_arg)

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_resource(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise MetalRefusal(
            f"Metal image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.storage_image_access_mode(
            self.texture_argument_resource_type(args[0])
        )
        if access is None:
            return
        if self.is_multisample_storage_image_resource(
            self.texture_argument_resource_type(args[0])
        ):
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise MetalRefusal(
                f"Metal image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got access::write"
            )
        if func_name == "imageStore" and access == "read":
            raise MetalRefusal(
                f"Metal image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got access::read"
            )
        if is_image_atomic_operation(func_name) and access != "read_write":
            raise MetalRefusal(
                f"Metal image operation '{func_name}' requires read_write "
                f"storage image access for {texture_name}: got access::{access}"
            )

    def image_atomic_value_arguments(self, func_name, args, image_type):
        has_sample = self.is_multisample_storage_image_resource(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, image_type, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args, image_type),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        format_label = image_format or self.resource_base_type(image_type)
        raise MetalRefusal(
            image_atomic_value_kind_error(
                "Metal",
                func_name,
                format_label,
                component_kind,
                expression_debug_name(value_arg),
                value_kind,
            )
        )

    def scalar_expected_kind(self):
        return numeric_scalar_type_kind(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_result_type(
        self, func_name, image_type, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise MetalRefusal(
            image_atomic_result_kind_error(
                "Metal", func_name, format_label, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        component_kind = resolve_image_atomic_component_kind(
            func_name,
            image_format,
            metal_storage_image_component_type(image_type),
            "Metal",
            self.resource_base_type(image_type),
        )
        self.validate_image_atomic_value_argument_types(
            func_name,
            args,
            image_type,
            component_kind,
            image_format,
        )
        self.validate_image_atomic_result_type(
            func_name,
            image_type,
            component_kind,
            image_format,
        )

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        if not self.function_call_matches_known_signature(func_name, args):
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.storage_image_access_mode(
                self.texture_argument_resource_type(args[index])
            )
            if self.is_multisample_storage_image_resource(
                self.texture_argument_resource_type(args[index])
            ):
                continue
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(
                required_access, read_write_label="read_write"
            )
            raise MetalRefusal(
                f"Metal function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got access::{actual_access}"
            )

    def metal_standard_array_element_type(self, type_name):
        qualified = re.fullmatch(r"((?:(?:const|volatile)\s+)+)(.+)", type_name)
        if qualified is not None:
            return qualified[1] + self.metal_standard_array_element_type(qualified[2])
        if re.fullmatch(
            r"packed_(?:char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float)[234]",
            type_name,
        ):
            return type_name
        native = self.metal_native_narrow_bitcast_storage_type(type_name)
        if native is not None:
            return native
        return self.map_type(type_name)

    def split_metal_array_resource_type(self, type_name):
        type_name = str(type_name or "").strip()
        if not type_name.startswith("array<") or not type_name.endswith(">"):
            return None

        body = type_name[len("array<") : -1]
        depth = 0
        for index, char in enumerate(body):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                return body[:index].strip(), body[index + 1 :].strip()
        return None

    def metal_array_resource_type(self, element_type, array_size):
        return f"array<{element_type}, {array_size or '1'}>"

    def function_parameter_resource_type(self, func_name, parameter):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        param_name = getattr(parameter, "name", None)
        function_hints = self.function_resource_array_size_hints.get(func_name, {})

        if self.is_array_type_node(raw_type):
            base_type = self.type_name_string(raw_type.element_type)
            if not self.is_resource_parameter_type(base_type):
                return None
            array_size = (
                self.safe_expression_to_string(raw_type.size)
                if raw_type.size is not None
                else function_hints.get(param_name, "")
            )
            return self.metal_array_resource_type(
                self.map_resource_type_with_format(base_type, parameter),
                array_size,
            )

        type_string = self.type_name_string(raw_type)
        if "[" in type_string and "]" in type_string:
            base_type, array_size = parse_array_type(type_string)
            if not self.is_resource_parameter_type(base_type):
                return None
            return self.metal_array_resource_type(
                self.map_resource_type_with_format(base_type, parameter),
                (
                    function_hints.get(param_name, "")
                    if array_size is None
                    else array_size
                ),
            )

        if not self.is_resource_parameter_type(type_string):
            return None
        return self.map_resource_type_with_format(type_string, parameter)

    def function_argument_resource_type(self, arg):
        resource_type = self.texture_argument_resource_type(arg)
        if resource_type is not None:
            array_size = self.texture_argument_resource_array_size(arg)
            if array_size is not None:
                return self.metal_array_resource_type(resource_type, array_size)
            return resource_type

        resource_type = self.sampler_argument_resource_type(arg)
        if resource_type is None:
            return None
        array_size = self.sampler_argument_resource_array_size(arg)
        if array_size is not None:
            return self.metal_array_resource_type(resource_type, array_size)
        return resource_type

    def normalize_function_resource_compatibility_type(self, resource_type):
        array_resource = self.split_metal_array_resource_type(resource_type)
        if array_resource is not None:
            element_type, array_size = array_resource
            return self.metal_array_resource_type(
                self.normalize_function_resource_compatibility_type(element_type),
                self.normalize_function_resource_array_size(array_size),
            )
        if self.is_storage_image_resource(resource_type):
            return self.storage_image_access_agnostic_type(resource_type)
        return self.resource_base_type(resource_type)

    def normalize_function_resource_array_size(self, array_size):
        literal_size = self.literal_int_resource_array_size(array_size)
        if literal_size is not None:
            return str(literal_size)
        return str(array_size or "")

    def literal_int_resource_array_size(self, array_size):
        value = self.literal_int_value(array_size, self.literal_int_constants)
        if value is not None:
            return value

        if not isinstance(array_size, str):
            return None
        try:
            parsed = py_ast.parse(array_size, mode="eval")
        except SyntaxError:
            return None
        return self.literal_int_python_expression_value(parsed.body)

    def literal_int_python_expression_value(self, expr):
        if isinstance(expr, py_ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, py_ast.Name):
            return self.literal_int_constants.get(expr.id)
        if isinstance(expr, py_ast.UnaryOp):
            operand = self.literal_int_python_expression_value(expr.operand)
            if operand is None:
                return None
            if isinstance(expr.op, py_ast.UAdd):
                return operand
            if isinstance(expr.op, py_ast.USub):
                return -operand
            return None
        if isinstance(expr, py_ast.BinOp):
            left = self.literal_int_python_expression_value(expr.left)
            right = self.literal_int_python_expression_value(expr.right)
            if left is None or right is None:
                return None
            if isinstance(expr.op, py_ast.Add):
                return left + right
            if isinstance(expr.op, py_ast.Sub):
                return left - right
            if isinstance(expr.op, py_ast.Mult):
                return left * right
        return None

    def resource_type_requires_function_resource_compatibility(self, resource_type):
        array_resource = self.split_metal_array_resource_type(resource_type)
        if array_resource is not None:
            element_type, _ = array_resource
            return self.resource_type_requires_function_resource_compatibility(
                element_type
            )

        base_type = self.resource_base_type(resource_type)
        return self.is_sampler_type(base_type) or str(base_type).startswith("texture")

    def validate_function_resource_argument_types(self, func_name, args):
        parameter_nodes = self.function_parameter_nodes.get(func_name)
        if not parameter_nodes:
            return
        if not self.function_call_matches_known_signature(func_name, args):
            return

        for index, parameter in enumerate(parameter_nodes):
            if index >= len(args):
                return
            expected_type = self.function_parameter_resource_type(func_name, parameter)
            if expected_type is None:
                continue

            actual_type = self.function_argument_resource_type(args[index])
            if not self.resource_type_requires_function_resource_compatibility(
                expected_type
            ):
                continue

            if self.normalize_function_resource_compatibility_type(
                expected_type
            ) == self.normalize_function_resource_compatibility_type(actual_type):
                continue

            actual_name = expression_debug_name(args[index])
            actual_type_label = actual_type or self.type_name_string(
                self.expression_result_type(args[index])
            )
            raise MetalRefusal(
                f"Metal function call '{func_name}' requires resource parameter "
                f"{getattr(parameter, 'name', None)} of type {expected_type}: "
                f"argument {actual_name} has {actual_type_label or 'non-resource'}"
            )

    def validate_integer_coordinate_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        coord_type = self.metal_referenced_value_type(
            self.expression_result_type(args[1])
        )
        if coord_type is None or self.is_integer_coordinate_type(coord_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "resource",
                func_name,
                "an integer",
                "coordinate",
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_coordinate_dimension_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.metal_referenced_value_type(
            self.expression_result_type(args[1])
        )
        coord_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(coord_type), self.map_type
        )
        if coord_dimension is None or coord_dimension == expected_dimension:
            return
        raise MetalRefusal(
            operation_dimension_argument_error(
                "Metal",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "coordinate",
                self.resource_base_type(texture_type),
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_offset_dimension_argument(self, func_name, args):
        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not offset_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for offset_index in offset_indices:
            offset_type = self.metal_referenced_value_type(
                self.expression_result_type(args[offset_index])
            )
            if offset_type is None:
                continue
            if not self.is_integer_coordinate_type(offset_type):
                raise MetalRefusal(
                    operation_argument_type_error(
                        "Metal",
                        "resource",
                        func_name,
                        "an integer",
                        "offset",
                        expression_debug_name(args[offset_index]),
                        self.type_name_string(offset_type),
                    )
                )
            offset_dimension = integer_coordinate_dimension_from_type_name(
                self.type_name_string(offset_type), self.map_type
            )
            if offset_dimension is None or offset_dimension == expected_dimension:
                continue
            raise MetalRefusal(
                operation_dimension_argument_error(
                    "Metal",
                    "resource",
                    func_name,
                    expected_dimension,
                    "integer",
                    "offset",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[offset_index]),
                    self.type_name_string(offset_type),
                )
            )

    def gradient_argument_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def query_lod_coordinate_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def validate_query_lod_coordinate_argument(self, func_name, args):
        coord_index = texture_query_lod_coordinate_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if coord_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_query_lod_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.metal_referenced_value_type(
            self.expression_result_type(args[coord_index])
        )
        if coord_type is None:
            return
        coord_dimension = self.query_lod_coordinate_dimension(coord_type)
        if coord_dimension is None:
            raise MetalRefusal(
                texture_query_lod_coordinate_type_error(
                    "Metal",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise MetalRefusal(
            texture_query_lod_coordinate_dimension_error(
                "Metal",
                func_name,
                expected_dimension,
                self.resource_base_type(texture_type),
                expression_debug_name(args[coord_index]),
                self.type_name_string(coord_type),
            )
        )

    def validate_gradient_dimension_arguments(self, func_name, args):
        gradient_indices = texture_gradient_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not gradient_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_gradient_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for gradient_index in gradient_indices:
            gradient_type = self.metal_referenced_value_type(
                self.expression_result_type(args[gradient_index])
            )
            if gradient_type is None:
                continue
            gradient_dimension = self.gradient_argument_dimension(gradient_type)
            if gradient_dimension is None:
                raise MetalRefusal(
                    operation_argument_type_error(
                        "Metal",
                        "resource",
                        func_name,
                        "a floating",
                        "gradient",
                        expression_debug_name(args[gradient_index]),
                        self.type_name_string(gradient_type),
                    )
                )
            if gradient_dimension == expected_dimension:
                continue
            raise MetalRefusal(
                operation_dimension_argument_error(
                    "Metal",
                    "resource",
                    func_name,
                    expected_dimension,
                    "floating",
                    "gradient",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[gradient_index]),
                    self.type_name_string(gradient_type),
                )
            )

    def is_scalar_floating_type(self, vtype):
        return is_floating_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_numeric_type(self, vtype):
        return is_numeric_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_integer_type(self, vtype):
        return is_integer_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def metal_scalar_geometric_call(self, func_name, args):
        """Lower scalar ``length``, ``dot``, ``distance`` and ``normalize``, which Metal declares only for vectors."""
        arity = {"length": 1, "normalize": 1, "dot": 2, "distance": 2}.get(func_name)
        if arity != len(args) or func_name in self.user_function_names:
            return None
        if not all(
            self.is_scalar_numeric_type(self.expression_result_type(arg))
            for arg in args
        ):
            return None
        operands = [self.generate_expression(arg) for arg in args]
        if func_name == "dot":
            return f"(({operands[0]}) * ({operands[1]}))"
        call_name = "sign" if func_name == "normalize" else "abs"
        if self.metal_function_name_is_shadowed(call_name):
            call_name = f"metal::{call_name}"
        if func_name == "distance":
            return f"{call_name}(({operands[0]}) - ({operands[1]}))"
        return f"{call_name}({operands[0]})"

    def texture_argument_diagnostic_type(self, arg):
        sampler_names = {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters
        argument_type = shared_texture_argument_diagnostic_type(
            arg,
            self.texture_resource_type,
            self.expression_name,
            self.expression_result_type,
            sampler_names,
        )
        # A reference argument such as `constant float &bias` names its referent's value.
        if argument_type is None:
            return None
        return self.reference_referent_type_name(argument_type) or argument_type

    @staticmethod
    def metal_referenced_value_type(vtype):
        """Return the value type a reference-typed expression reads as."""
        if isinstance(vtype, ReferenceType):
            return vtype.referenced_type
        if isinstance(vtype, str) and vtype.rstrip().endswith("&"):
            return vtype.rstrip().rstrip("&").rstrip()
        return vtype

    def validate_compare_argument(self, func_name, args):
        compare_index = texture_compare_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if compare_index is None:
            return
        compare_type = self.metal_referenced_value_type(
            self.expression_result_type(args[compare_index])
        )
        if compare_type is None or self.is_scalar_floating_type(compare_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "texture compare",
                func_name,
                "a scalar floating",
                "compare",
                expression_debug_name(args[compare_index]),
                self.type_name_string(compare_type),
            )
        )

    def validate_lod_argument(self, func_name, args):
        lod_index = texture_lod_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if lod_index is None:
            return
        lod_type = self.texture_argument_diagnostic_type(args[lod_index])
        if lod_type is None or self.is_scalar_numeric_type(lod_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "texture LOD",
                func_name,
                "a scalar numeric",
                "lod",
                expression_debug_name(args[lod_index]),
                self.type_name_string(lod_type),
            )
        )

    def validate_bias_argument(self, func_name, args):
        bias_index = texture_bias_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if bias_index is None:
            return
        bias_type = self.texture_argument_diagnostic_type(args[bias_index])
        if bias_type is None or self.is_scalar_numeric_type(bias_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "texture bias",
                func_name,
                "a scalar numeric",
                "bias",
                expression_debug_name(args[bias_index]),
                self.type_name_string(bias_type),
            )
        )

    def validate_mip_level_argument(self, func_name, args):
        level_index = texture_mip_level_argument_index(func_name, len(args))
        if level_index is None:
            return
        level_type = self.texture_argument_diagnostic_type(args[level_index])
        if level_type is None or self.is_scalar_integer_type(level_type):
            return
        raise MetalRefusal(
            operation_argument_type_error(
                "Metal",
                "resource",
                func_name,
                "a scalar integer",
                "mip/sample level",
                expression_debug_name(args[level_index]),
                self.type_name_string(level_type),
            )
        )
