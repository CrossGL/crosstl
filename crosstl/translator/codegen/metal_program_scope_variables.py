"""Declarations of Metal program-scope variables: the declared type and array shape, buffer-like variables (argument buffers, GLSL buffer blocks and buffer resources), ray-tracing objects, structured buffers, textures, samplers and plain values."""

from ..ast import (
    ArrayType,
    PointerType,
    ReferenceType,
)
from ..unhandled import (
    UNHANDLED,
)
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .image_access_contracts import (
    record_explicit_image_metadata,
)


class MetalProgramScopeVariableMixin:
    """Per-kind members of ``MetalCodeGen.declare_metal_program_scope_variable``."""

    def metal_program_scope_variable_shape(
        self,
        node,
    ):
        """The variable's type name, array suffix, array size and the number of resource bindings it occupies."""
        resource_count = 1
        array_size = None
        if getattr(node, "var_type", None) is not None:
            if (
                hasattr(node.var_type, "name")
                or hasattr(node.var_type, "element_type")
                or isinstance(node.var_type, (PointerType, ReferenceType))
            ):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):  # ArrayType
                    base_type = self.convert_type_node_to_string(
                        node.var_type.element_type
                    )
                    # An array of arrays: the outer dimension is the first suffix.
                    base_type, element_suffix = split_array_type_suffix(base_type)
                    array_size = (
                        self.expression_to_string(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    vtype = base_type
                    array_suffix = (
                        f"[{array_size}]" if array_size else "[]"
                    ) + element_suffix
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    )
                    element_type = node.var_type.element_type
                    while isinstance(element_type, ArrayType):
                        resource_count *= self.resource_array_count(element_type.size)
                        element_type = element_type.element_type
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
                    array_suffix = ""
            else:
                vtype = str(node.var_type)
                array_suffix = ""
        elif getattr(node, "vtype", None) is not None:
            vtype = node.vtype
            array_suffix = ""
        else:
            # No written type: take it from the initializer so a shader-scope
            # `let scale = 2.0` is `constant float scale = 2.0`, not
            # `constant None scale` (#2444).
            vtype = self.infer_metal_global_value_type(node)
            array_suffix = ""
        return resource_count, array_size, vtype, array_suffix

    def declare_metal_buffer_variable(
        self,
        emission,
        node,
        array_size,
        resource_count,
        var_name,
        vtype,
    ):
        """Declare an argument buffer, a lowered or unsupported GLSL buffer block, or a Metal buffer resource."""
        if self.is_metal_argument_buffer_global(node):
            binding = self.metal_argument_buffer_binding(node)
            declaration = self.format_metal_argument_buffer_global(
                node, vtype, var_name
            )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                resource_count,
                var_name,
                node,
            )
            emission.code += declaration
            return

        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(
                    emission.used_resource_bindings,
                    "buffer",
                    resource_count,
                )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                resource_count,
                var_name,
                node,
            )
            self.glsl_buffer_block_variables.append(
                (node, binding, lowered_block, array_size)
            )
            return

        if self.is_glsl_buffer_block_variable(node, vtype):
            emission.code += self.glsl_buffer_block_diagnostic(
                "Metal", vtype, var_name, node
            )
            self.unsupported_glsl_buffer_block_variables.add(var_name)
            self.unsupported_glsl_buffer_block_variable_types[var_name] = (
                self.type_name_string(vtype)
            )
            emission.code += self.unsupported_glsl_buffer_block_variable_placeholder(
                "Metal", vtype, var_name
            )
            return

        metal_buffer_address_space = self.metal_buffer_resource_address_space(
            node, vtype
        )
        if metal_buffer_address_space is not None:
            binding_count = self.metal_buffer_resource_binding_count(node, array_size)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(
                    emission.used_resource_bindings,
                    "buffer",
                    binding_count,
                )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                binding_count,
                var_name,
                node,
            )
            self.metal_buffer_resource_variables.append(
                (node, binding, vtype, array_size, metal_buffer_address_space)
            )
            return
        return UNHANDLED

    def declare_metal_ray_tracing_variable(
        self,
        emission,
        node,
        array_size,
        resource_count,
        var_name,
        vtype,
    ):
        """Declare an acceleration structure, a visible function table or an intersection function table."""
        if self.is_acceleration_structure_type(vtype):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(
                    emission.used_resource_bindings,
                    "buffer",
                    resource_count,
                )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                resource_count,
                var_name,
                node,
            )
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if array_size is not None:
                self.unsupported_metal_acceleration_structure_array_variables[
                    var_name
                ] = mapped_type
                emission.code += (
                    self.unsupported_metal_acceleration_structure_array_diagnostic(
                        var_name
                    )
                )
                return
            self.acceleration_structure_variables.append(
                (node, binding, mapped_type, array_size)
            )
            return

        if self.is_visible_function_table_type(vtype):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(
                    emission.used_resource_bindings,
                    "buffer",
                    resource_count,
                )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                resource_count,
                var_name,
                node,
            )
            if array_size is not None:
                self.unsupported_metal_ray_function_table_array_variables[var_name] = (
                    "visible_function_table"
                )
                emission.code += (
                    self.unsupported_metal_ray_function_table_array_diagnostic(
                        "visible_function_table", var_name
                    )
                )
                return
            mapped_type = self.map_visible_function_table_type(vtype)
            self.visible_function_table_variables.append(
                (node, binding, mapped_type, array_size)
            )
            return

        if self.is_intersection_function_table_type(vtype):
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            if binding is None:
                binding = self.lowest_free_resource_binding(
                    emission.used_resource_bindings,
                    "buffer",
                    resource_count,
                )
            binding = self.reserve_or_remap_resource_binding(
                emission.used_resource_bindings,
                emission.source_resource_bindings,
                "buffer",
                binding,
                resource_count,
                var_name,
                node,
            )
            if array_size is not None:
                self.unsupported_metal_ray_function_table_array_variables[var_name] = (
                    "intersection_function_table"
                )
                emission.code += (
                    self.unsupported_metal_ray_function_table_array_diagnostic(
                        "intersection_function_table", var_name
                    )
                )
                return
            mapped_type = self.map_intersection_function_table_type(vtype)
            self.intersection_function_table_variables.append(
                (node, binding, mapped_type, array_size)
            )
            return
        return UNHANDLED

    def declare_metal_structured_buffer_variable(
        self,
        emission,
        node,
        array_size,
        resource_count,
        var_name,
        vtype,
    ):
        """Declare a structured buffer with its length and counter companions."""
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer"}, ("b", "u", "t")
        )
        if binding is None:
            binding = self.lowest_free_resource_binding(
                emission.used_resource_bindings,
                "buffer",
                resource_count,
            )
        binding = self.reserve_or_remap_resource_binding(
            emission.used_resource_bindings,
            emission.source_resource_bindings,
            "buffer",
            binding,
            resource_count,
            var_name,
            node,
        )
        self.structured_buffer_variables.append((node, binding, vtype, array_size))
        # The companions take the first free slots after their buffer, so they sit next to it.
        companion_start = binding + resource_count
        if self.structured_buffer_requires_length(var_name):
            length_binding = self.next_available_resource_binding(
                emission.used_resource_bindings,
                "buffer",
                companion_start,
                resource_count,
            )
            companion_start = length_binding + resource_count
            length_name = self.structured_buffer_length_resource_name(var_name)
            self.reserve_resource_binding_range(
                emission.used_resource_bindings,
                "Metal",
                "buffer",
                length_binding,
                resource_count,
                length_name,
            )
            self.structured_buffer_length_variables.append(
                (node, length_binding, vtype, array_size)
            )
        if self.structured_buffer_requires_counter(vtype):
            counter_binding = self.next_available_resource_binding(
                emission.used_resource_bindings,
                "buffer",
                companion_start,
                resource_count,
            )
            counter_name = self.structured_buffer_counter_resource_name(var_name)
            self.reserve_resource_binding_range(
                emission.used_resource_bindings,
                "Metal",
                "buffer",
                counter_binding,
                resource_count,
                counter_name,
            )
            self.structured_buffer_counter_variables.append(
                (node, counter_binding, vtype, array_size)
            )

    def declare_metal_texture_variable(
        self,
        emission,
        node,
        array_size,
        resource_count,
        var_name,
        vtype,
    ):
        """Declare a texture or image and record its type, raw type and image metadata."""
        mapped_type = self.map_resource_type_with_format(vtype, node)
        binding = self.explicit_resource_binding_index(
            node, {"binding", "texture"}, ("t", "u")
        )
        if binding is None:
            binding = self.lowest_free_resource_binding(
                emission.used_resource_bindings,
                "texture",
                resource_count,
            )
        binding = self.reserve_or_remap_resource_binding(
            emission.used_resource_bindings,
            emission.source_resource_bindings,
            "texture",
            binding,
            resource_count,
            var_name,
            node,
        )
        self.texture_variables.append((node, binding, mapped_type, array_size))
        self.texture_variable_types[node.name] = mapped_type
        self.texture_variable_raw_types[node.name] = self.resource_base_type(vtype)
        record_explicit_image_metadata(
            node.name,
            node,
            self.attribute_value_to_string,
            image_formats=self.image_variable_formats,
        )

    def declare_metal_sampler_variable(
        self,
        emission,
        node,
        array_size,
        resource_count,
        var_name,
    ):
        """Declare a sampler or comparison sampler."""
        binding = self.explicit_resource_binding_index(
            node, {"binding", "sampler"}, ("s",)
        )
        if binding is None:
            binding = self.lowest_free_resource_binding(
                emission.used_resource_bindings,
                "sampler",
                resource_count,
            )
        binding = self.reserve_or_remap_resource_binding(
            emission.used_resource_bindings,
            emission.source_resource_bindings,
            "sampler",
            binding,
            resource_count,
            var_name,
            node,
        )
        self.sampler_variables.append((node, binding, array_size))

    def declare_metal_value_variable(
        self,
        emission,
        node,
        array_suffix,
        var_name,
        vtype,
    ):
        """Declare a program-scope value with its address space and initializer."""
        mapped_type = self.map_type(vtype)
        if var_name in self.metal_runtime_initialized_constants and not array_suffix:
            qualifier = self.global_variable_qualifier(node)
            self.record_metal_program_scope_value_global(
                var_name, vtype, qualifier, node
            )
            init_expr = self.generate_expression_with_expected(
                getattr(node, "initial_value", None),
                getattr(node, "var_type", vtype),
            )
            # Metal program-scope constants need constant expressions;
            # a source constant computed by a function call becomes an
            # accessor that each use calls.
            emission.code += (
                f"inline {mapped_type} {var_name}() {{\n"
                f"    return {init_expr};\n"
                "}\n"
            )
            return
        declaration = format_c_style_array_declaration(mapped_type, var_name)
        if array_suffix:
            declaration = f"{declaration}{array_suffix}"
        qualifier = self.global_variable_qualifier(node)
        declaration = f"{qualifier}{declaration}"
        resolved_var_type = getattr(node, "var_type", None) or vtype
        self.record_metal_program_scope_value_global(
            var_name, resolved_var_type, qualifier, node
        )
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            expected_type = getattr(node, "var_type", None) or vtype
            init_expr = self.generate_expression_with_expected(
                initial_value, expected_type
            )
            emission.code += f"{declaration} = {init_expr};\n"
        elif self.global_value_variable_requires_initializer(qualifier):
            self.refuse_program_scope_groupshared_global(var_name, node)
            init_expr = self.metal_program_scope_global_default_initializer(
                mapped_type, array_suffix
            )
            emission.code += f"{declaration} = {init_expr};\n"
        else:
            emission.code += f"{declaration};\n"
