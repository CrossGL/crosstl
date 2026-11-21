"""GLSL declarations of global resources for ``GLSLCodeGen.declare_glsl_global_resource``: buffer blocks, samplers, structured buffers, constant buffers, opaque resources and layout-bound uniforms."""

from ..refusals import (
    OpenGLRefusal,
)
from ..unhandled import (
    UNHANDLED,
)
from .array_utils import (
    format_c_style_array_declaration,
)


class OpenGLGlobalResourceMixin:
    """Global-resource declaration members of ``GLSLCodeGen``."""

    def declare_glsl_buffer_block_resource(
        self,
        emission,
        node,
        array_suffix,
        resource_count,
        source_var_name,
        var_name,
        vtype,
    ):
        """Declare a global that GLSL holds in a buffer block; return ``UNHANDLED`` when it needs no block."""
        if self.is_glsl_buffer_block_variable(node, vtype):
            self.record_glsl_buffer_block_variable(var_name, node, vtype)
            if self.is_shader_record_buffer_block(node):
                if self.explicit_resource_binding_index(node) is not None:
                    raise OpenGLRefusal(
                        "GLSL shaderRecordEXT buffer blocks cannot declare "
                        "binding layout qualifiers"
                    )
                declaration = self.glsl_buffer_block_declaration(
                    node, vtype, var_name, None, array_suffix
                )
                if declaration is not None:
                    emission.code += declaration
                    return

            binding_namespace = "buffer binding"
            resource_binding = self.opengl_resource_binding_for_declaration(
                node,
                emission.used_resource_bindings,
                emission.resource_binding_cursors,
                binding_namespace,
                resource_count,
            )
            declaration = self.glsl_buffer_block_declaration(
                node, vtype, var_name, resource_binding, array_suffix
            )
            if declaration is not None:
                self.reserve_resource_binding_range(
                    emission.used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                self.record_glsl_resource_binding(
                    node,
                    source_var_name,
                    var_name,
                    "buffer-block",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                emission.code += declaration
                self.advance_resource_binding(
                    emission.resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                return
        return UNHANDLED

    def declare_glsl_sampler_resource(
        self,
        emission,
        node,
        array_suffix,
        mapped_type,
        resource_count,
        source_var_name,
        var_name,
    ):
        """Declare a standalone sampler."""
        self.explicit_resource_binding_index(node)
        self.sampler_variables.add(var_name)
        self.glsl_sampler_registers[var_name] = (
            source_var_name,
            self.glsl_source_register(node),
        )
        if self.should_preserve_sampler_resource(source_var_name):
            binding_namespace = "sampler binding"
            resource_binding = self.opengl_resource_binding_for_declaration(
                node,
                emission.used_resource_bindings,
                emission.resource_binding_cursors,
                binding_namespace,
                resource_count,
            )
            self.reserve_resource_binding_range(
                emission.used_resource_bindings,
                "OpenGL",
                binding_namespace,
                resource_binding,
                resource_count,
                var_name,
            )
            self.record_glsl_resource_binding(
                node,
                source_var_name,
                var_name,
                "sampler",
                binding_namespace,
                resource_binding,
                resource_count,
            )
            declaration = format_c_style_array_declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            layout = self.glsl_resource_layout_prefix(
                binding=resource_binding, node=node
            )
            prefix = f"{layout} " if layout else ""
            emission.code += f"{prefix}uniform {declaration};\n"
            self.advance_resource_binding(
                emission.resource_binding_cursors,
                binding_namespace,
                resource_binding,
                resource_count,
            )
        return

    def declare_glsl_structured_buffer_resource(
        self,
        emission,
        node,
        array_size,
        resource_count,
        source_var_name,
        var_name,
        vtype,
    ):
        """Declare a structured buffer as a shader storage block."""
        self.record_structured_buffer_access_metadata(var_name, vtype, node)
        if source_var_name != var_name:
            self.record_structured_buffer_access_metadata(source_var_name, vtype, node)
        binding_namespace = "buffer binding"
        resource_binding = self.opengl_resource_binding_for_declaration(
            node,
            emission.used_resource_bindings,
            emission.resource_binding_cursors,
            binding_namespace,
            resource_count,
        )
        self.reserve_resource_binding_range(
            emission.used_resource_bindings,
            "OpenGL",
            binding_namespace,
            resource_binding,
            resource_count,
            var_name,
        )
        emission.code += self.structured_buffer_block_declaration(
            vtype, var_name, resource_binding, array_size, node
        )
        self.record_glsl_resource_binding(
            node,
            source_var_name,
            var_name,
            "structured-buffer",
            binding_namespace,
            resource_binding,
            resource_count,
        )
        self.advance_resource_binding(
            emission.resource_binding_cursors,
            binding_namespace,
            resource_binding,
            resource_count,
        )
        if self.structured_buffer_requires_counter(vtype):
            counter_binding = self.next_available_resource_binding(
                emission.used_resource_bindings,
                emission.resource_binding_cursors,
                binding_namespace,
                resource_count,
            )
            counter_name = self.structured_buffer_counter_resource_name(var_name)
            self.reserve_resource_binding_range(
                emission.used_resource_bindings,
                "OpenGL",
                binding_namespace,
                counter_binding,
                resource_count,
                counter_name,
            )
            emission.code += self.structured_buffer_counter_block_declaration(
                var_name, counter_binding, array_size
            )
            self.glsl_resource_binding_records.append(
                {
                    "name": counter_name,
                    "glslName": counter_name,
                    "kind": "structured-buffer-counter",
                    "namespace": "shader-storage-buffer",
                    "binding": int(counter_binding),
                    "count": (
                        max(resource_count, 1) if isinstance(resource_count, int) else 1
                    ),
                    "register": None,
                    "of": source_var_name,
                }
            )
            self.advance_resource_binding(
                emission.resource_binding_cursors,
                binding_namespace,
                counter_binding,
                resource_count,
            )
        return

    def declare_glsl_constant_buffer_resource(
        self,
        emission,
        node,
        array_suffix,
        resource_count,
        source_var_name,
        var_name,
        vtype,
    ):
        """Declare a constant buffer as a uniform block."""
        self.glsl_uniform_block_variable_types[var_name] = (
            self.constant_buffer_element_type(vtype)
        )
        if source_var_name != var_name:
            self.glsl_uniform_block_variable_types[source_var_name] = (
                self.constant_buffer_element_type(vtype)
            )
        binding_namespace = "uniform buffer binding"
        resource_binding = self.opengl_resource_binding_for_declaration(
            node,
            emission.used_resource_bindings,
            emission.resource_binding_cursors,
            binding_namespace,
            resource_count,
        )
        self.reserve_resource_binding_range(
            emission.used_resource_bindings,
            "OpenGL",
            binding_namespace,
            resource_binding,
            resource_count,
            var_name,
        )
        emission.code += self.glsl_constant_buffer_block_declaration(
            vtype,
            var_name,
            resource_binding,
            array_suffix,
            node=node,
        )
        self.record_glsl_resource_binding(
            node,
            source_var_name,
            var_name,
            "constant-buffer",
            binding_namespace,
            resource_binding,
            resource_count,
        )
        self.advance_resource_binding(
            emission.resource_binding_cursors,
            binding_namespace,
            resource_binding,
            resource_count,
        )
        return

    def declare_glsl_opaque_resource(
        self,
        emission,
        node,
        declaration,
        mapped_type,
        array_suffix,
        resource_count,
        source_var_name,
        var_name,
        vtype,
    ):
        """Declare a texture, image or other opaque resource uniform."""
        binding_namespace = (
            "image binding" if self.is_storage_image_type(vtype) else "texture binding"
        )
        resource_binding = self.opengl_resource_binding_for_declaration(
            node,
            emission.used_resource_bindings,
            emission.resource_binding_cursors,
            binding_namespace,
            resource_count,
        )
        self.reserve_resource_binding_range(
            emission.used_resource_bindings,
            "OpenGL",
            binding_namespace,
            resource_binding,
            resource_count,
            var_name,
        )
        layout = self.opaque_resource_layout(mapped_type, resource_binding, node)
        memory_qualifiers = self.resource_memory_qualifiers(node)
        if (
            var_name in self.glsl_write_only_unknown_format_images
            or source_var_name in self.glsl_write_only_unknown_format_images
        ) and "writeonly" not in memory_qualifiers.split():
            memory_qualifiers = " ".join(
                part for part in (memory_qualifiers, "writeonly") if part
            )
        precision_qualifiers = self.resource_precision_qualifiers(node)
        qualifier_parts = []
        if layout:
            qualifier_parts.append(layout)
        if memory_qualifiers:
            qualifier_parts.append(memory_qualifiers)
        qualifier_parts.append("uniform")
        if precision_qualifiers:
            qualifier_parts.append(precision_qualifiers)
        emission.code += self.glsl_vulkan_resources.declaration(
            source_var_name,
            var_name,
            " ".join(qualifier_parts),
            declaration,
            mapped_type,
            array_suffix,
        )
        self.record_glsl_resource_binding(
            node,
            source_var_name,
            var_name,
            ("image" if binding_namespace == "image binding" else "texture"),
            binding_namespace,
            resource_binding,
            resource_count,
        )
        self.advance_resource_binding(
            emission.resource_binding_cursors,
            binding_namespace,
            resource_binding,
            resource_count,
        )

    def declare_glsl_layout_bound_struct_uniform(
        self,
        emission,
        node,
        array_suffix,
        resource_count,
        source_var_name,
        var_name,
        vtype,
    ):
        """Declare a struct uniform that carries an explicit layout binding."""
        self.glsl_uniform_block_variable_types[var_name] = str(
            self.resource_base_type(vtype)
        )
        if source_var_name != var_name:
            self.glsl_uniform_block_variable_types[source_var_name] = str(
                self.resource_base_type(vtype)
            )
        binding_namespace = "uniform buffer binding"
        resource_binding = self.opengl_resource_binding_for_declaration(
            node,
            emission.used_resource_bindings,
            emission.resource_binding_cursors,
            binding_namespace,
            resource_count,
        )
        self.reserve_resource_binding_range(
            emission.used_resource_bindings,
            "OpenGL",
            binding_namespace,
            resource_binding,
            resource_count,
            var_name,
        )
        emission.code += self.glsl_struct_uniform_block_declaration(
            vtype,
            var_name,
            resource_binding,
            array_suffix,
            node=node,
        )
        self.record_glsl_resource_binding(
            node,
            source_var_name,
            var_name,
            "constant-buffer",
            binding_namespace,
            resource_binding,
            resource_count,
        )
        self.advance_resource_binding(
            emission.resource_binding_cursors,
            binding_namespace,
            resource_binding,
            resource_count,
        )
