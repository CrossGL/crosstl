"""Loose uniforms for SPIR-V: one std140 ``Uniform`` block per module.

CrossGL and GLSL accept a non-opaque ``uniform T name;`` outside any block, and HLSL treats a non-static
module-scope global as a member of the implicit ``$Globals`` constant buffer. Vulkan has neither: the
GL_KHR_vulkan_glsl extension removes non-opaque uniforms outside a block ("Changes to Chapter 4 of the OpenGL
Shading Language Specification"), and a SPIR-V ``Uniform`` variable must be a ``Block``-decorated struct
(Vulkan specification, "Shader Resource Interface", VUID-StandaloneSpirv-Uniform-06676).

The emitter used to give such a uniform ``Private`` storage. The module validated, but every invocation read its
own uninitialised copy, so the values the host set never reached the shader. The mixin here gathers every loose
uniform of the module, shader scope first and then each stage in declaration order, into one ``Block`` struct
with std140 offsets, as DXC does with ``$Globals`` and glslang's relaxed Vulkan rules do with
``gl_DefaultUniformBlock``. Each name resolves to its member through the host's ``cbuffer_members`` table, the
path HLSL ``cbuffer`` members already take.

Two forms have no faithful lowering and are refused with :class:`UnsupportedSPIRVFeatureError`:

* an initializer, because a Vulkan uniform block has no default contents for the host to inherit;
* one name declared with two different types, which GLSL 4.60 section 4.3.5 makes a link error.

The mixin relies on the host class's ``type_name_from_value``, ``is_glsl_buffer_block_node``,
``is_structured_buffer_declared_type_name``, ``is_resource_type_name``, ``struct_types``,
``map_resource_type_with_format``, ``storage_layout_type``, ``register_struct_type``, ``decorate_cbuffer_type``,
``create_variable``, ``next_available_resource_binding``, ``used_resource_bindings``, ``decorations``,
``uniform_buffers`` and ``cbuffer_members``.
"""

from __future__ import annotations

from .array_utils import parse_array_type

__all__ = [
    "DEFAULT_UNIFORM_BLOCK_TYPE",
    "DEFAULT_UNIFORM_BLOCK_VARIABLE",
    "SpirvDefaultUniformMixin",
]

DEFAULT_UNIFORM_BLOCK_TYPE = "CrossGLDefaultUniforms"
DEFAULT_UNIFORM_BLOCK_VARIABLE = "crossgl_default_uniforms"
_UNIFORM = "uniform"
_NOT_LOOSE = {"input", "in", "output", "out", "shared", "groupshared", "threadgroup"}


def _lowered_names(node, field):
    return {
        str(getattr(item, "name", item)).lower() for item in getattr(node, field, [])
    }


class SpirvDefaultUniformMixin:
    """Lower loose non-opaque uniforms to members of one module-wide std140 block."""

    def is_loose_uniform_node(self, node) -> bool:
        """True for a ``uniform`` variable that is neither a block, a struct nor a resource."""
        names = _lowered_names(node, "attributes") | _lowered_names(node, "qualifiers")
        if _UNIFORM not in names or names & _NOT_LOOSE:
            return False
        if self.is_glsl_buffer_block_node(node):
            return False
        type_source = getattr(node, "var_type", getattr(node, "vtype", None))
        type_name = self.type_name_from_value(type_source)
        if not type_name or self.is_structured_buffer_declared_type_name(type_name):
            return False
        base_type_name, _ = parse_array_type(type_name)
        if self.is_resource_type_name(base_type_name):
            return False
        return base_type_name not in self.struct_types

    def collect_loose_uniform_nodes(self, ast) -> list:
        nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            nodes.extend(getattr(stage, "local_variables", []) or [])
        return [node for node in nodes if self.is_loose_uniform_node(node)]

    def declare_default_uniform_block(self, ast):
        """Create the module's default uniform block before any function body is lowered."""
        from .SPIRV_codegen import UnsupportedSPIRVFeatureError

        self.default_uniform_node_ids = set()
        self.default_uniform_block = None
        self.default_uniform_block_bound = False
        members, member_types_by_name = [], {}
        for node in self.collect_loose_uniform_nodes(ast):
            location = getattr(node, "source_location", None)
            if getattr(node, "initial_value", None) is not None:
                raise UnsupportedSPIRVFeatureError(
                    "uniform initializer",
                    f"SPIR-V uniform '{node.name}' cannot take an initializer: a Vulkan uniform "
                    "block has no default contents (GL_KHR_vulkan_glsl removes uniform initializers)",
                    missing_capabilities=("spirv.uniform.initializer",),
                    source_location=location,
                )
            type_source = getattr(node, "var_type", getattr(node, "vtype", None))
            type_name = self.type_name_from_value(type_source)
            self.default_uniform_node_ids.add(id(node))
            if node.name in member_types_by_name:
                if member_types_by_name[node.name] != type_name:
                    raise UnsupportedSPIRVFeatureError(
                        "conflicting uniform declarations",
                        f"uniform '{node.name}' is declared as both "
                        f"{member_types_by_name[node.name]} and {type_name}; GLSL 4.60 section "
                        "4.3.5 requires one type per uniform name",
                        missing_capabilities=("spirv.uniform.conflicting_types",),
                        source_location=location,
                    )
                continue
            if node.name in self.cbuffer_members:
                raise UnsupportedSPIRVFeatureError(
                    "conflicting uniform declarations",
                    f"uniform '{node.name}' is also a constant buffer member",
                    missing_capabilities=("spirv.uniform.conflicting_types",),
                    source_location=location,
                )
            member_types_by_name[node.name] = type_name
            value_type = self.map_resource_type_with_format(type_source, node)
            members.append((self.storage_layout_type(value_type, "std140"), node.name))

        if not members:
            return None
        block_type = self.register_struct_type(DEFAULT_UNIFORM_BLOCK_TYPE, members)
        self.decorate_cbuffer_type(block_type, members)
        block = self.create_variable(
            block_type, "Uniform", DEFAULT_UNIFORM_BLOCK_VARIABLE
        )
        self.uniform_buffers.append(block)
        for index, (member_type, name) in enumerate(members):
            self.cbuffer_members[name] = (block, member_type, index)
        self.default_uniform_block = block
        return block

    def is_default_uniform_node(self, node) -> bool:
        return id(node) in getattr(self, "default_uniform_node_ids", ())

    def bind_default_uniform_block(self):
        """Give the block the next free binding of set 0, after every declared resource has its own."""
        block = getattr(self, "default_uniform_block", None)
        if block is None or getattr(self, "default_uniform_block_bound", False):
            return
        binding = self.next_available_resource_binding(0)
        self.used_resource_bindings.add((0, binding))
        self.decorations.append(f"OpDecorate %{block.id} DescriptorSet 0")
        self.decorations.append(f"OpDecorate %{block.id} Binding {binding}")
        self.default_uniform_block_bound = True
