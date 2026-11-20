"""Module-scope declarations of the WGSL generator: constant buffers, constants, global variables, stage resource parameters, textures, samplers, resource members and buffer blocks."""

import re

from ..refusals import WGSLRefusal
from .wgsl_workgroup_locals import WORKGROUP_QUALIFIERS


class WGSLGlobalDeclarationMixin:
    """Global declaration members of ``WGSLCodeGen``."""

    def generate_cbuffer(self, node):
        lines = [f"struct {self.type_identifier_name(node.name)} {{"]
        for member in getattr(node, "members", []) or []:
            self.validate_cbuffer_member(node, member)
            member_name = self.struct_member_identifier_name(node.name, member.name)
            prefix = (
                "@align(16) "
                if self.uniform_scalar_array_wrapper_name(member.member_type)
                is not None
                else ""
            )
            lines.append(
                f"    {prefix}{member_name}: "
                f"{self.uniform_buffer_member_type_name(member.member_type)},"
            )
        lines.append("};")

        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        instance_name = self.cbuffer_instance_name(node)
        lines.append(
            f"{attributes}\nvar<uniform> {instance_name}: "
            f"{self.type_identifier_name(node.name)};"
        )
        return "\n".join(lines)

    def generate_constant(self, node):
        value = (
            None
            if node.value is None
            else self.generate_module_scope_value(node.value, node.const_type)
        )
        name = self.module_identifier_name(node.name)
        type_name = self.type_name_string(node.const_type)
        specialized = self._specialization_plan.declaration(
            node.name, name, type_name, value
        )
        if specialized is not None:
            return specialized
        return f"const {name}: {type_name} = {value};"

    def generate_global_variable(self, node):
        sampled_texture_array = self.sampled_texture_array_info(node.var_type)
        if sampled_texture_array is not None:
            return self.generate_sampled_texture_array_global_variable(
                node, sampled_texture_array
            )
        resource_array_type = self.resource_array_element_type_name(node.var_type)
        if resource_array_type is not None:
            raise WGSLRefusal(
                "WGSL target cannot lower resource array "
                f"{node.name} of type {resource_array_type}; WebGPU/WGSL requires "
                "each texture, sampler, image, and storage-buffer element to be "
                "expanded to individual module-scope bindings"
            )
        sampled_texture_type = self.sampled_texture_type(node.var_type)
        if sampled_texture_type is not None:
            return self.generate_sampled_texture_global_variable(
                node, sampled_texture_type
            )
        if self.is_sampler_type(node.var_type):
            return self.generate_sampler_global_variable(node)
        storage_texture_type = self.storage_texture_type(node.var_type, node)
        if storage_texture_type is not None:
            return self.generate_storage_texture_global_variable(
                node, storage_texture_type
            )
        if self.is_buffer_pointer_type(node.var_type, node.qualifiers):
            return self.generate_buffer_pointer_global_variable(node)
        if self.is_glsl_buffer_block_variable(node):
            return self.generate_glsl_buffer_block_global_variable(node)

        qualifier_names = {str(qualifier).lower() for qualifier in node.qualifiers}
        address_space = "private"
        access = ""
        attributes = ""
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.var_type
        )
        if unsupported_storage_type is not None:
            raise WGSLRefusal(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        storage_buffer_access = self.structured_buffer_access(node.var_type)
        if storage_buffer_access:
            address_space = "storage"
            access = f", {storage_buffer_access}"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "uniform" in qualifier_names:
            if self.is_unsupported_uniform_array_stride(
                node.var_type, variable_name=node.name
            ):
                address_space = "storage"
                access = ", read"
            else:
                address_space = "uniform"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "buffer" in qualifier_names or "storage" in qualifier_names:
            address_space = "storage"
            access = ", read_write"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif qualifier_names & WORKGROUP_QUALIFIERS:
            address_space = "workgroup"
        self.require_atomic_address_space(
            None, node.name, node.var_type, address_space, access
        )

        initializer = ""
        if node.initial_value is not None and address_space == "private":
            value = self.generate_module_scope_value(node.initial_value, node.var_type)
            initializer = f" = {value}"
        prefix = f"{attributes}\n" if attributes else ""
        variable_name = self.module_identifier_name(node.name)
        type_name = self.type_name_string(
            self.atomic_declaration_type(None, node.name, node.var_type),
            allow_storage_resources=True,
        )
        resource_declarations = self.generate_resource_member_global_variables(
            node.name, node.var_type
        )
        if (
            initializer
            and not resource_declarations
            and qualifier_names & {"const", "constant"}
        ):
            # A module-scope constant is a WGSL ``const``: a ``var<private>`` initializer cannot read another
            # module-scope variable, so ``constant int B = A + 1;`` must not make ``A`` a variable.
            return f"const {variable_name}: {type_name}{initializer};"
        declaration = (
            f"{prefix}var<{address_space}{access}> {variable_name}: "
            f"{type_name}"
            f"{initializer};"
        )
        if resource_declarations:
            declaration += "\n" + "\n".join(resource_declarations)
        return declaration

    def generate_stage_resource_parameter(self, node):
        unsupported_storage_type = self.unsupported_storage_buffer_type_name(
            node.param_type
        )
        if unsupported_storage_type is not None:
            raise WGSLRefusal(
                "WGSL target does not support "
                f"{unsupported_storage_type} resources yet"
            )
        storage_buffer_access = self.structured_buffer_access(node.param_type)
        if storage_buffer_access:
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
            self.require_atomic_address_space(
                None, node.name, node.param_type, "storage", storage_buffer_access
            )
            param_type = self.atomic_declaration_type(None, node.name, node.param_type)
            return (
                f"{attributes}\nvar<storage, {storage_buffer_access}> "
                f"{self.module_identifier_name(node.name)}: "
                f"{self.type_name_string(param_type, allow_storage_resources=True)};"
            )
        uniform_type = self.stage_uniform_parameter_type(node)
        if uniform_type is not None:
            self.validate_uniform_binding_type(uniform_type)
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
            return (
                f"{attributes}\nvar<uniform> {self.module_identifier_name(node.name)}: "
                f"{self.type_name_string(uniform_type)};"
            )
        raise WGSLRefusal(
            "WGSL target cannot lower stage resource parameter "
            f"{node.name} of type {node.param_type}"
        )

    def generate_sampled_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for sampled texture "
                f"resource {node.name}"
            )
        texture_attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        sampler_attributes = self.sampler_binding_attributes_for_texture(
            texture_attributes,
            node=node,
        )
        texture_name = self.module_identifier_name(node.name)
        sampler_name = self.texture_sampler_name(texture_name)
        return (
            f"{texture_attributes}\nvar {texture_name}: {texture_type};\n"
            f"{sampler_attributes}\nvar {sampler_name}: "
            f"{self.companion_sampler_type(node.var_type)};"
        )

    def generate_sampled_texture_array_global_variable(self, node, array_info):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for sampled texture "
                f"resource array {node.name}"
            )
        array_size = self.resolve_array_size(node.var_type.size, node.name)
        if array_size is None:
            raise WGSLRefusal(
                "WGSL target cannot lower sampled texture resource array "
                f"{node.name}; WebGPU/WGSL requires project-level binding "
                "expansion to declare each texture and sampler element as an "
                "individual module-scope binding"
            )
        if array_size < 1:
            raise WGSLRefusal(
                "WGSL target cannot lower sampled texture resource array "
                f"{node.name} with size {array_size}; declare at least one "
                "texture/sampler binding element"
            )

        texture_type = array_info["texture_type"]
        element_type = array_info["element_type"]
        companion_sampler_type = self.companion_sampler_type(element_type)
        root_name = self.module_identifier_name(node.name)
        bindings = []
        declarations = []
        next_texture_binding = None
        for index in range(array_size):
            texture_name = self.safe_wgsl_identifier(f"{root_name}_{index}")
            texture_attributes = (
                self.explicit_binding_attributes(node)
                if index == 0
                else self.next_binding_attributes(
                    group=self.binding_group_for_resource_array(node),
                    start=next_texture_binding,
                )
            )
            if not texture_attributes:
                texture_attributes = self.next_binding_attributes(
                    group=self.binding_group_for_resource_array(node),
                    start=next_texture_binding,
                )
            sampler_attributes = self.sampler_binding_attributes_for_texture(
                texture_attributes
            )
            # Expanded elements follow their preceding sampler, even when the first binding is explicit.
            next_texture_binding = (
                int(re.search(r"@binding\((\d+)\)", sampler_attributes).group(1)) + 1
            )
            sampler_name = self.texture_sampler_name(texture_name)
            bindings.append(
                {
                    "texture_name": texture_name,
                    "sampler_name": sampler_name,
                    "index": index,
                }
            )
            declarations.append(
                f"{texture_attributes}\nvar {texture_name}: {texture_type};\n"
                f"{sampler_attributes}\nvar {sampler_name}: {companion_sampler_type};"
            )
        self._sampled_texture_array_bindings[node.name] = {
            "source_name": node.name,
            "root_name": root_name,
            "element_type": element_type,
            "texture_type": texture_type,
            "bindings": tuple(bindings),
        }
        return "\n".join(declarations)

    def generate_sampler_global_variable(self, node):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for sampler resource "
                f"{node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return (
            f"{attributes}\nvar {self.module_identifier_name(node.name)}: "
            f"{self.sampler_type(node.var_type)};"
        )

    def generate_storage_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for storage texture "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        self._module_storage_texture_access_modes[node.name] = (
            self.storage_texture_access(node)
        )
        return (
            f"{attributes}\nvar {self.module_identifier_name(node.name)}: "
            f"{texture_type};"
        )

    def generate_resource_member_global_variables(self, root_name, root_type):
        declarations = []
        for info in self.resource_paths_for_type(root_type):
            binding_name = self.resource_member_binding_name(root_name, info["path"])
            info = {**info, "binding_name": binding_name, "root_name": root_name}
            self._module_resource_bindings[(root_name, tuple(info["path"]))] = info
            declarations.append(self.generate_resource_member_binding(info))
        return declarations

    def generate_resource_member_binding(self, info):
        member = info["member"]
        binding_name = info["binding_name"]
        sampled_texture_type = self.sampled_texture_type(info["member_type"])
        if sampled_texture_type is not None:
            texture_attributes = (
                self.explicit_binding_attributes(member)
                or self.next_binding_attributes()
            )
            sampler_attributes = self.sampler_binding_attributes_for_texture(
                texture_attributes,
                node=member,
            )
            sampler_name = self.texture_sampler_name(binding_name)
            return (
                f"{texture_attributes}\nvar {binding_name}: {sampled_texture_type};\n"
                f"{sampler_attributes}\nvar {sampler_name}: "
                f"{self.companion_sampler_type(info['member_type'])};"
            )
        if self.is_sampler_type(info["member_type"]):
            attributes = (
                self.explicit_binding_attributes(member)
                or self.next_binding_attributes()
            )
            return (
                f"{attributes}\nvar {binding_name}: "
                f"{self.sampler_type(info['member_type'])};"
            )
        raise WGSLRefusal(
            "WGSL target does not support resource member "
            f"{info['owner_struct']}.{member.name} of type {info['resource_type_name']}; "
            "declare textures, samplers, and storage resources as module-scope "
            "bindings instead of user-struct fields"
        )

    def generate_buffer_pointer_global_variable(self, node):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for buffer pointer "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        # An atomic built-in on the buffer's elements makes them atomic<T> (wgsl_buffer_atomics).
        storage_type = self.atomic_declaration_type(None, node.name, node.var_type)
        return (
            f"{attributes}\nvar<storage, read_write> "
            f"{self.module_identifier_name(node.name)}: "
            f"{self.buffer_pointer_storage_type(storage_type)};"
        )

    def generate_glsl_buffer_block_global_variable(self, node):
        self.validate_glsl_buffer_block_variable(node)
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for GLSL buffer block "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        access = self.glsl_buffer_block_access(node)
        type_name = self.glsl_buffer_block_struct_name(node.var_type)
        self.require_atomic_address_space(
            None, node.name, node.var_type, "storage", access
        )
        return (
            f"{attributes}\nvar<storage, {access}> "
            f"{self.module_identifier_name(node.name)}: "
            f"{self.type_identifier_name(type_name)};"
        )
