"""Buffer resources in the WGSL generator: GLSL buffer blocks, storage access modes, buffer pointers, constant buffer members and uniform scalar arrays."""

from ..ast import (
    ArrayType,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    PointerType,
)
from ..refusals import WGSLRefusal
from .image_access_contracts import (
    explicit_image_access,
    merge_image_access_requirement,
)
from .wgsl_narrow_storage import narrow_storage_word_type


class WGSLBufferResourceMixin:
    """Buffer resource members of ``WGSLCodeGen``."""

    def is_glsl_buffer_block_variable(self, node):
        if self.is_buffer_pointer_type(
            getattr(node, "var_type", None), getattr(node, "qualifiers", [])
        ):
            return False
        if self.glsl_buffer_block_attribute(node) is not None:
            return True
        qualifier_names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "buffer" not in qualifier_names:
            return False
        struct_name = self.glsl_buffer_block_struct_name(
            getattr(node, "var_type", None)
        )
        return struct_name in self._structs_by_name

    def is_glsl_buffer_block_parameter(self, node):
        return self.glsl_buffer_block_attribute(node) is not None

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            if key == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        if attr is None:
            qualifier_names = {
                self.semantic_key(str(qualifier))
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if "buffer" in qualifier_names:
                return "std430"
            return None
        arguments = getattr(attr, "arguments", []) or []
        if not arguments:
            return None
        return self.semantic_key(self.generate_attribute_argument(arguments[0]))

    def glsl_buffer_block_struct_name(self, vtype):
        return self.struct_type_name(self.array_element_type(vtype))

    def glsl_buffer_block_access(self, node):
        qualifiers = {
            self.semantic_key(str(qualifier))
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            self.semantic_key(str(getattr(attr, "name", attr)))
            for attr in getattr(node, "attributes", []) or []
        }
        names = qualifiers | attributes
        if "readonly" in names:
            return "read"
        if "writeonly" in names:
            return "write"
        return "read_write"

    def validate_glsl_buffer_block_layout(self, node, resource_name):
        layout = self.glsl_buffer_block_layout(node)
        if layout not in self.SUPPORTED_GLSL_BUFFER_BLOCK_LAYOUTS:
            layout_name = layout or "unspecified"
            raise WGSLRefusal(
                "WGSL target only supports std430 GLSL buffer block layout "
                f"for {resource_name}; got {layout_name}"
            )

    def validate_glsl_buffer_block_variable(self, node):
        self.validate_glsl_buffer_block_layout(node, node.name)
        if isinstance(getattr(node, "var_type", None), ArrayType):
            raise WGSLRefusal(
                "WGSL target does not support GLSL buffer block arrays yet; "
                "declare each block as a separate storage binding"
            )
        struct_name = self.glsl_buffer_block_struct_name(node.var_type)
        if not struct_name:
            raise WGSLRefusal(
                "WGSL target requires GLSL buffer block resource "
                f"{node.name} to use a named struct type"
            )

    def validate_glsl_buffer_block_struct(self, node):
        members = list(getattr(node, "members", []) or [])
        for index, member in enumerate(members):
            member_type = getattr(member, "member_type", None)
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                continue
            if self.structured_buffer_element_type(
                self.array_element_type(member_type)
            ):
                raise WGSLRefusal(
                    "WGSL target does not support storage-buffer resource member "
                    f"{node.name}.{member.name} inside GLSL buffer blocks"
                )
            if (
                self.unsized_array_type(member_type) is not None
                and index != len(members) - 1
            ):
                raise WGSLRefusal(
                    "WGSL target requires runtime-sized array member "
                    f"{node.name}.{member.name} to be the final GLSL buffer block member"
                )

    def collect_glsl_buffer_block_struct_names(self, global_variables, functions):
        struct_names = set()
        for variable in global_variables:
            if not self.is_glsl_buffer_block_variable(variable):
                continue
            struct_name = self.glsl_buffer_block_struct_name(variable.var_type)
            if struct_name:
                struct_names.add(struct_name)
        for function in functions:
            for parameter in getattr(function, "parameters", []) or []:
                if not self.is_glsl_buffer_block_parameter(parameter):
                    continue
                struct_name = self.glsl_buffer_block_struct_name(parameter.param_type)
                if struct_name:
                    struct_names.add(struct_name)
        return struct_names

    def module_storage_access_modes(
        self, global_variables, stage_resource_parameters=()
    ):
        modes = {}
        for variable in global_variables:
            if self.is_glsl_buffer_block_variable(variable):
                modes[variable.name] = self.glsl_buffer_block_access(variable)
                continue
            access = self.structured_buffer_access(getattr(variable, "var_type", None))
            if access:
                modes[variable.name] = access
            elif "uniform" in {
                str(q).lower() for q in getattr(variable, "qualifiers", []) or []
            } and self.is_unsupported_uniform_array_stride(
                getattr(variable, "var_type", None),
                variable_name=getattr(variable, "name", None),
            ):
                modes[variable.name] = "read"
        for parameter in stage_resource_parameters:
            access = self.structured_buffer_access(
                getattr(parameter, "param_type", None)
            )
            if access:
                modes[parameter.name] = access
        return modes

    def used_access_mode_identifiers(
        self, global_variables, stage_resource_parameters, functions
    ):
        modes = set(
            self.module_storage_access_modes(
                global_variables, stage_resource_parameters
            ).values()
        )
        for variable in global_variables:
            qualifier_names = {
                str(qualifier).lower()
                for qualifier in getattr(variable, "qualifiers", []) or []
            }
            if (
                not self.structured_buffer_access(getattr(variable, "var_type", None))
                and not self.is_glsl_buffer_block_variable(variable)
                and ("buffer" in qualifier_names or "storage" in qualifier_names)
            ):
                modes.add("read_write")
        for function in functions:
            for parameter in getattr(function, "parameters", []) or []:
                if self.is_buffer_pointer_type(
                    getattr(parameter, "param_type", None),
                    getattr(parameter, "qualifiers", ()),
                ):
                    modes.add("read_write")
        modes.update(
            self.inferred_storage_texture_access_modes(
                global_variables, stage_resource_parameters, functions
            ).values()
        )
        for node in list(global_variables) + list(stage_resource_parameters):
            type_name = self.resource_type_name(self.declared_resource_type(node)) or ""
            if self.STORAGE_TEXTURE_DIMENSION_MAP.get(type_name) is None:
                continue
            explicit_access = explicit_image_access(
                node, self.generate_attribute_argument
            )
            if explicit_access is not None:
                modes.add(explicit_access)
                continue
            names = self.resource_qualifier_attribute_names(node)
            if "constant" in names:
                modes.add("read")
            elif names & {"rw", "coherent"}:
                modes.add("read_write")
        return modes & self.WGSL_ACCESS_MODE_IDENTIFIERS

    def inferred_storage_texture_access_modes(
        self, global_variables, stage_resource_parameters, functions
    ):
        storage_texture_names = {
            getattr(node, "name", "")
            for node in list(global_variables) + list(stage_resource_parameters)
            if self.STORAGE_TEXTURE_DIMENSION_MAP.get(
                self.resource_type_name(self.declared_resource_type(node)) or ""
            )
            is not None
        }
        storage_texture_names.discard("")
        if not storage_texture_names:
            return {}

        modes = {}
        for function in functions:
            body = getattr(function, "body", None)
            if body is None or not hasattr(body, "walk"):
                continue
            for node in body.walk():
                if not isinstance(node, FunctionCallNode):
                    continue
                access = self.storage_image_operation_access_requirement(
                    self.expression_name(node.function)
                )
                if access is None or not getattr(node, "arguments", None):
                    continue
                access_path = self.expression_access_path(node.arguments[0])
                if access_path is None:
                    continue
                root_name, path = access_path
                if path or root_name not in storage_texture_names:
                    continue
                modes[root_name] = merge_image_access_requirement(
                    modes.get(root_name), access
                )
        return modes

    def storage_image_operation_access_requirement(self, function_name):
        normalized_name = self.semantic_key(function_name)
        if normalized_name == "imageload":
            return "read"
        if normalized_name == "imagestore":
            return "write"
        if normalized_name.startswith("imageatomic"):
            return "read_write"
        return None

    def is_buffer_pointer_type(self, vtype, qualifiers=()):
        if not isinstance(vtype, PointerType):
            return False
        qualifier_names = {str(qualifier).lower() for qualifier in qualifiers or []}
        return bool(qualifier_names.intersection({"buffer", "storage"}))

    def buffer_pointer_element_type(self, vtype):
        if not isinstance(vtype, PointerType):
            raise WGSLRefusal("WGSL target expected a buffer pointer type")
        return vtype.pointee_type

    def buffer_pointer_storage_type(self, vtype):
        element = self.buffer_pointer_element_type(vtype)
        element_type = self.type_name_string(
            narrow_storage_word_type(element) or element
        )
        return f"array<{element_type}>"

    def buffer_pointer_parameter_type(self, vtype):
        return f"ptr<storage, {self.buffer_pointer_storage_type(vtype)}, read_write>"

    def validate_cbuffer_member(self, cbuffer, member):
        member_type = getattr(member, "member_type", None)
        element_type = self.array_element_type(member_type)
        resource_type_name = self.resource_type_name(element_type)
        if resource_type_name is not None and not self.is_resource_type_name(
            resource_type_name
        ):
            resource_type_name = None
        if resource_type_name:
            display_type_name = (
                str(element_type.name)
                if isinstance(element_type, NamedType)
                else str(element_type)
            )
            raise WGSLRefusal(
                "WGSL target does not support resource member "
                f"{cbuffer.name}.{member.name} of type {display_type_name} "
                "inside uniform buffers; declare resources as module-scope "
                "bindings instead"
            )
        if self.structured_buffer_element_type(element_type) is not None:
            raise WGSLRefusal(
                "WGSL target does not support storage-buffer resource member "
                f"{cbuffer.name}.{member.name} inside uniform buffers; "
                "declare StructuredBuffer resources as module-scope bindings"
            )
        if self.unsized_array_type(member_type) is not None:
            raise WGSLRefusal(
                "WGSL target does not support runtime-sized array member "
                f"{cbuffer.name}.{member.name} inside uniform buffers"
            )

    def uniform_buffer_struct_names(
        self, cbuffers, global_variables, stage_resource_parameters
    ):
        names = {
            getattr(cbuffer, "name", "")
            for cbuffer in cbuffers
            if getattr(cbuffer, "name", "")
        }
        for variable in global_variables:
            qualifier_names = {
                str(qualifier).lower()
                for qualifier in getattr(variable, "qualifiers", []) or []
            }
            if "uniform" not in qualifier_names:
                continue
            struct_name = self.struct_type_name(
                self.array_element_type(getattr(variable, "var_type", None))
            )
            if struct_name:
                names.add(struct_name)
        for parameter in stage_resource_parameters:
            uniform_type = self.stage_uniform_parameter_type(parameter)
            struct_name = self.struct_type_name(self.array_element_type(uniform_type))
            if struct_name:
                names.add(struct_name)
        names.discard("")
        return names

    def uniform_scalar_array_wrapper_names(self, cbuffers, structs):
        scalar_types = set()
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                scalar_type = self.uniform_scalar_array_element_type_name(
                    getattr(member, "member_type", None)
                )
                if scalar_type is not None:
                    scalar_types.add(scalar_type)
        for struct in structs:
            if getattr(struct, "name", "") not in self._uniform_buffer_struct_names:
                continue
            for member in getattr(struct, "members", []) or []:
                scalar_type = self.uniform_scalar_array_element_type_name(
                    getattr(member, "member_type", None)
                )
                if scalar_type is not None:
                    scalar_types.add(scalar_type)

        used_names = set(self._type_identifier_names.values())
        used_names.update(self._function_identifier_names.values())
        used_names.update(self._module_identifier_names.values())
        wrappers = {}
        suffixes = {"f32": "F32", "i32": "I32", "u32": "U32"}
        for scalar_type in sorted(scalar_types):
            base_name = f"UniformArrayElement{suffixes[scalar_type]}"
            wrapper_name = self.unique_wgsl_identifier(base_name, used_names)
            wrappers[scalar_type] = wrapper_name
            used_names.add(wrapper_name)
        return wrappers

    def is_uniform_scalar_array_access(self, array_expr):
        if isinstance(array_expr, IdentifierNode):
            if self.is_local_identifier(array_expr.name):
                return False
            member_type = self._cbuffer_member_types.get(array_expr.name)
            return self.uniform_scalar_array_wrapper_name(member_type) is not None
        if isinstance(array_expr, MemberAccessNode):
            object_type = self.array_element_type(
                self.expression_type(array_expr.object_expr)
            )
            struct_name = self.struct_type_name(object_type)
            if struct_name not in self._uniform_buffer_struct_names:
                return False
            member_type = self._struct_member_types.get(
                (struct_name, array_expr.member)
            )
            return self.uniform_scalar_array_wrapper_name(member_type) is not None
        return False

    def unsized_array_type(self, vtype):
        while isinstance(vtype, ArrayType):
            if vtype.size is None:
                return vtype
            vtype = vtype.element_type
        return None
