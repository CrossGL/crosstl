"""Storage textures, samplers and resource type names in the WGSL generator."""

from ..ast import NamedType
from ..refusals import WGSLRefusal
from .image_access_contracts import explicit_image_access


class WGSLStorageTextureMixin:
    """Storage texture members of ``WGSLCodeGen``."""

    def storage_texture_type(self, vtype, node):
        type_name = self.resource_type_name(vtype)
        dimension = self.STORAGE_TEXTURE_DIMENSION_MAP.get(type_name or "")
        if dimension is None:
            return None
        texture_format = self.storage_texture_format(node)
        access = self.storage_texture_access(node)
        return f"texture_storage_{dimension}<{texture_format}, {access}>"

    def storage_texture_format(self, node):
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            texture_format = self.STORAGE_TEXTURE_FORMAT_MAP.get(key)
            if texture_format is not None:
                return texture_format
        name = getattr(node, "name", "storage texture")
        raise WGSLRefusal(
            "WGSL target requires storage image resource "
            f"{name} to declare a representable image format such as "
            "layout(rgba8) or @rgba8"
        )

    def storage_texture_access(self, node):
        explicit_access = explicit_image_access(node, self.generate_attribute_argument)
        if explicit_access is not None:
            return explicit_access
        names = self.resource_qualifier_attribute_names(node)
        if "constant" in names:
            return "read"
        if names & {"rw", "coherent"}:
            return "read_write"
        name = getattr(node, "name", None)
        if name:
            inferred_access = self._inferred_storage_texture_access_modes.get(name)
            if inferred_access is not None:
                return inferred_access
        diagnostic_name = name or "storage texture"
        raise WGSLRefusal(
            "WGSL target requires storage image resource "
            f"{diagnostic_name} to declare an access mode "
            "(@readonly, @writeonly, or @readwrite) or use it in "
            "imageLoad/imageStore so access can be inferred"
        )

    def resource_qualifier_attribute_names(self, node):
        names = {
            self.semantic_key(str(qualifier))
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        names.update(
            self.semantic_key(str(getattr(attr, "name", attr)))
            for attr in getattr(node, "attributes", []) or []
        )
        return names

    def require_writable_storage_texture_resource(self, resource, function_name):
        resource_type = self.expression_type(resource)
        if (
            self.STORAGE_TEXTURE_DIMENSION_MAP.get(
                self.resource_type_name(resource_type) or ""
            )
            is None
        ):
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() to use a storage image "
                "resource"
            )
        access = self.storage_texture_access_for_expression(resource)
        if access == "read":
            raise WGSLRefusal(
                "WGSL target cannot store through read-only storage image "
                f"resource in {function_name}()"
            )

    def require_readable_storage_texture_resource(self, resource, function_name):
        resource_type = self.expression_type(resource)
        if (
            self.STORAGE_TEXTURE_DIMENSION_MAP.get(
                self.resource_type_name(resource_type) or ""
            )
            is None
        ):
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() to use a storage image "
                "resource"
            )
        access = self.storage_texture_access_for_expression(resource)
        if access == "write":
            raise WGSLRefusal(
                "WGSL target cannot load from write-only storage image "
                f"resource in {function_name}()"
            )

    def storage_texture_access_for_expression(self, expr):
        access_path = self.expression_access_path(expr)
        if access_path is None:
            return None
        root_name, path = access_path
        if path:
            return None
        return self._module_storage_texture_access_modes.get(root_name)

    def storage_texture_dimension_for_expression(self, expr):
        resource_type = self.expression_type(expr)
        return self.STORAGE_TEXTURE_DIMENSION_MAP.get(
            self.resource_type_name(resource_type) or ""
        )

    def storage_texture_coordinate_args(self, image, coords):
        coord_expr = self.generate_expression(coords)
        if self.storage_texture_dimension_for_expression(image) != "2d_array":
            return [coord_expr]
        return [f"({coord_expr}).xy", f"({coord_expr}).z"]

    def is_depth_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        return type_name in {
            "sampler2darrayshadow",
            "sampler2dshadow",
            "samplercubearrayshadow",
            "samplercubeshadow",
        }

    def is_sampler_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        return (
            type_name in self.SAMPLER_TYPE_NAMES
            or type_name in self.COMPARISON_SAMPLER_TYPE_NAMES
        )

    def is_comparison_sampler_type(self, vtype):
        return self.resource_type_name(vtype) in self.COMPARISON_SAMPLER_TYPE_NAMES

    def sampler_type(self, vtype):
        if self.is_comparison_sampler_type(vtype):
            return "sampler_comparison"
        return "sampler"

    def companion_sampler_type(self, texture_type):
        if self.is_depth_texture_type(texture_type):
            return "sampler_comparison"
        return "sampler"

    def resource_type_name(self, vtype):
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name).lower()
        if isinstance(vtype, str):
            return vtype.strip().lower()
        return None

    def struct_member_resource_type_name(self, member):
        member_type = getattr(member, "member_type", None)
        type_name = self.resource_type_name(member_type)
        if type_name is None or not self.is_resource_type_name(type_name):
            return None
        if isinstance(member_type, NamedType):
            return str(member_type.name)
        return str(member_type)

    def supported_struct_resource_member(self, member):
        return self.sampled_texture_type(
            getattr(member, "member_type", None)
        ) is not None or self.is_sampler_type(getattr(member, "member_type", None))
