"""Struct declarations and generic struct monomorphization in the WGSL generator."""

import copy
import re

from ..ast import (
    ArrayType,
    FunctionCallNode,
    GenericType,
    IdentifierNode,
    LiteralNode,
    MatrixType,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    StructMemberNode,
    StructNode,
    UnaryOpNode,
    VectorType,
)
from ..refusals import WGSLRefusal
from .enum_utils import sanitize_type_name
from .wgsl_clip_distances import binds_clip_distances, clip_distances_member_type
from .wgsl_primitive_index import binds_primitive_index
from .wgsl_stage_io import (
    member_interpolation,
    with_interpolation,
)
from .wgsl_unsigned_builtin_members import (
    scalar_builtin_of,
    unsigned_builtin_member_type,
)


class WGSLGenericStructMixin:
    """Generic struct members of ``WGSLCodeGen``."""

    def generate_struct(self, node):
        if getattr(node, "generic_params", None):
            raise WGSLRefusal(
                "WGSL target cannot emit unspecialized generic struct "
                f"{node.name}; use concrete type arguments before WGSL generation"
            )
        if node.name in self._glsl_buffer_block_struct_names:
            self.validate_glsl_buffer_block_struct(node)
        lines = [f"struct {self.type_identifier_name(node.name)} {{"]
        for member in node.members:
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                if node.name in self._glsl_buffer_block_struct_names:
                    raise WGSLRefusal(
                        "WGSL target does not support resource member "
                        f"{node.name}.{member.name} of type {resource_type_name} "
                        "inside GLSL buffer blocks; declare textures, samplers, "
                        "and storage resources as module-scope bindings"
                    )
                if self.supported_struct_resource_member(member):
                    continue
                raise WGSLRefusal(
                    "WGSL target does not support resource member "
                    f"{node.name}.{member.name} of type {resource_type_name}; "
                    "declare textures, samplers, and storage resources as "
                    "module-scope bindings instead of user-struct fields"
                )
            direction = self.struct_member_direction(node.name)
            attributes = self.wgsl_attributes(member.attributes, direction=direction)
            binding_kind, explicit_location = self.member_stage_binding(
                member, direction
            )
            builtin = self._stage_interface_builtin_members.get(
                (node.name, member.name)
            )
            if builtin is not None:
                attributes = f"@builtin({builtin})"
                binding_kind = "builtin"
                explicit_location = None
            implicit_location = self._stage_interface_member_locations.get(
                (node.name, member.name)
            )
            if implicit_location is not None:
                attributes = " ".join(
                    part
                    for part in (f"@location({implicit_location})", attributes)
                    if part
                )
            member_name = self.struct_member_identifier_name(node.name, member.name)
            member_type = member.member_type
            unsigned_builtin = scalar_builtin_of(attributes)
            if unsigned_builtin is not None:
                member_type = unsigned_builtin_member_type(
                    self, node.name, member, unsigned_builtin
                )
                if binds_primitive_index(attributes):
                    self._uses_primitive_index = True
            elif binds_clip_distances(attributes):
                member_type = clip_distances_member_type(self, node.name, member)
            stage_array = self.stage_interface_array_member_info_for_type(member_type)
            base_location = (
                implicit_location
                if implicit_location is not None
                else explicit_location if binding_kind == "location" else None
            )
            if stage_array is not None and base_location is not None:
                for index in range(stage_array["size"]):
                    field_name = self.stage_interface_array_field_name(
                        node.name, member.name, index
                    )
                    if "@location(" in attributes:
                        field_attributes = re.sub(
                            r"@location\(\d+\)",
                            f"@location({base_location + index})",
                            attributes,
                            count=1,
                        )
                    else:
                        field_attributes = " ".join(
                            part
                            for part in (
                                f"@location({base_location + index})",
                                attributes,
                            )
                            if part
                        )
                    field_attributes = with_interpolation(
                        field_attributes,
                        stage_array["element_type_name"],
                        member_interpolation(
                            self, node.name, member, base_location + index
                        ),
                        f"member {node.name}.{member.name}",
                    )
                    prefix = f"{field_attributes} " if field_attributes else ""
                    lines.append(
                        f"    {prefix}{field_name}: {stage_array['element_type_name']},"
                    )
                continue
            stage_matrix = self.stage_interface_matrix_member_info_for_type(member_type)
            if stage_matrix is not None and base_location is not None:
                for index in range(stage_matrix["columns"]):
                    field_name = self.stage_interface_array_field_name(
                        node.name, member.name, index
                    )
                    if "@location(" in attributes:
                        field_attributes = re.sub(
                            r"@location\(\d+\)",
                            f"@location({base_location + index})",
                            attributes,
                            count=1,
                        )
                    else:
                        field_attributes = " ".join(
                            part
                            for part in (
                                f"@location({base_location + index})",
                                attributes,
                            )
                            if part
                        )
                    field_attributes = with_interpolation(
                        field_attributes,
                        stage_matrix["column_type_name"],
                        member_interpolation(
                            self, node.name, member, base_location + index
                        ),
                        f"member {node.name}.{member.name}",
                    )
                    prefix = f"{field_attributes} " if field_attributes else ""
                    lines.append(
                        f"    {prefix}{field_name}: {stage_matrix['column_type_name']},"
                    )
                continue
            if node.name in self._uniform_buffer_struct_names:
                if self.uniform_scalar_array_wrapper_name(member_type) is not None:
                    attributes = " ".join(
                        attribute
                        for attribute in (attributes, "@align(16)")
                        if attribute
                    )
                self.require_atomic_address_space(
                    node.name, member.name, member_type, "uniform"
                )
                member_type_name = self.uniform_buffer_member_type_name(member_type)
            else:
                member_type_name = self.type_name_string(
                    self.atomic_declaration_type(node.name, member.name, member_type)
                )
            attributes = with_interpolation(
                attributes,
                member_type_name,
                member_interpolation(self, node.name, member, base_location),
                f"member {node.name}.{member.name}",
            )
            prefix = f"{attributes} " if attributes else ""
            lines.append(f"    {prefix}{member_name}: {member_type_name},")
        lines.append("};")
        return "\n".join(lines)

    def collect_generic_struct_definitions(self, structs):
        return {
            struct.name: struct
            for struct in structs
            if getattr(struct, "name", None) and getattr(struct, "generic_params", None)
        }

    def monomorphized_structs(self, ast, target_stage, source_structs):
        concrete_structs = [
            struct
            for struct in source_structs
            if not getattr(struct, "generic_params", None)
        ]
        self._generic_struct_specializations = {}
        self._generic_struct_specialized_names = {}

        if not self._generic_struct_definitions:
            return concrete_structs

        generic_parameter_names = self.generic_parameter_names(ast)
        pending = []

        def enqueue_specialization(base_name, generic_args):
            if base_name not in self._generic_struct_definitions:
                return
            if not generic_args:
                return
            vtype = NamedType(base_name, list(generic_args))
            if self.type_node_contains_generic_parameter(
                vtype, generic_parameter_names
            ):
                return
            key = self.generic_struct_type_key(vtype.name, vtype.generic_args)
            if (
                key in self._generic_struct_specializations
                or key in self._generic_struct_specialized_names
            ):
                return
            self._generic_struct_specialized_names[key] = (
                self.generic_struct_specialized_name_from_key(key)
            )
            pending.append(vtype)

        def enqueue_from_ast_value(value):
            if isinstance(value, NamedType):
                enqueue_specialization(value.name, value.generic_args)
                return
            if not isinstance(value, FunctionCallNode):
                return
            function = getattr(value, "function", None)
            if not isinstance(function, IdentifierNode):
                return
            parsed = self.generic_constructor_parts(function.name)
            if parsed is None:
                return
            enqueue_specialization(*parsed)

        self.visit_ast_values(ast, enqueue_from_ast_value)
        index = 0
        while index < len(pending):
            vtype = pending[index]
            index += 1
            key = self.generic_struct_type_key(vtype.name, vtype.generic_args)
            if key in self._generic_struct_specializations:
                continue
            specialization = self.build_generic_struct_specialization(vtype, key)
            self._generic_struct_specializations[key] = specialization
            self.visit_ast_values(specialization, enqueue_from_ast_value)

        return concrete_structs + list(self._generic_struct_specializations.values())

    def build_generic_struct_specialization(self, vtype, key):
        definition = self._generic_struct_definitions[vtype.name]
        generic_params = [
            param.name
            for param in getattr(definition, "generic_params", []) or []
            if getattr(param, "name", None)
        ]
        if len(vtype.generic_args) != len(generic_params):
            expected = len(generic_params)
            actual = len(vtype.generic_args)
            raise WGSLRefusal(
                "WGSL target cannot specialize generic struct "
                f"{vtype.name}; expected {expected} type argument"
                f"{'s' if expected != 1 else ''}, got {actual}"
            )

        substitutions = {
            param: copy.deepcopy(arg)
            for param, arg in zip(generic_params, vtype.generic_args)
        }
        members = []
        for member in getattr(definition, "members", []) or []:
            if not isinstance(member, StructMemberNode):
                raise WGSLRefusal(
                    "WGSL target cannot specialize generic struct "
                    f"{definition.name}; member {getattr(member, 'name', '<unnamed>')} "
                    f"uses unsupported {type(member).__name__}"
                )
            specialized = copy.copy(member)
            specialized.member_type = self.substitute_generic_struct_type(
                member.member_type, substitutions
            )
            members.append(specialized)

        return StructNode(
            self._generic_struct_specialized_names[key],
            members,
            attributes=list(getattr(definition, "attributes", []) or []),
            inheritance=list(getattr(definition, "inheritance", []) or []),
            visibility=getattr(definition, "visibility", "public"),
        )

    def substitute_generic_struct_type(self, vtype, substitutions):
        if isinstance(vtype, NamedType):
            if not vtype.generic_args and vtype.name in substitutions:
                return copy.deepcopy(substitutions[vtype.name])
            return NamedType(
                vtype.name,
                [
                    self.substitute_generic_struct_type(arg, substitutions)
                    for arg in vtype.generic_args
                ],
            )
        if isinstance(vtype, VectorType):
            return VectorType(
                self.substitute_generic_struct_type(vtype.element_type, substitutions),
                vtype.size,
            )
        if isinstance(vtype, MatrixType):
            return MatrixType(
                self.substitute_generic_struct_type(vtype.element_type, substitutions),
                vtype.rows,
                vtype.cols,
            )
        if isinstance(vtype, ArrayType):
            size = vtype.size
            # A size may be a value parameter: ``T items[N]`` in ``generic<T, N> struct Ring``.
            size_name = size.name if isinstance(size, IdentifierNode) else size
            if isinstance(size_name, str) and size_name.strip() in substitutions:
                size = substitutions[size_name.strip()]
            return ArrayType(
                self.substitute_generic_struct_type(vtype.element_type, substitutions),
                copy.deepcopy(size),
            )
        if isinstance(vtype, PointerType):
            return PointerType(
                self.substitute_generic_struct_type(vtype.pointee_type, substitutions),
                is_mutable=vtype.is_mutable,
            )
        if isinstance(vtype, ReferenceType):
            return ReferenceType(
                self.substitute_generic_struct_type(
                    vtype.referenced_type, substitutions
                ),
                is_mutable=vtype.is_mutable,
            )
        if isinstance(vtype, GenericType) and vtype.name in substitutions:
            return copy.deepcopy(substitutions[vtype.name])
        return copy.deepcopy(vtype)

    def generic_parameter_names(self, value):
        names = set()

        def collect(current):
            for param in getattr(current, "generic_params", []) or []:
                name = getattr(param, "name", None)
                if name:
                    names.add(name)

        self.visit_ast_values(value, collect)
        return names

    def type_node_contains_generic_parameter(self, vtype, generic_parameter_names):
        if isinstance(vtype, str):
            key = self.generic_type_key(vtype)
            if key in generic_parameter_names:
                return True
            parsed = self.generic_constructor_parts(vtype)
            if parsed is None:
                return False
            _, generic_args = parsed
            return any(
                self.type_node_contains_generic_parameter(arg, generic_parameter_names)
                for arg in generic_args
            )
        if isinstance(vtype, NamedType):
            if not vtype.generic_args and vtype.name in generic_parameter_names:
                return True
            return any(
                self.type_node_contains_generic_parameter(arg, generic_parameter_names)
                for arg in vtype.generic_args
            )
        if isinstance(vtype, GenericType):
            return vtype.name in generic_parameter_names
        if isinstance(vtype, VectorType):
            return self.type_node_contains_generic_parameter(
                vtype.element_type, generic_parameter_names
            )
        if isinstance(vtype, MatrixType):
            return self.type_node_contains_generic_parameter(
                vtype.element_type, generic_parameter_names
            )
        if isinstance(vtype, ArrayType):
            return self.type_node_contains_generic_parameter(
                vtype.element_type, generic_parameter_names
            )
        if isinstance(vtype, PointerType):
            return self.type_node_contains_generic_parameter(
                vtype.pointee_type, generic_parameter_names
            )
        if isinstance(vtype, ReferenceType):
            return self.type_node_contains_generic_parameter(
                vtype.referenced_type, generic_parameter_names
            )
        return False

    def generic_struct_type_key(self, base_name, generic_args):
        return (
            str(base_name),
            tuple(self.generic_type_key(arg) for arg in generic_args),
        )

    def generic_type_key(self, vtype):
        if isinstance(vtype, LiteralNode):
            # The argument of a value parameter: ``4`` in ``Ring<float, 4>``.
            if isinstance(vtype.value, bool):
                return "true" if vtype.value else "false"
            return str(vtype.value)
        if (
            isinstance(vtype, UnaryOpNode)
            and getattr(vtype, "operator", None) == "-"
            and isinstance(getattr(vtype, "operand", None), LiteralNode)
        ):
            return f"-{self.generic_type_key(vtype.operand)}"
        if isinstance(vtype, PrimitiveType):
            return self.PRIMITIVE_TYPE_MAP.get(vtype.name.lower(), vtype.name)
        if isinstance(vtype, VectorType):
            return f"vec{vtype.size}<{self.generic_type_key(vtype.element_type)}>"
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.generic_type_key(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            element = self.generic_type_key(vtype.element_type)
            if vtype.size is None:
                return f"array<{element}>"
            return f"array<{element}, {vtype.size}>"
        if isinstance(vtype, PointerType):
            return f"ptr<{self.generic_type_key(vtype.pointee_type)}>"
        if isinstance(vtype, ReferenceType):
            return self.generic_type_key(vtype.referenced_type)
        if isinstance(vtype, NamedType):
            if not vtype.generic_args:
                return self.map_type_name(vtype.name)
            args = ", ".join(self.generic_type_key(arg) for arg in vtype.generic_args)
            return f"{vtype.name}<{args}>"
        if isinstance(vtype, GenericType):
            return vtype.name
        if isinstance(vtype, str):
            match = re.fullmatch(r"\s*([A-Za-z_][0-9A-Za-z_]*)\s*<(.+)>\s*", vtype)
            if match is not None:
                args = self.split_generic_constructor_arguments(match.group(2))
                if args is not None:
                    rendered_args = ", ".join(
                        self.generic_type_key(arg) for arg in args
                    )
                    return f"{match.group(1)}<{rendered_args}>"
        return self.map_type_name(str(vtype))

    def generic_struct_specialized_name_from_key(self, key):
        base_name, generic_args = key
        suffix = "_".join(sanitize_type_name(arg) for arg in generic_args)
        return f"{sanitize_type_name(base_name)}_{suffix}" if suffix else base_name

    def generic_struct_specialized_type_name(self, vtype):
        generic_parameter_names = self.generic_parameter_names(
            self._generic_struct_definitions.get(vtype.name)
        )
        if self.type_node_contains_generic_parameter(vtype, generic_parameter_names):
            args = ", ".join(self.generic_type_key(arg) for arg in vtype.generic_args)
            raise WGSLRefusal(
                "WGSL target cannot specialize generic struct "
                f"{vtype.name}<{args}> because at least one type argument is "
                "still generic"
            )
        key = self.generic_struct_type_key(vtype.name, vtype.generic_args)
        if key not in self._generic_struct_specialized_names:
            raise WGSLRefusal(
                "WGSL target cannot emit generic struct "
                f"{self.generic_type_key(vtype)}; no concrete specialization was "
                "collected before WGSL generation"
            )
        return self.type_identifier_name(self._generic_struct_specialized_names[key])
