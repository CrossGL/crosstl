"""Sampled texture arrays, specialization constant planning, stage interface arrays and matrices, and array size resolution in the WGSL generator."""

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    LiteralNode,
    MatrixType,
    MemberAccessNode,
)
from ..refusals import WGSLRefusal
from .wgsl_specialization_constants import plan_wgsl_specialization_constants


class WGSLResourceArrayMixin:
    """Resource array members of ``WGSLCodeGen``."""

    def sampled_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        if type_name is None:
            return None
        return self.SAMPLED_TEXTURE_TYPE_MAP.get(type_name)

    def sampled_texture_array_info(self, vtype):
        if not isinstance(vtype, ArrayType):
            return None
        element_type = self.array_element_type(vtype)
        texture_type = self.sampled_texture_type(element_type)
        if texture_type is None:
            return None
        return {"element_type": element_type, "texture_type": texture_type}

    def sampled_texture_array_element_binding(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        if not isinstance(expr.array_expr, IdentifierNode):
            return None
        array_info = self._sampled_texture_array_bindings.get(expr.array_expr.name)
        if array_info is None:
            return None
        index = self.resolve_integer_expression(expr.index_expr)
        if index is None:
            return None
        bindings = array_info["bindings"]
        if index < 0 or index >= len(bindings):
            raise WGSLRefusal(
                "WGSL target sampled texture resource array "
                f"{expr.array_expr.name} index {index} is outside the lowered "
                f"binding range 0..{len(bindings) - 1}"
            )
        return bindings[index]

    def sampled_texture_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        if not isinstance(expr.array_expr, IdentifierNode):
            return None
        array_info = self._sampled_texture_array_bindings.get(expr.array_expr.name)
        if array_info is None:
            return None
        return array_info

    def stage_interface_array_member_info(self, struct_name, member_name):
        if struct_name not in self._stage_interface_struct_names:
            return None
        if (struct_name, member_name) in self._clip_distance_members:
            # The clip_distances built-in is one array<f32, N> member (wgsl_clip_distances).
            return None
        member_type = self._struct_member_types.get((struct_name, member_name))
        return self.stage_interface_array_member_info_for_type(member_type)

    def plan_specialization_constants(self, ast, structs):
        """Plan the overrides of ``ast``'s specialization constants; constants that follow one stop folding.

        A stage-interface array is flattened to one member per element of its default size, as its interface
        locations are assigned from that size, so it is the one array a dependent constant may size outside
        workgroup memory.
        """
        flattened = [
            member.member_type
            for struct in structs
            if getattr(struct, "name", "") in self._stage_interface_struct_names
            for member in getattr(struct, "members", []) or []
            if self.stage_interface_array_member_info_for_type(
                getattr(member, "member_type", None)
            )
            is not None
        ]
        self._specialization_plan = plan_wgsl_specialization_constants(
            ast, self.type_name_string, flattened
        )
        dependent = self._specialization_plan.dependent
        self._specialization_default_values = {
            name: value
            for name, value in self._constant_values.items()
            if name in dependent
        }
        self._constant_values = {
            name: value
            for name, value in self._constant_values.items()
            if name not in dependent
        }

    def stage_interface_array_member_info_for_type(self, member_type):
        if not isinstance(member_type, ArrayType):
            return None
        size = self.resolve_integer_expression(member_type.size)
        if size is None:
            size = self.resolve_integer_expression(
                member_type.size, self._specialization_default_values
            )
        if size is None or size < 1:
            return None
        element_type = member_type.element_type
        if isinstance(element_type, ArrayType):
            return None
        try:
            element_type_name = self.type_name_string(element_type)
            scalar_type = self.scalar_type_name(element_type)
            vector_shape = self.vector_shape(element_type)
        except ValueError:
            return None
        if scalar_type not in {"f32", "i32", "u32"} and vector_shape is None:
            return None
        if vector_shape is not None and vector_shape[0] not in {"f32", "i32", "u32"}:
            return None
        return {
            "size": size,
            "element_type": element_type,
            "element_type_name": element_type_name,
        }

    def stage_interface_matrix_member_info(self, struct_name, member_name):
        if struct_name not in self._stage_interface_struct_names:
            return None
        member_type = self._struct_member_types.get((struct_name, member_name))
        return self.stage_interface_matrix_member_info_for_type(member_type)

    def stage_interface_matrix_member_info_for_type(self, member_type):
        if not isinstance(member_type, MatrixType):
            return None
        try:
            matrix_type_name = self.type_name_string(member_type)
            element_type_name = self.type_name_string(member_type.element_type)
        except ValueError:
            return None
        if element_type_name not in {"f32", "i32", "u32"}:
            return None
        return {
            "columns": member_type.cols,
            "rows": member_type.rows,
            "matrix_type_name": matrix_type_name,
            "column_type_name": f"vec{member_type.rows}<{element_type_name}>",
        }

    def stage_interface_array_field_name(self, struct_name, member_name, index):
        member_name = self.struct_member_identifier_name(struct_name, member_name)
        return f"{member_name}_{index}"

    def stage_interface_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = expr.array_expr
        if not isinstance(array_expr, MemberAccessNode):
            return None
        object_type = self.array_element_type(
            self.expression_type(array_expr.object_expr)
        )
        struct_name = self.struct_type_name(object_type)
        if struct_name is None:
            return None
        member_info = self.stage_interface_array_member_info(
            struct_name, array_expr.member
        )
        if member_info is None:
            return None
        return {
            **member_info,
            "struct_name": struct_name,
            "member_name": array_expr.member,
            "object_expr": array_expr.object_expr,
            "index_expr": expr.index_expr,
        }

    def stage_interface_array_access_field(self, access, index):
        object_expr = self.generate_expression(access["object_expr"])
        field_name = self.stage_interface_array_field_name(
            access["struct_name"], access["member_name"], index
        )
        return f"{object_expr}.{field_name}"

    def generate_stage_interface_array_access(self, expr):
        access = self.stage_interface_array_access(expr)
        if access is None:
            return None
        index = self.resolve_integer_expression(expr.index_expr)
        if index is not None:
            if index < 0 or index >= access["size"]:
                raise WGSLRefusal(
                    "WGSL target stage interface array "
                    f"{access['member_name']} index {index} is outside the lowered "
                    f"field range 0..{access['size'] - 1}"
                )
            return self.stage_interface_array_access_field(access, index)

        index_expr = self.generate_expression(expr.index_expr)
        result = self.stage_interface_array_access_field(access, 0)
        for field_index in range(1, access["size"]):
            field = self.stage_interface_array_access_field(access, field_index)
            result = f"select({result}, {field}, ({index_expr} == {field_index}))"
        return result

    def stage_interface_matrix_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_type = self.array_element_type(self.expression_type(expr.object_expr))
        struct_name = self.struct_type_name(object_type)
        if struct_name is None:
            return None
        member_info = self.stage_interface_matrix_member_info(struct_name, expr.member)
        if member_info is None:
            return None
        return {
            **member_info,
            "struct_name": struct_name,
            "member_name": expr.member,
            "object_expr": expr.object_expr,
        }

    def stage_interface_matrix_access_field(self, access, index):
        object_expr = self.generate_expression(access["object_expr"])
        field_name = self.stage_interface_array_field_name(
            access["struct_name"], access["member_name"], index
        )
        return f"{object_expr}.{field_name}"

    def generate_stage_interface_matrix_access(self, expr):
        access = self.stage_interface_matrix_access(expr)
        if access is None:
            return None
        # Each column reads the object again, so a read whose object has
        # side effects would mix columns from different elements.
        self.require_repeatable_matrix_operands(
            "read of a split stage-interface member",
            (access["object_expr"],),
        )
        columns = ", ".join(
            self.stage_interface_matrix_access_field(access, index)
            for index in range(access["columns"])
        )
        return f"{access['matrix_type_name']}({columns})"

    def resolve_array_size(self, size, resource_name):
        resolved = self.resolve_integer_expression(size)
        if resolved is not None:
            return resolved
        if size is None:
            return None
        raise WGSLRefusal(
            "WGSL target cannot lower sampled texture resource array "
            f"{resource_name} with non-constant size {self.generate_expression(size)}; "
            "WebGPU/WGSL requires project-level binding expansion with a known "
            "element count"
        )

    def resolve_integer_expression(self, expr, constant_values=None):
        constant_values = (
            self._constant_values if constant_values is None else constant_values
        )
        if expr is None:
            return None
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            try:
                return int(expr, 0)
            except ValueError:
                return constant_values.get(expr)
        if isinstance(expr, LiteralNode):
            try:
                return int(expr.value)
            except (TypeError, ValueError):
                return None
        if isinstance(expr, IdentifierNode):
            return constant_values.get(expr.name)
        return None

    def binding_group_for_resource_array(self, node):
        group, _binding = self.resolved_explicit_binding_components(node)
        return str(group)
