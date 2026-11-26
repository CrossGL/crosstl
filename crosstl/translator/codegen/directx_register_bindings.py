"""Texture offsets and register bindings in the HLSL generator: reserved constants, global bindings and ranges, labels, integer literals, comparison samplers, metadata and explicit cbuffer registers."""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from .array_utils import evaluate_literal_int_expression
from .directx_values import (
    DirectXTextureOffsetError,
    _DirectXResourceRegisterAllocator,
)
from .hlsl_register_slots import next_shared_resource_slot
from .image_access_contracts import (
    is_projected_texture_basic_offset_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_lod_offset_operation,
    is_texture_compare_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_lod_offset_operation,
)
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class DirectXRegisterBindingMixin:
    """Register binding members of ``HLSLCodeGen``."""

    def global_resource_register_metadata(self, node):
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            if lowered_block["readonly"]:
                prefix = "t"
                attribute_names = {"binding", "texture"}
            else:
                prefix = "u"
                attribute_names = {"binding", "texture", "uav"}
        else:
            if self.is_glsl_buffer_block_variable(node, vtype):
                return None
            mapped_type = self.hlsl_struct_buffer_resource_type(
                node, vtype
            ) or self.map_resource_type_with_format(vtype, node)
            if (
                mapped_type.startswith("Texture")
                or self.is_hlsl_feedback_texture_type(mapped_type)
                or self.is_multisample_storage_image_resource_type(mapped_type)
                or self.is_hlsl_acceleration_structure_type(mapped_type)
            ):
                if self.is_hlsl_feedback_texture_type(mapped_type):
                    prefix = "u"
                    attribute_names = {"binding", "texture", "uav"}
                else:
                    prefix = "t"
                    attribute_names = {"binding", "texture"}
            elif self.is_hlsl_rw_texture_type(mapped_type):
                prefix = "u"
                attribute_names = {"binding", "texture", "uav"}
            elif self.is_hlsl_uav_buffer_type(mapped_type):
                prefix = "u"
                attribute_names = {"binding", "buffer", "texture", "uav"}
            elif self.is_hlsl_constant_buffer_type(mapped_type):
                prefix = "b"
                attribute_names = {"binding", "buffer"}
            elif self.is_hlsl_texture_buffer_type(mapped_type):
                prefix = "t"
                attribute_names = {"binding", "buffer", "texture"}
            elif self.is_hlsl_readonly_buffer_type(mapped_type):
                prefix = "t"
                attribute_names = {"binding", "buffer", "texture"}
            elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
                prefix = "s"
                attribute_names = {"binding", "sampler"}
            else:
                return None

        binding = self.explicit_resource_binding_index(node, attribute_names, (prefix,))
        if binding is None:
            return None
        space = self.explicit_resource_register_space(node)
        return prefix, binding, space, resource_count, var_name

    def new_directx_resource_register_allocator(self):
        allocator = _DirectXResourceRegisterAllocator()
        self.directx_resource_register_allocator = allocator
        return allocator

    def reserve_explicit_global_resource_registers(
        self, global_vars, used_registers, cbuffers=()
    ):
        self.directx_resource_register_overrides = {}
        relocatable_metadata = []
        # Fixed cbuffer registers are emitted after the globals, but an implicitly bound global resource must not
        # take them first, so they are reserved here too (generate_cbuffers re-reserves them idempotently).
        fixed_cbuffers = [
            (node, self.cbuffer_resource_register_metadata(node))
            for node in cbuffers
            if not self.directx_binding_allows_register_relocation(node)
        ]
        candidates = [
            (node, self.global_resource_register_metadata(node)) for node in global_vars
        ] + fixed_cbuffers
        for node, metadata in candidates:
            if metadata is None:
                continue
            if self.directx_binding_allows_register_relocation(node):
                relocatable_metadata.append((node, metadata))
                continue
            prefix, binding, space, resource_count, var_name = metadata
            self.reserve_resource_register_range(
                used_registers,
                prefix,
                binding,
                resource_count,
                var_name,
                space,
                provenance=self.directx_resource_binding_provenance(node),
            )

        for node, metadata in relocatable_metadata:
            prefix, binding, space, resource_count, var_name = metadata
            resolved_binding = self.reserve_relocatable_resource_register_range(
                used_registers,
                prefix,
                binding,
                resource_count,
                var_name,
                space,
                provenance=self.directx_resource_binding_provenance(node),
            )
            if resolved_binding != binding:
                self.record_directx_resource_register_override(
                    node, prefix, resolved_binding
                )

    def cbuffer_resource_register_metadata(self, node):
        register_prefix = "t" if self.is_hlsl_tbuffer_node(node) else "b"
        binding = self.explicit_resource_binding_index(
            node, {"binding", "buffer"}, (register_prefix,)
        )
        if binding is None:
            return None
        space = self.explicit_resource_register_space(node)
        return register_prefix, binding, space, 1, node.name

    def reserve_explicit_cbuffer_registers(self, cbuffers, used_registers):
        relocatable_metadata = []
        for node in cbuffers:
            metadata = self.cbuffer_resource_register_metadata(node)
            if metadata is None:
                continue
            if self.directx_binding_allows_register_relocation(node):
                relocatable_metadata.append((node, metadata))
                continue
            prefix, binding, space, resource_count, var_name = metadata
            self.reserve_resource_register_range(
                used_registers,
                prefix,
                binding,
                resource_count,
                var_name,
                space,
                provenance=self.directx_resource_binding_provenance(node),
            )

        for node, metadata in relocatable_metadata:
            prefix, binding, space, resource_count, var_name = metadata
            resolved_binding = self.reserve_relocatable_resource_register_range(
                used_registers,
                prefix,
                binding,
                resource_count,
                var_name,
                space,
                provenance=self.directx_resource_binding_provenance(node),
            )
            if resolved_binding != binding:
                self.record_directx_resource_register_override(
                    node, prefix, resolved_binding
                )

    def directx_binding_allows_register_relocation(self, node):
        attr_names = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        if "register" in attr_names:
            return False
        if self.directx_relocatable_binding_annotation(node):
            return True
        return "binding" in attr_names and "set" in attr_names

    def directx_relocatable_binding_annotation(self, node):
        if not hasattr(node, "get_annotation"):
            return False
        return bool(node.get_annotation("directx.relocatable_binding", False))

    def reserve_relocatable_resource_register_range(
        self,
        used_registers,
        register_prefix,
        preferred,
        count,
        name,
        space=None,
        provenance=None,
    ):
        if isinstance(used_registers, _DirectXResourceRegisterAllocator):
            count = max(count or 1, 1)
            end = preferred + count - 1
            namespace = (register_prefix, space)
            if used_registers.has_allocation(namespace, preferred, end, name):
                return preferred

            if used_registers.range_is_available(namespace, preferred, end):
                binding = preferred
            else:
                cursor = used_registers.relocation_cursors.get(namespace, preferred)
                binding = used_registers.next_available(
                    namespace, max(preferred, cursor), count
                )
            self.reserve_resource_register_range(
                used_registers,
                register_prefix,
                binding,
                count,
                name,
                space,
                provenance=provenance,
            )
            used_registers.advance_relocation_cursor(namespace, binding, count)
            return binding

        try:
            self.reserve_resource_register_range(
                used_registers,
                register_prefix,
                preferred,
                count,
                name,
                space,
                provenance=provenance,
            )
            return preferred
        except ValueError:
            cursor = {space: preferred}
            binding = self.next_available_resource_register(
                used_registers, register_prefix, cursor, space, count
            )
            self.reserve_resource_register_range(
                used_registers,
                register_prefix,
                binding,
                count,
                name,
                space,
                provenance=provenance,
            )
            return binding

    def record_directx_resource_register_override(self, node, register_prefix, binding):
        self.directx_resource_register_overrides.setdefault(id(node), {})[
            register_prefix
        ] = binding

    def next_resource_register(self, register_cursors, space):
        return register_cursors.get(space, 0)

    def next_available_resource_register(
        self, used_registers, register_prefix, register_cursors, space, count
    ):
        count = max(count or 1, 1)
        binding = self.next_resource_register(register_cursors, space)
        if isinstance(used_registers, _DirectXResourceRegisterAllocator):
            return next_shared_resource_slot(
                used_registers, register_prefix, register_cursors, space, binding, count
            )

        ranges = used_registers.get((register_prefix, space), [])
        while True:
            end = binding + count - 1
            conflict_end = None
            for used_start, used_end, *_ in ranges:
                if binding <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding
            binding = conflict_end + 1

    def advance_resource_register(self, register_cursors, space, start, count):
        count = max(count or 1, 1)
        register_cursors[space] = max(register_cursors.get(space, 0), start + count)

    def reserve_resource_register_range(
        self,
        used_registers,
        register_prefix,
        start,
        count,
        name,
        space=None,
        provenance=None,
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        namespace = (register_prefix, space)
        if isinstance(used_registers, _DirectXResourceRegisterAllocator):
            if used_registers.has_allocation(namespace, start, end, name):
                return
            used_record = used_registers.conflicting_allocation(namespace, start, end)
            if used_record is None:
                used_registers.add_allocation(namespace, start, end, name, provenance)
                return

            used_start, used_end, used_name = used_record[:3]
            used_provenance = used_record[3] if len(used_record) > 3 else None
            self.raise_resource_register_conflict(
                register_prefix,
                start,
                end,
                name,
                provenance,
                space,
                used_start,
                used_end,
                used_name,
                used_provenance,
            )

        ranges = used_registers.setdefault(namespace, [])
        for used_record in ranges:
            used_start, used_end, used_name = used_record[:3]
            used_provenance = used_record[3] if len(used_record) > 3 else None
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                self.raise_resource_register_conflict(
                    register_prefix,
                    start,
                    end,
                    name,
                    provenance,
                    space,
                    used_start,
                    used_end,
                    used_name,
                    used_provenance,
                )
        ranges.append((start, end, name, provenance))

    def raise_resource_register_conflict(
        self,
        register_prefix,
        start,
        end,
        name,
        provenance,
        space,
        used_start,
        used_end,
        used_name,
        used_provenance,
    ):
        provenance_details = self.resource_binding_conflict_provenance_details(
            name,
            provenance,
            used_name,
            used_provenance,
        )
        provenance_suffix = f" ({provenance_details})" if provenance_details else ""
        raise DirectXRefusal(
            f"Conflicting DirectX resource binding for '{name}': "
            f"{self.resource_register_range_label(register_prefix, start, end, space)} "
            f"overlaps '{used_name}' "
            f"{self.resource_register_range_label(register_prefix, used_start, used_end, space)}"
            f"{provenance_suffix}"
        )

    def resource_register_range_label(self, register_prefix, start, end, space=None):
        if start == end:
            label = f"{register_prefix}{start}"
        else:
            label = f"{register_prefix}{start}-{register_prefix}{end}"
        if space:
            return f"{label}, {space}"
        return label

    def resource_binding_conflict_provenance_details(
        self, name, provenance, used_name, used_provenance
    ):
        details = []
        if provenance:
            details.append(f"{name}: {provenance}")
        if used_provenance:
            details.append(f"{used_name}: {used_provenance}")
        return "; ".join(details)

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def hlsl_texture_offset_type_dimension(self, type_value):
        type_name = self.type_name_string(type_value)
        normalized = "".join(str(type_name or "").split()).lower()
        if normalized in {
            "char",
            "short",
            "int",
            "long",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uchar",
            "ushort",
            "uint",
            "ulong",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "i8",
            "i16",
            "i32",
            "u8",
            "u16",
            "u32",
        }:
            return 1
        match = re.fullmatch(r"(?:i|u)?vec([234])(?:<(?:i|u)\d+>)?", normalized)
        if match is None:
            match = re.fullmatch(r"(?:int|uint)([234])", normalized)
        return int(match.group(1)) if match is not None else None

    def hlsl_texture_offset_constructor_parts(self, expr):
        if isinstance(expr, FunctionCallNode):
            return self.function_call_name(expr), list(
                getattr(expr, "arguments", getattr(expr, "args", [])) or []
            )
        if isinstance(expr, ConstructorNode):
            return self.type_name_string(getattr(expr, "constructor_type", None)), list(
                getattr(expr, "arguments", []) or []
            )
        return None, None

    def hlsl_texture_offset_constant_value(
        self,
        expr,
        *,
        scalar_constants=None,
        offset_constants=None,
    ):
        scalar_constants = (
            self.hlsl_current_visible_int_constants()
            if scalar_constants is None
            else scalar_constants
        )
        offset_constants = (
            self.current_hlsl_texture_offset_constants or {}
            if offset_constants is None
            else offset_constants
        )
        scalar_constants = dict(scalar_constants)
        scalar_constants.update(
            {
                name: value[0]
                for name, value in offset_constants.items()
                if len(value) == 1
            }
        )

        if expr is None:
            return None
        if isinstance(expr, bool):
            return None
        if isinstance(expr, int):
            return (expr,)
        if isinstance(expr, str):
            if expr in offset_constants:
                return tuple(offset_constants[expr])
            scalar = self.literal_int_value(expr, scalar_constants)
            return (int(scalar),) if scalar is not None else None

        name = getattr(expr, "name", None)
        if isinstance(name, str):
            if name in offset_constants:
                return tuple(offset_constants[name])
            if name in scalar_constants:
                return (int(scalar_constants[name]),)

        constructor_name, constructor_args = self.hlsl_texture_offset_constructor_parts(
            expr
        )
        constructor_dimension = self.hlsl_texture_offset_type_dimension(
            constructor_name
        )
        if constructor_dimension is not None:
            components = []
            for argument in constructor_args:
                value = self.hlsl_texture_offset_constant_value(
                    argument,
                    scalar_constants=scalar_constants,
                    offset_constants=offset_constants,
                )
                if value is None:
                    return None
                components.extend(value)
            if len(components) == 1 and constructor_dimension > 1:
                components *= constructor_dimension
            if len(components) != constructor_dimension:
                return None
            return tuple(int(component) for component in components)

        if isinstance(expr, CastNode):
            dimension = self.hlsl_texture_offset_type_dimension(
                getattr(expr, "target_type", None)
            )
            value = self.hlsl_texture_offset_constant_value(
                getattr(expr, "expression", None),
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            if dimension is None or value is None:
                return None
            if len(value) == 1 and dimension > 1:
                value *= dimension
            return tuple(value) if len(value) == dimension else None

        if isinstance(expr, UnaryOpNode):
            value = self.hlsl_texture_offset_constant_value(
                getattr(expr, "operand", None),
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if value is None or operator not in {"+", "-", "~"}:
                return None
            if operator == "+":
                return value
            if operator == "-":
                return tuple(-component for component in value)
            return tuple(~component for component in value)

        if isinstance(expr, BinaryOpNode):
            left = self.hlsl_texture_offset_constant_value(
                getattr(expr, "left", None),
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            right = self.hlsl_texture_offset_constant_value(
                getattr(expr, "right", None),
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            if left is None or right is None:
                return None
            if len(left) == 1 and len(right) > 1:
                left *= len(right)
            if len(right) == 1 and len(left) > 1:
                right *= len(left)
            if len(left) != len(right):
                return None
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            values = []
            for lhs, rhs in zip(left, right):
                value = self.hlsl_texture_offset_binary_component(lhs, operator, rhs)
                if value is None:
                    return None
                values.append(value)
            return tuple(values)

        if isinstance(expr, TernaryOpNode):
            condition = self.literal_int_value(
                getattr(expr, "condition", None), scalar_constants
            )
            if condition is None:
                return None
            selected = (
                getattr(expr, "true_expr", None)
                if condition
                else getattr(expr, "false_expr", None)
            )
            return self.hlsl_texture_offset_constant_value(
                selected,
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )

        if isinstance(expr, (MemberAccessNode, SwizzleNode)):
            components = getattr(expr, "member", None)
            vector_expr = getattr(expr, "object_expr", None)
            if components is None:
                components = getattr(expr, "components", None)
                vector_expr = getattr(expr, "vector_expr", None)
            value = self.hlsl_texture_offset_constant_value(
                vector_expr,
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            indices = self.hlsl_texture_offset_swizzle_indices(components)
            if (
                value is None
                or indices is None
                or any(i >= len(value) for i in indices)
            ):
                return None
            return tuple(value[index] for index in indices)

        scalar = self.literal_int_value(expr, scalar_constants)
        return (int(scalar),) if scalar is not None else None

    def hlsl_texture_offset_binary_component(self, left, operator, right):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/" and right != 0:
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient
        if operator == "%" and right != 0:
            quotient = abs(left) // abs(right)
            quotient = -quotient if (left < 0) != (right < 0) else quotient
            return left - quotient * right
        if operator == "<<" and right >= 0:
            return left << right
        if operator == ">>" and right >= 0:
            return left >> right
        if operator == "&":
            return left & right
        if operator == "|":
            return left | right
        if operator == "^":
            return left ^ right
        return None

    def hlsl_texture_offset_swizzle_indices(self, components):
        if not isinstance(components, str) or not components:
            return None
        groups = ("xyzw", "rgba", "stpq")
        indices = []
        for component in components:
            index = next(
                (group.find(component) for group in groups if component in group), -1
            )
            if index < 0:
                return None
            indices.append(index)
        return tuple(indices)

    def hlsl_immediate_texture_offset(
        self,
        func_name,
        offset_expression,
        texture_type,
        target_method,
        *,
        call_node=None,
    ):
        value = self.hlsl_texture_offset_constant_value(offset_expression)
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        texture_shape = self.sampled_texture_shape_type(texture_type) or "Texture"
        qualified_method = f"{texture_shape}.{target_method}"
        blocking_expression = self.hlsl_texture_offset_expression_label(
            offset_expression
        )
        function_name = self.current_function_name or "<global>"
        source_location = getattr(call_node, "source_location", None) or getattr(
            offset_expression, "source_location", None
        )

        reason = None
        detail = None
        if value is None or (
            expected_dimension is not None and len(value) != expected_dimension
        ):
            reason = "runtime-offset-requires-immediate"
            detail = (
                f"offset expression '{blocking_expression}' is runtime-derived, but "
                f"HLSL {qualified_method} requires a compile-time immediate offset"
            )
        elif any(component < -8 or component > 7 for component in value):
            reason = "immediate-offset-out-of-range"
            detail = (
                f"offset expression '{blocking_expression}' resolves to "
                f"{self.hlsl_texture_offset_value_label(value)}, but HLSL "
                f"{qualified_method} immediate components must be between -8 and 7"
            )

        if reason is not None:
            raise DirectXTextureOffsetError(
                f"DirectX texture operation '{func_name}' in function "
                f"'{function_name}' cannot preserve source sampling semantics: "
                f"{detail}. Use a literal or const offset in range, ensure every "
                "reachable helper call supplies one concrete value, or target a "
                "backend/profile with programmable texture offsets.",
                operation=func_name,
                target_method=qualified_method,
                blocking_expression=blocking_expression,
                function_name=function_name,
                reason=reason,
                source_location=source_location,
            )

        return self.hlsl_render_texture_offset_value(value)

    def hlsl_texture_offset_expression_label(self, expression):
        constructor_name, constructor_args = self.hlsl_texture_offset_constructor_parts(
            expression
        )
        if constructor_name is not None:
            arguments = ", ".join(
                self.hlsl_texture_offset_expression_label(argument)
                for argument in constructor_args
            )
            return f"{constructor_name}({arguments})"
        return expression_debug_name(expression)

    def hlsl_render_texture_offset_value(self, value):
        if len(value) == 1:
            return str(value[0])
        return f"int{len(value)}({', '.join(str(component) for component in value)})"

    def hlsl_texture_offset_value_label(self, value):
        if len(value) == 1:
            return str(value[0])
        return f"({', '.join(str(component) for component in value)})"

    def collect_hlsl_texture_offset_global_constants(self, constants):
        resolved = {}
        pending = list(constants or [])
        changed = True
        while changed:
            changed = False
            remaining = []
            for constant in pending:
                name = getattr(constant, "name", None)
                dimension = self.hlsl_texture_offset_type_dimension(
                    getattr(constant, "const_type", None)
                )
                if not name or dimension is None or dimension == 1:
                    continue
                value = self.hlsl_texture_offset_constant_value(
                    getattr(constant, "value", None),
                    scalar_constants=self.literal_int_constants,
                    offset_constants=resolved,
                )
                if value is None:
                    remaining.append(constant)
                    continue
                if len(value) == dimension:
                    resolved[name] = value
                    changed = True
            pending = remaining
        return resolved

    def hlsl_texture_offset_analysis_functions(self, ast, target_stage):
        functions = [
            function
            for function in getattr(ast, "functions", []) or []
            if getattr(function, "name", None) not in self.hlsl_omitted_function_names
            and should_emit_qualified_function(
                target_stage, function_stage_name(function)
            )
        ]
        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            functions.extend(getattr(stage, "local_functions", []) or [])
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
        deduplicated = []
        seen = set()
        for function in functions:
            if id(function) in seen or getattr(function, "body", None) is None:
                continue
            seen.add(id(function))
            deduplicated.append(function)
        return deduplicated

    def hlsl_texture_offset_operation_requires_immediate(self, operation):
        return (
            is_texture_sample_basic_offset_operation(operation)
            or is_texture_sample_lod_offset_operation(operation)
            or is_texture_sample_grad_offset_operation(operation)
            or is_projected_texture_basic_offset_operation(operation)
            or is_projected_texture_lod_offset_operation(operation)
            or is_projected_texture_grad_offset_operation(operation)
            or is_texture_compare_offset_operation(operation)
        )

    def collect_hlsl_texture_offset_parameter_constants(self, ast, target_stage):
        functions = self.hlsl_texture_offset_analysis_functions(ast, target_stage)
        if not any(
            self.hlsl_texture_offset_operation_requires_immediate(
                self.function_call_name(node)
            )
            for function in functions
            for node in self.walk_ast(getattr(function, "body", None))
            if isinstance(node, FunctionCallNode)
        ):
            return {}

        functions_by_name = {}
        for function in functions:
            name = getattr(function, "name", None)
            if name:
                functions_by_name.setdefault(name, []).append(function)

        mutated_parameter_indices = self.hlsl_texture_offset_mutated_parameter_indices(
            functions, functions_by_name
        )

        proven = {}
        for _ in range(len(functions) + 1):
            incoming = {id(function): [] for function in functions}
            for caller in functions:
                scalar_constants = self.hlsl_initial_int_constants(caller)
                offset_constants = self.hlsl_initial_texture_offset_constants(caller)
                for parameter_name, value in proven.get(id(caller), {}).items():
                    offset_constants[parameter_name] = value
                self.collect_hlsl_texture_offset_calls_from_body(
                    getattr(caller, "body", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )

            updated = {}
            for function in functions:
                calls = incoming[id(function)]
                if not calls:
                    continue
                parameter_values = {}
                parameters = list(
                    getattr(function, "parameters", getattr(function, "params", []))
                    or []
                )
                for index, parameter in enumerate(parameters):
                    if index in mutated_parameter_indices[id(function)]:
                        continue
                    values = [arguments[index] for arguments in calls]
                    first = values[0]
                    if first is not None and all(
                        value == first for value in values[1:]
                    ):
                        parameter_values[getattr(parameter, "name", "")] = first
                if parameter_values:
                    updated[id(function)] = parameter_values
            if updated == proven:
                break
            proven = updated
        return proven

    def hlsl_texture_offset_call_target(self, call, functions_by_name):
        name = self.function_call_name(call)
        candidates = list(functions_by_name.get(name, []))
        if not candidates and isinstance(name, str) and "::" in name:
            candidates = list(functions_by_name.get(name.rsplit("::", 1)[-1], []))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        argument_count = len(
            getattr(call, "arguments", getattr(call, "args", [])) or []
        )
        arity_matches = [
            candidate
            for candidate in candidates
            if len(
                getattr(candidate, "parameters", getattr(candidate, "params", [])) or []
            )
            == argument_count
        ]
        return arity_matches[0] if len(arity_matches) == 1 else None

    def hlsl_texture_offset_write_root_name(self, expression):
        if isinstance(expression, str):
            return expression
        if isinstance(
            expression, (ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        ):
            child = getattr(
                expression,
                "array_expr",
                getattr(
                    expression,
                    "object_expr",
                    getattr(expression, "pointer_expr", None),
                ),
            )
            return self.hlsl_texture_offset_write_root_name(child)
        if isinstance(expression, SwizzleNode):
            return self.hlsl_texture_offset_write_root_name(
                getattr(expression, "vector_expr", None)
            )
        name = getattr(expression, "name", None)
        if isinstance(name, str):
            return name
        return None

    def hlsl_texture_offset_mutated_parameter_indices(
        self, functions, functions_by_name
    ):
        parameter_indices = {}
        mutated = {}
        for function in functions:
            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            parameter_indices[id(function)] = {
                getattr(parameter, "name", None): index
                for index, parameter in enumerate(parameters)
                if getattr(parameter, "name", None)
            }
            mutated[id(function)] = {
                index
                for index, parameter in enumerate(parameters)
                if set(self.hlsl_parameter_qualifiers(parameter)) & {"out", "inout"}
            }

            for node in self.walk_ast(getattr(function, "body", None)):
                target = None
                if isinstance(node, AssignmentNode):
                    target = getattr(node, "target", getattr(node, "left", None))
                elif isinstance(node, UnaryOpNode) and getattr(
                    node, "operator", getattr(node, "op", None)
                ) in {"++", "--"}:
                    target = getattr(node, "operand", None)
                root_name = self.hlsl_texture_offset_write_root_name(target)
                index = parameter_indices[id(function)].get(root_name)
                if index is not None:
                    mutated[id(function)].add(index)

        changed = True
        while changed:
            changed = False
            for function in functions:
                caller_parameters = parameter_indices[id(function)]
                for node in self.walk_ast(getattr(function, "body", None)):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee = self.hlsl_texture_offset_call_target(
                        node, functions_by_name
                    )
                    if callee is None:
                        continue
                    arguments = list(
                        getattr(node, "arguments", getattr(node, "args", [])) or []
                    )
                    callee_parameters = list(
                        getattr(
                            callee,
                            "parameters",
                            getattr(callee, "params", []),
                        )
                        or []
                    )
                    for index in mutated[id(callee)]:
                        if index >= len(arguments) or index >= len(callee_parameters):
                            continue
                        if not set(
                            self.hlsl_parameter_qualifiers(callee_parameters[index])
                        ) & {"out", "inout"}:
                            continue
                        root_name = self.hlsl_texture_offset_write_root_name(
                            arguments[index]
                        )
                        caller_index = caller_parameters.get(root_name)
                        if (
                            caller_index is not None
                            and caller_index not in mutated[id(function)]
                        ):
                            mutated[id(function)].add(caller_index)
                            changed = True
        return mutated

    def collect_hlsl_texture_offset_calls_from_expression(
        self,
        expression,
        functions_by_name,
        scalar_constants,
        offset_constants,
        incoming,
    ):
        if expression is None:
            return
        for node in self.walk_ast(expression):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.hlsl_texture_offset_call_target(node, functions_by_name)
            if callee is None:
                continue
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            parameters = list(
                getattr(callee, "parameters", getattr(callee, "params", [])) or []
            )
            values = []
            for index, parameter in enumerate(parameters):
                argument = (
                    arguments[index]
                    if index < len(arguments)
                    else getattr(parameter, "default_value", None)
                )
                values.append(
                    self.hlsl_texture_offset_constant_value(
                        argument,
                        scalar_constants=scalar_constants,
                        offset_constants=offset_constants,
                    )
                )
            incoming[id(callee)].append(values)

    def hlsl_update_texture_offset_analysis_constants(
        self, statement, scalar_constants, offset_constants
    ):
        if isinstance(statement, VariableNode):
            name = getattr(statement, "name", None)
            if not name:
                return
            scalar_constants.pop(name, None)
            offset_constants.pop(name, None)
            if "const" not in getattr(statement, "qualifiers", []):
                return
            value = self.hlsl_texture_offset_constant_value(
                getattr(statement, "initial_value", None),
                scalar_constants=scalar_constants,
                offset_constants=offset_constants,
            )
            if value is None:
                return
            if len(value) == 1:
                scalar_constants[name] = value[0]
            else:
                offset_constants[name] = value
            return

        assignment = self.hlsl_assignment_from_statement(statement)
        if assignment is None:
            return
        target_name = self.expression_name(
            getattr(assignment, "target", getattr(assignment, "left", None))
        )
        if target_name:
            scalar_constants.pop(target_name, None)
            offset_constants.pop(target_name, None)

    def collect_hlsl_texture_offset_calls_from_body(
        self,
        body,
        functions_by_name,
        scalar_constants,
        offset_constants,
        incoming,
    ):
        for statement in self.hlsl_statement_body_items(body):
            if isinstance(statement, VariableNode):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "initial_value", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                self.hlsl_update_texture_offset_analysis_constants(
                    statement, scalar_constants, offset_constants
                )
                continue

            if isinstance(statement, BlockNode):
                self.collect_hlsl_texture_offset_calls_from_body(
                    statement,
                    functions_by_name,
                    dict(scalar_constants),
                    dict(offset_constants),
                    incoming,
                )
                continue

            if isinstance(statement, IfNode):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "condition", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                for else_if_condition in (
                    getattr(statement, "else_if_conditions", []) or []
                ):
                    self.collect_hlsl_texture_offset_calls_from_expression(
                        else_if_condition,
                        functions_by_name,
                        scalar_constants,
                        offset_constants,
                        incoming,
                    )
                branches = [
                    getattr(
                        statement, "then_branch", getattr(statement, "if_body", None)
                    ),
                    *list(getattr(statement, "else_if_bodies", []) or []),
                    getattr(
                        statement, "else_branch", getattr(statement, "else_body", None)
                    ),
                ]
                for branch in branches:
                    if branch is not None:
                        self.collect_hlsl_texture_offset_calls_from_body(
                            branch,
                            functions_by_name,
                            dict(scalar_constants),
                            dict(offset_constants),
                            incoming,
                        )
                continue

            if isinstance(statement, ForNode):
                loop_scalars = dict(scalar_constants)
                loop_offsets = dict(offset_constants)
                initializers = getattr(statement, "init", None)
                if not isinstance(initializers, list):
                    initializers = [initializers] if initializers is not None else []
                for initializer in initializers:
                    self.collect_hlsl_texture_offset_calls_from_expression(
                        initializer,
                        functions_by_name,
                        loop_scalars,
                        loop_offsets,
                        incoming,
                    )
                    self.hlsl_update_texture_offset_analysis_constants(
                        initializer, loop_scalars, loop_offsets
                    )
                for expression in (
                    getattr(statement, "condition", None),
                    getattr(statement, "update", None),
                ):
                    self.collect_hlsl_texture_offset_calls_from_expression(
                        expression,
                        functions_by_name,
                        loop_scalars,
                        loop_offsets,
                        incoming,
                    )
                self.collect_hlsl_texture_offset_calls_from_body(
                    getattr(statement, "body", None),
                    functions_by_name,
                    loop_scalars,
                    loop_offsets,
                    incoming,
                )
                continue

            if isinstance(statement, ForInNode):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "iterable", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                loop_scalars = dict(scalar_constants)
                loop_offsets = dict(offset_constants)
                pattern_name = self.hlsl_declaration_identifier_name(
                    getattr(statement, "pattern", None)
                )
                loop_scalars.pop(pattern_name, None)
                loop_offsets.pop(pattern_name, None)
                self.collect_hlsl_texture_offset_calls_from_body(
                    getattr(statement, "body", None),
                    functions_by_name,
                    loop_scalars,
                    loop_offsets,
                    incoming,
                )
                continue

            if isinstance(statement, (WhileNode, DoWhileNode, LoopNode)):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "condition", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                self.collect_hlsl_texture_offset_calls_from_body(
                    getattr(statement, "body", None),
                    functions_by_name,
                    dict(scalar_constants),
                    dict(offset_constants),
                    incoming,
                )
                continue

            if isinstance(statement, SwitchNode):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "expression", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                for case in getattr(statement, "cases", []) or []:
                    self.collect_hlsl_texture_offset_calls_from_expression(
                        getattr(case, "value", None),
                        functions_by_name,
                        scalar_constants,
                        offset_constants,
                        incoming,
                    )
                    self.collect_hlsl_texture_offset_calls_from_body(
                        getattr(case, "statements", []),
                        functions_by_name,
                        dict(scalar_constants),
                        dict(offset_constants),
                        incoming,
                    )
                self.collect_hlsl_texture_offset_calls_from_body(
                    getattr(statement, "default_case", None),
                    functions_by_name,
                    dict(scalar_constants),
                    dict(offset_constants),
                    incoming,
                )
                continue

            if isinstance(statement, MatchNode):
                self.collect_hlsl_texture_offset_calls_from_expression(
                    getattr(statement, "expression", None),
                    functions_by_name,
                    scalar_constants,
                    offset_constants,
                    incoming,
                )
                for arm in getattr(statement, "arms", []) or []:
                    arm_scalars = dict(scalar_constants)
                    arm_offsets = dict(offset_constants)
                    pattern_name = self.hlsl_declaration_identifier_name(
                        getattr(arm, "pattern", None)
                    )
                    arm_scalars.pop(pattern_name, None)
                    arm_offsets.pop(pattern_name, None)
                    self.collect_hlsl_texture_offset_calls_from_expression(
                        getattr(arm, "guard", None),
                        functions_by_name,
                        arm_scalars,
                        arm_offsets,
                        incoming,
                    )
                    self.collect_hlsl_texture_offset_calls_from_body(
                        getattr(arm, "body", None),
                        functions_by_name,
                        arm_scalars,
                        arm_offsets,
                        incoming,
                    )
                continue

            self.collect_hlsl_texture_offset_calls_from_expression(
                statement,
                functions_by_name,
                scalar_constants,
                offset_constants,
                incoming,
            )
            self.hlsl_update_texture_offset_analysis_constants(
                statement, scalar_constants, offset_constants
            )

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if not name:
                    continue

                visible_constants.pop(name, None)
                if "const" not in getattr(node, "qualifiers", []):
                    continue

                value = self.literal_int_value(
                    getattr(node, "initial_value", None), visible_constants
                )
                if value is not None:
                    visible_constants[name] = value

        return visible_constants

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None

    def texture_call_parts(self, args, func_name=None):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        if explicit_sampler:
            sampler_name = self.generate_expression(args[1])
            if self.texture_call_uses_comparison_sampler(func_name):
                sampler_name = self.hlsl_comparison_sampler_alias_expression(
                    args[1], sampler_name
                )
        elif self.implicit_call_uses_regular_sampler(func_name):
            member_sampler = self.hlsl_implicit_member_sampler_name(
                args[0],
                self.hlsl_member_sampler_info_for_call(func_name, args),
            )
            sampler_name = self.current_implicit_texture_regular_samplers.get(
                texture_base_name,
                self.global_implicit_texture_regular_samplers.get(
                    texture_base_name,
                    self.current_implicit_texture_samplers.get(
                        texture_base_name,
                        member_sampler or f"{texture_base_name}Sampler",
                    ),
                ),
            )
        else:
            member_sampler = self.hlsl_implicit_member_sampler_name(
                args[0],
                self.hlsl_member_sampler_info_for_call(func_name, args),
            )
            sampler_name = self.current_implicit_texture_samplers.get(
                texture_base_name, member_sampler or f"{texture_base_name}Sampler"
            )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_name, coord, extra_args

    def texture_call_uses_comparison_sampler(self, func_name):
        return bool(
            func_name
            and (
                is_texture_compare_operation(func_name)
                or is_texture_gather_compare_operation(func_name)
            )
        )

    def hlsl_comparison_sampler_alias_name(self, sampler_name):
        return f"{sampler_name}_cglComparison"

    def hlsl_comparison_sampler_alias_expression(self, sampler_arg, rendered):
        sampler_name = self.expression_name(sampler_arg)
        alias_name = self.global_mixed_sampler_aliases.get(sampler_name)
        if not alias_name:
            return rendered
        if rendered == sampler_name:
            return alias_name
        if rendered.startswith(f"{sampler_name}["):
            return f"{alias_name}{rendered[len(sampler_name):]}"
        return rendered
