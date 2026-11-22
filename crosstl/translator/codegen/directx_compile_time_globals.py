"""Compile-time globals in the HLSL generator: cbuffers and packoffsets, fixed array returns, recursive function checks, global constant initializers and GLSL built-in blocks."""

import re
from hashlib import sha256

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ArrayType,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    StructMemberNode,
    StructNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..float_limit_constants import render_float_limit_constant
from ..refusals import DirectXRefusal
from ..resource_identity import resource_identity_marker
from ..standard_constants import render_standard_math_constant
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
)
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .bfloat_constants import (
    bfloat16_constant_bits,
    scalar_constant_value,
)
from .buffer_member_layout import (
    member_explicit_offset,
    member_is_packed,
    member_is_row_major,
    packoffset_text,
)
from .directx_values import (
    DirectXCompileTimeGlobalError,
    DirectXMappedOverloadError,
    DirectXRecursiveFunctionError,
    DirectXWorkgroupSizeError,
)
from .enum_utils import (
    default_value_expression,
    sanitize_type_name,
)
from .glsl_block_layout import (
    BLOCK_LAYOUTS,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    hlsl_packoffsets_for_block,
    member_explicit_alignment,
)
from .hlsl_buffer_layout import (
    HlslLayoutError,
    cbuffer_placement,
)
from .hlsl_glsl_block_registers import (
    HlslRegisterBlockPlan,
    hlsl_cbuffer_register_plan,
    hlsl_members_hold_struct_placements,
)
from .hlsl_host_workgroup_size import hlsl_host_numthreads
from .hlsl_push_constants import (
    hlsl_push_constant_block,
    is_push_constant,
)
from .hlsl_refusals import DirectXBufferLayoutError
from .host_workgroup_size_ids import is_host_sized_workgroup
from .specialized_workgroup_size import require_fixed_workgroup_size
from .stage_utils import (
    collect_stage_local_cbuffers,
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
)


class DirectXCompileTimeGlobalMixin:
    """Compile time global members of ``HLSLCodeGen``."""

    def directx_compile_time_global_error(
        self,
        node,
        expression,
        *,
        reason,
        detail=None,
        operation=None,
        operand_type=None,
        result_type=None,
    ):
        name = getattr(node, "name", getattr(node, "variable_name", "<unnamed>"))
        explanation = detail or reason.replace("-", " ")
        source_location = getattr(expression, "source_location", None) or getattr(
            node, "source_location", None
        )
        raise DirectXCompileTimeGlobalError(
            f"DirectX compile-time global '{name}' cannot be lowered: {explanation}",
            variable_name=name,
            expression_kind=(
                expression.__class__.__name__ if expression is not None else None
            ),
            detail=detail,
            operation=operation,
            operand_type=operand_type,
            result_type=result_type,
            reason=reason,
            source_location=source_location,
        )

    def is_directx_compile_time_global_candidate(self, node):
        if getattr(node, "initial_value", None) is None:
            return False
        if id(node) in self.directx_specialization_constant_global_ids:
            return False

        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"const", "constant"}):
            return False
        if qualifiers.intersection(
            {
                "buffer",
                "coherent",
                "device",
                "global",
                "groupshared",
                "in",
                "inout",
                "input",
                "out",
                "output",
                "readonly",
                "readwrite",
                "restrict",
                "shared",
                "storage",
                "thread",
                "threadgroup",
                "threadgroup_imageblock",
                "uniform",
                "volatile",
                "workgroup",
                "writeonly",
            }
        ):
            return False
        if getattr(node, "resource_qualifiers", None):
            return False
        if getattr(node, "semantic", None) is not None:
            return False
        if getattr(node, "attributes", None):
            return False

        vtype = getattr(node, "var_type", getattr(node, "vtype", None))
        return not self.is_directx_compile_time_resource_type(vtype, node=node)

    def is_directx_compile_time_resource_type(self, vtype, *, node=None, seen=None):
        if isinstance(vtype, (PointerType, ReferenceType)):
            return True
        if isinstance(vtype, ArrayType):
            return self.is_directx_compile_time_resource_type(
                vtype.element_type,
                node=node,
                seen=seen,
            )

        type_name = self.type_name_string(vtype)
        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return self.is_directx_compile_time_resource_type(
                base_type,
                node=node,
                seen=seen,
            )

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if (
            self.is_resource_parameter_type(vtype)
            or self.is_sampler_type(vtype)
            or self.is_image_type(vtype)
            or mapped_type.startswith("Texture")
            or self.is_hlsl_feedback_texture_type(mapped_type)
            or self.is_hlsl_rw_texture_type(mapped_type)
            or self.is_hlsl_acceleration_structure_type(mapped_type)
            or self.is_hlsl_uav_buffer_type(mapped_type)
            or self.is_hlsl_readonly_buffer_type(mapped_type)
            or self.is_hlsl_constant_buffer_type(mapped_type)
            or self.is_hlsl_texture_buffer_type(mapped_type)
            or mapped_type in {"SamplerState", "SamplerComparisonState"}
        ):
            return True

        mapped_value_type = self.map_type(vtype)
        seen = set(seen or ())
        if mapped_value_type in seen:
            return False
        fields = self.hlsl_struct_constructor_fields(mapped_value_type)
        if fields is None:
            return False
        seen.add(mapped_value_type)
        return any(
            self.is_directx_compile_time_resource_type(
                field_type,
                seen=seen,
            )
            for _field_name, field_type in fields
        )

    def validate_directx_compile_time_global_type(
        self,
        node,
        vtype,
        *,
        seen=None,
    ):
        if isinstance(vtype, (PointerType, ReferenceType)):
            self.directx_compile_time_global_error(
                node,
                vtype,
                reason="unrepresentable-value-type",
                detail=f"pointer/reference type '{self.type_name_string(vtype)}'",
            )

        mapped_type = self.map_type(vtype)
        array_type = self.hlsl_outer_array_type(mapped_type)
        if array_type is not None:
            _array_type, extent_text, element_type = array_type
            extent = evaluate_literal_int_expression(
                extent_text,
                self.literal_int_constants,
            )
            if extent is None:
                self.directx_compile_time_global_error(
                    node,
                    vtype,
                    reason="array-extent-unresolved",
                    detail=f"array extent '{extent_text}' is not a known integer constant",
                )
            if extent <= 0:
                self.directx_compile_time_global_error(
                    node,
                    vtype,
                    reason="array-extent-invalid",
                    detail=f"array extent '{extent}' must be positive",
                )
            self.validate_directx_compile_time_global_type(
                node,
                element_type,
                seen=seen,
            )
            return

        if (
            self.is_scalar_value_type(mapped_type)
            or self.is_vector_value_type(mapped_type)
            or self.hlsl_matrix_shape(mapped_type) is not None
        ):
            return

        fields = self.hlsl_struct_constructor_fields(mapped_type)
        if fields is not None:
            if not fields:
                self.directx_compile_time_global_error(
                    node,
                    vtype,
                    reason="unrepresentable-value-type",
                    detail=f"empty struct type '{mapped_type}'",
                )
            seen = set(seen or ())
            if mapped_type in seen:
                self.directx_compile_time_global_error(
                    node,
                    vtype,
                    reason="recursive-value-type",
                    detail=f"recursive struct type '{mapped_type}'",
                )
            seen.add(mapped_type)
            for _field_name, field_type in fields:
                self.validate_directx_compile_time_global_type(
                    node,
                    field_type,
                    seen=seen,
                )
            return

        self.directx_compile_time_global_error(
            node,
            vtype,
            reason="unrepresentable-value-type",
            detail=f"value type '{mapped_type}'",
        )

    def directx_compile_time_constructor_type(self, type_name):
        mapped_type = self.map_type(type_name)
        if (
            self.is_scalar_value_type(mapped_type)
            or self.is_vector_value_type(mapped_type)
            or self.hlsl_matrix_shape(mapped_type) is not None
        ):
            return mapped_type
        if self.hlsl_struct_constructor_fields(mapped_type) is not None:
            return mapped_type
        return None

    def directx_compile_time_bitcast_contract(self, node, expression):
        function_name = self.function_call_name(expression)
        if function_name in self.function_return_types:
            return None

        arguments = list(
            getattr(expression, "arguments", getattr(expression, "args", [])) or []
        )
        explicit_target = self.hlsl_metal_as_type_target_type(function_name)
        is_bitcast = (
            function_name in self.HLSL_BITCAST_FUNCTION_TARGETS
            or explicit_target is not None
        )
        if not is_bitcast:
            return None
        if len(arguments) != 1:
            self.directx_compile_time_global_error(
                node,
                expression,
                reason="bitcast-arity",
                detail=f"{function_name} expects exactly one operand",
                operation=function_name,
            )

        operand = arguments[0]
        operand_type = self.map_type(self.expression_result_type(operand))
        result_type = explicit_target
        if result_type is None:
            result_type = self.hlsl_bitcast_result_type(function_name, arguments)
        result_type = self.map_type(result_type)
        operand_info = self.hlsl_explicit_bitcast_type_info(operand_type)
        result_info = self.hlsl_explicit_bitcast_type_info(result_type)
        if operand_info is None or result_info is None:
            self.directx_compile_time_global_error(
                node,
                expression,
                reason="bitcast-type-unresolved",
                detail=(
                    f"{function_name} cannot resolve an exact bitcast from "
                    f"'{operand_type}' to '{result_type}'"
                ),
                operation=function_name,
                operand_type=operand_type,
                result_type=result_type,
            )

        identity = operand_info["type"] == result_info["type"]
        native_shape = (
            operand_info["component_width"] == 32
            and result_info["component_width"] == 32
            and operand_info["lanes"] == result_info["lanes"]
            and operand_info["width"] == result_info["width"]
        )
        if not identity and not native_shape:
            self.directx_compile_time_global_error(
                node,
                expression,
                reason="bitcast-shape-unsupported",
                detail=(
                    f"{function_name} cannot preserve the {operand_info['width']}-bit "
                    f"'{operand_info['type']}' payload as '{result_info['type']}' "
                    "in an HLSL constant initializer"
                ),
                operation=function_name,
                operand_type=operand_info["type"],
                result_type=result_info["type"],
            )
        return {
            "operation": function_name,
            "operand": operand,
            "operand_type": operand_info["type"],
            "result_type": result_info["type"],
        }

    def validate_directx_compile_time_global_expression(
        self,
        node,
        expression,
        known_constant_names,
        candidate_names,
    ):
        def reject(reason, current, detail=None):
            self.directx_compile_time_global_error(
                node,
                current,
                reason=reason,
                detail=detail,
            )

        def validate(current):
            if isinstance(current, (bool, int, float, LiteralNode)):
                return
            if isinstance(current, (IdentifierNode, VariableNode)):
                name = getattr(current, "name", None)
                if name in known_constant_names:
                    return
                if name and render_standard_math_constant(name, "directx") is not None:
                    return
                if name and render_float_limit_constant(name, "directx") is not None:
                    return
                reason = (
                    "forward-reference" if name in candidate_names else "runtime-value"
                )
                reject(reason, current, name)
            if isinstance(current, ArrayLiteralNode):
                for element in getattr(current, "elements", []) or []:
                    validate(element)
                return
            if isinstance(current, MemberAccessNode):
                validate(
                    getattr(current, "object", getattr(current, "object_expr", None))
                )
                return
            if isinstance(current, SwizzleNode):
                validate(current.vector_expr)
                return
            if isinstance(current, ArrayAccessNode):
                validate(
                    getattr(current, "array", getattr(current, "array_expr", None))
                )
                validate(
                    getattr(current, "index", getattr(current, "index_expr", None))
                )
                return
            if isinstance(current, BinaryOpNode):
                validate(current.left)
                validate(current.right)
                return
            if isinstance(current, UnaryOpNode):
                operator = self.map_operator(current.op)
                if operator not in {"+", "-", "!", "~"}:
                    reject("invalid-unary-operator", current, operator)
                validate(current.operand)
                return
            if isinstance(current, TernaryOpNode):
                validate(current.condition)
                validate(current.true_expr)
                validate(current.false_expr)
                return
            if isinstance(current, CastNode):
                if (
                    self.directx_compile_time_constructor_type(current.target_type)
                    is None
                ):
                    reject(
                        "unrepresentable-cast-type",
                        current,
                        self.type_name_string(current.target_type),
                    )
                validate(current.expression)
                return
            if isinstance(current, ConstructorNode):
                constructor_type = self.type_name_string(current.constructor_type)
                if self.directx_compile_time_constructor_type(constructor_type) is None:
                    reject("unrepresentable-constructor", current, constructor_type)
                for argument in list(current.arguments or []) + list(
                    (current.named_arguments or {}).values()
                ):
                    validate(argument)
                return
            if isinstance(current, FunctionCallNode):
                function_name = self.function_call_name(current)
                bitcast = self.directx_compile_time_bitcast_contract(node, current)
                if bitcast is not None:
                    try:
                        validate(bitcast["operand"])
                    except DirectXCompileTimeGlobalError as error:
                        if error.reason not in {
                            "forward-reference",
                            "function-call",
                            "runtime-value",
                        }:
                            raise
                        self.directx_compile_time_global_error(
                            node,
                            current,
                            reason=f"bitcast-operand-{error.reason}",
                            detail=error.detail,
                            operation=bitcast["operation"],
                            operand_type=bitcast["operand_type"],
                            result_type=bitcast["result_type"],
                        )
                    return
                if self.directx_compile_time_constructor_type(function_name) is None:
                    reject("function-call", current, function_name)
                for argument in (
                    getattr(current, "arguments", getattr(current, "args", [])) or []
                ):
                    validate(argument)
                return
            reject("unrepresentable-expression", current)

        validate(expression)

    def prepare_directx_compile_time_globals(self, ast, global_vars):
        self.directx_compile_time_global_node_ids = set()
        self.directx_omitted_compile_time_global_node_ids = set()
        candidates = [
            node
            for node in global_vars or []
            if self.is_directx_compile_time_global_candidate(node)
        ]
        candidate_names = {
            getattr(node, "name", getattr(node, "variable_name", None))
            for node in candidates
        }
        candidate_names.discard(None)
        known_constant_names = {
            getattr(node, "name", None)
            for node in getattr(ast, "constants", []) or []
            if getattr(node, "name", None)
        }
        known_constant_names.update(self.directx_specialization_constant_types)
        known_constant_names.update(getattr(self, "enum_variant_constants", {}))
        known_constant_names.update(self.HLSL_GLSL_BUILTIN_INT_CONSTANTS)

        for node in candidates:
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            name = getattr(node, "name", getattr(node, "variable_name", None))
            struct_fields = self.hlsl_struct_constructor_fields(self.map_type(vtype))
            if struct_fields == []:
                self.validate_directx_compile_time_global_expression(
                    node,
                    getattr(node, "initial_value", None),
                    known_constant_names,
                    candidate_names,
                )
                referenced = any(
                    isinstance(candidate, IdentifierNode) and candidate.name == name
                    for candidate in self.walk_ast(ast)
                )
                if name and not referenced:
                    self.directx_omitted_compile_time_global_node_ids.add(id(node))
                    known_constant_names.add(name)
                    continue
            self.validate_directx_compile_time_global_type(node, vtype)
            self.validate_directx_compile_time_global_expression(
                node,
                getattr(node, "initial_value", None),
                known_constant_names,
                candidate_names,
            )
            self.directx_compile_time_global_node_ids.add(id(node))
            if name:
                known_constant_names.add(name)

            if self.map_type(vtype) not in {"int", "uint", "int64_t", "uint64_t"}:
                continue
            value = evaluate_literal_int_expression(
                getattr(node, "initial_value", None),
                self.literal_int_constants,
            )
            if value is not None and name:
                self.literal_int_constants[name] = value

    def hlsl_compile_time_default_initializer(self, node, expression, expected_type):
        mapped_type = self.map_type(expected_type)
        array_type = self.hlsl_outer_array_type(mapped_type)
        if array_type is not None:
            _array_type, extent_text, element_type = array_type
            extent = evaluate_literal_int_expression(
                extent_text,
                self.literal_int_constants,
            )
            if extent is None or extent <= 0:
                self.directx_compile_time_global_error(
                    node,
                    expression,
                    reason="array-extent-unresolved",
                    detail=f"array extent '{extent_text}' is not a positive integer constant",
                )
            elements = [
                self.hlsl_compile_time_default_initializer(
                    node,
                    expression,
                    element_type,
                )
                for _index in range(extent)
            ]
            return "{" + ", ".join(elements) + "}"

        fields = self.hlsl_struct_constructor_fields(mapped_type)
        if fields is not None:
            values = [
                self.hlsl_compile_time_default_initializer(
                    node,
                    expression,
                    field_type,
                )
                for _field_name, field_type in fields
            ]
            return "{" + ", ".join(values) + "}"
        if self.is_vector_value_type(mapped_type):
            return f"({mapped_type})0"
        return default_value_expression(self, mapped_type)

    def hlsl_compile_time_initializer_expression(
        self,
        node,
        expression,
        expected_type,
    ):
        if self.is_hlsl_bfloat16_type(expected_type):
            numeric_value = self.hlsl_bfloat16_constant_numeric_value(expression)
            if numeric_value is not None:
                return f"0x{bfloat16_constant_bits(numeric_value):04x}u"
        mapped_type = self.map_type(expected_type)
        array_type = self.hlsl_outer_array_type(mapped_type)
        if array_type is not None:
            if not isinstance(expression, ArrayLiteralNode):
                self.directx_compile_time_global_error(
                    node,
                    expression,
                    reason="array-initializer-shape",
                    detail="fixed arrays require a brace initializer",
                )
            _mapped_type, extent, element_type = (
                self.hlsl_fixed_array_aggregate_contract(expression, mapped_type)
            )
            elements = list(getattr(expression, "elements", []) or [])
            if self.hlsl_is_zero_aggregate_initializer(expression):
                elements = []
            if len(elements) > extent:
                self.hlsl_aggregate_error(
                    expression,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=extent,
                )
            rendered = [
                self.hlsl_compile_time_initializer_expression(
                    node,
                    element,
                    element_type,
                )
                for element in elements
            ]
            rendered.extend(
                self.hlsl_compile_time_default_initializer(
                    node,
                    expression,
                    element_type,
                )
                for _index in range(len(elements), extent)
            )
            return "{" + ", ".join(rendered) + "}"

        fields = self.hlsl_struct_constructor_fields(mapped_type)
        if fields is not None:
            initializer = self.hlsl_struct_initializer_components(
                expected_type,
                expression,
            )
            if initializer is not None:
                _type_name, initializer_fields, _rendered_args, field_exprs = (
                    initializer
                )
                rendered = []
                for (_field_name, field_type), field_expression in zip(
                    initializer_fields,
                    field_exprs,
                ):
                    if field_expression is None:
                        rendered.append(
                            self.hlsl_compile_time_default_initializer(
                                node,
                                expression,
                                field_type,
                            )
                        )
                    else:
                        rendered.append(
                            self.hlsl_compile_time_initializer_expression(
                                node,
                                field_expression,
                                field_type,
                            )
                        )
                return "{" + ", ".join(rendered) + "}"
            if isinstance(expression, (ArrayLiteralNode, ConstructorNode)) or (
                isinstance(expression, FunctionCallNode)
                and self.function_call_name(expression)
                == self.type_name_string(expected_type)
            ):
                self.directx_compile_time_global_error(
                    node,
                    expression,
                    reason="struct-initializer-shape",
                    detail=f"initializer does not match struct type '{mapped_type}'",
                )

        if isinstance(expression, ArrayLiteralNode):
            return self.hlsl_contextual_aggregate_initializer(expression, mapped_type)
        if self.hlsl_matrix_shape(mapped_type) is not None:
            # Matrix constructors take CrossGL's column-major operand order; the
            # expression generator owns that reordering.
            return self.generate_constant_expression(expression, expected_type)
        if isinstance(expression, CastNode):
            target_type = self.map_type(expression.target_type)
            operand = self.hlsl_compile_time_initializer_expression(
                node,
                expression.expression,
                expression.target_type,
            )
            return f"{target_type}({operand})"
        if isinstance(expression, ConstructorNode):
            constructor_type = self.map_type(expression.constructor_type)
            arguments = [
                self.generate_constant_expression(argument)
                for argument in expression.arguments
            ]
            return f"{constructor_type}({', '.join(arguments)})"
        return self.generate_constant_expression(expression, expected_type)

    def generate_directx_compile_time_global_declaration(self, node):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        vtype = getattr(node, "var_type", getattr(node, "vtype", None))
        declaration = format_c_style_array_declaration(self.map_type(vtype), name)
        initializer = self.hlsl_compile_time_initializer_expression(
            node,
            getattr(node, "initial_value", None),
            vtype,
        )
        return f"static const {declaration} = {initializer};\n"

    def referenced_hlsl_glsl_builtin_int_constants(self, ast):
        declared_names = {
            getattr(node, "name", None) for node in getattr(ast, "constants", []) or []
        }
        declared_names.update(
            getattr(node, "name", getattr(node, "variable_name", None))
            for node in getattr(ast, "global_variables", []) or []
        )
        used_names = {
            getattr(node, "name", None)
            for node in self.walk_ast(ast)
            if hasattr(node, "__class__") and "Identifier" in str(node.__class__)
        }
        return {
            name: value
            for name, value in self.HLSL_GLSL_BUILTIN_INT_CONSTANTS.items()
            if name in used_names and name not in declared_names
        }

    def generate_hlsl_glsl_builtin_constant_declarations(self, ast):
        declarations = []
        for name, value in sorted(
            self.referenced_hlsl_glsl_builtin_int_constants(ast).items()
        ):
            declarations.append(f"static const int {name} = {value};")
        return "\n".join(declarations) + "\n\n" if declarations else ""

    def hlsl_bfloat16_constant_numeric_value(self, expr):
        return scalar_constant_value(
            expr,
            constants=self.literal_int_constants,
            function_names=self.function_return_types,
        )

    def generate_constant_expression(self, expr, expected_type=None):
        if self.is_hlsl_bfloat16_type(expected_type):
            numeric_value = self.hlsl_bfloat16_constant_numeric_value(expr)
            if numeric_value is not None:
                bits = bfloat16_constant_bits(numeric_value)
                return f"0x{bits:04x}u"
        value_code = self.generate_expression_with_expected(expr, expected_type)
        if self.is_hlsl_bfloat16_type(expected_type) and any(
            helper in value_code
            for helper in (
                "__crossgl_bfloat16_from_float",
                "__crossgl_bfloat16_from_int",
                "__crossgl_bfloat16_from_uint",
                "__crossgl_bfloat16_to_float",
            )
        ):
            raise self.directx_bfloat16_unsupported(
                "DirectX compile-time bfloat16 values require a constant-foldable "
                "scalar initializer",
                operation="compile-time bfloat16 initializer",
                source_type=expected_type,
                reason="non-constant-bfloat16-initializer",
                source_location=getattr(expr, "source_location", None),
            )
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def hlsl_cbuffer_nodes(self, ast, target_stage=None):
        return list(getattr(ast, "cbuffers", []) or []) + collect_stage_local_cbuffers(
            ast, target_stage
        )

    def is_hlsl_tbuffer_node(self, node):
        return any(
            str(getattr(attr, "name", "")).lower() == "tbuffer"
            for attr in getattr(node, "attributes", []) or []
        )

    def collect_cbuffer_member_types(self, cbuffers):
        member_types = {}
        for cbuffer in cbuffers or []:
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if isinstance(member, ArrayNode):
                    member_type = getattr(
                        member, "element_type", getattr(member, "vtype", "float")
                    )
                    type_name = self.type_name_string(member_type)
                    if getattr(member, "size", None):
                        type_name = f"{type_name}[{member.size}]"
                else:
                    member_type = getattr(
                        member,
                        "member_type",
                        getattr(member, "vtype", "float"),
                    )
                    type_name = self.type_name_string(member_type)
                if type_name:
                    member_types[member_name] = type_name
        return member_types

    def cbuffer_validation_shader_view(self, ast, cbuffers, target_stage=None):
        class CbufferValidationView:
            pass

        view = CbufferValidationView()
        view.cbuffers = list(cbuffers or [])
        view.structs = list(
            getattr(ast, "structs", []) or []
        ) + collect_stage_local_structs(ast, target_stage)
        view.global_variables = list(getattr(ast, "global_variables", []) or [])
        view.global_variables.extend(
            collect_stage_local_variables(
                ast, target_stage, self.is_stage_local_resource_variable
            )
        )
        view.constants = list(getattr(ast, "constants", []) or [])
        return view

    def generate_cbuffers(
        self,
        ast,
        cbuffers=None,
        target_stage=None,
        used_resource_registers=None,
        texture_registers=None,
        cbuffer_registers=None,
    ):
        code = ""
        cbuffers = list(
            cbuffers if cbuffers is not None else getattr(ast, "cbuffers", [])
        )
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise DirectXRefusal(
                f"Duplicate cbuffer name(s) in DirectX output: {names}"
            )

        validation_view = self.cbuffer_validation_shader_view(
            ast, cbuffers, target_stage
        )
        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(
            validation_view
        )
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise DirectXRefusal(
                "Cbuffer name(s) conflict with existing DirectX declaration(s): "
                f"{names}"
            )

        duplicate_members = collect_duplicate_cbuffer_member_names(cbuffers)
        if duplicate_members:
            names = ", ".join(sorted(duplicate_members))
            raise DirectXRefusal(
                f"Ambiguous cbuffer member name(s) in DirectX output: {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(
            validation_view
        )
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise DirectXRefusal(
                "Cbuffer member name(s) conflict with DirectX global declaration(s): "
                f"{names}"
            )
        if used_resource_registers is None:
            used_resource_registers = self.new_directx_resource_register_allocator()
        texture_registers = texture_registers if texture_registers is not None else {}
        cbuffer_registers = cbuffer_registers if cbuffer_registers is not None else {}
        self.reserve_explicit_cbuffer_registers(cbuffers, used_resource_registers)
        self.reset_hlsl_glsl_block_registers()
        for node in cbuffers:
            is_tbuffer = self.is_hlsl_tbuffer_node(node)
            buffer_keyword = "tbuffer" if is_tbuffer else "cbuffer"
            register_prefix = "t" if is_tbuffer else "b"
            space = self.explicit_resource_register_space(node)
            register_cursors = texture_registers if is_tbuffer else cbuffer_registers
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, (register_prefix,)
            )
            if binding is None:
                binding = self.next_available_resource_register(
                    used_resource_registers,
                    register_prefix,
                    register_cursors,
                    space,
                    1,
                )
            self.reserve_resource_register_range(
                used_resource_registers,
                register_prefix,
                binding,
                1,
                node.name,
                space,
                provenance=self.directx_resource_binding_provenance(node),
            )
            self.advance_resource_register(register_cursors, space, binding, 1)
            register = self.resource_register_suffix(register_prefix, binding, node)
            code += resource_identity_marker(node, node.name)
            if isinstance(node, StructNode) or (
                hasattr(node, "name") and hasattr(node, "members")
            ):
                packoffsets = self.hlsl_block_packoffsets(ast, node)
                if isinstance(packoffsets, HlslRegisterBlockPlan):
                    block = self.hlsl_glsl_block_register_buffer(
                        ast,
                        node,
                        packoffsets.layout,
                        packoffsets.layout_type,
                        buffer_keyword,
                        register,
                    )
                else:
                    block = f"{buffer_keyword} {node.name}{register} {{\n"
                    for member in getattr(node, "members", []) or []:
                        declaration = self.generate_hlsl_cbuffer_member_declaration(
                            member, packoffsets.get(getattr(member, "name", None))
                        )
                        block += f"    {declaration};\n"
                    block += "};\n"
                if is_push_constant(node):
                    block = hlsl_push_constant_block(self, ast, node, block)
                code += block
        return code

    def hlsl_block_packoffsets(self, ast, node):
        """``packoffset`` arguments that keep the members of a GLSL block at the offsets of its layout.

        A GLSL uniform or push-constant block keeps the std140, std430 or scalar layout the host writes
        (``@ buffer_layout(...)``). HLSL packing places a ``vec3`` after a ``float``, and any member after an array, a
        matrix or a struct, at other bytes, so every member gets the ``packoffset`` of its offset unless HLSL packing
        already agrees. A struct, array or matrix whose inner layout differs between the two rule sets (a std430
        ``float[]`` has a 4-byte stride, a constant-buffer array a 16-byte one) returns an
        :class:`HlslRegisterBlockPlan`: the block is then read through its registers (``hlsl_glsl_block_registers``).
        """
        layout = declared_buffer_layout(getattr(node, "attributes", None))
        if layout not in BLOCK_LAYOUTS:
            return self.hlsl_cbuffer_explicit_packoffsets(ast, node)
        structs = {
            struct.name: struct
            for struct in getattr(ast, "structs", []) or []
            if getattr(struct, "name", None)
        }

        def struct_members(name):
            struct = structs.get(name)
            if struct is None:
                return None
            return [
                self.hlsl_block_member(member)
                for member in getattr(struct, "members", []) or []
                if getattr(member, "name", None)
                and not isinstance(member, FunctionNode)
            ]

        try:
            layout_type = crossgl_struct_layout_type(
                node.name,
                [
                    self.hlsl_block_member(member)
                    for member in getattr(node, "members", []) or []
                    if getattr(member, "name", None)
                ],
                struct_members,
                (node.name,),
                self.hlsl_constant_int_value,
            )
        except HlslLayoutError as exc:
            raise DirectXBufferLayoutError(node.name, layout, exc) from exc
        try:
            offsets = hlsl_packoffsets_for_block(layout_type, layout)
        except HlslLayoutError:
            # packoffset cannot move an array element, a matrix vector or a
            # nested struct member: read the block through its registers.
            return HlslRegisterBlockPlan(layout, layout_type)
        if offsets is None:
            return {}
        return {
            name: packoffset_text(offset)
            for (name, _member_type), offset in zip(layout_type.members, offsets)
        }

    def hlsl_cbuffer_explicit_packoffsets(self, ast, node):
        """``packoffset`` arguments for the ``@ offset(N)`` members of a constant buffer packed by HLSL rules.

        A constant buffer imported from SPIR-V (``@ buffer_layout(hlsl_cbuffer)`` with ``@ offset`` on its
        members) or written in CrossGL can move a member past the byte HLSL packing gives it. Each member keeps
        its offset; a member without one follows the previous member by the packing rules, as GLSL's
        ``layout(offset = N)`` places the members after it (GLSL 4.60 section 4.4.5). Returns ``{}`` when no
        member names an offset or HLSL packing already agrees.
        """
        members = [
            member
            for member in getattr(node, "members", []) or []
            if getattr(member, "name", None) and not isinstance(member, FunctionNode)
        ]
        structs = {
            struct.name: struct
            for struct in getattr(ast, "structs", []) or []
            if getattr(struct, "name", None)
        }
        explicit = any(member_explicit_offset(member) is not None for member in members)
        if not explicit and not hlsl_members_hold_struct_placements(
            structs, members, self.hlsl_block_member
        ):
            return {}

        def struct_members(name):
            struct = structs.get(name)
            if struct is None:
                return None
            return [
                self.hlsl_block_member(member)
                for member in getattr(struct, "members", []) or []
                if getattr(member, "name", None)
                and not isinstance(member, FunctionNode)
            ]

        try:
            layout_type = crossgl_struct_layout_type(
                node.name,
                [self.hlsl_block_member(member) for member in members],
                struct_members,
                (node.name,),
                self.hlsl_constant_int_value,
            )
            # A struct member's placement that moves a scalar from where HLSL packs it is read from registers.
            plan = hlsl_cbuffer_register_plan(layout_type)
            if plan is not None:
                return plan
            if not explicit:
                return {}
            offsets = []
            natural = []
            cursor = natural_cursor = 0
            for (name, member_type), explicit in zip(
                layout_type.members, layout_type.offsets
            ):
                natural_start, natural_size = cbuffer_placement(
                    member_type, natural_cursor
                )
                natural.append(natural_start)
                natural_cursor = natural_start + natural_size
                if explicit is None:
                    start, size = cbuffer_placement(member_type, cursor)
                else:
                    if explicit < cursor:
                        raise HlslLayoutError(
                            f"member '{name}' at byte {explicit} overlaps the member before it"
                        )
                    start, size = cbuffer_placement(member_type, explicit)
                    if start != explicit:
                        raise HlslLayoutError(
                            f"member '{name}' at byte {explicit} breaks the constant-buffer "
                            f"packing of its type, which starts at byte {start}"
                        )
                offsets.append(start)
                cursor = start + size
            if offsets == natural:
                return {}
            return {
                name: packoffset_text(offset)
                for (name, _member_type), offset in zip(layout_type.members, offsets)
            }
        except HlslLayoutError as exc:
            raise DirectXBufferLayoutError(node.name, "explicit-offset", exc) from exc

    def hlsl_block_member(self, member):
        """``(name, type, row_major, offset, alignment, packed)`` of a uniform-block or struct member."""
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            member_type = ArrayType(element_type, getattr(member, "size", None))
        elif hasattr(member, "member_type"):
            member_type = member.member_type
        else:
            member_type = getattr(member, "vtype", "float")
        return (
            member.name,
            member_type,
            member_is_row_major(member),
            member_explicit_offset(member),
            member_explicit_alignment(member),
            member_is_packed(member),
        )

    def generate_hlsl_cbuffer_member_declaration(self, member, packoffset=None):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            member_type = self.hlsl_bfloat16_storage_type(
                element_type,
                operation=f"cbuffer member '{member.name}'",
                source_location=getattr(member, "source_location", None),
            )
            size = getattr(member, "size", None)
            size_text = self.expression_to_string(size) if size is not None else "1"
            declaration = f"{member_type} {member.name}[{size_text}]"
        else:
            if hasattr(member, "member_type"):
                source_member_type = member.member_type
            else:
                source_member_type = getattr(member, "vtype", "float")
            member_type = self.hlsl_bfloat16_storage_type(
                source_member_type,
                operation=f"cbuffer member '{member.name}'",
                source_location=getattr(member, "source_location", None),
            )
            if (getattr(member, "annotations", {}) or {}).get("directx.half_storage"):
                member_type = self.hlsl_half_storage.physical_type(source_member_type)
            declaration = format_c_style_array_declaration(member_type, member.name)

        layout_qualifier = self.hlsl_cbuffer_member_layout_qualifier(member)
        if layout_qualifier:
            declaration = f"{layout_qualifier} {declaration}"

        packoffset = packoffset or self.hlsl_cbuffer_member_packoffset(member)
        if packoffset:
            declaration = f"{declaration} : packoffset({packoffset})"
        return declaration

    def hlsl_cbuffer_member_layout_qualifier(self, member):
        for qualifier in getattr(member, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name in {"row_major", "column_major"}:
                return qualifier_name

        for attr in getattr(member, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name in {"row_major", "column_major"}:
                return attr_name
        return None

    def hlsl_cbuffer_member_packoffset(self, member):
        for attr in getattr(member, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name != "packoffset":
                continue
            arguments = getattr(attr, "arguments", None) or getattr(attr, "args", [])
            if arguments:
                return self.expression_to_string(arguments[0])
        return None

    def generate_compute_numthreads(self, execution_config=None):
        require_fixed_workgroup_size(
            execution_config,
            "HLSL",
            "Direct3D has no specialization constants, and numthreads takes literals",
        )
        if is_host_sized_workgroup(execution_config):
            return hlsl_host_numthreads()
        x, y, z = self.validate_hlsl_compute_workgroup_size(
            compute_local_size(execution_config)
        )
        return f"[numthreads({x}, {y}, {z})]\n"

    def validate_hlsl_compute_workgroup_size(self, values):
        resolved = []
        for value in values:
            text = str(value).strip()
            if not re.fullmatch(r"[+-]?[0-9]+", text):
                return tuple(values)
            resolved.append(int(text))
        if any(value <= 0 for value in resolved):
            raise DirectXWorkgroupSizeError(
                "DirectX compute workgroup dimensions must be positive integers",
                workgroup_size=resolved,
                reason="non-positive-dimension",
            )
        if resolved[0] > 1024 or resolved[1] > 1024 or resolved[2] > 64:
            raise DirectXWorkgroupSizeError(
                "DirectX compute workgroup dimensions exceed HLSL limits "
                "(x <= 1024, y <= 1024, z <= 64)",
                workgroup_size=resolved,
                reason="dimension-limit-exceeded",
            )
        if resolved[0] * resolved[1] * resolved[2] > 1024:
            raise DirectXWorkgroupSizeError(
                "DirectX compute workgroup size must not exceed 1024 threads",
                workgroup_size=resolved,
                reason="thread-count-limit-exceeded",
            )
        return tuple(resolved)

    def generate_stage_numthreads(self, func, shader_type, execution_config=None):
        if shader_type not in {"mesh", "task", "amplification", "object"}:
            return ""

        arguments = self.hlsl_stage_attribute_arguments(func, "numthreads")
        if arguments:
            if len(arguments) != 3:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage numthreads requires exactly "
                    "three arguments"
                )
            values = [
                self.hlsl_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            self.validate_hlsl_mesh_thread_group_size(
                shader_type,
                [self.hlsl_constant_int_value(argument) for argument in arguments],
            )
            return f"[numthreads({', '.join(values)})]\n"

        if is_host_sized_workgroup(execution_config):
            return hlsl_host_numthreads()
        values = compute_local_size(execution_config)
        self.validate_hlsl_mesh_thread_group_size(shader_type, values)
        return f"[numthreads({values[0]}, {values[1]}, {values[2]})]\n"

    def hlsl_alias_resolved_type_text(self, raw_type):
        """``raw_type``'s spelling with type aliases resolved, so ``typedef float Triple[3]`` reads as ``float[3]``."""
        type_text = self.type_name_string(raw_type)
        aliases = getattr(self, "hlsl_type_aliases", {})
        seen = set()
        while type_text in aliases and type_text not in seen:
            seen.add(type_text)
            type_text = self.type_name_string(aliases[type_text])
        return type_text

    def hlsl_fixed_array_return_components(self, raw_return_type):
        type_text = self.hlsl_alias_resolved_type_text(raw_return_type)
        if not type_text:
            return None
        base_type, array_suffix = split_array_type_suffix(type_text)
        if not array_suffix:
            return None
        match = re.fullmatch(r"\[(\d+)\]", array_suffix.strip())
        if match is None:
            return None
        component_count = int(match.group(1))
        if component_count <= 0:
            return None
        return type_text, base_type, component_count

    def hlsl_fixed_array_return_vector_type(self, raw_return_type):
        components = self.hlsl_fixed_array_return_components(raw_return_type)
        if components is None:
            return None
        _type_text, base_type, component_count = components
        if component_count not in {2, 3, 4}:
            return None
        if self.is_hlsl_bfloat16_type(base_type):
            # bfloat16 lanes are uint payloads; a uintN vector would coerce
            # each element numerically, so keep the logical element type in
            # the fixed-array struct lowering instead.
            return None
        mapped_base = self.map_type(base_type)
        if not self.is_scalar_value_type(mapped_base):
            return None
        return f"{mapped_base}{component_count}"

    def hlsl_fixed_array_return_field_name(self, index):
        return f"value{index}"

    def hlsl_fixed_array_return_struct_type(self, raw_return_type):
        components = self.hlsl_fixed_array_return_components(raw_return_type)
        if components is None:
            return None
        type_text, base_type, component_count = components
        vector_type = self.hlsl_fixed_array_return_vector_type(type_text)
        if vector_type is not None:
            return None

        registered_name = self.hlsl_fixed_array_return_structs.get(type_text)
        if registered_name is not None:
            return registered_name

        mapped_base = self.map_type(base_type)
        base_name = sanitize_type_name(mapped_base or base_type or "value") or "value"
        candidate_name = f"CrossGLFixedArray_{base_name}_{component_count}"
        struct_name = candidate_name
        existing_metadata = self.hlsl_fixed_array_return_struct_metadata.get(
            struct_name
        )
        if (struct_name in self.structs_by_name or existing_metadata is not None) and (
            existing_metadata is None or existing_metadata.get("type_text") != type_text
        ):
            digest = sha256(type_text.encode("utf-8")).hexdigest()[:8]
            struct_name = f"{candidate_name}_{digest}"

        members = [
            StructMemberNode(
                self.hlsl_fixed_array_return_field_name(index),
                base_type,
            )
            for index in range(component_count)
        ]
        struct_node = StructNode(struct_name, members)
        field_types = {member.name: base_type for member in members}

        self.hlsl_fixed_array_return_structs[type_text] = struct_name
        self.hlsl_fixed_array_return_struct_nodes[struct_name] = struct_node
        self.hlsl_fixed_array_return_struct_metadata[struct_name] = {
            "type_text": type_text,
            "base_type": base_type,
            "size": component_count,
        }
        self.structs_by_name[struct_name] = struct_node
        self.struct_member_types[struct_name] = field_types
        return struct_name

    def refresh_hlsl_fixed_array_return_struct_member_types(self):
        for (
            struct_name,
            metadata,
        ) in self.hlsl_fixed_array_return_struct_metadata.items():
            self.struct_member_types[struct_name] = {
                self.hlsl_fixed_array_return_field_name(index): metadata["base_type"]
                for index in range(metadata["size"])
            }

    def hlsl_fixed_array_return_type_name(self, raw_return_type):
        vector_type = self.hlsl_fixed_array_return_vector_type(raw_return_type)
        if vector_type is not None:
            return vector_type
        return self.hlsl_fixed_array_return_struct_type(raw_return_type)

    def hlsl_function_return_type_name(self, raw_return_type):
        fixed_array_type = self.hlsl_fixed_array_return_type_name(raw_return_type)
        if fixed_array_type is not None:
            return fixed_array_type
        return self.type_name_string(raw_return_type)

    def hlsl_fixed_array_value_expression(
        self,
        rendered,
        expected_type,
        source_type,
    ):
        if not rendered:
            return None
        components = self.hlsl_fixed_array_return_components(source_type)
        if components is None:
            return None
        _type_text, _base_type, component_count = components
        lowered_type = self.hlsl_fixed_array_return_type_name(source_type)
        if lowered_type is None:
            return None
        expected = self.map_type(expected_type)
        if expected != self.map_type(lowered_type):
            return None
        values = [f"{rendered}[{index}]" for index in range(component_count)]
        if lowered_type in self.hlsl_fixed_array_return_struct_metadata:
            # HLSL constructs only numeric types; a structure needs the generated constructor function.
            return self.format_struct_constructor_expression(lowered_type, values)
        return f"{lowered_type}({', '.join(values)})"

    def hlsl_direct_recursive_call_nodes(self, function):
        source_name = getattr(function, "name", None)
        if not source_name:
            return {}
        function_signature = self.hlsl_source_function_signature(function)
        recursive_calls = {}
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            if self.function_call_name(node) != source_name:
                continue
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            try:
                resolved = self.resolve_hlsl_function_overload(
                    source_name,
                    arguments,
                    call_node=node,
                )
            except DirectXMappedOverloadError:
                continue
            if resolved is None:
                continue
            if self.hlsl_source_function_signature(resolved) == function_signature:
                recursive_calls[id(node)] = node
        return recursive_calls

    @staticmethod
    def hlsl_tail_recursive_argument_is_same_parameter(argument, parameter_name):
        if isinstance(argument, str):
            return argument.strip() == parameter_name
        if isinstance(argument, (IdentifierNode, VariableNode)):
            return getattr(argument, "name", None) == parameter_name
        return False

    def hlsl_validate_tail_recursive_call_arguments(self, function, call):
        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        target_name = self.hlsl_function_declaration_name(function)
        if len(parameters) != len(arguments):
            raise DirectXRecursiveFunctionError(
                "DirectX tail-recursion lowering requires an exact parameter and "
                "argument correspondence",
                function_name=target_name,
                cycle=(target_name, target_name),
                reason="tail-argument-count-mismatch",
                source_location=getattr(call, "source_location", None),
            )
        for parameter, argument in zip(parameters, arguments):
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                raise DirectXRecursiveFunctionError(
                    "DirectX tail-recursion lowering requires named parameters",
                    function_name=target_name,
                    cycle=(target_name, target_name),
                    reason="tail-parameter-unnamed",
                    source_location=getattr(call, "source_location", None),
                )
            if self.hlsl_tail_recursive_argument_is_same_parameter(
                argument, parameter_name
            ):
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if (
                qualifiers.intersection(
                    {"const", "constant", "readonly", "out", "inout"}
                )
                or isinstance(parameter_type, (ArrayType, PointerType, ReferenceType))
                or self.is_directx_compile_time_resource_type(parameter_type)
            ):
                raise DirectXRecursiveFunctionError(
                    "DirectX tail-recursion lowering cannot rebind pointer, "
                    "resource, const, out, or inout parameter "
                    f"'{parameter_name}'",
                    function_name=target_name,
                    cycle=(target_name, target_name),
                    reason="tail-parameter-rebind-unsupported",
                    source_location=getattr(call, "source_location", None),
                )
