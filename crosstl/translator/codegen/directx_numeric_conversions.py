"""Numeric conversions in the HLSL generator: bfloat16, integer and floating vectors, local statements, contextual and narrowing conversions, byte values, binary arithmetic and tail recursion."""

import re

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    narrow_integer_shape,
)
from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .builtin_result_types import (
    C_FAMILY,
    inferred_value_type,
)
from .directx_values import (
    DirectXContextualConversionError,
    DirectXRecursiveFunctionError,
)


class DirectXNumericConversionMixin:
    """Numeric conversion members of ``HLSLCodeGen``."""

    def hlsl_tail_recursive_void_call_ids(self, function):
        recursive_calls = self.hlsl_direct_recursive_call_nodes(function)
        if not recursive_calls:
            return set()
        target_name = self.hlsl_function_declaration_name(function)
        first_call = next(iter(recursive_calls.values()))
        return_type = self.type_name_string(getattr(function, "return_type", None))
        if self.map_type(return_type or "void") != "void":
            raise DirectXRecursiveFunctionError(
                "DirectX does not support recursive non-void functions",
                function_name=target_name,
                cycle=(target_name, target_name),
                reason="non-void-recursion-unsupported",
                source_location=getattr(first_call, "source_location", None),
            )

        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            if getattr(parameter, "name", None)
        }
        if any(
            isinstance(node, (VariableNode, ArrayNode))
            and getattr(node, "name", None) in parameter_names
            for node in self.walk_ast(getattr(function, "body", []))
        ):
            raise DirectXRecursiveFunctionError(
                "DirectX cannot lower tail recursion when a local declaration "
                "shadows a parameter",
                function_name=target_name,
                cycle=(target_name, target_name),
                reason="tail-parameter-shadowed",
                source_location=getattr(first_call, "source_location", None),
            )

        lowered = set()
        valid = True

        def body_statements(body):
            if body is None:
                return []
            if isinstance(body, list):
                return body
            statements = getattr(body, "statements", None)
            if statements is not None:
                return list(statements or [])
            return [body]

        def direct_call(statement):
            if isinstance(statement, FunctionCallNode):
                return statement
            if isinstance(statement, ExpressionStatementNode):
                expression = getattr(statement, "expression", None)
                return expression if isinstance(expression, FunctionCallNode) else None
            return None

        def contains_recursive_call(value):
            return any(id(node) in recursive_calls for node in self.walk_ast(value))

        def visit_sequence(body, tail_position):
            statements = body_statements(body)
            for index, statement in enumerate(statements):
                visit_statement(
                    statement, tail_position and index == len(statements) - 1
                )

        def visit_statement(statement, tail_position):
            nonlocal valid
            call = direct_call(statement)
            if call is not None and id(call) in recursive_calls:
                if not tail_position:
                    valid = False
                    return
                lowered.add(id(call))
                return
            if isinstance(statement, BlockNode):
                visit_sequence(statement, tail_position)
                return
            if isinstance(statement, IfNode):
                conditions = [
                    getattr(
                        statement, "condition", getattr(statement, "if_condition", None)
                    ),
                    *(getattr(statement, "else_if_conditions", []) or []),
                ]
                if any(contains_recursive_call(condition) for condition in conditions):
                    valid = False
                    return
                visit_sequence(getattr(statement, "if_body", None), tail_position)
                for branch in getattr(statement, "else_if_bodies", []) or []:
                    visit_sequence(branch, tail_position)
                visit_sequence(getattr(statement, "else_body", None), tail_position)
                return
            if contains_recursive_call(statement):
                valid = False

        visit_sequence(getattr(function, "body", []), True)
        if not valid or lowered != set(recursive_calls):
            raise DirectXRecursiveFunctionError(
                "DirectX does not support recursive calls unless every direct "
                "self-call is a safely lowerable void tail call",
                function_name=target_name,
                cycle=(target_name, target_name),
                reason="non-tail-recursion-unsupported",
                source_location=getattr(first_call, "source_location", None),
            )
        for call in recursive_calls.values():
            self.hlsl_validate_tail_recursive_call_arguments(function, call)
        return lowered

    def generate_hlsl_tail_recursive_call_statement(self, expression, indent):
        if (
            not isinstance(expression, FunctionCallNode)
            or id(expression) not in self.current_hlsl_tail_recursive_call_ids
        ):
            return None
        function = self.current_hlsl_tail_recursive_function
        if function is None:
            return None
        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        arguments = list(
            getattr(expression, "arguments", getattr(expression, "args", [])) or []
        )
        if len(parameters) != len(arguments):
            return None

        pending = []
        used_names = set(self.current_identifier_reserved_names)
        used_names.update(self.local_variable_types)
        for parameter, argument in zip(parameters, arguments):
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                return None
            if self.hlsl_tail_recursive_argument_is_same_parameter(
                argument, parameter_name
            ):
                continue
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            rendered = self.generate_expression_with_expected(argument, parameter_type)
            temporary = self.hlsl_unique_local_identifier(
                f"{parameter_name}_tail_value", used_names
            )
            used_names.add(temporary)
            self.current_identifier_reserved_names.add(temporary)
            self.local_variable_types[temporary] = self.type_name_string(parameter_type)
            self.local_variable_source_types[temporary] = self.type_name_string(
                parameter_type
            )
            pending.append((parameter_name, parameter_type, temporary, rendered))

        indent_str = "    " * indent
        code = ""
        for _parameter_name, parameter_type, temporary, rendered in pending:
            declaration = format_c_style_array_declaration(
                self.map_type(parameter_type), temporary
            )
            code += f"{indent_str}{declaration} = {rendered};\n"
        for parameter_name, _parameter_type, temporary, _rendered in pending:
            target = self.hlsl_identifier_name(parameter_name)
            code += f"{indent_str}{target} = {temporary};\n"
        code += f"{indent_str}continue;\n"
        return code

    def generate_hlsl_parameter_epilogue(self, indent):
        return "".join(
            f"{'    ' * indent}{statement}\n"
            for statement in self.current_hlsl_parameter_epilogue
        )

    def generate_hlsl_return_value_statement(self, value, indent):
        """Evaluate the result before pointer writeback, preserving precision."""
        indent_str = "    " * indent
        return_type = self.current_hlsl_precise_return_type
        epilogue = self.generate_hlsl_parameter_epilogue(indent)
        if return_type is None and not epilogue:
            return f"{indent_str}return {value};\n"

        precise = "precise " if return_type is not None else ""
        return_type = return_type or self.map_type(self.current_function_return_type)
        if return_type == "void":
            return f"{indent_str}{value};\n{epilogue}{indent_str}return;\n"

        used_names = set(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.function_return_types)
        used_names.update(self.structs_by_name)
        used_names.update(self.hlsl_type_aliases)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(self.current_identifier_reserved_names)
        name = self.hlsl_unique_local_identifier(
            "crossglPreciseReturn" if precise else "crossglReturn", used_names
        )
        self.current_identifier_reserved_names.add(name)
        source_type = self.type_name_string(self.current_function_return_type)
        self.local_variable_types[name] = source_type
        self.local_variable_source_types[name] = source_type
        declaration = format_c_style_array_declaration(return_type, name)
        return (
            f"{indent_str}{precise}{declaration} = {value};\n"
            f"{epilogue}"
            f"{indent_str}return {name};\n"
        )

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        aggregate_return = self.generate_hlsl_aggregate_conditional_return(
            stmt.expression, indent
        )
        if aggregate_return is not None:
            return aggregate_return

        aggregate_value = self.render_hlsl_aggregate_conditional_value_expression(
            stmt.expression,
            return_type,
            indent,
            context="return",
        )
        if aggregate_value is not None:
            aggregate_code, value = aggregate_value
            return aggregate_code + self.generate_hlsl_return_value_statement(
                value, indent
            )

        value = self.generate_expression_with_expected(stmt.expression, return_type)
        return self.generate_hlsl_return_value_statement(value, indent)

    def infer_hlsl_global_types(self, global_vars):
        """Resolve untyped globals before declaration and expression lowering."""
        self.hlsl_inferred_global_node_ids = set()
        for node in global_vars:
            if not hasattr(node, "var_type") or node.var_type is not None:
                continue
            vtype = inferred_value_type(
                self, getattr(node, "initial_value", None), C_FAMILY
            )
            if not vtype:
                self.refuse_lowering(
                    f"global declaration: the type of '{getattr(node, 'name', '?')}' "
                    "is not written and cannot be inferred from its initializer"
                )
            node.var_type = vtype
            node.vtype = vtype
            self.global_variable_types[node.name] = vtype
            self.hlsl_inferred_global_node_ids.add(id(node))

    def local_variable_declared_type(self, stmt):
        vtype = getattr(stmt, "var_type", None)
        if vtype is None:
            vtype = getattr(stmt, "vtype", None)
        vtype_name = self.type_name_string(vtype)
        if isinstance(vtype_name, str) and vtype_name.strip().lower() in {
            "auto",
            "let",
        }:
            inferred_type = inferred_value_type(
                self, getattr(stmt, "initial_value", None), C_FAMILY
            )
            if inferred_type is not None:
                return inferred_type
            return self.refuse_untyped_local(stmt)
        if vtype is None:
            vtype = inferred_value_type(
                self, getattr(stmt, "initial_value", None), C_FAMILY
            )
        if not vtype:
            return self.refuse_untyped_local(stmt)
        fixed_array_value_type = self.hlsl_fixed_array_local_value_type(
            vtype,
            getattr(stmt, "initial_value", None),
        )
        return fixed_array_value_type or vtype

    def refuse_untyped_local(self, stmt):
        """Refuse a local whose type is neither written nor inferable from its initializer."""
        return self.refuse_lowering(
            f"local declaration: the type of '{getattr(stmt, 'name', '?')}' is not "
            "written and cannot be inferred from its initializer",
            "float",
        )

    def hlsl_fixed_array_local_value_type(self, declared_type, initializer):
        if initializer is None:
            return None
        declared_components = self.hlsl_fixed_array_return_components(declared_type)
        if declared_components is None:
            return None

        initializer_source_type = self.hlsl_source_expression_type(initializer)
        if initializer_source_type is None or (
            self.hlsl_source_type_signature(initializer_source_type)
            != self.hlsl_source_type_signature(declared_type)
        ):
            return None

        lowered_type = self.hlsl_fixed_array_return_type_name(declared_type)
        initializer_value_type = self.expression_result_type(initializer)
        if self.map_type(initializer_value_type) != self.map_type(lowered_type):
            return None
        return lowered_type

    def hlsl_source_variable_declared_type(self, stmt):
        vtype = getattr(stmt, "var_type", None)
        if vtype is None:
            vtype = getattr(stmt, "vtype", None)
        vtype_name = self.type_name_string(vtype)
        if (
            not isinstance(vtype, (PointerType, ReferenceType))
            and isinstance(vtype_name, str)
            and vtype_name.strip().lower() in {"auto", "let"}
        ):
            return self.hlsl_source_expression_type(
                getattr(stmt, "initial_value", None)
            ) or self.local_variable_declared_type(stmt)
        if vtype is None:
            return self.hlsl_source_expression_type(
                getattr(stmt, "initial_value", None)
            ) or self.local_variable_declared_type(stmt)
        return vtype

    def local_variable_qualifier(self, node, *, omit_const=False):
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        rendered = []
        if qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"}:
            rendered.append("groupshared")
        if "buffer" in qualifiers:
            rendered.append("static")
        if self.hlsl_has_precise_modifier(node) or self.hlsl_precise_function_local(
            node
        ):
            rendered.append("precise")
        if "const" in qualifiers and not omit_const:
            rendered.append("const")
        return f"{' '.join(rendered)} " if rendered else ""

    def hlsl_has_precise_modifier(self, node):
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        if "precise" in qualifiers:
            return True
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() == "precise":
                return True
        return False

    def hlsl_precise_modifier_prefix(self, node):
        return "precise " if self.hlsl_has_precise_modifier(node) else ""

    def hlsl_matrix_orientation_prefix(self, node):
        """``row_major `` or ``column_major `` for a declaration that sets its matrix orientation."""
        qualifier = self.hlsl_cbuffer_member_layout_qualifier(node)
        return f"{qualifier} " if qualifier else ""

    def hlsl_canonical_compile_time_type_text(self, type_text):
        type_text = str(type_text)
        for name, value in sorted(self.literal_int_constants.items()):
            type_text = type_text.replace(f"[{name}]", f"[{value}]")
        return type_text

    def hlsl_named_declaration_signature(self, node):
        declared_type = getattr(
            node,
            "var_type",
            getattr(node, "vtype", getattr(node, "param_type", None)),
        )
        qualifiers = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"groupshared", "shared", "threadgroup", "workgroup"}:
                normalized = "groupshared"
            qualifiers.append(normalized)
        return (
            node.__class__.__name__,
            self.hlsl_canonical_compile_time_type_text(
                self.type_name_string(declared_type)
            ),
            tuple(sorted(qualifiers)),
            repr(getattr(node, "attributes", None)),
        )

    def deduplicate_hlsl_groupshared_declarations(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                declarations.append(node)
                continue

            signature = self.hlsl_named_declaration_signature(node)
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = signature
                declarations.append(node)
                continue
            if existing != signature:
                raise DirectXRefusal(
                    f"Conflicting groupshared variable declaration for '{name}'"
                )
        return declarations

    def generate_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        if not lines:
            return ""

        result = ""
        for line in lines:
            terminator = "" if line.endswith((";", "{", "}")) else ";"
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if vtype.__class__ is str:
            return vtype
        if isinstance(vtype, ReferenceType) or hasattr(vtype, "referenced_type"):
            return self.convert_type_node_to_string(vtype)
        if isinstance(vtype, PointerType) or hasattr(vtype, "pointee_type"):
            return self.convert_type_node_to_string(vtype)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def generate_expression_with_expected(
        self, expr, expected_type, *, source_location=None
    ):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            rendered = self.generate_expression(expr)
            source_type = self.expression_result_type(expr)
            fixed_array_value = self.hlsl_fixed_array_value_expression(
                rendered,
                expected_type,
                source_type,
            )
            if fixed_array_value is not None:
                return fixed_array_value
            return self.hlsl_narrowing_cast_expression(
                rendered,
                expected_type,
                source_type,
                source_location=(
                    source_location
                    if source_location is not None
                    else getattr(expr, "source_location", None)
                ),
            )
        finally:
            self.current_expression_expected_type = previous_expected_type

    def require_hlsl_bfloat16_helper(self, helper_name):
        self.required_hlsl_bfloat16_helpers.add(helper_name)

    def hlsl_bfloat16_to_float_expression(self, rendered):
        self.require_hlsl_bfloat16_helper("to_float")
        return f"__crossgl_bfloat16_to_float(uint({rendered}))"

    _HLSL_BFLOAT16_VECTOR_SPELLINGS = (
        re.compile(
            r"(?:metal::)?(?:vec|vector)\s*<\s*(?:bfloat|bfloat16|bfloat16_t)"
            r"\s*,\s*([234])\s*>"
        ),
        re.compile(r"(?:bfloat16vec|bfloat)([234])"),
    )

    def hlsl_bfloat16_vector_width(self, vtype):
        """Return N for a source ``vec<bfloat16_t, N>`` value type."""

        type_name = self.type_name_string(vtype)
        if not isinstance(type_name, str):
            return None
        for pattern in self._HLSL_BFLOAT16_VECTOR_SPELLINGS:
            match = pattern.fullmatch(type_name.strip())
            if match is not None:
                return int(match.group(1))
        return None

    def hlsl_bfloat16_vector_to_float_expression(self, rendered, width):
        # DirectX carries bfloat16 lanes in the low 16 bits of uint lanes.
        self.require_hlsl_bfloat16_helper("to_float")
        self.require_hlsl_bfloat16_helper("to_float_vectors")
        return f"__crossgl_bfloat16_to_float(uint{width}({rendered}))"

    def hlsl_float_to_bfloat16_expression(self, rendered):
        self.require_hlsl_bfloat16_helper("from_float")
        return f"__crossgl_bfloat16_from_float(float({rendered}))"

    def hlsl_groupshared_bfloat16_array_element(self, target):
        """Whether ``target`` is one scalar of a bfloat16 groupshared array of
        arrays, such as MLX Winograd's ``threadgroup T Os[M][M][BO]``.

        One-dimensional groupshared arrays bind as workgroup pointers, whose
        stores already narrow; arrays of arrays keep their subscripts and are
        declared with the same ``uint16_t`` storage.
        """

        depth, root = 0, target
        while isinstance(root, ArrayAccessNode):
            root, depth = root.array, depth + 1
        name = getattr(root, "name", None)
        if depth < 2 or not isinstance(root, (IdentifierNode, VariableNode)):
            return False
        # A hoisted groupshared array is read through the alias its activation records; the alias is a fresh
        # global name, so an identifier without one is not a hoisted array.
        alias = self.current_identifier_aliases.get(name)
        if alias not in self.hlsl_hoisted_groupshared_aliases_by_declaration.values():
            return False
        element_type, array_suffix = split_array_type_suffix(
            str(self.global_variable_types.get(alias) or "")
        )
        return array_suffix.count("[") == depth and self.is_hlsl_bfloat16_type(
            element_type
        )

    def hlsl_bfloat16_storage_assignment_expression(
        self, target, rendered, target_type, *, source_type=None
    ):
        storage_expression = None
        if isinstance(target, ArrayAccessNode):
            storage_expression = target.array
        elif (
            isinstance(target, UnaryOpNode)
            and not getattr(target, "is_postfix", False)
            and target.op == "*"
        ):
            storage_expression = target.operand
        if storage_expression is None:
            return rendered

        binding = self.hlsl_resource_pointer_binding(storage_expression)
        if binding is None and self.hlsl_groupshared_bfloat16_array_element(target):
            # Register bfloat16 values and conversions are uint low-bit
            # payloads; the uint16_t element takes the 16-bit payload.
            storage_type = self.hlsl_bfloat16_storage_type(
                target_type,
                operation="groupshared assignment",
                source_location=getattr(target, "source_location", None),
            )
            return f"{storage_type}({rendered})"
        if binding is None or binding.get("access") not in {"write", "read_write"}:
            return rendered
        source_element_type = binding.get("source_element_type")
        logical_target_type = (
            target_type
            if self.is_hlsl_bfloat16_type(target_type)
            else source_element_type
        )
        physical_element_type = binding.get("element_type")
        physical_bfloat_store = (
            self.is_hlsl_bfloat16_type(source_type)
            and getattr(self, "requires_hlsl_bfloat16_storage", False)
            and self.map_type(physical_element_type) == "uint16_t"
        )
        if (
            not self.is_hlsl_bfloat16_type(logical_target_type)
            and not physical_bfloat_store
        ):
            return rendered
        resource_name = self.hlsl_resource_type_name(binding.get("resource_type"))
        if (
            resource_name not in {
                "RWBuffer",
                "RWStructuredBuffer",
                "RasterizerOrderedBuffer",
                "RasterizerOrderedStructuredBuffer",
            }
            and binding.get("kind") != "workgroup-pointer"
        ):
            return rendered

        storage_type = self.hlsl_bfloat16_storage_type(
            (
                logical_target_type
                if self.is_hlsl_bfloat16_type(logical_target_type)
                else source_type
            ),
            operation="storage assignment",
            source_location=getattr(target, "source_location", None),
        )
        return f"{storage_type}({rendered})"

    def hlsl_bfloat16_conversion_expression(
        self, rendered, expected_type, source_type, *, source_location=None
    ):
        registered_conversion = (
            self.hlsl_registered_structure_scalar_conversion_expression(
                rendered,
                expected_type,
                source_type,
                source_location=source_location,
            )
        )
        if registered_conversion is not None:
            return registered_conversion

        vector_conversion = self.hlsl_bfloat16_vector_conversion_expression(
            rendered, expected_type, source_type, source_location=source_location
        )
        if vector_conversion is not None:
            return vector_conversion

        expected_is_bfloat = self.is_hlsl_bfloat16_type(expected_type)
        source_is_bfloat = self.is_hlsl_bfloat16_type(source_type)
        if not expected_is_bfloat and not source_is_bfloat:
            return None
        if expected_is_bfloat and source_is_bfloat:
            return rendered

        if expected_is_bfloat:
            mapped_source = self.map_type(source_type)
            if not self.is_scalar_value_type(mapped_source):
                raise self.directx_bfloat16_unsupported(
                    "DirectX exact bfloat16 conversion requires a scalar numeric "
                    f"source, got '{self.type_name_string(source_type)}'",
                    operation="numeric conversion to bfloat16",
                    source_type=source_type,
                    reason="unsupported-conversion-source",
                    source_location=source_location,
                )
            if mapped_source == "bool":
                rendered = f"(({rendered}) ? 1.0 : 0.0)"
            if mapped_source in {"int", "uint"}:
                self.require_hlsl_bfloat16_helper("from_uint")
                self.require_hlsl_bfloat16_helper(f"from_{mapped_source}")
                return f"__crossgl_bfloat16_from_{mapped_source}({mapped_source}({rendered}))"
            if mapped_source in {"int64_t", "uint64_t"}:
                helper = "from_int64" if mapped_source == "int64_t" else "from_uint64"
                self.require_hlsl_bfloat16_helper("from_uint64")
                self.require_hlsl_bfloat16_helper(helper)
                return f"__crossgl_bfloat16_{helper}({mapped_source}({rendered}))"
            return self.hlsl_float_to_bfloat16_expression(rendered)

        decoded = self.hlsl_bfloat16_to_float_expression(rendered)
        mapped_expected = self.map_type(expected_type)
        if mapped_expected == "float":
            return decoded
        if mapped_expected == "bool":
            return f"({decoded} != 0.0)"
        if self.is_scalar_value_type(mapped_expected):
            if mapped_expected == "float16_t":
                return self.hlsl_half_conversion_expression(
                    decoded, 1, ("float",), source_location=source_location
                )
            return f"{mapped_expected}({decoded})"
        raise self.directx_bfloat16_unsupported(
            "DirectX exact bfloat16 conversion requires a scalar numeric target, "
            f"got '{self.type_name_string(expected_type)}'",
            operation="numeric conversion from bfloat16",
            source_type=source_type,
            reason="unsupported-conversion-target",
            source_location=source_location,
        )

    def hlsl_integer_conversion_bit_width(self, type_info):
        if type_info is None:
            return None
        return {
            "int64_t": 64,
            "uint64_t": 64,
            "int": 32,
            "uint": 32,
            "int16_t": 16,
            "uint16_t": 16,
            "min16int": 16,
            "min16uint": 16,
            "min12int": 12,
        }.get(type_info["base_type"])

    def hlsl_contextual_integer_narrowing_expression(
        self,
        rendered,
        expected_type,
        source_type,
        *,
        source_location=None,
    ):
        """Render one source-defined integer narrowing conversion when required.

        HLSL constructors preserve the source assignment's numeric conversion and
        evaluate their argument exactly once. They are only valid for identical
        scalar/vector shapes, so unequal lane counts remain diagnostics instead
        of being rewritten into a different computation.
        """
        expected_info = self.hlsl_integer_arithmetic_type_info(expected_type)
        source_info = self.hlsl_integer_arithmetic_type_info(source_type)
        if expected_info is None or source_info is None:
            return None

        expected_bits = self.hlsl_integer_conversion_bit_width(expected_info)
        source_bits = self.hlsl_integer_conversion_bit_width(source_info)
        if expected_bits is None or source_bits is None:
            return None
        if expected_bits >= source_bits:
            return None
        if expected_info["width"] != source_info["width"]:
            raise DirectXContextualConversionError(
                "DirectX cannot lower source integer narrowing from "
                f"'{self.map_type(source_type)}' to '{self.map_type(expected_type)}' "
                "because the scalar/vector lane shapes differ",
                source_type=self.map_type(source_type),
                target_type=self.map_type(expected_type),
                reason="integer-lane-shape-mismatch",
                source_location=source_location,
            )
        return f"{expected_info['mapped_type']}({rendered})"

    def hlsl_contextual_floating_narrowing_expression(
        self,
        rendered,
        expected_type,
        source_type,
        *,
        source_location=None,
    ):
        """Render an explicit source-defined floating narrowing conversion.

        DXC diagnoses implicit float/double to native or minimum-precision
        assignments under ``-Wconversion``. Native binary16 uses explicit
        nearest-even rounding; minimum-precision types retain HLSL casts. Limit this
        lowering to identical scalar/vector shapes; shape conversion remains a
        separate constructor or semantic-analysis contract.
        """
        expected_info = self.hlsl_floating_arithmetic_type_info(expected_type)
        source_info = self.hlsl_floating_arithmetic_type_info(source_type)
        if expected_info is None or source_info is None:
            return None

        precision_bits = {
            "double": 64,
            "float": 32,
            "half": 16,
            "float16_t": 16,
            "min16float": 16,
            "min10float": 10,
        }
        expected_bits = precision_bits.get(expected_info["base_type"])
        source_bits = precision_bits.get(source_info["base_type"])
        if (
            expected_bits is None
            or source_bits is None
            or expected_bits >= source_bits
            or expected_info["width"] != source_info["width"]
        ):
            return None
        if expected_info["base_type"] == "float16_t":
            return self.hlsl_half_conversion_expression(
                rendered,
                expected_info["width"],
                (source_type,),
                source_location=source_location,
            )
        return f"{expected_info['mapped_type']}({rendered})"

    def hlsl_contextual_wide_integer_to_float_expression(
        self,
        rendered,
        expected_type,
        source_type,
        *,
        source_location=None,
    ):
        """Make source-defined int64/uint64-to-float conversions explicit."""
        expected_info = self.hlsl_floating_arithmetic_type_info(expected_type)
        source_info = self.hlsl_integer_arithmetic_type_info(source_type)
        if (
            expected_info is None
            or expected_info["base_type"] != "float"
            or source_info is None
            or source_info["base_type"] not in {"int64_t", "uint64_t"}
        ):
            return None
        if expected_info["width"] != source_info["width"]:
            raise DirectXContextualConversionError(
                "DirectX cannot lower source 64-bit-integer-to-float conversion "
                f"from '{source_info['mapped_type']}' to "
                f"'{expected_info['mapped_type']}' because the scalar/vector lane "
                "shapes differ",
                source_type=source_info["mapped_type"],
                target_type=expected_info["mapped_type"],
                reason="wide-integer-floating-lane-shape-mismatch",
                source_location=source_location,
            )
        return self.hlsl_wide_integer_float_conversion_expression(
            rendered,
            expected_info,
            source_info,
            source_location=source_location,
        )

    def hlsl_narrowing_cast_expression(
        self, rendered, expected_type, source_type, *, source_location=None
    ):
        expected_name = self.type_name_string(expected_type)
        source_name = self.type_name_string(source_type)
        if not rendered or not expected_name or not source_name:
            return rendered

        registered_conversion = (
            self.hlsl_registered_structure_scalar_conversion_expression(
                rendered,
                expected_name,
                source_name,
                source_location=source_location,
            )
        )
        if registered_conversion is not None:
            return registered_conversion

        bfloat_conversion = self.hlsl_bfloat16_conversion_expression(
            rendered,
            expected_name,
            source_name,
        )
        if bfloat_conversion is not None:
            return bfloat_conversion

        byte_conversion = self.hlsl_byte_conversion_expression(rendered, expected_name)
        if byte_conversion is not None:
            return byte_conversion

        expected = self.map_type(expected_name)
        source = self.map_type(source_name)
        if not expected or not source or expected == source:
            return rendered
        unsigned_to_float = self.hlsl_unsigned_float_conversion_expression(
            rendered, expected, source, source_location=source_location
        )
        if unsigned_to_float is not None:
            return unsigned_to_float
        expected_info = self.hlsl_floating_arithmetic_type_info(expected)
        source_info = self.hlsl_floating_arithmetic_type_info(source)
        if (
            expected_info is not None
            and expected_info["base_type"] in {"float", "double"}
            and source_info is not None
            and source_info["base_type"] == "float16_t"
            and expected_info["width"] == source_info["width"]
        ):
            return self.hlsl_native_16_bit_arithmetic_operand(
                rendered, source_info, expected_info["base_type"]
            )
        if expected == "complex64_t" and self.is_scalar_value_type(source):
            self.required_hlsl_complex64_helpers.add("__crossgl_complex64_make")
            return f"__crossgl_complex64_make({rendered}, 0.0)"
        floating_narrowing = self.hlsl_contextual_floating_narrowing_expression(
            rendered,
            expected,
            source,
            source_location=source_location,
        )
        if floating_narrowing is not None:
            return floating_narrowing
        wide_integer_to_float = self.hlsl_contextual_wide_integer_to_float_expression(
            rendered,
            expected,
            source,
            source_location=source_location,
        )
        if wide_integer_to_float is not None:
            return wide_integer_to_float
        expected_integer = self.hlsl_integer_arithmetic_type_info(expected)
        source_integer = self.hlsl_integer_arithmetic_type_info(source)
        expected_floating = self.hlsl_floating_arithmetic_type_info(expected)
        if (
            expected_floating is not None
            and expected_floating["base_type"] == "float16_t"
            and source_integer is not None
            and expected_floating["width"] == source_integer["width"]
        ):
            width = expected_floating["width"]
            floating_type = "float" + (str(width) if width > 1 else "")
            return self.hlsl_half_conversion_expression(
                f"{floating_type}({rendered})",
                width,
                (source,),
                source_location=source_location,
            )
        integer_narrowing = self.hlsl_contextual_integer_narrowing_expression(
            rendered,
            expected,
            source,
            source_location=source_location,
        )
        if integer_narrowing is not None:
            return integer_narrowing
        if (
            expected_integer is not None
            and expected_integer["minimum_precision"]
            and source_integer is not None
            and expected_integer["width"] == source_integer["width"]
        ):
            signed_source_bases = {"int", "int64_t", "uint64_t"}
            unsigned_source_bases = {"uint", "uint64_t"}
            valid_source_bases = (
                unsigned_source_bases
                if expected_integer["base_type"] == "min16uint"
                else signed_source_bases
            )
            if source_integer["base_type"] in valid_source_bases:
                return f"{expected}({rendered})"
        return rendered

    def hlsl_byte_integer_shape(self, value_type):
        source_name = self.type_name_string(value_type)
        aliases = getattr(self, "hlsl_type_aliases", {})
        seen = set()
        while source_name in aliases and source_name not in seen:
            seen.add(source_name)
            source_name = self.type_name_string(aliases[source_name])
        shape = narrow_integer_shape(source_name)
        return shape if shape is not None and shape[1] == 8 else None

    def hlsl_byte_conversion_expression(self, rendered, value_type):
        shape = self.hlsl_byte_integer_shape(value_type)
        if shape is None:
            return None
        kind, _bits, width = shape
        suffix = str(width) if width > 1 else ""
        # Shift as unsigned, then extend the retained sign bit in signed storage.
        # Each expression is evaluated once, including calls and indexed reads.
        if kind is ArithmeticScalarKind.SIGNED_INTEGER:
            return f"(int{suffix}(uint{suffix}({rendered}) << 24u) >> 24)"
        return f"(uint{suffix}({rendered}) & 255u)"

    def hlsl_byte_update_expression(self, node, target, operator, value=None):
        value_type = self.expression_result_type(target)
        if self.hlsl_byte_integer_shape(value_type) is None:
            return None
        owner = target
        while isinstance(owner, MemberAccessNode):
            owner = owner.object_expr
        if not isinstance(owner, (IdentifierNode, VariableNode)):
            raise DirectXContextualConversionError(
                "DirectX byte updates require a stable variable or field",
                source_type=value_type,
                target_type=value_type,
                reason="byte-update-lvalue-unsupported",
                source_location=getattr(node, "source_location", None),
            )
        lhs = self.generate_expression(target)
        promoted = self.map_type(value_type).replace("uint", "int")
        rhs = (
            self.generate_expression_with_expected(value, None)
            if value is not None
            else "1"
        )
        binary = operator[0] if operator in {"++", "--"} else operator[:-1]
        converted = self.hlsl_byte_conversion_expression(
            f"({promoted}({lhs}) {binary} ({rhs}))", value_type
        )
        updated = f"({lhs} = {converted})"
        if value is None and getattr(node, "is_postfix", False):
            inverse = "-" if operator == "++" else "+"
            previous = self.hlsl_byte_conversion_expression(
                f"({promoted}({lhs}) {inverse} 1)", value_type
            )
            return f"({updated}, {previous})"
        return updated

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "float16_t",
            "half",
            "min16float",
            "min10float",
            "double",
            "int64_t",
            "uint64_t",
            "int16_t",
            "uint16_t",
            "int",
            "min16int",
            "min12int",
            "uint",
            "min16uint",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float2",
            "float3",
            "float4",
            "float16_t2",
            "float16_t3",
            "float16_t4",
            "half2",
            "half3",
            "half4",
            "min16float2",
            "min16float3",
            "min16float4",
            "min10float2",
            "min10float3",
            "min10float4",
            "int64_t2",
            "int64_t3",
            "int64_t4",
            "int16_t2",
            "int16_t3",
            "int16_t4",
            "uint64_t2",
            "uint64_t3",
            "uint64_t4",
            "uint16_t2",
            "uint16_t3",
            "uint16_t4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "min16int2",
            "min16int3",
            "min16int4",
            "min12int2",
            "min12int3",
            "min12int4",
            "uint2",
            "uint3",
            "uint4",
            "min16uint2",
            "min16uint3",
            "min16uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def vector_component_type(self, vtype):
        if self.hlsl_bfloat16_vector_lanes(vtype) is not None:
            # Lanes of a bfloat16 vector are bfloat16 payloads, not uint values.
            return "bfloat"
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("float16_t"):
            return "float16_t"
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("min16float"):
            return "min16float"
        if mapped_type.startswith("min10float"):
            return "min10float"
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("uint64_t"):
            return "uint64_t"
        if mapped_type.startswith("int64_t"):
            return "int64_t"
        if mapped_type.startswith("uint16_t"):
            return "uint16_t"
        if mapped_type.startswith("int16_t"):
            return "int16_t"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("min16uint"):
            return "min16uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("min16int"):
            return "min16int"
        if mapped_type.startswith("min12int"):
            return "min12int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def hlsl_matrix_shape(self, vtype):
        mapped_type = self.map_type(vtype)
        for component_type in (
            "min16float",
            "min10float",
            "min16uint",
            "min16int",
            "min12int",
            "uint64_t",
            "int64_t",
            "uint16_t",
            "int16_t",
            "float16_t",
            "double",
            "float",
            "half",
            "uint",
            "int",
            "bool",
        ):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if (
                len(suffix) == 3
                and suffix[0] in {"2", "3", "4"}
                and suffix[1] == "x"
                and suffix[2] in {"2", "3", "4"}
            ):
                return component_type, suffix[0], suffix[2]
        return None

    def hlsl_numeric_matrix_type(self, vtype):
        # Boolean matrices have no product; their compound assignments are refused elsewhere.
        shape = self.hlsl_matrix_shape(vtype)
        return shape is not None and shape[0] != "bool"

    def hlsl_matrix_dimensions(self, vtype):
        matrix_shape = self.hlsl_matrix_shape(vtype)
        if matrix_shape is None:
            return None
        component_type, rows, columns = matrix_shape
        return component_type, int(rows), int(columns)

    def hlsl_binary_multiply_uses_mul(self, expr):
        """Return whether HLSL should lower a CrossGL multiply to mul(...)."""
        op = getattr(expr, "operator", getattr(expr, "op", None))
        if op != "*":
            return False

        left_type = self.expression_result_type(getattr(expr, "left", None))
        right_type = self.expression_result_type(getattr(expr, "right", None))
        left_is_matrix = self.hlsl_matrix_shape(left_type) is not None
        right_is_matrix = self.hlsl_matrix_shape(right_type) is not None
        left_is_vector = self.is_vector_value_type(left_type)
        right_is_vector = self.is_vector_value_type(right_type)

        return (
            left_is_matrix
            and (right_is_matrix or right_is_vector)
            or right_is_matrix
            and left_is_vector
        )

    def hlsl_integer_arithmetic_type_info(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        for base_type in self.HLSL_INTEGER_ARITHMETIC_BASE_TYPES:
            if mapped_type == base_type:
                width = 1
            elif mapped_type in {f"{base_type}2", f"{base_type}3", f"{base_type}4"}:
                width = int(mapped_type[-1])
            else:
                continue
            return {
                "mapped_type": mapped_type,
                "base_type": base_type,
                "width": width,
                "minimum_precision": (
                    base_type in self.HLSL_MINIMUM_PRECISION_INTEGER_BASE_TYPES
                ),
            }
        return None

    def hlsl_floating_arithmetic_type(self, vtype):
        return self.hlsl_floating_arithmetic_type_info(vtype) is not None

    def hlsl_floating_arithmetic_type_info(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        for base_type in (
            "float16_t",
            "min16float",
            "min10float",
            "double",
            "float",
            "half",
        ):
            if mapped_type == base_type:
                width = 1
            elif mapped_type in {f"{base_type}2", f"{base_type}3", f"{base_type}4"}:
                width = int(mapped_type[-1])
            else:
                continue
            return {
                "mapped_type": mapped_type,
                "base_type": base_type,
                "width": width,
                "native_16_bit": base_type == "float16_t",
            }
        return None

    def hlsl_floating_self_comparison(self, expression, rendered, operator):
        left, right = expression.left, expression.right
        if (
            operator not in {"==", "!="}
            or not isinstance(left, IdentifierNode)
            or not isinstance(right, IdentifierNode)
            or left.name != right.name
            or left.name not in self.local_variable_types
        ):
            return None
        source_type = self.local_variable_types[left.name]
        if any(
            marker in (self.type_name_string(source_type) or "")
            for marker in ("*", "&")
        ):
            return None
        comparison = ">" if operator == "!=" else "<="
        if self.is_hlsl_bfloat16_type(source_type):
            return f"((uint({rendered}) & 0x7fffu) {comparison} 0x7f80u)"
        info = self.hlsl_floating_arithmetic_type_info(source_type)
        if info is None or info["base_type"] not in {"float", "float16_t"}:
            return None
        # Optimized DXC folds floating self-comparisons even on precise locals.
        # Private scalar/vector payload classification preserves the NaN branch.
        if info["native_16_bit"]:
            return f"((asuint16({rendered}) & 0x7fffu) {comparison} 0x7c00u)"
        return f"((asuint({rendered}) & 0x7fffffffu) {comparison} 0x7f800000u)"

    def hlsl_integer_floating_binary_contract(
        self,
        node,
        operator,
        left_type,
        right_type,
    ):
        """Preserve rounding for uint32 and warning-clean wide integer operands."""
        if operator not in {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="}:
            return None

        left_floating = self.hlsl_floating_arithmetic_type_info(left_type)
        right_floating = self.hlsl_floating_arithmetic_type_info(right_type)
        left_integer = self.hlsl_integer_arithmetic_type_info(left_type)
        right_integer = self.hlsl_integer_arithmetic_type_info(right_type)

        if (
            left_floating is not None
            and left_floating["base_type"] == "float"
            and right_integer is not None
            and right_integer["base_type"] in {"uint", "int64_t", "uint64_t"}
        ):
            floating = left_floating
            integer = right_integer
            integer_side = "right"
        elif (
            right_floating is not None
            and right_floating["base_type"] == "float"
            and left_integer is not None
            and left_integer["base_type"] in {"uint", "int64_t", "uint64_t"}
        ):
            floating = right_floating
            integer = left_integer
            integer_side = "left"
        else:
            return None

        if (
            floating["width"] > 1
            and integer["width"] > 1
            and floating["width"] != integer["width"]
        ):
            raise DirectXContextualConversionError(
                "DirectX cannot lower mixed float and integer arithmetic "
                f"from '{integer['mapped_type']}' to "
                f"'{floating['mapped_type']}' because the vector lane shapes differ",
                source_type=integer["mapped_type"],
                target_type=floating["mapped_type"],
                reason="wide-integer-floating-lane-shape-mismatch",
                source_location=getattr(node, "source_location", None),
            )

        result_width = max(floating["width"], integer["width"])
        result_type = "float" if result_width == 1 else f"float{result_width}"
        integer_float_type = (
            "float" if integer["width"] == 1 else f"float{integer['width']}"
        )
        return {
            "floating": floating,
            "integer": integer,
            "integer_side": integer_side,
            "integer_float_type": integer_float_type,
            "result_type": result_type,
        }

    def hlsl_integer_floating_binary_expression(
        self,
        expr,
        left,
        right,
        operator,
    ):
        contract = self.hlsl_integer_floating_binary_contract(
            expr,
            operator,
            self.expression_result_type(getattr(expr, "left", None)),
            self.expression_result_type(getattr(expr, "right", None)),
        )
        if contract is None:
            return None

        value = left if contract["integer_side"] == "left" else right
        converted = self.hlsl_unsigned_float_conversion_expression(
            value,
            contract["integer_float_type"],
            contract["integer"]["mapped_type"],
            source_location=getattr(expr, "source_location", None),
        )
        if converted is None:
            converted = f"{contract['integer_float_type']}({value})"
        if contract["integer_side"] == "left":
            left = converted
        else:
            right = converted
        return f"({left} {operator} {right})"
