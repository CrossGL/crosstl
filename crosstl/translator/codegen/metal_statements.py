"""Statements of the Metal generator: atomic initializers, option payloads, ray query types, narrow conversions, value type predicates, expression statements and discards, signed built-in members, lowered stage I/O assignments, if, loop, switch and match statements, statement bodies and identifier uses, and packed-byte, fast, safe and precise math calls."""

import re

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CastNode,
    ContinueNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
)
from ..math_namespaces import (
    METAL_FAST_MATH_FUNCTIONS,
    METAL_MATH_NAMESPACE_ANNOTATION,
    METAL_PRECISE_MATH_FUNCTIONS,
)
from ..refusals import MetalRefusal
from ..structure_conversions import ScalarKind
from .array_utils import split_array_type_suffix
from .enum_utils import (
    generic_enum_specialized_type_name,
    generic_type_parts,
)
from .float_remainder_assignment import float_remainder_assignment
from .for_initializers import join_for_initializer
from .generic_struct_utils import generic_struct_specialized_type_name
from .match_utils import (
    generate_ordered_conditional_match,
    generate_switch_match,
    is_switch_lowerable_match,
)
from .matrix_compound_assignment import matrix_product_assignment
from .metal_array_copies import generate_metal_array_copy
from .metal_debug_printf import metal_debug_printf_statement
from .metal_safe_math import precise_namespace_eligible


class MetalStatementMixin:
    """Statement members of ``MetalCodeGen``."""

    def literal_int_array_size_from_type_name(self, type_name):
        if not isinstance(type_name, str) or "[" not in type_name:
            return None

        _base_type, array_suffix = split_array_type_suffix(type_name)
        close_bracket = array_suffix.find("]")
        if close_bracket == -1:
            return None

        size_expr = array_suffix[1:close_bracket].strip()
        if not size_expr:
            return None
        return self.literal_int_resource_array_size(size_expr)

    def metal_atomic_value_type(self, atomic_type):
        return {
            "atomic_bool": "bool",
            "atomic_int": "int",
            "atomic_uint": "uint",
        }.get(str(atomic_type), atomic_type)

    def metal_atomic_zero_value(self, atomic_type):
        return {
            "atomic_bool": "false",
            "atomic_int": "0",
            "atomic_uint": "0u",
        }.get(str(atomic_type), "0")

    def generate_metal_atomic_array_initializer_stores(
        self, name, var_type, initial_value, indent_str
    ):
        array_parts = self.metal_atomic_array_type_parts(var_type)
        if array_parts is None or not isinstance(initial_value, ArrayLiteralNode):
            return None

        atomic_type, declared_size = array_parts
        elements = list(getattr(initial_value, "elements", []) or [])
        store_count = declared_size if declared_size is not None else len(elements)
        value_type = self.metal_atomic_value_type(atomic_type)
        zero_value = self.metal_atomic_zero_value(atomic_type)
        lines = []
        for index in range(store_count):
            if index < len(elements):
                value = self.generate_expression_with_expected(
                    elements[index], value_type
                )
            else:
                value = zero_value
            lines.append(
                f"{indent_str}atomic_store_explicit(&{name}[{index}], {value}, "
                "memory_order_relaxed);\n"
            )
        if declared_size is not None and len(elements) > declared_size:
            self.refuse_lowering(
                f"atomic array initializer: '{name}' has {len(elements)} initializers "
                f"for {declared_size} elements"
            )
        return "".join(lines)

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if (
            hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
            or isinstance(vtype, (PointerType, ReferenceType))
        ):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def option_payload_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        if not isinstance(type_name, str):
            return None

        base_name, generic_args = generic_type_parts(type_name.strip())
        if base_name.rsplit("::", 1)[-1] != "Option" or len(generic_args) != 1:
            return None
        return generic_args[0]

    def lowerable_option_payload_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        payload_type = self.option_payload_type_name(type_name)
        if payload_type is None:
            return None
        if generic_enum_specialized_type_name(self, type_name) is not None:
            return None
        if generic_struct_specialized_type_name(self, type_name) is not None:
            return None
        return payload_type

    def expected_option_payload_type_name(self):
        expected_type = self.current_expression_expected_type
        option_payload = self.lowerable_option_payload_type_name(
            self.current_expression_expected_type
        ) or self.lowerable_option_payload_type_name(self.current_function_return_type)
        if option_payload is not None:
            return option_payload
        if self.option_payload_type_name(expected_type) is not None:
            return None
        return expected_type

    def option_constructor_payload_expression(self, expr):
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = getattr(expr, "name", None)
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None
        if func_name.rsplit("::", 1)[-1] != "Some":
            return None

        args = list(getattr(expr, "args", getattr(expr, "arguments", [])) or [])
        payload_type = self.expected_option_payload_type_name()
        if payload_type is None:
            return None
        if args:
            return self.generate_expression_with_expected(args[0], payload_type)
        return self.metal_default_value_expression(payload_type)

    def option_none_default_expression(self):
        payload_type = self.expected_option_payload_type_name()
        if payload_type is None:
            return None
        return self.metal_default_value_expression(payload_type)

    def is_metal_ray_query_type_name(self, type_name):
        type_name = str(type_name or "").strip()
        return (
            type_name == "RayQuery"
            or type_name == "rayQueryEXT"
            or type_name.startswith("RayQuery<")
        )

    def requires_metal_builtin_ray_desc(self, type_name):
        type_name = str(type_name or "").strip()
        return type_name == "RayDesc" and type_name not in getattr(
            self, "structs_by_name", {}
        )

    def require_metal_ray_query_runtime(self):
        self.required_metal_ray_query_runtime = True

    def metal_ray_query_method_return_type(self, operation):
        return self.METAL_RAY_QUERY_RETURN_TYPES.get(str(operation))

    def is_metal_ray_query_method(self, operation):
        return self.metal_ray_query_method_return_type(operation) is not None

    def metal_ray_query_helper_name(self, operation):
        return f"cgl_ray_query_{self.metal_snake_case_name(operation)}"

    def metal_snake_case_name(self, name):
        text = str(name)
        result = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0:
                previous = text[index - 1]
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if previous.islower() or next_char.islower():
                    result.append("_")
            result.append(char.lower())
        return "".join(result)

    def ray_query_operation_from_function_call(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None
        func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
        if not isinstance(func_expr, MemberAccessNode):
            return None
        operation = str(getattr(func_expr, "member", ""))
        if not self.is_metal_ray_query_method(operation):
            return None
        receiver_type = self.expression_result_type(
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        )
        if receiver_type is not None and self.is_metal_ray_query_type_name(
            self.type_name_string(receiver_type)
        ):
            return operation
        return None

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        expected_type_name = self.type_name_string(expected_type)
        self.current_expression_expected_type = expected_type_name
        try:
            rendered = self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type
        rendered = self.metal_implicit_vector_conversion(
            expr, rendered, expected_type_name
        )
        # Native Metal math on bfloat produces float/half values and Metal does
        # not implicitly convert those values back to bfloat.  Expected-type
        # boundaries are the source-language conversion point, so preserve the
        # source semantics with an explicit native construction.  Applying this
        # uniformly also covers ternaries/arithmetic without guessing their
        # target overload result; an existing bfloat construction is idempotent
        # and is left intact to keep output stable.
        is_bfloat_constructor = (
            self.is_discarded_metal_type_constructor(expr)
            and self.map_type(self.function_call_name(expr)) == "bfloat"
        )
        if self.map_type(expected_type_name) == "bfloat" and not is_bfloat_constructor:
            return f"bfloat({rendered})"
        return (
            self.metal_narrow_integer_conversion_expression(rendered, expected_type)
            or rendered
        )

    def metal_narrow_integer_conversion_expression(self, rendered, value_type):
        native = self.metal_native_narrow_bitcast_storage_type(value_type)
        if native is None or not re.fullmatch(r"u?(?:char|short)[234]?", native):
            return None
        carrier = self.map_type(value_type)
        if self.metal_narrow_is_normalized(rendered, carrier, native):
            # Narrow vector arithmetic already wrapped the value.
            return rendered
        # Keep arithmetic carriers widened, but preserve the source conversion.
        return f"{carrier}({native}({rendered}))"

    def metal_direct_narrow_float_conversion(self, constructor, arguments, rendered):
        """``uchar(f)`` of a floating ``f``: converted directly, which keeps its result (MSL 4.1 section 8.6).

        Converting through the 32-bit carrier first, ``uchar(uint(f))``, would wrap a value the direct conversion
        saturates on Apple GPUs.
        """
        native = self.metal_native_narrow_bitcast_storage_type(constructor)
        if (
            native is None
            or len(arguments) != 1
            or not re.fullmatch(r"u?(?:char|short)[234]?", native)
        ):
            return None
        operand = self.map_type(
            self.type_name_string(self.expression_result_type(arguments[0])) or ""
        )
        if not re.fullmatch(r"(?:float|half|bfloat)[234]?", str(operand or "")):
            return None
        return f"{self.map_type(constructor)}({native}({rendered}))"

    def metal_narrow_update_repeatable_lvalue(self, target):
        """Return whether ``target`` can be read and written by repeating it.

        The widened update renders the lvalue twice, so every subexpression
        must name the same storage on both evaluations: variables, fields and
        array elements whose indices are side-effect-free.
        """
        if isinstance(target, (IdentifierNode, VariableNode)):
            return True
        if isinstance(target, MemberAccessNode):
            return self.metal_narrow_update_repeatable_lvalue(target.object_expr)
        if isinstance(target, ArrayAccessNode):
            return self.metal_narrow_update_repeatable_lvalue(
                target.array_expr
            ) and self.metal_narrow_update_repeatable_index(target.index_expr)
        return False

    def metal_narrow_update_repeatable_index(self, index):
        if isinstance(index, LiteralNode):
            return True
        if isinstance(index, (IdentifierNode, VariableNode, MemberAccessNode)):
            return self.metal_narrow_update_repeatable_lvalue(index)
        if isinstance(index, ArrayAccessNode):
            return self.metal_narrow_update_repeatable_lvalue(index)
        if isinstance(index, BinaryOpNode):
            return self.metal_narrow_update_repeatable_index(
                index.left
            ) and self.metal_narrow_update_repeatable_index(index.right)
        if isinstance(index, UnaryOpNode):
            return str(index.operator) not in {
                "++",
                "--",
            } and self.metal_narrow_update_repeatable_index(index.operand)
        if isinstance(index, CastNode):
            return self.metal_narrow_update_repeatable_index(
                getattr(index, "expression", None)
            )
        return False

    def generate_metal_bfloat_promoted_argument(self, argument):
        rendered = self.generate_expression(argument)
        argument_type = self.expression_result_type(argument)
        if self.map_type(argument_type) == "bfloat":
            return f"float({rendered})"
        return rendered

    METAL_STRUCTURE_PROMOTION_SCALAR_KINDS = {
        "bool": ScalarKind.BOOLEAN,
        "char": ScalarKind.SIGNED_INTEGER,
        "short": ScalarKind.SIGNED_INTEGER,
        "int": ScalarKind.SIGNED_INTEGER,
        "long": ScalarKind.SIGNED_INTEGER,
        "int64_t": ScalarKind.SIGNED_INTEGER,
        "uchar": ScalarKind.UNSIGNED_INTEGER,
        "ushort": ScalarKind.UNSIGNED_INTEGER,
        "uint": ScalarKind.UNSIGNED_INTEGER,
        "ulong": ScalarKind.UNSIGNED_INTEGER,
        "uint64_t": ScalarKind.UNSIGNED_INTEGER,
        "half": ScalarKind.FLOATING,
        "float": ScalarKind.FLOATING,
        "bfloat": ScalarKind.FLOATING,
        "double": ScalarKind.FLOATING,
    }

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "half",
            "bfloat",
            "double",
            "int",
            "uint",
            "long",
            "ulong",
            "int64_t",
            "uint64_t",
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
            "half2",
            "half3",
            "half4",
            "bfloat2",
            "bfloat3",
            "bfloat4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "long2",
            "long3",
            "long4",
            "ulong2",
            "ulong3",
            "ulong4",
            "bool2",
            "bool3",
            "bool4",
        }

    def is_matrix_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        mapped_type, array_suffix = split_array_type_suffix(self.map_type(vtype))
        if array_suffix:
            return False
        for prefix in ("float", "half", "double"):
            if not mapped_type.startswith(prefix):
                continue
            dimensions = mapped_type[len(prefix) :].split("x", 1)
            return (
                len(dimensions) == 2
                and dimensions[0] in {"2", "3", "4"}
                and dimensions[1] in {"2", "3", "4"}
            )
        return False

    def is_builtin_value_constructor_type(self, vtype):
        return (
            self.is_scalar_value_type(vtype)
            or self.is_vector_value_type(vtype)
            or self.is_matrix_value_type(vtype)
        )

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type in {"bfloat2", "bfloat3", "bfloat4"}:
            return "bfloat"
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type in {"ulong2", "ulong3", "ulong4"}:
            return "ulong"
        if mapped_type in {"long2", "long3", "long4"}:
            return "long"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def vector_value_width(self, vtype):
        mapped_type = self.map_type(vtype)
        if not mapped_type or not mapped_type[-1:].isdigit():
            return None
        if self.is_matrix_value_type(mapped_type):
            return None
        if self.vector_component_type(mapped_type) is None:
            return None
        return int(mapped_type[-1])

    def infer_metal_global_value_type(self, node):
        """The type of a shader-scope declaration with no written type (#2444).

        Take it from the initializer: a literal carries its own type, and a
        more involved expression goes through expression_result_type. When
        neither gives a type the declaration is refused instead of guessed.
        """
        initial_value = getattr(node, "initial_value", None)
        literal_type = getattr(initial_value, "literal_type", None)
        if literal_type is not None:
            return self.type_name_string(literal_type)
        inferred = self.expression_result_type(initial_value)
        if inferred:
            return inferred
        var_name = getattr(node, "name", getattr(node, "variable_name", "?"))
        # "float" only lets generation reach the program's other refusals.
        return self.refuse_lowering(
            f"global declaration: the type of '{var_name}' is not written and "
            "cannot be inferred from its initializer",
            "float",
        )

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        expr_node = getattr(stmt, "expression", stmt)
        discard_statement = self.generate_fragment_discard_statement(expr_node)
        if discard_statement is not None:
            return discard_statement
        debug_print = metal_debug_printf_statement(self, expr_node)
        if debug_print is not None:
            return debug_print

        if hasattr(stmt, "expression"):
            if isinstance(stmt.expression, (AssignmentNode, BackendAssignmentNode)):
                return self.generate_assignment(stmt.expression)
            return self.generate_discarded_expression(stmt.expression)
        else:
            return self.generate_discarded_expression(stmt)

    def is_discarded_metal_texel_atomic(self, expr, rendered):
        """Whether ``rendered`` selects ``.x`` of an image atomic's returned texel."""
        if not isinstance(expr, FunctionCallNode) or not rendered.endswith(").x"):
            return False
        function = getattr(expr, "function", None) or getattr(expr, "name", None)
        function_name = getattr(function, "name", function)
        return isinstance(function_name, str) and function_name.startswith(
            "imageAtomic"
        )

    def is_discarded_metal_type_constructor(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return False
        function = getattr(expr, "function", None) or getattr(expr, "name", None)
        function_name = getattr(function, "name", function)
        return isinstance(function_name, str) and (
            function_name in self.type_mapping
            or function_name in self.metal_type_aliases
        )

    def generate_fragment_discard_statement(self, expr):
        """Lower CrossGL/HLSL fragment-kill expressions to Metal syntax."""
        if isinstance(expr, IdentifierNode) and expr.name == "discard":
            return "discard_fragment()"

        if not isinstance(expr, FunctionCallNode):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        func_name = getattr(func_expr, "name", func_expr)
        args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        if func_name == "discard" and not args:
            return "discard_fragment()"
        if func_name != "clip" or len(args) != 1:
            return None

        predicate = self.generate_clip_discard_predicate(args[0])
        return f"if ({predicate}) {{\n    discard_fragment();\n}}"

    def generate_clip_discard_predicate(self, expr):
        """Return the Metal predicate for HLSL-style ``clip(expr)``."""
        rendered = self.generate_expression(expr)
        expr_type = self.expression_result_type(expr)
        width = self.vector_value_width(expr_type)
        if width is None:
            return f"({rendered}) < 0.0"

        mapped_type = self.map_type(expr_type)
        component_type = self.vector_component_type(mapped_type) or "float"
        zero = "0.0" if component_type in {"float", "half", "double"} else "0"
        return f"any(({rendered}) < {mapped_type}({zero}))"

    def metal_is_floating_value_type(self, vtype):
        """Whether ``vtype`` is a floating-point scalar or vector."""
        if vtype is None:
            return False
        mapped = self.map_type(vtype)
        if not isinstance(mapped, str) or not (
            self.is_scalar_value_type(mapped) or self.is_vector_value_type(mapped)
        ):
            return False
        return self.vector_component_type(mapped) in {
            "float",
            "half",
            "double",
            "bfloat",
        }

    def generate_assignment(self, node):
        if hasattr(node, "target") and hasattr(node, "value"):
            target = node.target
            value = node.value
        else:
            target = node.left
            value = node.right
        op = getattr(node, "operator", "=")
        # MSL has no *= for two matrices; v *= m keeps its MSL row-vector meaning.
        matrix_product = matrix_product_assignment(
            target,
            value,
            op,
            type_of=self.expression_result_type,
            is_matrix=self.is_matrix_value_type,
            is_vector=lambda _vtype: False,
            backend="metal",
        )
        if matrix_product is not None:
            return self.generate_assignment(matrix_product)
        # MSL has no floating %=; x = x % y lowers the remainder through fmod.
        remainder = float_remainder_assignment(
            target,
            value,
            op,
            type_of=self.expression_result_type,
            is_floating=self.metal_is_floating_value_type,
            backend="metal",
        )
        if remainder is not None:
            return self.generate_assignment(remainder)
        if op == "=":
            # Copied element by element, so the value itself is never rendered: a whole buffer block array has no
            # Metal expression.
            array_copy = generate_metal_array_copy(self, target, value)
            if array_copy is not None:
                return array_copy
        rhs = self.generate_expression_with_expected(
            value, self.metal_assignment_value_type(target, value, op)
        )
        if op == "=":
            converted = self.metal_registered_structure_assignment(target, value)
            if converted is not None:
                rhs = converted

        reference_declaration = self.generate_local_reference_assignment_declaration(
            target, value, rhs
        )
        if reference_declaration is not None:
            return reference_declaration

        stage_io_assignment = self.generate_metal_stage_io_lowered_assignment(
            target, rhs, op
        )
        if stage_io_assignment is not None:
            return stage_io_assignment
        if self.is_metal_signed_builtin_member(target):
            return self.generate_metal_signed_builtin_member_assignment(target, rhs, op)

        mesh_output_store = self.generate_metal_mesh_output_assignment(
            target, rhs, op, getattr(node, "value", getattr(node, "right", None))
        )
        if mesh_output_store is not None:
            return mesh_output_store

        block_store = self.generate_glsl_buffer_block_store(target, rhs, op)
        if block_store is not None:
            return block_store
        unsupported_store = self.unsupported_glsl_buffer_block_assignment_diagnostic(
            target
        )
        if unsupported_store is not None:
            return unsupported_store
        readonly_mesh_payload_store = (
            self.readonly_metal_mesh_payload_assignment_diagnostic(target)
        )
        if readonly_mesh_payload_store is not None:
            return readonly_mesh_payload_store
        readonly_raw_buffer_store = self.readonly_raw_buffer_assignment_diagnostic(
            target
        )
        if readonly_raw_buffer_store is not None:
            return readonly_raw_buffer_store
        readonly_parameter_store = self.readonly_metal_parameter_assignment_diagnostic(
            target
        )
        if readonly_parameter_store is not None:
            return readonly_parameter_store
        readonly_global_store = (
            self.readonly_metal_program_scope_global_assignment_diagnostic(target)
        )
        if readonly_global_store is not None:
            return readonly_global_store
        readonly_mesh_payload_alias = (
            self.readonly_metal_mesh_payload_alias_assignment_diagnostic(target, value)
        )
        if readonly_mesh_payload_alias is not None:
            return readonly_mesh_payload_alias
        address_space_assignment = self.address_space_assignment_diagnostic(
            target, value, op
        )
        if address_space_assignment is not None:
            return address_space_assignment

        layout_store = self.metal_hlsl_layout_store_target(target)
        if layout_store is not None:
            lhs, write_conversion = layout_store
            if write_conversion and op == "=":
                rhs = f"{write_conversion}({rhs})"
            return f"{lhs} {op} {rhs}"
        if op != "=":
            byte_update = self.metal_narrow_integer_update_expression(
                node, target, op, value
            )
            if byte_update is not None:
                return byte_update

        lhs = self.generate_expression(target)
        native_vector = self.metal_native_narrow_bitcast_storage_type(
            self.expression_result_type(target)
        )
        member_target = target
        while isinstance(member_target, ArrayAccessNode):
            member_target = member_target.array
        if (
            op == "="
            and isinstance(member_target, MemberAccessNode)
            and native_vector is not None
            and native_vector[-1:] in {"2", "3", "4"}
            and self.member_lookup_type_name(
                self.expression_result_type(member_target.object)
            )
            in self.structs_by_name
        ):
            # Aggregate storage retains narrow vectors; arithmetic values do not.
            rhs = f"{native_vector}({rhs})"
        if op == "=" and self.pointer_assignment_needs_address(target, value):
            rhs = f"&{rhs}"
        matrix_update = self.metal_matrix_compound_assignment(
            target, value, op, lhs, rhs
        )
        if matrix_update is not None:
            return matrix_update
        return f"{lhs} {op} {rhs}"

    def is_metal_signed_builtin_member(self, expr):
        """Whether ``expr`` reads a ``uint``-declared member that holds GLSL's ``int`` built-in value."""
        if not self.metal_signed_builtin_members or not isinstance(
            expr, MemberAccessNode
        ):
            return False
        object_type = self.expression_result_type(getattr(expr, "object", None))
        struct_name = self.type_name_string(self.member_lookup_type_name(object_type))
        return str(getattr(expr, "member", "")) in (
            self.metal_signed_builtin_members.get(struct_name, ())
        )

    def generate_metal_signed_builtin_member_storage(self, expr):
        """The ``uint`` storage of a signed built-in member, without the conversion its reads take."""
        self.metal_signed_member_raw_reads.add(id(expr))
        try:
            return self.generate_expression(expr)
        finally:
            self.metal_signed_member_raw_reads.discard(id(expr))

    def generate_metal_stage_io_lowered_assignment(self, target, rhs, op):
        if isinstance(target, MemberAccessNode):
            info = self.metal_stage_io_lowered_member_info(target)
            if info is None or info["lowering"].get("kind") != "matrix":
                return None
            lowering = info["lowering"]
            temp_name = self.next_metal_temp_variable("stage_io_matrix")
            lines = [f"{lowering['mapped_type']} {temp_name} = {rhs}"]
            for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                lines.append(
                    f"{info['object_code']}.{field_name} {op} {temp_name}[{index}]"
                )
            return "\n".join(lines)

        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            if not isinstance(array_expr, MemberAccessNode):
                return None
            info = self.metal_stage_io_lowered_member_info(array_expr)
            if info is None or info["lowering"].get("kind") != "array":
                return None
            lowering = info["lowering"]
            index_expr = getattr(target, "index", None)
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            if literal_index is not None and 0 <= literal_index < lowering["size"]:
                field_name = lowering["fields"][literal_index][0]
                return f"{info['object_code']}.{field_name} {op} {rhs}"

            rendered_index = self.generate_expression(index_expr)
            temp_name = self.next_metal_temp_variable("stage_io_array")
            lines = [f"{lowering['element_type']} {temp_name} = {rhs}"]
            lines.append(f"switch ({rendered_index}) {{")
            for index, (field_name, _field_type) in enumerate(lowering["fields"]):
                lines.append(f"case {index}:")
                lines.append(
                    f"    {info['object_code']}.{field_name} {op} {temp_name};"
                )
                lines.append("    break;")
            lines.append("default:")
            lines.append("    break;")
            lines.append("}")
            return "\n".join(lines)

        return None

    def metal_stage_io_lowered_member_info(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_type = self.expression_result_type(object_expr)
        object_type = self.member_lookup_type_name(object_type)
        struct_name = self.type_name_string(object_type)
        member_name = str(getattr(expr, "member", ""))
        lowering = self.metal_stage_io_member_lowerings.get(struct_name, {}).get(
            member_name
        )
        if lowering is None:
            return None
        return {
            "struct_name": struct_name,
            "member_name": member_name,
            "object_code": self.generate_expression_with_expected(object_expr, None),
            "lowering": lowering,
        }

    def generate_metal_stage_io_lowered_member_access(self, expr):
        info = self.metal_stage_io_lowered_member_info(expr)
        if info is None or info["lowering"].get("kind") != "matrix":
            return None
        lowering = info["lowering"]
        columns = ", ".join(
            f"{info['object_code']}.{field_name}"
            for field_name, _field_type in lowering["fields"]
        )
        return f"{lowering['mapped_type']}({columns})"

    def generate_metal_stage_io_lowered_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        if not isinstance(array_expr, MemberAccessNode):
            return None
        info = self.metal_stage_io_lowered_member_info(array_expr)
        if info is None or info["lowering"].get("kind") != "array":
            return None
        lowering = info["lowering"]
        index_expr = getattr(expr, "index", None)
        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is not None and 0 <= literal_index < lowering["size"]:
            field_name = lowering["fields"][literal_index][0]
            return f"{info['object_code']}.{field_name}"
        helper_name = self.metal_stage_io_array_getter_name(
            info["struct_name"], info["member_name"]
        )
        return f"{helper_name}({info['object_code']}, {self.generate_expression(index_expr)})"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        code += self.generate_scoped_statement_body(if_body, indent + 1)
        code += f"{indent_str}}}"

        else_branch = getattr(node, "else_branch", None)
        if isinstance(else_branch, IfNode):
            nested = self.generate_if(else_branch, indent)
            if not nested.startswith(f"{indent_str}if ("):
                raise MetalRefusal(
                    "Metal else-if chain did not render as an if statement"
                )
            return code + " else " + nested[len(indent_str) :]
        if else_branch:
            code += " else {\n"
            code += self.generate_scoped_statement_body(else_branch, indent + 1)
            code += f"{indent_str}}}"

        return code + "\n"

    def generate_for(self, node, indent):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            init = self.generate_for_initializer(getattr(node, "init", None))

            condition = (
                self.generate_expression(node.condition)
                if getattr(node, "condition", None)
                else ""
            )

            if isinstance(getattr(node, "update", None), list):
                update = ", ".join(
                    self.generate_discarded_expression(expression).strip().rstrip(";")
                    for expression in node.update
                )
            else:
                update = (
                    self.generate_discarded_expression(node.update)
                    if getattr(node, "update", None)
                    else ""
                )

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

            code += self.generate_scoped_statement_body(node.body, indent + 1)

            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_block(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}{{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "statements", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_scoped_statement_body(self, body, indent):
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        # Read-only locals end with their block, so an outer variable of the same name
        # is writable again after it.
        previous_readonly = set(self.current_readonly_metal_parameters)
        previous_readonly_reasons = dict(self.current_readonly_metal_parameter_reasons)
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )
            self.current_readonly_metal_parameters = previous_readonly
            self.current_readonly_metal_parameter_reasons = previous_readonly_reasons

    def metal_for_in_array_element_type(self, iterable_type, *, allow_unsized=False):
        type_name = self.type_name_string(iterable_type)
        if not type_name:
            return None
        type_name = self.resolve_metal_type_alias(type_name)
        base_type, suffix = split_array_type_suffix(type_name)
        if suffix:
            match = re.fullmatch(r"\[([^\]]*)\](.*)", suffix)
            if match is not None and (allow_unsized or match.group(1)):
                return f"{base_type}{match.group(2)}"
        return self.metal_array_element_type(type_name)

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}do {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}} while ({condition});\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for case in getattr(node, "cases", []) or []:
            value = getattr(case, "value", None)
            if value is None:
                label = "default"
            else:
                label = f"case {self.generate_expression(value)}"
            code += self.generate_switch_case(
                label, getattr(case, "statements", []), indent + 1
            )

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            code += self.generate_switch_case(
                "default",
                getattr(default_case, "statements", default_case),
                indent + 1,
            )

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        expression_type = self.type_name_string(
            self.expression_result_type(getattr(node, "expression", None))
        )
        if is_switch_lowerable_match(node) and expression_type != "bool":
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "Metal")

    def format_match_condition(self, condition):
        condition = str(condition).strip()
        if self.condition_has_single_outer_parentheses(condition):
            return condition
        return f"({condition})"

    def match_expression_requires_fallback_assignment(self):
        return True

    def generate_match_expression_diagnostic_assignment(
        self,
        target_variable,
        target_type,
        reason,
        indent,
        target_name,
    ):
        """Assign the default value on the path no arm takes; refuse an arm that produces no value.

        The unmatched path is reached only by a subject the arms do not cover, which a well-formed value-position
        match excludes; Metal still needs the variable assigned on that path. An arm without a value is a
        malformed match, not an unreachable path.
        """
        indent_str = "    " * indent
        fallback = self.metal_default_value_expression(target_type)
        if reason != "no wildcard arm handles remaining cases":
            self.refuse_lowering(f"match expression for '{target_variable}': {reason}")
        return f"{indent_str}{target_variable} = {fallback};\n"

    def condition_has_single_outer_parentheses(self, condition):
        if not (condition.startswith("(") and condition.endswith(")")):
            return False
        depth = 0
        for index, char in enumerate(condition):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(condition) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def statement_body_terminates(self, body):
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]

        return bool(statements) and isinstance(
            statements[-1], (BreakNode, ContinueNode, ReturnNode)
        )

    def statement_body_has_statements(self, body):
        if hasattr(body, "statements"):
            return bool(body.statements)
        if isinstance(body, list):
            return bool(body)
        return body is not None

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_statement_body(self, body, indent):
        if hasattr(body, "statements"):
            statements = list(body.statements)
        elif isinstance(body, list):
            statements = list(body)
        elif body is not None:
            statements = [body]
        else:
            statements = []

        code = ""
        previous_unused_declarations = self.current_unused_local_declaration_names
        try:
            for index, stmt in enumerate(statements):
                self.current_unused_local_declaration_names = (
                    self.unused_local_declaration_names_for_statement(
                        stmt,
                        statements[index + 1 :],
                    )
                )
                code += self.generate_statement(stmt, indent)
        finally:
            self.current_unused_local_declaration_names = previous_unused_declarations
        return code

    def unused_local_declaration_names_for_statement(self, stmt, remaining_statements):
        name = getattr(stmt, "name", None)
        if not name or not isinstance(stmt, (VariableNode, ArrayNode)):
            return set()
        if self.statement_sequence_uses_identifier(remaining_statements, name):
            return set()
        return {name}

    def statement_sequence_uses_identifier(self, statements, name):
        for stmt in statements:
            if (
                isinstance(stmt, (VariableNode, ArrayNode))
                and getattr(stmt, "name", None) == name
            ):
                return False
            if self.statement_uses_identifier(stmt, name):
                return True
        return False

    def statement_uses_identifier(self, stmt, name):
        if isinstance(stmt, VariableNode):
            return self.node_uses_identifier(getattr(stmt, "initial_value", None), name)
        if isinstance(stmt, ArrayNode):
            return False
        return self.node_uses_identifier(stmt, name)

    def statement_body_uses_identifier(self, body, name):
        if hasattr(body, "statements"):
            return self.statement_sequence_uses_identifier(body.statements, name)
        if isinstance(body, list):
            return self.statement_sequence_uses_identifier(body, name)
        return self.node_uses_identifier(body, name)

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, list):
            return join_for_initializer(
                [self.generate_for_initializer(item) for item in init],
                [isinstance(item, VariableNode) for item in init],
                "metal",
            )
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            initializer = self.generate_statement(init, 0).strip().rstrip(";")
            return self.strip_unused_metal_declaration_attribute(initializer)
        return self.generate_discarded_expression(init).strip().rstrip(";")

    # HLSL pack_u8/pack_s8 keep the low byte of each lane, pack_clamp_u8 and
    # pack_clamp_s8 saturate each lane to [0, 255] or [-128, 127], and all four
    # put lane x in the least significant byte. Metal narrows each lane to
    # uchar4 (unsigned narrowing keeps the low byte) or to char4 after the
    # clamp, then reinterprets the four bytes as a uint, which puts lane x in
    # the least significant byte on the little-endian Apple GPUs.
    # unpack_u8u32/unpack_s8s32 (and the 16-bit forms) reinterpret the uint as
    # uchar4 or char4 and widen each lane, sign-extending the signed lanes.
    # Each entry is (result type, template); expression_result_type reports the
    # result type so HLSL's implicit conversions of the result get a constructor.
    PACKED_BYTE_INTRINSICS = {
        "pack_u8": ("uint", "as_type<uint>(uchar4(uint4({value})))"),
        "pack_s8": ("uint", "as_type<uint>(uchar4(uint4({value})))"),
        "pack_clamp_u8": (
            "uint",
            "as_type<uint>(uchar4(clamp(int4({value}), int4(0), int4(255))))",
        ),
        "pack_clamp_s8": (
            "uint",
            "as_type<uint>(char4(clamp(int4({value}), int4(-128), int4(127))))",
        ),
        "unpack_u8u32": ("uint4", "uint4(as_type<uchar4>(uint({value})))"),
        "unpack_s8s32": ("int4", "int4(as_type<char4>(uint({value})))"),
        "unpack_u8u16": ("ushort4", "ushort4(as_type<uchar4>(uint({value})))"),
        "unpack_s8s16": ("short4", "short4(as_type<char4>(uint({value})))"),
    }

    def generate_packed_byte_call(self, func_name, args):
        entry = self.PACKED_BYTE_INTRINSICS.get(func_name)
        if entry is None or func_name in self.user_function_names:
            return None
        template = entry[1]
        if len(args) != 1:
            raise MetalRefusal(
                f"Metal {func_name} alias requires 1 argument; got {len(args)}"
            )
        return template.format(value=self.generate_expression(args[0]))

    def metal_function_name_is_shadowed(self, func_name):
        return (
            func_name in self.local_variable_types
            or func_name in self.user_function_names
        )

    # Metal's fast:: overloads take float scalars and vectors; half and bfloat
    # scalars convert implicitly, exactly as in the source call.
    METAL_FAST_MATH_ARGUMENT_TYPES = frozenset(
        {"float", "float2", "float3", "float4", "half", "bfloat"}
    )

    def metal_fast_math_call(self, expr, func_name):
        """Return whether a builtin call keeps its source ``metal::fast::`` name."""
        if func_name not in METAL_FAST_MATH_FUNCTIONS:
            return False
        get_annotation = getattr(expr, "get_annotation", None)
        if not callable(get_annotation):
            return False
        if get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) != "fast":
            return False
        if self.metal_function_name_is_shadowed(func_name):
            return False
        for argument in expr.args:
            argument_type = self.expression_result_type(argument)
            if argument_type is None:
                return False
            if self.map_type(argument_type) not in self.METAL_FAST_MATH_ARGUMENT_TYPES:
                return False
        return True

    # Under -fno-fast-math the plain binary32 overloads of these functions
    # match metal::precise:: bit for bit, but the binary16 overloads are native
    # half implementations (``tan(-0.0h)`` is +0), so half operands need the
    # namespace back.  metal::precise:: declares binary32 overloads only: a
    # half scalar converts to float implicitly, a half vector does not, so
    # half vectors keep the plain overload instead of an ill-formed call.
    METAL_PRECISE_MATH_ARGUMENT_TYPES = frozenset({"half"})

    def metal_safe_math_call(self, expr, func_name):
        """Whether a builtin call takes ``metal::precise::`` to keep IEEE semantics (metal_safe_math.py)."""
        if (
            not getattr(self, "metal_safe_math", False)
            or func_name in self.user_function_names
            or self.metal_function_name_is_shadowed(func_name)
        ):
            return False
        argument_types = []
        for argument in expr.args or []:
            argument_type = self.expression_result_type(argument)
            if argument_type is None:
                return False
            argument_types.append(self.map_type(argument_type))
        return precise_namespace_eligible(func_name, argument_types)

    def metal_precise_math_call(self, expr, func_name):
        """Return whether a builtin call keeps its source ``metal::precise::`` name."""
        if func_name not in METAL_PRECISE_MATH_FUNCTIONS:
            return False
        get_annotation = getattr(expr, "get_annotation", None)
        if not callable(get_annotation):
            return False
        if get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) != "precise":
            return False
        if self.metal_function_name_is_shadowed(func_name) or not expr.args:
            return False
        for argument in expr.args:
            argument_type = self.expression_result_type(argument)
            if argument_type is None:
                return False
            if (
                self.map_type(argument_type)
                not in self.METAL_PRECISE_MATH_ARGUMENT_TYPES
            ):
                return False
        return True

    def metal_explicit_as_type_target(self, func_name, args):
        if (
            not isinstance(func_name, str)
            or func_name in self.user_function_names
            or len(args or []) != 1
        ):
            return None
        match = re.fullmatch(r"(?:(?:metal)::)?as_type<(.+)>", func_name)
        if match is None:
            return None
        target_type = match.group(1).strip()
        packed_target = self.metal_explicit_packed_vector_type(target_type)
        if packed_target is not None:
            return packed_target
        return self.metal_native_narrow_bitcast_storage_type(
            target_type
        ) or self.map_type(target_type)

    def metal_explicit_packed_vector_type(self, value_type):
        """Return an exact packed-vector identity that cannot be normalized."""
        source_name = self.type_name_string(value_type)
        raw_type = self.resolve_metal_type_alias(source_name)
        normalized = re.sub(r"^(?:metal::)+", "", str(raw_type or "").strip())
        if re.fullmatch(
            r"packed_(?:char|uchar|short|ushort|int|uint|half|float)[234]",
            normalized,
        ) or re.fullmatch(
            r"packed_(?:vector|vec)\s*<.+,\s*[234]\s*>",
            normalized,
        ):
            return normalized
        return None

    def metal_explicit_bitcast_type_info(self, value_type):
        """Return the exact native scalar/vector storage width for ``as_type``."""
        source_name = self.type_name_string(value_type)
        if not source_name:
            return None
        raw_type = self.resolve_metal_type_alias(source_name)
        if not raw_type or self.metal_explicit_packed_vector_type(raw_type) is not None:
            return None
        mapped_type = self.metal_native_narrow_bitcast_storage_type(
            raw_type
        ) or self.map_type(raw_type)
        normalized = re.sub(r"^(?:metal::)+", "", str(mapped_type or "").strip())
        scalar_widths = {
            "char": 8,
            "uchar": 8,
            "int8": 8,
            "uint8": 8,
            "int8_t": 8,
            "uint8_t": 8,
            "short": 16,
            "ushort": 16,
            "int16": 16,
            "uint16": 16,
            "int16_t": 16,
            "uint16_t": 16,
            "half": 16,
            "float16": 16,
            "float16_t": 16,
            "bfloat": 16,
            "bfloat16": 16,
            "bfloat16_t": 16,
            "int": 32,
            "uint": 32,
            "int32": 32,
            "uint32": 32,
            "int32_t": 32,
            "uint32_t": 32,
            "float": 32,
            "long": 64,
            "ulong": 64,
            "int64": 64,
            "uint64": 64,
            "int64_t": 64,
            "uint64_t": 64,
            "double": 64,
        }
        component_type = normalized
        lanes = 1
        vector_match = re.fullmatch(r"(.+?)([234])", normalized)
        if vector_match is not None and vector_match.group(1) in scalar_widths:
            component_type = vector_match.group(1)
            lanes = int(vector_match.group(2))
        else:
            generic_match = re.fullmatch(
                r"(?:vector|vec)\s*<\s*(.+?)\s*,\s*([234])\s*>",
                normalized,
            )
            if generic_match is not None:
                component_type = re.sub(
                    r"^(?:metal::)+", "", generic_match.group(1).strip()
                )
                lanes = int(generic_match.group(2))
        component_width = scalar_widths.get(component_type)
        if component_width is None:
            return None
        # Metal's ordinary three-component vectors have the same size and
        # alignment as their four-component counterparts.  ``as_type`` is
        # constrained by native object size, not by the number of semantic
        # lanes, so uint3 <-> uint4 (and corresponding scalar families) is a
        # valid 128-bit bitcast.  Packed vectors are rejected before mapping,
        # because normalizing them to ordinary vectors would change storage.
        storage_lanes = 4 if lanes == 3 else lanes
        return {
            "type": normalized,
            "component": component_type,
            "componentWidth": component_width,
            "lanes": lanes,
            "storageLanes": storage_lanes,
            "totalWidth": component_width * storage_lanes,
        }
