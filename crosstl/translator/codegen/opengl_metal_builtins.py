"""Metal built-ins in the GLSL generator: statement termination, float selection and SIMD helpers, atomic fences, function bodies and floating remainder calls."""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerAccessNode,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
)
from ..float_limit_constants import render_float_limit_constant
from ..math_namespaces import (
    METAL_FREE_ARITHMETIC_OPERATOR_PREFIXES,
    METAL_MATH_NAMESPACE_ANNOTATION,
)
from ..refusals import OpenGLRefusal
from ..standard_constants import render_standard_math_constant
from ..validation import expression_debug_name
from .enum_utils import (
    enum_value_expression,
    generate_enum_constructor_call,
    generate_enum_constructor_expression,
)
from .for_initializers import join_for_initializer
from .glsl_draw_parameters import (
    draw_parameter_spelling,
    vertex_builtin_spelling,
)
from .glsl_version_floor import glsl_version_reaches
from .opengl_values import (
    GLSL_FLOAT_TO_INTEGER_RULE_HELPERS,
    OpenGLAtomicFenceLoweringError,
    OpenGLFloatingRemainderError,
    OpenGLIsFiniteError,
)
from .stage_utils import normalize_stage_name


class OpenGLMetalBuiltinMixin:
    """Metal builtin members of ``GLSLCodeGen``."""

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
        previous_local_variable_source_types = dict(self.local_variable_source_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        previous_type_aliases = dict(self.current_type_aliases)
        previous_workgroup_pointer_aliases = dict(
            self.current_glsl_workgroup_pointer_aliases
        )
        previous_storage_pointer_aliases = dict(
            self.current_glsl_storage_pointer_aliases
        )
        previous_scalar_array_views = dict(self.current_glsl_scalar_array_views)
        previous_compile_time_constant_state = (
            self.glsl_compile_time_constant_scope_state()
        )
        previous_workgroup_pointer_scope_depth = self.glsl_workgroup_pointer_scope_depth
        try:
            self.glsl_workgroup_pointer_scope_depth = (
                previous_workgroup_pointer_scope_depth + 1
            )
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.local_variable_source_types = previous_local_variable_source_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_type_aliases = previous_type_aliases
            self.current_glsl_workgroup_pointer_aliases = (
                previous_workgroup_pointer_aliases
            )
            self.current_glsl_storage_pointer_aliases = previous_storage_pointer_aliases
            self.current_glsl_scalar_array_views = previous_scalar_array_views
            self.restore_glsl_compile_time_constant_scope_state(
                previous_compile_time_constant_state
            )
            self.glsl_workgroup_pointer_scope_depth = (
                previous_workgroup_pointer_scope_depth
            )

    def generate_statement_body(self, body, indent):
        code = ""
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent)
        elif body is not None:
            code += self.generate_statement(body, indent)
        return code

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, list):
            return join_for_initializer(
                [self.generate_for_initializer(item) for item in init],
                [isinstance(item, VariableNode) for item in init],
                "opengl",
            )
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            return self.generate_statement(init, 0).strip().rstrip(";")
        return self.generate_expression(init).strip().rstrip(";")

    def glsl_target_builtin_identifier(self, name):
        stage_name = normalize_stage_name(
            getattr(self, "current_stage_entry_type", None)
        )
        if stage_name == "compute":
            quadgroup_alias = {
                "gl_QuadGroupID": "(gl_LocalInvocationIndex / 4u)",
                "gl_QuadGroupInvocationID": "(gl_LocalInvocationIndex % 4u)",
            }.get(name)
            if quadgroup_alias is not None:
                return quadgroup_alias
        if stage_name == "vertex" and name == "gl_VertexIndex":
            return "gl_VertexID"
        if stage_name == "vertex":
            return vertex_builtin_spelling(self, name)
        return draw_parameter_spelling(self, name)

    def glsl_identifier_is_resolved(self, name):
        symbol_tables = (
            self.local_variable_types,
            self.local_variable_source_types,
            self.global_variable_types,
            self.glsl_global_identifier_names,
            self.global_type_aliases,
            self.current_type_aliases,
            self.function_return_types,
            self.structs_by_name,
            getattr(self, "enum_variant_constants", {}),
            self.current_resource_aliases,
            self.current_identifier_aliases,
            self.current_stage_parameter_aliases,
            self.current_texture_parameters,
            self.texture_variable_types,
            self.current_glsl_workgroup_pointer_aliases,
            self.current_glsl_storage_pointer_aliases,
            self.glsl_global_workgroup_pointer_aliases,
            self.glsl_global_storage_pointer_aliases,
            self.current_stage_inputs,
            self.current_stage_outputs,
            self.glsl_uniform_block_variable_types,
        )
        if any(name in symbols for symbols in symbol_tables):
            return True
        if any(
            str(path).rsplit("::", 1)[-1] == name
            for path in getattr(self, "enum_variant_constants", {})
        ):
            return True
        symbol_sets = (
            self.current_sampler_parameters,
            self.sampler_variables,
            self.stage_entry_resource_declaration_names,
            self.glsl_buffer_block_variable_names,
            self.glsl_hoisted_shared_alias_names,
            self.flattened_stage_variables,
            getattr(self, "enum_type_names", set()),
            getattr(self, "enum_struct_type_names", set()),
        )
        return any(name in symbols for symbols in symbol_sets)

    def generate_glsl_identifier_expression(self, name):
        stage_alias = self.current_stage_parameter_aliases.get(name)
        if stage_alias is not None:
            return stage_alias

        target_name = self.glsl_target_builtin_identifier(name)
        if target_name != name or self.glsl_identifier_is_resolved(name):
            return target_name

        # GLSL defines none of the C/Metal float limit macros (NAN, INFINITY,
        # FLT_MAX, ...). Render them from their exact binary32 encodings.
        limit_constant = render_float_limit_constant(name, "opengl")
        if limit_constant is not None:
            return limit_constant
        standard_constant = render_standard_math_constant(name, "opengl")
        return standard_constant if standard_constant is not None else target_name

    def is_glsl_builtin_option_none_expression(self, expr):
        if not getattr(self, "glsl_builtin_option_available", False):
            return False
        if isinstance(expr, str):
            return expr == "None"
        return getattr(expr, "name", None) == "None"

    def glsl_builtin_option_constructor_name(self, func_name):
        if not getattr(self, "glsl_builtin_option_available", False):
            return func_name
        if func_name == "Some":
            return "Option::Some"
        if func_name == "None":
            return "Option::None"
        return func_name

    def generate_glsl_builtin_option_none_value(self):
        if not getattr(self, "glsl_builtin_option_available", False):
            return None
        return self.glsl_enum_constructor_call("Option::None", [])

    def glsl_rewrite_generic_enum_constructor_call(self, rendered):
        if rendered is None:
            return None
        for (type_text, variant_name), target_name in sorted(
            self.glsl_generic_enum_constructor_names.items()
        ):
            specialization = self.generic_enum_specializations[type_text]
            source_name = f"{specialization['struct_name']}_{variant_name}_make"
            prefix = f"{source_name}("
            if rendered.startswith(prefix):
                return f"{target_name}{rendered[len(source_name) :]}"
        return rendered

    def glsl_enum_constructor_call(self, path, arguments):
        return self.glsl_rewrite_generic_enum_constructor_call(
            generate_enum_constructor_call(self, path, arguments)
        )

    def glsl_enum_constructor_expression(self, expression):
        return self.glsl_rewrite_generic_enum_constructor_call(
            generate_enum_constructor_expression(self, expression)
        )

    def glsl_enum_value_expression(self, path):
        return self.glsl_rewrite_generic_enum_constructor_call(
            enum_value_expression(self, path)
        )

    def generate_glsl_builtin_option_none_comparison(self, left, op, right):
        if op not in {"==", "!="}:
            return None
        left_is_none = self.is_glsl_builtin_option_none_expression(left)
        right_is_none = self.is_glsl_builtin_option_none_expression(right)
        if left_is_none == right_is_none:
            return None

        subject = right if left_is_none else left
        subject_expr = self.generate_expression(subject)
        operator = "==" if op == "==" else "!="
        none_name = self.enum_variant_constants.get("Option::None", "Option_None")
        return f"({subject_expr}.variant {operator} {none_name})"

    def glsl_expression_reads_storage_buffer(self, expression):
        """True when ``expression`` names a value stored in a storage buffer.

        Follows member and element selection, and copy construction of a
        structure from a value of the same structure type, back to its root:
        a structured buffer load, an element of a storage pointer or buffer
        root, or the pointee of a storage pointer.
        """
        aliases = self.glsl_storage_pointer_aliases()
        node = expression
        while True:
            if isinstance(node, FunctionCallNode):
                arguments = list(getattr(node, "arguments", None) or [])
                name = self.function_call_name(node)
                if not isinstance(name, str) or name in self.function_return_types:
                    return False
                if name == "buffer_load":
                    return len(arguments) == 2
                if len(arguments) != 1 or self.glsl_struct_node(name) is None:
                    return False
                argument_type = self.glsl_source_expression_type(arguments[0])
                if argument_type is None or self.map_type(
                    argument_type
                ) != self.map_type(name):
                    return False
                node = arguments[0]
                continue
            if isinstance(node, MemberAccessNode):
                node = node.object
                continue
            if isinstance(node, PointerAccessNode):
                return (
                    self.glsl_storage_pointer_binding(node.pointer_expr, aliases)
                    is not None
                )
            if isinstance(node, ArrayAccessNode):
                if self.glsl_storage_pointer_binding(node.array, aliases) is not None:
                    return True
                node = node.array
                continue
            if isinstance(node, UnaryOpNode) and self.map_operator(node.op) == "*":
                return (
                    self.glsl_storage_pointer_binding(node.operand, aliases) is not None
                )
            return False

    def glsl_copied_storage_structure_arm(self, arm, rendered):
        """Copy a structure read from a storage buffer before ``?:`` selects it.

        glslang 12 stores a structure loaded from a storage buffer straight
        into the selection's function-scope result variable. The loaded value
        has the buffer's explicitly laid out structure type, so the store mixes
        two structure types and spirv-val rejects the module. A by-value helper
        call makes glslang copy the value member-wise into a function-scope
        parameter first; the arm is still evaluated only when it is selected.
        """
        if self.glsl_generating_global_initializer:
            return rendered
        source_type = self.glsl_source_expression_type(arm)
        if source_type is None or self.glsl_struct_node(source_type) is None:
            return rendered
        if not self.glsl_expression_reads_storage_buffer(arm):
            return rendered
        mapped = self.map_type(source_type)
        name = self.glsl_storage_structure_copy_helper_names.get(mapped)
        if name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            name = self.glsl_unique_identifier(f"crossgl_copy_{mapped}", used_names)
            self.glsl_module_used_identifier_names.add(name)
            self.glsl_storage_structure_copy_helper_names[mapped] = name
        return f"{name}({rendered})"

    def generate_glsl_storage_structure_copy_helpers(self):
        return "".join(
            f"{mapped} {name}({mapped} value) {{\n    return value;\n}}\n\n"
            for mapped, name in sorted(
                self.glsl_storage_structure_copy_helper_names.items()
            )
        )

    def glsl_signed_remainder_type(self, left_type, right_type):
        common = self.glsl_common_arithmetic_type(left_type, right_type, "%")
        info = self.glsl_value_type_info(common)
        if info is None or info["family"] != "int":
            return None
        return self.map_type(common)

    def generate_glsl_signed_remainder(self, node, left, right, left_type, right_type):
        mapped = self.glsl_signed_remainder_type(left_type, right_type)
        if mapped is None:
            return None
        width = self.glsl_value_type_info(mapped)["width"]
        if width > 1:
            if self.glsl_value_type_info(left_type)["width"] == 1:
                left = f"{mapped}({left})"
            if self.glsl_value_type_info(right_type)["width"] == 1:
                right = f"{mapped}({right})"
        if self.glsl_generating_global_initializer:
            if not all(
                self.glsl_side_effect_free_expression(operand)
                for operand in (node.left, node.right)
            ):
                self.glsl_arithmetic_conversion_error(
                    node,
                    "%",
                    left_type,
                    right_type,
                    mapped,
                    "signed-remainder-global-side-effects",
                )
            return f"({left} - ({left} / {right}) * {right})"
        name = self.glsl_signed_remainder_helper_names.get(mapped)
        if name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            name = self.glsl_unique_identifier(
                f"crossgl_signed_remainder_{mapped}", used_names
            )
            self.glsl_module_used_identifier_names.add(name)
            self.glsl_signed_remainder_helper_names[mapped] = name
        return f"{name}({left}, {right})"

    def generate_glsl_signed_remainder_helpers(self):
        # GLSL '%' does not guarantee the C-like sign for negative operands.
        return "".join(
            f"{mapped} {name}({mapped} left, {mapped} right) {{\n"
            "    return left - (left / right) * right;\n}\n\n"
            for mapped, name in sorted(self.glsl_signed_remainder_helper_names.items())
        )

    def generate_glsl_float_selection(self, node, condition, true_expr, false_expr):
        # Only already-evaluated values may cross the helper call boundary.
        # Calls, indexing and other lazy arms retain native conditional evaluation.
        if self.glsl_generating_global_initializer:
            return None
        if not self.glsl_side_effect_free_expression(node.condition) or not all(
            isinstance(arm, (VariableNode, IdentifierNode, LiteralNode))
            for arm in (node.true_expr, node.false_expr)
        ):
            return None
        result_type = self.glsl_source_expression_type(node)
        mapped = self.map_type(result_type) if result_type is not None else None
        if mapped not in {"float", "vec2", "vec3", "vec4"}:
            return None
        for builtin in ("floatBitsToUint", "uintBitsToFloat"):
            if (
                builtin in self.function_return_types
                or builtin in self.global_variable_types
            ):
                self.glsl_scalar_conversion_error(
                    node,
                    result_type,
                    result_type,
                    "conditional-target-builtin-shadowed",
                )
        name = self.glsl_float_selection_helper_names.get(mapped)
        if name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            name = self.glsl_unique_identifier(
                f"crossgl_select_bits_{mapped}", used_names
            )
            self.glsl_module_used_identifier_names.add(name)
            self.glsl_float_selection_helper_names[mapped] = name
        return f"{name}({condition}, {true_expr}, {false_expr})"

    def generate_glsl_float_selection_helpers(self):
        code = ""
        for mapped, name in sorted(self.glsl_float_selection_helper_names.items()):
            mask_type = "uint" if mapped == "float" else "uvec" + mapped[-1]
            code += (
                f"{mapped} {name}(bool condition, {mapped} yes, {mapped} no) {{\n"
                f"    {mask_type} mask = {mask_type}(0u - uint(condition));\n"
                "    return uintBitsToFloat((floatBitsToUint(yes) & mask) | "
                "(floatBitsToUint(no) & ~mask));\n"
                "}\n\n"
            )
        return code

    def glsl_float_to_integer_conversion_info(self, expression):
        """Integer type info when ``expression`` converts floating values."""
        if not isinstance(expression, FunctionCallNode):
            return None
        arguments = list(getattr(expression, "arguments", []) or [])
        name = self.function_call_name(expression)
        if len(arguments) != 1:
            return None
        if name in GLSL_FLOAT_TO_INTEGER_RULE_HELPERS:
            # The ftoi_nan and convert_sat helpers replace int(x) and uint(x) of a float x.
            target = self.glsl_value_type_info(self.function_return_types.get(name))
        elif name in self.function_return_types:
            return None
        elif self.glsl_constructor_type(name) is None:
            return None
        else:
            target = self.glsl_value_type_info(name)
        if target is None or target["family"] not in {"int", "uint"}:
            return None
        source = self.glsl_value_type_info(
            self.glsl_source_expression_type(arguments[0])
        )
        if (
            source is None
            or source["family"] != "float"
            or source["width"] not in {1, target["width"]}
        ):
            return None
        return target

    def glsl_integer_float_conversion_applies(self, integer, float_type):
        if integer is None or self.glsl_generating_global_initializer:
            return False
        target = self.glsl_value_type_info(float_type)
        return (
            target is not None
            and target["family"] == "float"
            and target["bits"] == 32
            and integer["width"] in {1, target["width"]}
        )

    def generate_glsl_integer_float_conversion(
        self, node, generated, integer, float_type
    ):
        """Convert a float-to-integer result back to binary32 exactly.

        Drivers fold ``float(int(x))`` into ``trunc(x)``, which keeps the sign
        of a zero quotient; the integer round trip always produces +0.0.
        Global initializers are constant expressions folded exactly by the
        GLSL front end, so they keep the plain constructor.
        """
        if not self.glsl_integer_float_conversion_applies(integer, float_type):
            return None
        target = self.glsl_value_type_info(float_type)
        for builtin in ("floatBitsToUint", "uintBitsToFloat", "equal"):
            if (
                builtin in self.function_return_types
                or builtin in self.global_variable_types
            ):
                self.glsl_scalar_conversion_error(
                    node,
                    integer["source"],
                    float_type,
                    "conversion-target-builtin-shadowed",
                )
        mapped = integer["mapped"]
        name = self.glsl_integer_float_helper_names.get(mapped)
        if name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            name = self.glsl_unique_identifier(
                f"crossgl_integer_float_{mapped}", used_names
            )
            self.glsl_module_used_identifier_names.add(name)
            self.glsl_integer_float_helper_names[mapped] = name
        converted = f"{name}({generated})"
        if integer["width"] == 1 and target["width"] != 1:
            return f"{target['mapped']}({converted})"
        return converted

    def generate_glsl_integer_float_helpers(self):
        code = ""
        for mapped, name in sorted(self.glsl_integer_float_helper_names.items()):
            info = self.glsl_value_type_info(mapped)
            width = info["width"]
            if width == 1:
                result, is_zero = "float", f"uint(value == {mapped}(0))"
            else:
                result = f"vec{width}"
                is_zero = f"uvec{width}(equal(value, {mapped}(0)))"
            code += (
                f"{result} {name}({mapped} value) {{\n"
                f"    return uintBitsToFloat(floatBitsToUint({result}(value)) & "
                f"~({is_zero} << 31u));\n"
                "}\n\n"
            )
        return code

    def atomic_fence_operand_identifier(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.atomic_fence_operand_text(expr.left)} | "
                f"{self.atomic_fence_operand_text(expr.right)}"
            )
        name = self.atomic_fence_operand_identifier(expr)
        return name if name is not None else expression_debug_name(expr)

    def collect_atomic_fence_memory_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_atomic_fence_memory_flags(expr.left)
            right = self.collect_atomic_fence_memory_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        name = self.atomic_fence_operand_identifier(expr)
        if name in self.GLSL_ATOMIC_FENCE_MEMORY_FLAGS:
            return frozenset({name})
        return None

    def opengl_atomic_fence_contract_error(
        self,
        reason,
        args,
        *,
        source_location=None,
    ):
        rendered = [self.atomic_fence_operand_text(arg) for arg in args]
        return OpenGLAtomicFenceLoweringError(
            reason,
            memory_flags=rendered[0] if rendered else None,
            memory_order=rendered[1] if len(rendered) > 1 else None,
            thread_scope=rendered[2] if len(rendered) > 2 else None,
            source_location=source_location,
        )

    def validate_glsl_atomic_thread_fence_contract(
        self,
        args,
        *,
        source_location=None,
        validate_stage=True,
    ):
        if len(args) != 3:
            raise self.opengl_atomic_fence_contract_error(
                "invalid-argument-count",
                args,
                source_location=source_location,
            )

        flags = self.collect_atomic_fence_memory_flags(args[0])
        order = self.atomic_fence_operand_identifier(args[1])
        scope = self.atomic_fence_operand_identifier(args[2])
        if flags is None:
            raise self.opengl_atomic_fence_contract_error(
                "unknown-memory-flags",
                args,
                source_location=source_location,
            )
        if order not in self.GLSL_ATOMIC_FENCE_MEMORY_ORDERS:
            raise self.opengl_atomic_fence_contract_error(
                "unknown-memory-order",
                args,
                source_location=source_location,
            )
        if scope not in self.GLSL_ATOMIC_FENCE_THREAD_SCOPES:
            raise self.opengl_atomic_fence_contract_error(
                "unknown-thread-scope",
                args,
                source_location=source_location,
            )

        if scope != "thread_scope_threadgroup":
            reason = (
                "unsupported-system-thread-scope"
                if scope == "thread_scope_system"
                else "unsupported-thread-scope"
            )
            raise self.opengl_atomic_fence_contract_error(
                reason,
                args,
                source_location=source_location,
            )
        if order != "memory_order_acq_rel":
            raise self.opengl_atomic_fence_contract_error(
                "unsupported-memory-order",
                args,
                source_location=source_location,
            )
        if flags != frozenset({"mem_threadgroup"}):
            raise self.opengl_atomic_fence_contract_error(
                "unsupported-memory-flags",
                args,
                source_location=source_location,
            )
        if validate_stage and (
            normalize_stage_name(
                self.current_stage_entry_type or self.glsl_atomic_fence_stage
            )
            != "compute"
        ):
            raise self.opengl_atomic_fence_contract_error(
                "unsupported-shader-stage",
                args,
                source_location=source_location,
            )

    def validate_glsl_atomic_fences_before_resource_qualifiers(
        self, ast, target_stage=None
    ):
        if "atomicThreadFence" in self.function_return_types:
            return
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if not isinstance(node, FunctionCallNode):
                    continue
                if self.function_call_name(node) != "atomicThreadFence":
                    continue
                self.validate_glsl_atomic_thread_fence_contract(
                    node.args,
                    source_location=getattr(node, "source_location", None),
                    validate_stage=False,
                )

    def generate_glsl_atomic_thread_fence(self, args, *, source_location=None):
        self.validate_glsl_atomic_thread_fence_contract(
            args,
            source_location=source_location,
        )
        return "memoryBarrierShared()"

    def synchronization_function_call(self, func_name, args, *, source_location=None):
        if not func_name or func_name in self.function_return_types:
            return None
        if func_name == "subgroupExecutionBarrier":
            if self.software_subgroup_width is not None:
                return self.generate_glsl_software_subgroup_operation(
                    func_name, args, source_location=source_location
                )
            if args:
                raise self.glsl_software_subgroup_error(
                    "OpenGL subgroup execution barriers require zero arguments",
                    operation=func_name,
                    reason="invalid-argument-count",
                    source_location=source_location,
                )
            return "subgroupBarrier()"
        if self.software_subgroup_width is not None and not args:
            # Software subgroups are slices of the workgroup, so a workgroup-scope fence orders their memory.
            widened = self.GLSL_SUBGROUP_FENCE_WIDENINGS.get(func_name)
            if widened is not None:
                return f"{widened}()"
            if func_name == "subgroupBarrier":
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroups have no subgroup barrier over buffer, image and shared memory",
                    operation=func_name,
                    reason="subgroup-memory-barrier-unsupported",
                    source_location=source_location,
                )
        if func_name == "atomicThreadFence":
            return self.generate_glsl_atomic_thread_fence(
                args,
                source_location=source_location,
            )
        if args:
            return None
        if func_name in {"workgroupBarrier", "workgroupExecutionBarrier"}:
            return "barrier()"
        if func_name in {"deviceMemoryBarrier", "allMemoryBarrier"}:
            # GLSL has no device-scope fence limited to buffers and images; memoryBarrier() orders those and
            # shared memory (GLSL 4.60 section 8.16), which covers both.
            return "memoryBarrier()"
        return None

    def collect_metal_simd_group_placeholder_functions(self, functions):
        return {
            func.name
            for func in functions or []
            if self.is_metal_simd_group_placeholder_function(func)
            and self.is_plain_metal_simd_group_helper_name(func.name)
            and func.name in self.GLSL_METAL_SIMD_GROUP_PLAIN_HELPER_OPERATIONS
        }

    def is_metal_simd_group_placeholder_function(self, func):
        func_name = getattr(func, "name", None)
        if not self.is_plain_metal_simd_group_helper_name(func_name):
            return False
        return self.is_empty_function_body(getattr(func, "body", []))

    def should_elide_metal_simd_group_placeholder_function(self, func):
        return (
            getattr(func, "name", None)
            in self.glsl_metal_simd_group_placeholder_functions
        )

    def should_elide_glsl_fixed_array_vector_pointer_view_helper(self, func):
        return id(func) in getattr(
            self,
            "glsl_fixed_array_vector_pointer_view_helper_ids",
            set(),
        )

    def should_elide_glsl_software_subgroup_helper(self, func):
        omitted = getattr(
            self,
            "glsl_software_subgroup_omitted_helper_function_ids",
            set(),
        )
        source_function_ids = {
            id(func),
            getattr(func, "_glsl_resource_source_function_id", None),
            getattr(func, "_generic_source_function_id", None),
        }
        source_function_ids.discard(None)
        return bool(source_function_ids & omitted)

    def should_elide_glsl_function(self, func):
        return (
            self.should_elide_metal_simd_group_placeholder_function(func)
            or self.should_elide_glsl_fixed_array_vector_pointer_view_helper(func)
            or self.should_elide_glsl_software_subgroup_helper(func)
        )

    def is_empty_function_body(self, body):
        if body is None:
            return False
        if hasattr(body, "statements"):
            return len(body.statements) == 0
        if isinstance(body, list):
            return len(body) == 0
        return False

    def glsl_unsupported_metal_simd_placeholder_function_body(self, func, indent):
        if self.should_elide_glsl_function(func):
            return None
        if not self.is_metal_simd_group_placeholder_function(func):
            return None

        return_type = (
            self.type_name_string(getattr(func, "return_type", None)) or "void"
        )
        mapped_return_type = self.map_type(return_type)
        if mapped_return_type == "void":
            return None

        indent_str = "    " * indent
        fallback = self.refuse_lowering(
            f"wave intrinsic: {func.name} has no OpenGL subgroup equivalent",
            self.zero_value_expression(return_type),
        )
        return f"{indent_str}return {fallback};\n"

    def glsl_wave_operation_name(self, operation):
        if not isinstance(operation, str):
            return operation
        mapped = self.GLSL_METAL_SIMD_GROUP_HELPER_OPERATIONS.get(operation)
        if mapped is None:
            return operation
        if not self.is_plain_metal_simd_group_helper_name(operation):
            return mapped
        if operation in self.glsl_metal_simd_group_placeholder_functions:
            return mapped
        if operation not in self.function_return_types:
            return mapped
        return operation

    def is_plain_metal_simd_group_helper_name(self, func_name):
        if not isinstance(func_name, str):
            return False
        if func_name in self.GLSL_METAL_SIMD_GROUP_PLAIN_HELPER_OPERATIONS:
            return True
        return func_name.startswith(
            (
                "simd_prefix_",
                "simd_broadcast",
                "simd_shuffle",
                "simd_vote",
                "simd_is_",
            )
        )

    def is_metal_simd_group_helper_name(self, func_name):
        if not isinstance(func_name, str):
            return False
        if func_name.startswith(self.GLSL_METAL_SIMD_GROUP_HELPER_PREFIX):
            return True
        if not self.is_plain_metal_simd_group_helper_name(func_name):
            return False
        return (
            func_name in self.glsl_metal_simd_group_placeholder_functions
            or func_name not in self.function_return_types
        )

    def is_mangled_metal_simd_group_helper_name(self, func_name):
        return isinstance(func_name, str) and func_name.startswith("metal_u3a_u3asimd_")

    def generate_glsl_wave_op_expression(self, node):
        # A predicated reduction already ran outside its guard; its source
        # expression reads that result.
        override = getattr(self, "glsl_software_subgroup_result_overrides", {}).get(
            id(node)
        )
        if override is not None:
            return override
        return self.generate_glsl_wave_operation(
            node.operation,
            node.arguments,
            source_location=getattr(node, "source_location", None),
        )

    def generate_saturate_call(self, func_name, args):
        if func_name != "saturate" or func_name in self.function_return_types:
            return None
        if len(args) != 1:
            raise OpenGLRefusal("OpenGL saturate alias requires 1 argument")

        value = self.generate_expression(args[0])
        zero, one = self.saturate_bound_literals(args[0])
        return f"clamp({value}, {zero}, {one})"

    def glsl_isfinite_result_type(self, func_name, args, *, source_type=False):
        if func_name not in self.GLSL_ISFINITE_NAMES or len(args) != 1:
            return None
        argument_type = (
            self.glsl_source_expression_type(args[0])
            if source_type
            else self.expression_result_type(args[0])
        )
        info = self.glsl_value_type_info(argument_type)
        if info is None:
            return "bool"
        return "bool" if info["width"] == 1 else f"bvec{info['width']}"

    def glsl_isfinite_error(
        self,
        message,
        *,
        operand_type=None,
        result_type=None,
        reason,
        source_location=None,
    ):
        return OpenGLIsFiniteError(
            message,
            operand_type=operand_type,
            result_type=result_type,
            reason=reason,
            source_location=source_location,
        )

    def generate_glsl_isfinite_call(self, func_name, args, *, call_node=None):
        if func_name not in self.GLSL_ISFINITE_NAMES:
            return None

        resolved_overload = self.resolve_glsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 1:
            raise self.glsl_isfinite_error(
                "OpenGL isfinite requires exactly one operand",
                reason="invalid-arity",
                source_location=source_location,
            )

        source_type = self.glsl_source_expression_type(args[0])
        display_type = (
            self.type_name_string(source_type) if source_type is not None else None
        )
        if source_type is None:
            raise self.glsl_isfinite_error(
                "OpenGL isfinite requires a statically known operand type",
                operand_type=display_type,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        info = self.glsl_signbit_type_info(source_type)
        if info is None:
            raise self.glsl_isfinite_error(
                "OpenGL isfinite supports only 32-bit float scalar and vector "
                f"operands, got {display_type}",
                operand_type=display_type,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        value = self.generate_expression_with_expected(args[0], info["type"])
        exponent_mask = "0x7f800000u"
        if info["lanes"] == 1:
            return f"((floatBitsToUint({value}) & {exponent_mask}) != {exponent_mask})"
        mask_type = f"uvec{info['lanes']}"
        bits = f"(floatBitsToUint({value}) & {mask_type}({exponent_mask}))"
        return f"notEqual({bits}, {mask_type}({exponent_mask}))"

    def generate_reciprocal_call(self, func_name, args):
        if func_name != "rcp" or func_name in self.function_return_types:
            return None
        if len(args) != 1:
            raise OpenGLRefusal("OpenGL rcp alias requires 1 argument")

        value_expr = args[0]
        value_type = self.map_type(self.expression_result_type(value_expr))
        component_type = self.vector_component_type(value_type) or value_type
        if component_type not in {"float", "double"}:
            return None

        value = self.generate_expression(value_expr)
        if self.is_scalar_value_type(value_type) or self.is_vector_value_type(
            value_type
        ):
            return f"(1.0 / {value})"
        return None

    def glsl_ieee_helper_name(self, operation, value_type):
        identity = ("ieee-math", operation, value_type)
        helper_name = self.glsl_module_generated_identifier_names.get(identity)
        if helper_name is None:
            used_names = self.glsl_module_used_identifier_names | getattr(
                self, "glsl_numeric_helper_reserved_names", set()
            )
            stem = operation.replace("-", "_")
            helper_name = self.glsl_unique_identifier(
                f"crossgl_{stem}_{value_type}", used_names
            )
            self.glsl_module_used_identifier_names.add(helper_name)
            self.glsl_module_generated_identifier_names[identity] = helper_name
        return helper_name

    def glsl_ieee_value_info(self, args):
        """Return the shared binary32/64 float type of ``args``, else None."""
        infos = [
            self.glsl_value_type_info(self.glsl_source_expression_type(arg))
            for arg in args
        ]
        if not infos or any(info is None for info in infos):
            return None
        widest = max(infos, key=lambda info: info["width"])
        if any(
            info["family"] != "float"
            or info["bits"] != widest["bits"]
            or info["bits"] not in {32, 64}
            or info["width"] not in {1, widest["width"]}
            for info in infos
        ):
            return None
        if widest["width"] not in {1, 2, 3, 4}:
            return None
        return widest

    def generate_glsl_ieee_math_call(self, func_name, args, *, call_node=None):
        """Lower calls whose GLSL builtin loses IEEE semantics on Mesa."""
        if func_name in self.function_return_types or func_name in (
            self.local_variable_types
        ):
            return None
        if func_name == "fma":
            operation = "fma"
            if len(args) != 3:
                return None
        elif func_name in self.GLSL_PRECISE_ODD_FUNCTIONS:
            get_annotation = getattr(call_node, "get_annotation", None)
            if not callable(get_annotation) or (
                get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) != "precise"
            ):
                return None
            if len(args) != 1:
                return None
            operation = f"precise-{func_name}"
        elif func_name == "atan2":
            if len(args) != 2 or not self.glsl_version_supports_bit_casts():
                return None
            operation = "atan2"
        else:
            return None
        info = self.glsl_ieee_value_info(args)
        if info is None or (operation != "fma" and info["bits"] != 32):
            return None
        if operation == "atan2" and any(
            self.glsl_narrow_float_width(self.glsl_source_expression_type(arg))
            is not None
            for arg in args
        ):
            return None
        value_type = info["mapped"]
        if operation == "fma" and all(
            self.glsl_half_width(self.glsl_source_expression_type(arg)) is not None
            for arg in args
        ):
            # A binary32 fma of binary16 values rounds twice; the helper rounds the exact sum once.
            operation = "half-fma"
        rendered = [
            self.generate_expression_with_expected(arg, value_type) for arg in args
        ]
        helper_name = self.glsl_ieee_helper_name(operation, value_type)
        self.required_glsl_metal_math_helpers.add(
            (helper_name, operation, value_type, "", info["width"])
        )
        if operation == "half-fma":
            if info["width"] != 1:
                scalar_name = self.glsl_ieee_helper_name("half-fma", "float")
                self.required_glsl_metal_math_helpers.add(
                    (scalar_name, "half-fma", "float", "", 1)
                )
            return f"{helper_name}({', '.join(rendered)})"
        if operation == "atan2":
            if info["width"] != 1:
                # Vector lanes call the scalar helper.
                scalar_name = self.glsl_ieee_helper_name("atan2", "float")
                self.required_glsl_metal_math_helpers.add(
                    (scalar_name, "atan2", "float", "", 1)
                )
            return f"{helper_name}({', '.join(rendered)})"
        # Like the builtin it replaces, a helper computes in the physical float
        # type, so a unary call's logical half or bfloat16 result is rounded.
        return self.glsl_apply_narrow_float_contract(
            f"{helper_name}({', '.join(rendered)})",
            self.glsl_componentwise_unary_result_type(func_name, args),
            call_node,
        )

    def generate_glsl_identical_difference(self, expr, left, right):
        """Keep ``x - x`` exact: Mesa folds it to 0, which hides inf and NaN."""
        if not (
            isinstance(expr.left, IdentifierNode)
            and isinstance(expr.right, IdentifierNode)
            and expr.left.name == expr.right.name
            and left == right
        ):
            return None
        info = self.glsl_ieee_value_info([expr.left])
        if info is None:
            return None
        helper_name = self.glsl_ieee_helper_name("difference", info["mapped"])
        self.required_glsl_metal_math_helpers.add(
            (helper_name, "difference", info["mapped"], "", info["width"])
        )
        return f"{helper_name}({left}, {right})"

    def glsl_is_metal_free_arithmetic_operator(self, func):
        """Return whether ``func`` is a Metal free arithmetic operator on floats."""
        name = str(getattr(func, "name", "") or "")
        if not name.startswith(METAL_FREE_ARITHMETIC_OPERATOR_PREFIXES):
            return False
        return_type = self.type_name_string(getattr(func, "return_type", None))
        member_types = list(self.glsl_struct_member_types(return_type).values()) or [
            return_type
        ]
        for member_type in member_types:
            info = self.glsl_value_type_info(member_type)
            if (
                info is None
                or info["family"] != "float"
                or self.glsl_narrow_float_width(member_type) is not None
            ):
                return False
        return True

    def generate_glsl_ieee_operator_arithmetic(self, expr, op, left, right):
        """Keep a Metal free operator's binary32 scalar arithmetic IEEE-exact."""
        operation = self.GLSL_IEEE_OPERATOR_HELPERS.get(op)
        if operation is None:
            return None
        operands = [expr.left, expr.right]
        info = self.glsl_ieee_value_info(operands)
        if (
            info is None
            or info["bits"] != 32
            or info["width"] != 1
            or any(
                self.glsl_narrow_float_width(self.glsl_source_expression_type(value))
                is not None
                for value in operands
            )
        ):
            return None
        value_type = info["mapped"]
        helper_name = self.glsl_ieee_helper_name(operation, value_type)
        self.required_glsl_metal_math_helpers.add(
            (helper_name, operation, value_type, "", 1)
        )
        return f"{helper_name}({left}, {right})"

    def glsl_exactness_test_local_ids(self, function):
        """Ids of binary32 locals whose only value is a sum tested for equality.

        MLX's ``log1p`` tests whether rounding absorbed ``x`` with ``float
        xp1 = 1.0f + x; if (xp1 == 1.0f)`` and divides by ``xp1 - 1.0f``.
        Mesa rewrites a non-precise ``(a + b) == a`` into ``b == 0`` and
        ``(1.0f + x) - 1.0f`` into ``x``.  Both NIR rules are inexact-only, so
        declaring the local ``precise`` keeps its sum, and everything the sum
        depends on, IEEE-exact.  Only a local assigned once, by its
        initializer, compares that sum itself; accumulators that are later
        reassigned, such as an attention kernel's running sum, stay as they
        are.

        Returns a mapping from each local's id to its single value, so the
        declaration can confirm, with the scope then in effect, that the sum
        is a floating one (``glsl_exactness_test_value_has_float_sum``).
        """
        body = getattr(function, "body", None)
        if body is None:
            return {}
        declarations = {}
        compared = set()
        assignments = {}

        def has_sum(value):
            return value is not None and any(
                isinstance(node, BinaryOpNode)
                and self.map_operator(getattr(node, "op", None)) in {"+", "-"}
                for node in self.walk_ast(value)
            )

        for node in self.walk_ast(body):
            if isinstance(node, VariableNode):
                declarations.setdefault(node.name, []).append(node)
                value = getattr(node, "initial_value", None)
                if value is not None:
                    assignments.setdefault(node.name, []).append(value)
            elif isinstance(node, AssignmentNode):
                target = getattr(node, "target", None)
                if isinstance(target, IdentifierNode):
                    # Compound assignments read the local they rewrite.
                    operator = self.map_operator(getattr(node, "operator", None))
                    assignments.setdefault(target.name, []).append(
                        getattr(node, "value", None) if operator == "=" else None
                    )
            elif isinstance(node, UnaryOpNode):
                operand = getattr(node, "operand", None)
                if isinstance(operand, IdentifierNode) and str(
                    getattr(node, "operator", "")
                ) in {"++", "--", "INCREMENT", "DECREMENT"}:
                    assignments.setdefault(operand.name, []).append(None)
            elif isinstance(node, BinaryOpNode) and self.map_operator(
                getattr(node, "op", None)
            ) in {"==", "!="}:
                compared.update(
                    value.name
                    for value in (node.left, node.right)
                    if isinstance(value, IdentifierNode)
                )
        result = {}
        for name in sorted(compared):
            found = declarations.get(name, ())
            values = assignments.get(name, ())
            if len(found) != 1 or len(values) != 1 or not has_sum(values[0]):
                continue
            declared = getattr(found[0], "var_type", None)
            info = self.glsl_value_type_info(declared)
            if (
                info is not None
                and info["family"] == "float"
                and info["bits"] == 32
                and self.glsl_narrow_float_width(declared) is None
            ):
                result[id(found[0])] = values[0]
        return result

    def glsl_exactness_test_value_has_float_sum(self, value):
        """Whether ``value`` contains a ``+`` or ``-`` that is not integral.

        An integer difference, such as ``float(n - 8)``, cannot be contracted
        or reassociated by a floating optimization, so it needs no
        ``precise``.  An operation whose type is unknown in the current scope
        keeps the qualifier.
        """
        for node in self.walk_ast(value):
            if not isinstance(node, BinaryOpNode) or self.map_operator(
                getattr(node, "op", None)
            ) not in {"+", "-"}:
                continue
            info = self.glsl_value_type_info(self.glsl_source_expression_type(node))
            if info is None or info["family"] == "float":
                return True
        return False

    def glsl_floating_remainder_helper(self, value_type):
        existing = self.glsl_floating_remainder_helper_names.get(value_type)
        if existing is not None:
            return existing
        name = self.glsl_unique_identifier(
            f"crossgl_fmod_{value_type}",
            self.glsl_module_used_identifier_names
            | getattr(self, "glsl_numeric_helper_reserved_names", set()),
        )
        self.glsl_module_used_identifier_names.add(name)
        self.glsl_floating_remainder_helper_names[value_type] = name
        return name

    def generate_glsl_floating_remainder_call(self, func_name, args, *, call_node=None):
        if func_name not in {"fmod", "metal::fmod", "metal_u3a_u3afmod"}:
            return None
        if (
            self.resolve_glsl_function_overload(func_name, args, call_node=call_node)
            is not None
        ):
            return None
        types = tuple(self.glsl_source_expression_type(arg) for arg in args)

        def reject(reason):
            raise OpenGLFloatingRemainderError(
                f"OpenGL cannot preserve floating remainder for {types}: {reason}",
                operand_types=tuple(self.type_name_string(value) for value in types),
                target_profile=self.current_glsl_version_line,
                reason=reason,
                source_location=getattr(call_node, "source_location", None),
            )

        if len(args) != 2:
            reject("invalid-arity")
        if any(self.glsl_narrow_float_width(value) is not None for value in types):
            reject("unsupported-narrow-remainder-profile")
        infos = tuple(self.glsl_value_type_info(value) for value in types)
        if any(info is None for info in infos):
            reject("unresolved-operand-type")
        if any(
            info["family"] != "float"
            or info["bits"] not in {32, 64}
            or info["width"] not in {1, 2, 3, 4}
            for info in infos
        ):
            reject("unsupported-operand-type")
        if infos[0]["bits"] != infos[1]["bits"]:
            reject("operand-type-mismatch")
        if infos[0]["width"] != infos[1]["width"] and 1 not in {
            info["width"] for info in infos
        }:
            reject("operand-shape-mismatch")
        value = max(infos, key=lambda info: info["width"])
        minimum = 400 if value["bits"] == 64 else 330
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL" or not glsl_version_reaches(
            self, f"a {value['bits']}-bit floating remainder", minimum
        ):
            reject("unsupported-profile")
        builtins = (
            {"floatBitsToUint", "uintBitsToFloat"}
            if value["bits"] == 32
            else {"packDouble2x32", "unpackDouble2x32"}
        )
        if any(
            name in self.function_return_types
            or name in self.global_variable_types
            or name in self.local_variable_types
            for name in builtins
        ):
            reject("target-builtin-shadowed")
        scalar = "float" if value["bits"] == 32 else "double"
        self.glsl_floating_remainder_helper(scalar)
        helper = self.glsl_floating_remainder_helper(value["mapped"])
        rendered = [
            self.generate_expression_with_expected(arg, value["mapped"]) for arg in args
        ]
        rendered = [
            (
                f"{value['mapped']}({expression})"
                if info["width"] == 1 and value["width"] != 1
                else expression
            )
            for info, expression in zip(infos, rendered)
        ]
        return f"{helper}({', '.join(rendered)})"
