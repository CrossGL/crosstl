"""GLSL emission of HLSL ``precise`` arithmetic.

HLSL ``precise`` forbids value-changing floating-point optimizations for the
marked values and for every operation that contributes to them. GLSL attaches
``precise`` to variables only, so these members capture each contributing
expression in a ``precise`` local: function parameters, compound stores,
selected values and call arguments. They also reject target profiles that have
no ``precise`` qualifier.
"""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .resource_arrays import is_private_pointer_parameter


class GLSLHlslPreciseMixin:
    """HLSL ``precise`` lowering members of ``GLSLCodeGen``."""

    GLSL_PRECISE_QUALIFIER_PATTERN = re.compile(r"\bprecise\b")

    GLSL_COMMENT_PATTERN = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)

    def glsl_code_uses_precise_qualifier(self, code):
        """Return whether generated code, outside comments, uses ``precise``.

        ``precise`` is a keyword from desktop GLSL 4.00 and GLSL ES 3.20, and
        identifiers spelled ``precise`` are escaped, so every match outside a
        comment is the qualifier.
        """
        code = self.GLSL_COMMENT_PATTERN.sub("", str(code))
        return self.GLSL_PRECISE_QUALIFIER_PATTERN.search(code) is not None

    def ensure_glsl_precise_support(self, code):
        """Select a profile that has ``precise`` when the output uses it.

        Desktop profiles below 4.00 are raised to 4.00. An ES profile below
        3.20 cannot keep the source's evaluation order, so it is rejected
        instead of silently relaxing it.
        """
        if not self.glsl_code_uses_precise_qualifier(code):
            return code
        version_line = self.current_glsl_version_line
        version_number = self.glsl_version_number(version_line)
        if self.glsl_version_is_es(version_line):
            if (version_number or 0) >= 320:
                return code
            raise self.GLSL_PRECISE_QUALIFIER_ERROR(
                f"{self.GLSL_TARGET_DISPLAY_NAME} target profile "
                f"'{version_line}' has no precise qualifier to keep the "
                "source's floating-point evaluation order; "
                f"{self.GLSL_PRECISE_QUALIFIER_REMEDY}",
                version_line=version_line,
            )
        if version_number is None or version_number >= 400:
            return code
        if self.glsl_extension_enabled("GL_ARB_gpu_shader5"):
            return code
        upgraded_version = self.glsl_version_line_with_minimum(version_line, 400)
        lines = str(code).splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip().startswith("#version"):
                continue
            line_end = "\n" if line.endswith("\n") else ""
            lines[index] = f"{upgraded_version}{line_end}"
            break
        self.current_glsl_version_line = upgraded_version
        return "".join(lines)

    def glsl_written_value_names(self, body):
        """Return the root names that assignments and increments in ``body`` write."""
        names = set()
        for node in self.walk_ast(body):
            target = None
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--"}:
                target = getattr(node, "operand", None)
            if target is not None:
                names.update(self.glsl_mutation_target_names(target))
        return names

    def glsl_precise_parameter_form(
        self,
        parameter,
        raw_param_type,
        param_type,
        parameter_qualifiers,
        shader_type,
        written_value_names,
    ):
        """Return how a parameter of a precise helper function stays precise.

        ``"qualifier"`` marks an output parameter ``precise``, which makes the
        assignments to it precise. A value parameter that the body writes is
        received under another name and copied into a precise local with its
        own name (``"copy"``): GLSL applies ``precise`` to the operations an
        l-value consumes, and compilers do not apply it to input parameters.
        """
        if not self.current_glsl_precise_function or shader_type is not None:
            return None
        if is_private_pointer_parameter(parameter):
            return None
        if self.is_opaque_resource_type(param_type) or not self.glsl_precise_value_type(
            raw_param_type
        ):
            return None
        if parameter_qualifiers:
            return "qualifier"
        if getattr(parameter, "name", None) in written_value_names:
            return "copy"
        return None

    def glsl_precise_parameter_input_name(self, parameter_name):
        return self.glsl_precise_synthetic_name(f"{parameter_name}Input")

    def glsl_precise_synthetic_name(self, base_name):
        """Allocate a function-local name that no global or function shadows."""
        used_names = set(self.current_glsl_synthetic_local_names)
        used_names.update(
            self.glsl_module_identifier_name(name)
            for name in self.glsl_global_identifier_names
        )
        used_names.update(self.function_definitions)
        name = self.glsl_synthetic_local_identifier(base_name, used_names=used_names)
        self.current_glsl_synthetic_local_names.add(name)
        return name

    GLSL_PRECISE_ARITHMETIC_OPERATORS = {"+", "-", "*", "/"}

    def glsl_expression_has_float_arithmetic(self, expression, unknown_is_float=False):
        """Return whether evaluating ``expression`` adds, subtracts, multiplies
        or divides floating-point values.

        Arguments of user function calls are not part of the scan: they are
        captured where the call is generated, so the call contributes only its
        result. With ``unknown_is_float``, an operation whose type cannot be
        resolved counts as floating point.
        """
        pending = [expression]
        visited = set()
        while pending:
            node = pending.pop()
            if node is None or isinstance(node, (str, int, float, bool)):
                continue
            if isinstance(node, (list, tuple, set)):
                pending.extend(node)
                continue
            if isinstance(node, dict):
                pending.extend(node.values())
                continue
            if id(node) in visited or not hasattr(node, "__dict__"):
                continue
            visited.add(id(node))
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node) in self.function_definitions
            ):
                continue
            if isinstance(node, BinaryOpNode):
                operator = self.map_operator(
                    getattr(node, "op", getattr(node, "operator", None))
                )
                if operator in self.GLSL_PRECISE_ARITHMETIC_OPERATORS:
                    info = self.glsl_value_type_info(
                        self.glsl_source_expression_type(node)
                    )
                    if info is None and unknown_is_float:
                        return True
                    if info is not None and info["family"] == "float":
                        return True
            pending.extend(
                value
                for key, value in vars(node).items()
                if key not in {"parent", "annotations", "source_location"}
            )
        return False

    def glsl_precise_capture(self, generated, capture_type, base_name):
        """Assign a generated value to a precise local where it is evaluated.

        GLSL keeps the evaluation order of an expression only when a precise
        l-value consumes its result, and controlling expressions are never
        precise by themselves. Assigning the value to a precise local declared
        at function entry keeps the order of evaluation and short-circuiting.
        """
        name = self.glsl_precise_synthetic_name(base_name)
        self.current_glsl_precise_captures.append((capture_type, name))
        return f"({name} = {generated})"

    def glsl_precise_condition(self, expression, generated):
        if self.current_glsl_precise_captures is None:
            return generated
        if not self.glsl_expression_has_float_arithmetic(expression):
            return generated
        return self.glsl_precise_capture(generated, "bool", "crossglPreciseCondition")

    def glsl_precise_local_target(self, target):
        """Return whether an assignment target is a local of the function.

        Locals and output parameters of a precise function are precise, and a
        written value parameter is copied into a precise local.
        """
        return any(
            name in self.local_variable_types
            for name in self.glsl_mutation_target_names(target)
            if "." not in name
        )

    GLSL_PRECISE_COMPOUND_OPERATORS = {"+=", "-=", "*=", "/="}

    def glsl_precise_compound_store(
        self, target, generated_target, op, right, value_type
    ):
        """Evaluate a compound float store to non-local storage precisely.

        ``target op= value`` becomes ``target = (capture = (target op value))``
        when evaluating the target expression twice has no side effects.
        """
        if self.current_glsl_precise_captures is None:
            return None
        if op not in self.GLSL_PRECISE_COMPOUND_OPERATORS:
            return None
        if self.glsl_precise_local_target(target):
            return None
        info = self.glsl_value_type_info(value_type)
        if info is None or info["family"] != "float":
            return None
        if not self.glsl_precise_value_type(value_type):
            return None
        mapped = self.map_type(value_type)
        if "[" in mapped or not self.glsl_expression_is_side_effect_free(target):
            return None
        value = f"({generated_target} {op[:-1]} {right})"
        captured = self.glsl_precise_capture(value, mapped, "crossglPreciseValue")
        return f"{generated_target} = {captured}"

    def glsl_expression_is_side_effect_free(self, expression):
        for node in self.walk_ast(expression):
            if isinstance(node, AssignmentNode):
                return False
            if isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--"}:
                return False
            if isinstance(node, FunctionCallNode) and (
                self.function_call_name(node) in self.function_definitions
            ):
                return False
        return True

    def glsl_precise_stored_value(self, target, value_node, generated, value_type):
        """Capture a value stored to storage that is not a precise local."""
        if self.current_glsl_precise_captures is None:
            return generated
        if target is not None and self.glsl_precise_local_target(target):
            return generated
        if not self.glsl_expression_has_float_arithmetic(value_node):
            return generated
        if not self.glsl_precise_value_type(value_type):
            return generated
        mapped = self.map_type(value_type)
        if "[" in mapped:
            return generated
        return self.glsl_precise_capture(generated, mapped, "crossglPreciseValue")

    @staticmethod
    def glsl_expression_is_variable_access(expression):
        """Return whether ``expression`` reads a variable or a part of one."""
        while isinstance(expression, (MemberAccessNode, ArrayAccessNode)):
            if isinstance(expression, MemberAccessNode):
                expression = expression.object
            else:
                expression = expression.array
        return isinstance(expression, (str, IdentifierNode, VariableNode))

    def glsl_precise_selected_value(self, base, generated):
        """Capture a computed value before a swizzle, member or index selects
        part of it.

        A precise l-value makes every operation that contributes to it
        precise, but compilers that track precise through variables (glslang)
        stop at a selection from a computed value such as ``(b - a).yx``, so
        the arithmetic that computes it would lose its evaluation order.
        """
        if self.current_glsl_precise_captures is None:
            return generated
        if self.glsl_expression_is_variable_access(base):
            return generated
        if not self.glsl_expression_has_float_arithmetic(base):
            return generated
        base_type = self.expression_result_type(base)
        if not self.glsl_precise_value_type(base_type):
            return generated
        mapped = self.map_type(base_type)
        if "[" in mapped:
            return generated
        return self.glsl_precise_capture(generated, mapped, "crossglPreciseValue")

    def glsl_precise_function_local(self, node):
        """Return whether a local of the current precise function becomes precise.

        GLSL applies ``precise`` to the operations whose results are consumed by
        a precise l-value in the same function. A source function marked
        precise evaluates all of its arithmetic as written, so every local value
        it computes is declared precise.
        """
        if not getattr(self, "current_glsl_precise_function", False):
            return False
        declared_type = self.local_variable_declared_type(node)
        if not self.glsl_precise_value_type(declared_type):
            return False
        return not self.glsl_precise_local_has_no_float_operation(node, declared_type)

    def glsl_precise_local_has_no_float_operation(self, node, declared_type):
        """Return whether ``precise`` cannot constrain a local's value.

        ``precise`` orders the floating-point operations that produce a value.
        A boolean or integer scalar or vector local has none when neither its
        initializer nor any write to it has floating-point arithmetic. A
        floating-point local has none when it is never written and its
        initializer only reinterprets the bits of such a value. An arithmetic
        operation of unknown type counts as floating point.
        """
        info = self.glsl_value_type_info(declared_type)
        name = getattr(node, "name", None)
        if info is None or not isinstance(name, str):
            return False
        writes = getattr(self, "current_glsl_precise_value_writes", None) or {}
        initial_value = getattr(node, "initial_value", None)
        if info["family"] == "float":
            if name in writes or not self.glsl_is_bit_reinterpretation(initial_value):
                return False
            return not self.glsl_expression_has_float_arithmetic(
                initial_value.args, unknown_is_float=True
            )
        if writes.get(name, False):
            return False
        return not self.glsl_expression_has_float_arithmetic(
            initial_value, unknown_is_float=True
        )

    def glsl_is_bit_reinterpretation(self, expression):
        """Return whether ``expression`` calls a builtin bit reinterpretation."""
        if not isinstance(expression, FunctionCallNode):
            return False
        name = self.function_call_name(expression)
        if not isinstance(name, str) or name in self.function_definitions:
            return False
        native_bitcasts = {
            "floatBitsToInt",
            "floatBitsToUint",
            "intBitsToFloat",
            "uintBitsToFloat",
        }
        return (
            name in native_bitcasts
            or name in self.GLSL_BITCAST_FUNCTIONS
            or self.metal_as_type_target(name) is not None
        )

    def glsl_precise_value_writes(self, body):
        """Map each root name written in ``body`` to whether a write to it may
        evaluate floating-point arithmetic.

        A compound arithmetic assignment adds its own operation, which counts
        as floating point when its operand is floating point or of unknown
        type. Increments and decrements of floating-point values belong to
        floating-point locals, which every write keeps precise.

        The map is built before the body is generated, so the locals that the
        body declares are given their declared types for the scan.
        """
        source_types = self.local_variable_source_types
        body_types = {
            name: declared_type
            for name, declared_type in self.glsl_body_declared_local_types(body).items()
            if name not in source_types
        }
        source_types.update(body_types)
        try:
            return self.glsl_scan_precise_value_writes(body)
        finally:
            for name in body_types:
                source_types.pop(name, None)

    def glsl_body_declared_local_types(self, body):
        """Map each local that ``body`` declares with one explicit type to it.

        A name declared with different types in different scopes is left out,
        so the operations that use it keep an unknown type.
        """
        declared = {}
        conflicting = set()
        for node in self.walk_ast(body):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            declared_type = self.type_name_string(
                getattr(node, "var_type", None) or getattr(node, "vtype", None)
            )
            if not isinstance(name, str) or declared_type in {None, "", "auto"}:
                continue
            if declared.setdefault(name, declared_type) != declared_type:
                conflicting.add(name)
        return {
            name: declared_type
            for name, declared_type in declared.items()
            if name not in conflicting
        }

    def glsl_scan_precise_value_writes(self, body):
        writes = {}
        for node in self.walk_ast(body):
            target = None
            float_write = False
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                value = getattr(node, "value", getattr(node, "right", None))
                float_write = self.glsl_expression_has_float_arithmetic(
                    value, unknown_is_float=True
                )
                operator = self.map_operator(getattr(node, "operator", "=") or "=")
                if not float_write and str(operator)[:-1] in (
                    self.GLSL_PRECISE_ARITHMETIC_OPERATORS
                ):
                    info = self.glsl_value_type_info(
                        self.glsl_source_expression_type(value)
                    )
                    float_write = info is None or info["family"] == "float"
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--"}:
                target = getattr(node, "operand", None)
            if target is None:
                continue
            for name in self.glsl_mutation_target_names(target):
                writes[name] = writes.get(name, False) or float_write
        return writes

    def glsl_precise_argument(self, argument, generated, param_type):
        """Capture a value argument of a user function in a precise local.

        The argument is evaluated in the caller but consumed by the callee's
        parameter, so the caller's precise l-values do not reach it.
        """
        if self.current_glsl_precise_captures is None or param_type is None:
            return generated
        if not self.glsl_expression_has_float_arithmetic(argument):
            return generated
        if not self.glsl_precise_value_type(param_type):
            return generated
        mapped = self.map_type(param_type)
        if "[" in mapped:
            return generated
        return self.glsl_precise_capture(generated, mapped, "crossglPreciseArgument")
