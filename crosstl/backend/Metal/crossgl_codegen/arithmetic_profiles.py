"""Profiled comparison, division, contraction, multiplication, additive, remainder and half widening helpers."""

from ....translator.division_math import binary32_division_support
from ....translator.fused_math import (
    FMA_HELPER_KEYS,
    binary32_fma_support,
)
from ....translator.remainder_math import binary32_remainder_support
from ..implicit_constructor_conversions import converting_constructor_argument
from ..MetalAst import (
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    InitializerListNode,
    UnaryOpNode,
    VectorConstructorNode,
)
from .support import (
    MetalAdditiveProfileError,
    MetalComparisonProfileError,
    MetalContractionProfileError,
    MetalDivisionProfileError,
    MetalHalfRemainderProfileError,
    MetalHalfWideningProfileError,
    MetalMultiplicationProfileError,
    MetalRemainderProfileError,
)


class MetalArithmeticProfileMixin:
    """Members of ``MetalToCrossGLConverter``: profiled comparison, division, contraction, multiplication, additive, remainder and half widening helpers."""

    def generate_profiled_metal_comparison(self, expression, is_main=False):
        operands = (expression.left, expression.right)
        types = [self.expression_metal_type(operand) for operand in operands]
        if any(
            self.metal_pointer_pointee_type_once(vtype) is not None for vtype in types
        ):
            return None
        conversion = self.metal_builtin_lowered_conversion_plan(expression.op, operands)
        if conversion is not None:
            types = conversion[1]
        narrow = self.metal_bfloat_arithmetic_plan(expression.op, *types)
        vectors = [self.metal_small_vector_type_parts(vtype) for vtype in types]
        elements = [
            vector[0] if vector is not None else vtype
            for vector, vtype in zip(vectors, types)
        ]
        infos = [
            self.metal_scalar_arithmetic_type_info(element) for element in elements
        ]

        def unsupported(reason):
            raise MetalComparisonProfileError(
                self.binary32_comparison_profile,
                ", ".join(str(vtype) for vtype in types),
                reason,
                getattr(expression, "source_location", None),
            )

        if any(info is None for info in infos):
            unsupported(
                "comparison operand types must be resolved before selecting a profile"
            )
        if all(info[0] == "integer" for info in infos):
            return None
        if narrow is not None:
            computation = narrow[0]
            vector = self.metal_small_vector_type_parts(computation)
            element, width = vector or (computation, 1)
        elif any(vectors):
            element, width = vectors[0] or vectors[1]
            if all(vectors) and vectors[0] != vectors[1]:
                unsupported(
                    "implicit conversion between distinct vector types is unsupported"
                )
        else:
            element = self.metal_scalar_binary_result_type("+", *types)
            width = 1
        element = self.normalized_metal_type(self.resolve_type_alias(element))
        if element != "float" and element not in self.metal_source_bfloat_types:
            return None
        if width not in {1, 2, 3, 4} or self.current_function is None:
            unsupported("profiled comparisons require a function and one to four lanes")
        mapped = "float" if width == 1 else f"vec{width}"
        arguments = []
        for index, operand in enumerate(operands):
            value = self.generate_expression(operand, is_main)
            if conversion is not None and conversion[0][index] is not None:
                function = conversion[0][index]
                name = self.sanitize_identifier(self.function_output_name(function))
                value = f"{name}({value})"
            if narrow is not None and narrow[1][index] is not None:
                value = f"{self.map_type(narrow[1][index])}({value})"
            arguments.append(f"{mapped}({value})")
        self.required_metal_comparisons.add((expression.op, width))
        name = self.metal_comparison_helper_name(expression.op, width)
        return f"{name}({', '.join(arguments)})"

    def metal_comparison_helper_name(self, operator, width):
        operation = {
            "==": "equal",
            "!=": "not_equal",
            "<": "less",
            "<=": "less_equal",
            ">": "greater",
            ">=": "greater_equal",
        }[operator]
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            ("comparison", operator, width),
            f"__crossgl_metal_compare_{operation}_float{suffix}",
        )

    def generate_metal_comparison_support_code(self, indent=0):
        if not self.required_metal_comparisons:
            return ""
        bits = self.metal_precise_math_unique_helper_name(
            "comparison-bits", "__crossgl_compare_bits"
        )
        # Integer ordering keeps the selected comparison policy independent of
        # target floating-point flush modes, without modifying the operands.
        code = (
            f"@metal_static\nuint {bits}(uint a, uint b) {{\n"
            "    uint ax = a & 0x7fffffffu;\n"
            "    uint bx = b & 0x7fffffffu;\n"
            "    if (ax > 0x7f800000u || bx > 0x7f800000u) { return 3u; }\n"
        )
        if self.binary32_comparison_profile == "flush-subnormals":
            code += (
                "    if (ax < 0x00800000u) { a &= 0x80000000u; }\n"
                "    if (bx < 0x00800000u) { b &= 0x80000000u; }\n"
            )
        code += (
            "    if (a == b || ((a | b) & 0x7fffffffu) == 0u) { return 0u; }\n"
            "    uint ak = (a & 0x80000000u) != 0u ? ~a : a | 0x80000000u;\n"
            "    uint bk = (b & 0x80000000u) != 0u ? ~b : b | 0x80000000u;\n"
            "    return ak < bk ? 1u : 2u;\n"
            "}\n"
        )
        conditions = {
            "==": "c == 0u",
            "!=": "c != 0u",
            "<": "c == 1u",
            "<=": "c <= 1u",
            ">": "c == 2u",
            ">=": "c == 0u || c == 2u",
        }
        for operator in sorted(
            {operator for operator, _ in self.required_metal_comparisons}
        ):
            name = self.metal_comparison_helper_name(operator, 1)
            code += (
                f"@metal_static\nbool {name}(float a, float b) {{\n"
                f"    uint c = {bits}(asuint(a), asuint(b));\n"
                f"    return {conditions[operator]};\n}}\n"
            )
        for operator, width in sorted(self.required_metal_comparisons):
            if width == 1:
                continue
            scalar = self.metal_comparison_helper_name(operator, 1)
            name = self.metal_comparison_helper_name(operator, width)
            lanes = ", ".join(
                f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nbvec{width} {name}(vec{width} a, vec{width} b) {{\n"
                f"    return bvec{width}({lanes});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_arithmetic_profile(self, operator):
        if operator == "/":
            return self.binary32_division_profile, MetalDivisionProfileError, "division"
        if operator == "*":
            return (
                self.binary32_multiplication_profile,
                MetalMultiplicationProfileError,
                "multiplication",
            )
        return (
            self.binary32_additive_profile,
            MetalAdditiveProfileError,
            "addition" if operator == "+" else "subtraction",
        )

    def metal_profiled_arithmetic_type(
        self, result_type, source_location=None, operator="/"
    ):
        profile, error_type, operation = self.metal_arithmetic_profile(operator)
        if (
            operator in {"+", "-"}
            and self.metal_pointer_pointee_type_once(result_type) is not None
        ):
            return None
        info = self.metal_math_builtin_type_info(result_type)
        if info is None or (
            info["category"] == "object"
            and self.normalized_metal_type(result_type) not in self.struct_member_types
        ):
            raise error_type(
                profile,
                result_type,
                "the computation type is unresolved; materialize the source type "
                f"before selecting a {operation} profile",
                source_location,
            )
        element = self.normalized_metal_type(
            self.resolve_type_alias(info["element_type"])
        )
        if element != "float" and info["category"] != "bfloat":
            return None
        if info["width"] not in {1, 2, 3, 4}:
            raise error_type(
                profile,
                result_type,
                f"the resolved {operation} width must be between one and four lanes",
                source_location,
            )
        return info

    def render_profiled_metal_division(
        self, left, right, result_type, source_location=None
    ):
        return self.render_profiled_metal_arithmetic(
            left, right, result_type, source_location
        )

    def render_profiled_metal_arithmetic(
        self, left, right, result_type, source_location=None, operator="/"
    ):
        if (
            operator not in {"/", "+", "-", "*"}
            or self.metal_arithmetic_profile(operator)[0] is None
        ):
            return None
        info = self.metal_profiled_arithmetic_type(
            result_type, source_location, operator
        )
        if info is None:
            return None
        width = info["width"]
        if operator == "/":
            self.required_metal_division_widths.add(width)
            name = self.metal_division_helper_name(width)
        elif operator == "*":
            self.required_metal_multiplication_widths.add(width)
            name = self.metal_multiplication_helper_name(width)
        else:
            self.required_metal_additive_operations.add((operator, width))
            name = self.metal_additive_helper_name(operator, width)
        mapped = "float" if width == 1 else f"vec{width}"
        call = f"{name}({mapped}({left}), {mapped}({right}))"
        if info["category"] == "bfloat":
            call = f"{self.map_type(result_type)}({call})"
        return call

    def generate_profiled_metal_arithmetic(self, expression, is_main=False):
        operator = expression.op
        profile, error_type, operation = self.metal_arithmetic_profile(operator)
        result_type = self.metal_binary_expression_type(expression)
        info = self.metal_profiled_arithmetic_type(
            result_type, getattr(expression, "source_location", None), operator
        )
        if info is None:
            return None
        if self.current_function is None:
            raise error_type(
                profile,
                result_type,
                f"global constant {operation} cannot call a runtime arithmetic helper; "
                "provide the explicitly rounded constant or move the operation into a function",
                getattr(expression, "source_location", None),
            )
        arguments = (expression.left, expression.right)
        conversion = self.metal_builtin_lowered_conversion_plan(operator, arguments)
        types = [self.expression_metal_type(argument) for argument in arguments]
        if conversion is not None:
            types = conversion[1]
        narrow = self.metal_bfloat_arithmetic_plan(operator, *types)
        rendered = []
        for index, argument in enumerate(arguments):
            value = self.generate_expression(argument, is_main)
            if conversion is not None and conversion[0][index] is not None:
                function = conversion[0][index]
                name = self.sanitize_identifier(self.function_output_name(function))
                value = f"{name}({value})"
            if narrow is not None and narrow[1][index] is not None:
                value = f"{self.map_type(narrow[1][index])}({value})"
            if self.binary16_widening_profile is not None:
                value = self.render_metal_half_widening(
                    value, types[index], result_type
                )
            rendered.append(value)
        return self.render_profiled_metal_arithmetic(
            *rendered, result_type, operator=operator
        )

    def metal_division_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"division-float{suffix}", f"__crossgl_metal_divide_float{suffix}"
        )

    def generate_profiled_metal_arithmetic_assignment(self, node, is_main):
        operator = node.operator[:-1]
        profile, error_type, operation = self.metal_arithmetic_profile(operator)
        result_type = self.metal_binary_expression_type(
            BinaryOpNode(node.left, operator, node.right)
        )
        if self.wide_vector_type_info(result_type) is not None:
            return None
        info = self.metal_profiled_arithmetic_type(
            result_type, getattr(node, "source_location", None), operator
        )
        if info is None:
            return None
        if not all(
            self.discarded_expression_is_proven_side_effect_free(value)
            for value in (node.left, node.right)
        ):
            raise error_type(
                profile,
                result_type,
                f"compound operands are not proven side-effect-free; explicitly sequence them before {operation}",
                getattr(node, "source_location", None),
            )
        if self.small_vector_index_info(node.left) is not None:
            return None
        target_type = self.map_type(
            self.metal_source_overload_value_type(self.expression_metal_type(node.left))
        )
        computation_type = self.map_type(result_type)
        right = self.generate_expression(node.right, is_main)
        conversion = self.metal_builtin_lowered_conversion_plan(
            operator, (node.left, node.right)
        )
        if conversion is not None:
            if conversion[0][0] is not None:
                raise error_type(
                    profile,
                    result_type,
                    "compound destination requires a source conversion; explicitly convert and assign the result",
                    getattr(node, "source_location", None),
                )
            if conversion[0][1] is not None:
                function = conversion[0][1]
                name = self.sanitize_identifier(self.function_output_name(function))
                right = f"{name}({right})"
        right = f"{computation_type}({right})"
        resource = self.is_structured_buffer_element_access(node.left)
        index_type = None
        if resource:
            index_type = self.map_type(self.expression_metal_type(node.left.index))
            index_info = self.metal_scalar_arithmetic_type_info(
                self.expression_metal_type(node.left.index)
            )
            if index_info is None or index_info[0] != "integer":
                raise error_type(
                    profile,
                    result_type,
                    "the buffer index type is unresolved",
                )
            arguments = [
                self.generate_without_structured_buffer_index_lowering(
                    node.left.array, is_main
                ),
                self.generate_expression(node.left.index, is_main),
                right,
            ]
        else:
            qualifiers = self.metal_addressable_storage_qualifiers(node.left)
            if qualifiers is None or any(q != "thread" for q in qualifiers):
                raise error_type(
                    profile,
                    result_type,
                    f"compound {operation} requires writable thread storage or a direct buffer element",
                    getattr(node, "source_location", None),
                )
            arguments = [self.generate_expression(node.left, is_main), right]
        key = (target_type, result_type, index_type)
        assignments = self.metal_division_assignments
        prefix = "division"
        helper_prefix = "divide"
        if operator == "*":
            assignments = self.metal_multiplication_assignments
            prefix = "multiplication"
            helper_prefix = "multiply"
        elif operator != "/":
            assignments = self.metal_additive_assignments
            key = (operator, *key)
            prefix = "additive"
            helper_prefix = "add" if operator == "+" else "subtract"
        if key not in assignments:
            name = self.metal_precise_math_unique_helper_name(
                (f"{prefix}-assign", key), f"__crossgl_metal_{helper_prefix}_assign"
            )
            expression = self.render_profiled_metal_arithmetic(
                "value", "right", result_type, operator=operator
            )
            if resource:
                parameters = f"device {target_type}* data, {index_type} index, {computation_type} right"
                load = f"{target_type} value = data[index];"
                store = "data[index] = updated;"
            else:
                parameters = f"inout {target_type} value, {computation_type} right"
                load = ""
                store = "value = updated;"
            assignments[key] = (
                name,
                f"@metal_static\n{target_type} {name}({parameters}) {{\n"
                f"    {load}\n    {target_type} updated = {target_type}({expression});\n"
                f"    {store}\n    return updated;\n}}\n",
            )
        return f"{assignments[key][0]}({', '.join(arguments)})"

    def generate_metal_division_support_code(self, indent=0):
        if not (
            self.required_metal_division_widths
            or self.required_metal_half_remainder_widths
            or self.required_metal_precise_atan2_widths
        ):
            return ""
        bits = self.metal_precise_math_unique_helper_name(
            "division-bits", "__crossgl_divide_bits"
        )
        scalar = self.metal_division_helper_name(1)
        flush = "true" if self.binary32_division_profile == "rne-flush" else "false"
        code = binary32_division_support(bits)
        if self.required_metal_division_widths:
            code += (
                f"@metal_static\nfloat {scalar}(float a, float b) {{\n"
                f"    return asfloat({bits}(asuint(a), asuint(b), {flush}));\n"
                "}\n"
            )
        for width in sorted(self.required_metal_division_widths - {1}):
            arguments = ", ".join(
                f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nvec{width} {self.metal_division_helper_name(width)}"
                f"(vec{width} a, vec{width} b) {{\n"
                f"    return vec{width}({arguments});\n"
                "}\n"
            )
        code += "".join(
            body for _name, body in self.metal_division_assignments.values()
        )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def generate_metal_fma_call(self, expression, is_main=False):
        if (
            self.binary32_fma_profile is None
            or self.metal_math_builtin_namespace_mode(expression.name) == "fast"
            or self.resolve_metal_math_builtin_name(expression.name, expression.args)
            != "fma"
        ):
            return None
        result_type = self.metal_math_builtin_result_type(expression)
        info = self.metal_math_builtin_type_info(result_type)
        if (
            info is None
            or self.normalized_metal_type(info["element_type"]) != "float"
            or info["width"] not in {1, 2, 3, 4}
        ):
            return None
        width = info["width"]
        self.required_metal_fma_widths.add(width)
        mapped_type = "float" if width == 1 else f"vec{width}"
        arguments = ", ".join(
            f"{mapped_type}({self.generate_expression(argument, is_main)})"
            for argument in expression.args
        )
        return f"{self.metal_fma_helper_name(width)}({arguments})"

    def metal_fma_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"fma-float{suffix}", f"__crossgl_metal_fma_float{suffix}"
        )

    def generate_metal_fma_support_code(self, indent=0):
        if not (
            self.required_metal_fma_widths
            or self.required_metal_fmuladd_widths
            or self.required_metal_half_remainder_widths
            or (
                self.required_metal_additive_operations
                and self.binary32_additive_profile != "native-precise"
            )
            or self.required_metal_multiplication_widths
        ):
            return ""
        names = {
            key: self.metal_precise_math_unique_helper_name(
                f"fma-{key}", f"__crossgl_fma_{key}"
            )
            for key in FMA_HELPER_KEYS
        }
        scalar = self.metal_fma_helper_name(1)
        flush = "true" if self.binary32_fma_profile == "rne-flush" else "false"
        code = binary32_fma_support(names)
        if self.required_metal_fma_widths:
            code += (
                f"@metal_static\nfloat {scalar}(float a, float b, float c) {{\n"
                f"    return asfloat({names['bits']}(asuint(a), asuint(b), asuint(c), {flush}));\n"
                "}\n"
            )
        for width in sorted(self.required_metal_fma_widths - {1}):
            vector = self.metal_fma_helper_name(width)
            arguments = ", ".join(
                f"{scalar}(a.{lane}, b.{lane}, c.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nvec{width} {vector}(vec{width} a, vec{width} b, vec{width} c) {{\n"
                f"    return vec{width}({arguments});\n"
                "}\n"
            )
        if self.required_metal_fmuladd_widths:
            contracted = self.metal_fmuladd_helper_name(1)
            contraction_flush = (
                "true" if self.floating_contraction_profile == "rne-flush" else "false"
            )
            code += (
                f"@metal_static\nfloat {contracted}(float a, float b, float c) {{\n"
                f"    return asfloat({names['bits']}(asuint(a), asuint(b), asuint(c), "
                f"{contraction_flush}));\n"
                "}\n"
            )
            for width in sorted(self.required_metal_fmuladd_widths - {1}):
                vector = self.metal_fmuladd_helper_name(width)
                arguments = ", ".join(
                    f"{contracted}(a.{lane}, b.{lane}, c.{lane})"
                    for lane in "xyzw"[:width]
                )
                code += (
                    f"@metal_static\nvec{width} {vector}(vec{width} a, vec{width} b, vec{width} c) {{\n"
                    f"    return vec{width}({arguments});\n"
                    "}\n"
                )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_fmuladd_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"fmuladd-float{suffix}", f"__crossgl_metal_fmuladd_float{suffix}"
        )

    @staticmethod
    def metal_contraction_constant_operand(operand):
        """Return whether Metal folds the operand before forming a product."""
        while isinstance(operand, UnaryOpNode) and operand.op in {"-", "+"}:
            operand = operand.operand
        return isinstance(operand, str)

    def metal_contraction_family(self, metal_type):
        """Return the floating family and width of a contractible value type."""
        if metal_type is None:
            return None
        if self.metal_pointer_pointee_type_once(metal_type) is not None:
            return None
        info = self.metal_math_builtin_type_info(metal_type)
        if info is None:
            return None
        if info["category"] == "bfloat":
            family = "bfloat"
        elif info["category"] == "floating":
            element = self.normalized_metal_type(
                self.resolve_type_alias(info["element_type"])
            )
            if element not in {"float", "half"}:
                return None
            family = element
        else:
            return None
        return family, info["width"]

    def metal_contraction_plan(self, expression):
        """Mirror clang's expression-level fmuladd selection for one addition.

        Metal's compiler forms a fused multiply-add when an addition or
        subtraction directly consumes a floating product of the same type in
        the same expression; the left product is preferred. Negated products,
        widened or narrowed products, and products of two constants are not
        fused.
        """
        if (
            self.floating_contraction_profile is None
            or not isinstance(expression, BinaryOpNode)
            or expression.op not in {"+", "-"}
            or self.current_function is None
        ):
            return None
        result_type = self.metal_binary_expression_type(expression)
        family = self.metal_contraction_family(result_type)
        if family is None:
            return None
        for index, operand in enumerate((expression.left, expression.right)):
            operand = self.metal_contraction_product(operand, family)
            if operand is None:
                continue
            if all(
                self.metal_contraction_constant_operand(value)
                for value in (operand.left, operand.right)
            ):
                continue
            location = getattr(operand, "source_location", None)
            if (
                self.resolve_metal_free_operator(
                    "*", (operand.left, operand.right), location
                )
                is not None
            ):
                continue
            product_type = self.metal_binary_expression_type(operand)
            if self.metal_contraction_family(product_type) != family:
                continue
            if family[1] not in {1, 2, 3, 4}:
                raise MetalContractionProfileError(
                    self.floating_contraction_profile,
                    result_type,
                    "the contracted width must be between one and four lanes",
                    getattr(expression, "source_location", None),
                )
            addend = expression.right if index == 0 else expression.left
            return {
                "result_type": result_type,
                "family": family,
                "product": operand,
                "product_index": index,
                "addend": addend,
                "negate_product": expression.op == "-" and index == 1,
                "negate_addend": expression.op == "-" and index == 0,
            }
        return None

    def metal_contraction_product(self, operand, family):
        """Return the product an operand evaluates without a conversion.

        Casts to the product's own type emit no instruction, so the source
        compiler still sees the product as the addition's operand.
        """
        while True:
            if isinstance(operand, BinaryOpNode) and operand.op == "*":
                return operand
            if isinstance(operand, CastNode):
                target, inner = operand.target_type, operand.expression
            elif isinstance(operand, VectorConstructorNode) and len(operand.args) == 1:
                target, inner = operand.vector_type, operand.args[0]
            else:
                return None
            if self.metal_contraction_family(str(target)) != family:
                return None
            operand = inner

    def render_metal_contraction_operand(self, operand, computation_type, is_main):
        """Convert one operand to the computation type, then widen to binary32."""
        value = self.generate_expression(operand, is_main)
        family, width = self.metal_contraction_family(computation_type)
        wide = "float" if width == 1 else f"vec{width}"
        source = self.metal_contraction_family(self.expression_metal_type(operand))
        if source != (family, width) and family != "float":
            # Apply the source's usual arithmetic conversion before widening.
            value = f"{self.map_type(computation_type)}({value})"
        return f"{wide}({value})"

    def generate_metal_contracted_arithmetic(self, expression, is_main=False):
        plan = self.metal_contraction_plan(expression)
        if plan is None:
            return None
        result_type = plan["result_type"]
        family, width = plan["family"]
        product = plan["product"]
        # binary16 and bfloat16 values widen exactly to binary32.
        operands = [
            self.render_metal_contraction_operand(value, result_type, is_main)
            for value in (product.left, product.right, plan["addend"])
        ]
        if plan["negate_product"]:
            operands[0] = f"-{operands[0]}"
        if plan["negate_addend"]:
            operands[2] = f"-{operands[2]}"
        if family == "half":
            # binary16 operands give an exact binary32 product, and the single
            # binary32 sum rounds to binary16 without double-rounding error.
            return (
                f"{self.map_type(result_type)}"
                f"({operands[0]} * {operands[1]} + {operands[2]})"
            )
        self.required_metal_fmuladd_widths.add(width)
        call = f"{self.metal_fmuladd_helper_name(width)}({', '.join(operands)})"
        if family == "bfloat":
            # Metal fuses bfloat arithmetic in binary32, then rounds once.
            call = f"{self.map_type(result_type)}({call})"
        return call

    def generate_metal_contracted_assignment(self, node, is_main):
        if (
            self.floating_contraction_profile is None
            or node.operator not in {"+=", "-="}
            or self.current_function is None
        ):
            return None
        operator = node.operator[:-1]
        expression = BinaryOpNode(node.left, operator, node.right)
        if hasattr(node, "source_location"):
            expression.source_location = node.source_location
        plan = self.metal_contraction_plan(expression)
        if plan is None or plan["product_index"] != 1:
            return None
        location = getattr(node, "source_location", None)
        if self.metal_contraction_family(
            self.expression_metal_type(node.left)
        ) != self.metal_contraction_family(plan["result_type"]):
            raise MetalContractionProfileError(
                self.floating_contraction_profile,
                plan["result_type"],
                "the compound destination requires a conversion; explicitly "
                "convert and assign the contracted result",
                location,
            )
        if not self.discarded_expression_is_proven_side_effect_free(node.left):
            raise MetalContractionProfileError(
                self.floating_contraction_profile,
                plan["result_type"],
                "the compound destination is not proven side-effect-free; "
                "explicitly sequence it before the contracted update",
                location,
            )
        assignment = AssignmentNode(node.left, expression, "=")
        if hasattr(node, "source_location"):
            assignment.source_location = node.source_location
        return self.generate_assignment(assignment, is_main)

    def metal_multiplication_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            ("multiplication", width), f"__crossgl_metal_multiply_float{suffix}"
        )

    def generate_metal_multiplication_support_code(self, indent=0):
        if not self.required_metal_multiplication_widths:
            return ""
        bits = self.metal_precise_math_unique_helper_name(
            "fma-bits", "__crossgl_fma_bits"
        )
        scalar = self.metal_multiplication_helper_name(1)
        flush = (
            "true" if self.binary32_multiplication_profile == "rne-flush" else "false"
        )
        # A same-sign zero addend preserves the product's signed zero without
        # introducing another rounding or relying on target floating-point modes.
        code = (
            f"@metal_static\nfloat {scalar}(float a, float b) {{\n"
            "    uint left = asuint(a);\n    uint right = asuint(b);\n"
            "    uint zero = (left ^ right) & 0x80000000u;\n"
            f"    return asfloat({bits}(left, right, zero, {flush}));\n}}\n"
        )
        for width in sorted(self.required_metal_multiplication_widths - {1}):
            name = self.metal_multiplication_helper_name(width)
            lanes = ", ".join(
                f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nvec{width} {name}(vec{width} a, vec{width} b) {{\n"
                f"    return vec{width}({lanes});\n}}\n"
            )
        code += "".join(
            body for _name, body in self.metal_multiplication_assignments.values()
        )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_additive_helper_name(self, operator, width):
        operation = "add" if operator == "+" else "subtract"
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            ("additive", operator, width), f"__crossgl_metal_{operation}_float{suffix}"
        )

    def generate_metal_additive_support_code(self, indent=0):
        if not self.required_metal_additive_operations:
            return ""
        bits = self.metal_precise_math_unique_helper_name(
            "fma-bits", "__crossgl_fma_bits"
        )
        flush = "true" if self.binary32_additive_profile == "rne-flush" else "false"
        operations = self.required_metal_additive_operations | {
            (operator, 1)
            for operator, _width in self.required_metal_additive_operations
        }
        code = ""
        for operator, width in sorted(operations, key=lambda item: (item[1], item[0])):
            name = self.metal_additive_helper_name(operator, width)
            if width == 1 and self.binary32_additive_profile == "native-precise":
                code += (
                    f"@metal_static\n@precise\nfloat {name}(float a, float b) {{\n"
                    f"    precise float result = a {operator} b;\n"
                    "    return result;\n}\n"
                )
            elif width == 1:
                right = "asuint(b)" if operator == "+" else "asuint(b) ^ 0x80000000u"
                # Multiplication by exactly one introduces no extra rounding.
                code += (
                    f"@metal_static\nfloat {name}(float a, float b) {{\n"
                    f"    return asfloat({bits}(asuint(a), 0x3f800000u, {right}, {flush}));\n}}\n"
                )
            else:
                scalar = self.metal_additive_helper_name(operator, 1)
                lanes = ", ".join(
                    f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
                )
                code += (
                    f"@metal_static\nvec{width} {name}(vec{width} a, vec{width} b) {{\n"
                    f"    return vec{width}({lanes});\n}}\n"
                )
        code += "".join(
            body for _name, body in self.metal_additive_assignments.values()
        )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def generate_metal_remainder_call(self, expression, is_main=False):
        if (
            self.binary32_remainder_profile is None
            or self.metal_math_builtin_namespace_mode(expression.name) == "fast"
            or str(expression.name).rsplit("::", 1)[-1] != "fmod"
        ):
            return None
        selected = self.selected_metal_callable(expression)
        bfloat_wrapper = self.is_materialized_metal_stdlib_wrapper(selected) and (
            self.normalized_metal_type(
                self.resolve_type_alias(
                    self.selected_metal_callable_return_type(selected)
                )
            )
            in self.metal_source_bfloat_types
        )
        if bfloat_wrapper:
            result_type = "float"
        elif (
            self.resolve_metal_math_builtin_name(expression.name, expression.args)
            == "fmod"
        ):
            result_type = self.metal_math_builtin_result_type(expression)
        else:
            return None
        info = self.metal_math_builtin_type_info(result_type)
        if (
            info is None
            or self.normalized_metal_type(self.resolve_type_alias(info["element_type"]))
            != "float"
        ):
            return None
        width = info["width"]
        if width not in {1, 2, 3, 4} or self.current_function is None:
            raise MetalRemainderProfileError(
                self.binary32_remainder_profile,
                result_type,
                "profiled binary32 remainder requires a function and one to four lanes",
                getattr(expression, "source_location", None),
            )
        self.required_metal_remainder_widths.add(width)
        mapped = "float" if width == 1 else f"vec{width}"
        arguments = ", ".join(
            f"{mapped}({self.generate_expression(argument, is_main)})"
            for argument in expression.args
        )
        return f"{self.metal_remainder_helper_name(width)}({arguments})"

    def metal_remainder_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"remainder-float{suffix}", f"__crossgl_metal_remainder_float{suffix}"
        )

    def generate_metal_remainder_support_code(self, indent=0):
        if not self.required_metal_remainder_widths:
            return ""
        bits = self.metal_precise_math_unique_helper_name(
            "remainder-bits", "__crossgl_remainder_bits"
        )
        scalar = self.metal_remainder_helper_name(1)
        flush = (
            "true"
            if self.binary32_remainder_profile == "flush-arithmetic-subnormals"
            else "false"
        )
        code = binary32_remainder_support(bits)
        code += (
            f"@metal_static\nfloat {scalar}(float a, float b) {{\n"
            f"    return asfloat({bits}(asuint(a), asuint(b), {flush}));\n"
            "}\n"
        )
        for width in sorted(self.required_metal_remainder_widths - {1}):
            arguments = ", ".join(
                f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nvec{width} {self.metal_remainder_helper_name(width)}"
                f"(vec{width} a, vec{width} b) {{\n"
                f"    return vec{width}({arguments});\n"
                "}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def generate_metal_half_remainder_call(self, expression, is_main=False):
        if (
            self.binary16_remainder_profile is None
            or self.resolve_metal_math_builtin_name(expression.name, expression.args)
            != "fmod"
        ):
            return None
        result_type = self.metal_math_builtin_result_type(expression)
        info = self.metal_math_builtin_type_info(result_type)
        if (
            info is None
            or self.normalized_metal_type(self.resolve_type_alias(info["element_type"]))
            != "half"
        ):
            return None
        width = info["width"]
        if width not in {1, 2, 3, 4} or self.current_function is None:
            raise MetalHalfRemainderProfileError(
                self.binary16_remainder_profile,
                result_type,
                "profiled half remainder requires a function and one to four lanes",
                getattr(expression, "source_location", None),
            )
        self.required_metal_half_remainder_widths.add(width)
        mapped = "float16" if width == 1 else f"half{width}"
        arguments = ", ".join(
            f"{mapped}({self.generate_expression(argument, is_main)})"
            for argument in expression.args
        )
        return f"{self.metal_half_remainder_helper_name(width)}({arguments})"

    def metal_half_remainder_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"remainder-half{suffix}", f"__crossgl_metal_remainder_half{suffix}"
        )

    def generate_metal_half_remainder_support_code(self, indent=0):
        if not self.required_metal_half_remainder_widths:
            return ""
        divide = self.metal_precise_math_unique_helper_name(
            "division-bits", "__crossgl_divide_bits"
        )
        fma = self.metal_precise_math_unique_helper_name(
            "fma-bits", "__crossgl_fma_bits"
        )
        scalar = self.metal_half_remainder_helper_name(1)
        # The selected profile rounds division, multiplication and subtraction
        # separately in binary32, then narrows once. Integer helpers prevent
        # target contraction or reciprocal approximations from changing it.
        code = (
            f"@metal_static\nfloat16 {scalar}(float16 a, float16 b) {{\n"
            "    uint x = asuint(float(a));\n"
            "    uint y = asuint(float(b));\n"
            f"    uint quotient = {divide}(x, y, false);\n"
            "    uint exponent = (quotient >> 23u) & 255u;\n"
            "    if (exponent < 127u) { quotient &= 0x80000000u; }\n"
            "    else if (exponent < 150u) {\n"
            "        quotient &= ~((1u << (150u - exponent)) - 1u);\n"
            "    }\n"
            f"    uint product = {fma}(quotient, y, (quotient ^ y) & 0x80000000u, false);\n"
            f"    uint difference = {fma}(product ^ 0x80000000u, 0x3f800000u, x, false);\n"
            "    return float16(asfloat(difference));\n"
            "}\n"
        )
        for width in sorted(self.required_metal_half_remainder_widths - {1}):
            arguments = ", ".join(
                f"{scalar}(a.{lane}, b.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nhalf{width} {self.metal_half_remainder_helper_name(width)}"
                f"(half{width} a, half{width} b) {{\n"
                f"    return half{width}({arguments});\n"
                "}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def generate_metal_value_argument(self, argument, parameter, is_main):
        converted = converting_constructor_argument(self, argument, parameter, is_main)
        if converted is not None:
            return converted
        if self.binary16_widening_profile is not None and isinstance(
            argument, InitializerListNode
        ):
            return self.generate_initializer_value(
                argument, is_main, self.metal_source_overload_parameter_type(parameter)
            )
        value = self.generate_expression(argument, is_main)
        if self.binary16_widening_profile is not None:
            value = self.render_metal_half_widening(
                value,
                self.expression_metal_type(argument),
                self.metal_source_overload_parameter_type(parameter),
                getattr(argument, "source_location", None),
            )
        return value

    def render_metal_half_widening(
        self, value, source_type, target_type, source_location=None
    ):
        if self.binary16_widening_profile is None or target_type is None:
            return value
        source = self.metal_math_builtin_type_info(
            self.metal_source_overload_value_type(source_type)
        )
        target = self.metal_math_builtin_type_info(
            self.metal_source_overload_value_type(target_type)
        )
        if source is None or target is None:
            return value
        if (
            self.normalized_metal_type(self.resolve_type_alias(source["element_type"]))
            != "half"
            or self.normalized_metal_type(
                self.resolve_type_alias(target["element_type"])
            )
            != "float"
        ):
            return value
        width = source["width"]
        if width not in {1, 2, 3, 4} or self.current_function is None:
            raise MetalHalfWideningProfileError(source_type, source_location)
        self.required_metal_half_widening_widths.add(width)
        return f"{self.metal_half_widening_helper_name(width)}({value})"

    def generate_metal_half_widening_constructor(self, target_type, arguments, is_main):
        if self.binary16_widening_profile is None:
            return None
        target = self.metal_math_builtin_type_info(target_type)
        if (
            target is None
            or self.normalized_metal_type(
                self.resolve_type_alias(target["element_type"])
            )
            != "float"
        ):
            return None
        rendered = []
        for argument in arguments:
            value = self.generate_expression(argument, is_main)
            rendered.append(
                self.render_metal_half_widening(
                    value,
                    self.expression_metal_type(argument),
                    target_type,
                    getattr(argument, "source_location", None),
                )
            )
        return f"{self.map_type(target_type)}({', '.join(rendered)})"

    def metal_half_widening_helper_name(self, width):
        suffix = "" if width == 1 else str(width)
        return self.metal_precise_math_unique_helper_name(
            f"widen-half{suffix}", f"__crossgl_metal_widen_half{suffix}"
        )

    def generate_metal_half_widening_support_code(self, indent=0):
        if not self.required_metal_half_widening_widths:
            return ""
        scalar = self.metal_half_widening_helper_name(1)
        # Classify the stored half before widening. Copies and bitcasts do not
        # use this helper, so their sign and payload bits remain untouched.
        code = (
            f"@metal_static\nfloat {scalar}(float16 value) {{\n"
            "    uint bits = uint(as_type<uint16>(value));\n"
            "    if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0u) {\n"
            "        return asfloat(0x7fc00000u);\n"
            "    }\n"
            "    return float(value);\n"
            "}\n"
        )
        for width in sorted(self.required_metal_half_widening_widths - {1}):
            lanes = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"@metal_static\nvec{width} {self.metal_half_widening_helper_name(width)}"
                f"(half{width} value) {{\n"
                f"    return vec{width}({lanes});\n}}\n"
            )
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())
