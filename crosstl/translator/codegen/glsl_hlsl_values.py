"""GLSL emission of HLSL value semantics.

HLSL overloads its intrinsics for every numeric component type, applies the
usual arithmetic conversions to their operands, zero-initializes static
storage and defines ``fmod`` and integer ``dot``. GLSL does not, so these
members make the conversions and helpers explicit: unified builtin operand
types, float remainder, integer dot, componentwise equality, signed image
coordinates and static global zero values.
"""

import re

from ..ast import FunctionCallNode
from .array_utils import split_array_type_suffix


class GLSLHlslValueMixin:
    """HLSL value-conversion and intrinsic members of ``GLSLCodeGen``."""

    def glsl_value_contains_call(self, value, function_name):
        if value is None:
            return False
        for node in self.walk_ast(value):
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node) == function_name
            ):
                return True
        return False

    def glsl_static_global_zero_value(self, vtype):
        """Zero value of a static global: HLSL zero-initializes static storage."""
        mapped_type = str(self.map_type(vtype) or "")
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        if not array_suffix:
            return self.zero_value_expression(base_type)
        extents = re.findall(r"\[\s*(\d+)\s*\]", array_suffix)
        if len(extents) != 1 or array_suffix.strip() != f"[{extents[0]}]":
            return None
        count = int(extents[0])
        element = self.zero_value_expression(base_type)
        return f"{base_type}[{count}]({', '.join([element] * count)})"

    def generate_glsl_float_remainder(self, left_expr, operator, right_expr):
        """Floating-point ``%``, which GLSL lacks, with truncated quotients.

        CrossGL, HLSL and Metal define the floating-point remainder with the sign
        of the dividend (``x - y * trunc(x / y)``); GLSL ``mod`` floors the
        quotient instead, so a helper with the source semantics is emitted.
        """
        if operator != "%":
            return None
        left = self.glsl_value_type_info(self.glsl_source_expression_type(left_expr))
        right = self.glsl_value_type_info(self.glsl_source_expression_type(right_expr))
        if left is None or right is None:
            return None
        families = {left["family"], right["family"]}
        if not families & {"float", "double"} or families - {
            "float",
            "double",
            "int",
            "uint",
        }:
            return None
        family = "double" if "double" in families else "float"
        width = max(left["width"], right["width"])
        if {left["width"], right["width"]} - {1, width}:
            return None
        value_type = self.glsl_vector_type_name(family, width)
        operands = []
        for expression, info in ((left_expr, left), (right_expr, right)):
            if info["family"] == family and info["width"] == width:
                operands.append(self.generate_expression(expression))
            else:
                operands.append(self.glsl_converted_operand(expression, value_type))
        self.glsl_float_remainder_types.add(value_type)
        if self.current_glsl_precise_function:
            self.glsl_precise_float_remainder_types.add(value_type)
        return f"crossgl_remainder({operands[0]}, {operands[1]})"

    def generate_glsl_float_remainder_helpers(self):
        helpers = ""
        precise_types = getattr(self, "glsl_precise_float_remainder_types", set())
        for value_type in sorted(getattr(self, "glsl_float_remainder_types", ())):
            if value_type in precise_types:
                # A precise caller evaluates the remainder as written, so the
                # product and difference must not be contracted either.
                helpers += (
                    f"{value_type} crossgl_remainder({value_type} x, {value_type} y) {{\n"
                    f"    precise {value_type} crossglPreciseReturn = "
                    "x - y * trunc(x / y);\n"
                    "    return crossglPreciseReturn;\n"
                    "}\n\n"
                )
                continue
            helpers += (
                f"{value_type} crossgl_remainder({value_type} x, {value_type} y) {{\n"
                "    return x - y * trunc(x / y);\n"
                "}\n\n"
            )
        return helpers

    def glsl_integer_dot_plan(self, arguments):
        """Return ``(family, width, infos)`` for a ``dot`` of 32-bit integers.

        CrossGL, HLSL and Metal define ``dot`` for integer vectors and return the
        component type; operands of mixed signedness use ``uint`` like the other
        usual arithmetic conversions. GLSL only defines ``dot`` for floating point.
        """
        if len(arguments) != 2:
            return None
        infos = []
        for argument in arguments:
            info = self.glsl_value_type_info(self.glsl_source_expression_type(argument))
            if info is None or info["narrow"] is not None or info["bits"] != 32:
                return None
            infos.append(info)
        families = {info["family"] for info in infos}
        if families - {"int", "uint"}:
            return None
        widths = {info["width"] for info in infos}
        if len(widths) != 1:
            return None
        family = "uint" if "uint" in families else "int"
        return family, widths.pop(), infos

    def generate_glsl_integer_dot(self, arguments):
        """Sum the component products of an integer ``dot`` through a helper."""
        plan = self.glsl_integer_dot_plan(arguments)
        if plan is None:
            return None
        family, width, infos = plan
        value_type = self.glsl_vector_type_name(family, width)
        rendered = [
            (
                self.generate_expression(argument)
                if info["family"] == family
                else self.glsl_converted_operand(argument, value_type)
            )
            for argument, info in zip(arguments, infos)
        ]
        if width == 1:
            return f"({rendered[0]} * {rendered[1]})"
        self.glsl_integer_dot_types.add(value_type)
        return f"crossgl_integer_dot({rendered[0]}, {rendered[1]})"

    def generate_glsl_integer_dot_helpers(self):
        helpers = ""
        for value_type in sorted(getattr(self, "glsl_integer_dot_types", ())):
            info = self.glsl_value_type_info(value_type)
            if info is None:
                continue
            components = "xyzw"[: info["width"]]
            products = " + ".join(f"a.{name} * b.{name}" for name in components)
            helpers += (
                f"{info['family']} crossgl_integer_dot({value_type} a, "
                f"{value_type} b) {{\n"
                f"    return {products};\n"
                "}\n\n"
            )
        return helpers

    def generate_glsl_componentwise_equality(self, left_expr, operator, right_expr):
        """Componentwise ``==``/``!=`` where the context expects a bool vector.

        GLSL ``==`` on vectors compares whole aggregates and yields one ``bool``;
        CrossGL comparisons of vectors are componentwise. When the value is used
        as a bool vector the comparison becomes ``equal``/``notEqual``.
        """
        function_name = {"==": "equal", "!=": "notEqual"}.get(operator)
        if function_name is None:
            return None
        expected = self.glsl_value_type_info(self.current_expression_expected_type)
        if expected is None or expected["family"] != "bool" or expected["width"] < 2:
            return None
        left_type = self.glsl_source_expression_type(left_expr)
        right_type = self.glsl_source_expression_type(right_expr)
        vector_type = self.glsl_relational_vector_type(left_type, right_type)
        if vector_type is None:
            return None
        plan = self.glsl_arithmetic_conversion_plan(
            left_type,
            right_type,
            operator,
            fail_closed=False,
        )
        if plan is None:
            left = self.glsl_vector_relational_operand(
                left_expr, left_type, vector_type
            )
            right = self.glsl_vector_relational_operand(
                right_expr, right_type, vector_type
            )
        else:
            left = self.generate_expression_with_expected(
                left_expr, plan.left_target_type
            )
            right = self.generate_expression_with_expected(
                right_expr, plan.right_target_type
            )
        return f"{function_name}({left}, {right})"

    def glsl_atomic_operand_family(self, argument):
        info = self.glsl_value_type_info(self.glsl_source_expression_type(argument))
        if (
            info is None
            or info["width"] != 1
            or info["narrow"] is not None
            or info["bits"] not in (None, 32)
        ):
            # Only 32-bit scalars share the int/uint atomic operand families;
            # 64-bit integers have no core GLSL atomics.
            return None
        return info["family"]

    GLSL_UNIFIED_COMPONENT_RANKS = {
        "bool": 0,
        "int": 1,
        "uint": 2,
        "float": 3,
        "double": 4,
    }

    # Builtins whose operands share one component type in GLSL, mapped to the
    # operand positions that GLSL also accepts as scalars next to vectors.
    GLSL_UNIFIED_BUILTIN_SCALAR_POSITIONS = {
        "max": (frozenset(), frozenset({1})),
        "min": (frozenset(), frozenset({1})),
        "clamp": (frozenset(), frozenset({1, 2})),
        "mix": (frozenset(), frozenset({2})),
        "step": (frozenset(), frozenset({0})),
        "smoothstep": (frozenset(), frozenset({0, 1})),
        "pow": (frozenset(),),
        "atan": (frozenset(),),
        "fma": (frozenset(),),
        "dot": (frozenset(),),
        "distance": (frozenset(),),
    }

    GLSL_UNIFIED_FLOAT_BUILTINS = frozenset(
        {"pow", "mix", "step", "smoothstep", "atan", "fma", "dot", "distance"}
    )

    GLSL_UNIFIED_SCALAR_RESULT_BUILTINS = frozenset({"dot", "distance"})

    GLSL_FLOAT_BIT_CASTS = {
        ("intBitsToFloat", "uint"): "uintBitsToFloat",
        ("uintBitsToFloat", "int"): "intBitsToFloat",
    }

    def glsl_vector_type_name(self, family, width):
        if width == 1:
            return family
        prefix = {"float": "", "double": "d", "int": "i", "uint": "u", "bool": "b"}
        return f"{prefix[family]}vec{width}"

    def glsl_unified_builtin_plan(self, func_name, args):
        """Plan explicit operand conversions for a builtin GLSL does not resolve.

        HLSL and CrossGL convert builtin operands with the usual arithmetic
        conversions (``int`` to ``uint``, integers to floating point) and promote
        scalars to the vector width of the other operands, so ``max(1, v)`` and
        ``pow(color, 0.5)`` are valid. GLSL provides no such overloads. Returns
        ``(family, width, operands)`` where ``operands`` are
        ``(info, target type or None)`` pairs, or ``None`` when GLSL accepts the
        call as written.
        """
        if not isinstance(func_name, str):
            return None
        shapes = self.GLSL_UNIFIED_BUILTIN_SCALAR_POSITIONS.get(func_name)
        if shapes is None or func_name in self.function_return_types:
            return None
        arguments = list(args or [])
        if not arguments:
            return None
        described = []
        for argument in arguments:
            info = self.glsl_value_type_info(self.glsl_source_expression_type(argument))
            if info is None or info["narrow"] is not None:
                return None
            if info["family"] not in self.GLSL_UNIFIED_COMPONENT_RANKS:
                return None
            described.append(info)
        if (
            func_name == "mix"
            and len(described) == 3
            and described[2]["family"] == "bool"
        ):
            # A boolean selector selects (GLSL 4.60 section 8.3); it must never become a weight.
            return None
        families = {info["family"] for info in described}
        common = max(families, key=self.GLSL_UNIFIED_COMPONENT_RANKS.get)
        if func_name in self.GLSL_UNIFIED_FLOAT_BUILTINS and common in {
            "bool",
            "int",
            "uint",
        }:
            if func_name in {"dot", "distance"}:
                return None
            common = "float"
        if common == "bool":
            return None
        width = max(info["width"] for info in described)
        if any(info["width"] not in {1, width} for info in described):
            return None
        scalar_positions = frozenset(
            index for index, info in enumerate(described) if info["width"] == 1
        )
        shape_supported = width == 1 or scalar_positions in shapes
        operands = []
        changed = False
        for info in described:
            target_width = info["width"]
            if not shape_supported and info["width"] == 1:
                target_width = width
            if info["family"] == common and target_width == info["width"]:
                operands.append((info, None))
                continue
            operands.append((info, self.glsl_vector_type_name(common, target_width)))
            changed = True
        if not changed:
            return None
        return common, width, operands

    def glsl_unified_builtin_result_type(self, func_name, args):
        plan = self.glsl_unified_builtin_plan(func_name, args)
        if plan is None:
            return None
        common, width, _operands = plan
        if func_name in self.GLSL_UNIFIED_SCALAR_RESULT_BUILTINS:
            return common
        return self.glsl_vector_type_name(common, width)

    def glsl_converted_operand(self, argument, target_type):
        value = getattr(argument, "value", None)
        if (
            target_type in {"float", "double", "uint"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            and "Literal" in type(argument).__name__
        ):
            return f"{value}.0" if target_type != "uint" else f"{value}u"
        return f"{target_type}({self.generate_expression(argument)})"

    def generate_glsl_converted_builtin_call(self, func_name, args):
        """Spell the operand conversions of builtins GLSL does not convert itself."""
        arguments = list(args or [])
        if not isinstance(func_name, str) or func_name in self.function_return_types:
            return None
        if func_name in {"intBitsToFloat", "uintBitsToFloat"} and len(arguments) == 1:
            rendered_argument = self.generate_expression(arguments[0])
            info = self.glsl_value_type_info(
                self.glsl_source_expression_type(arguments[0])
            )
            family = info["family"] if info is not None else None
            if re.fullmatch(r"\(?\d+u\)?", rendered_argument.strip()):
                # Integer literals above INT_MAX are emitted as uint literals.
                family = "uint"
            replacement = self.GLSL_FLOAT_BIT_CASTS.get((func_name, family))
            if replacement is None:
                return None
            return f"{replacement}({rendered_argument})"
        if func_name == "asdouble" and len(arguments) == 2:
            low, high = (
                (
                    self.glsl_converted_operand(argument, "uint")
                    if self.glsl_atomic_operand_family(argument) != "uint"
                    else self.generate_expression(argument)
                )
                for argument in arguments
            )
            return f"packDouble2x32(uvec2({low}, {high}))"
        if func_name == "dot" and len(arguments) == 2:
            integer_dot = self.generate_glsl_integer_dot(arguments)
            if integer_dot is not None:
                return integer_dot
        if func_name == "mix" and len(arguments) == 3:
            boolean_mix = self.generate_glsl_boolean_mix(arguments)
            if boolean_mix is not None:
                return boolean_mix
        if func_name == "atomicCompareExchange" and len(arguments) == 3:
            # Returns the original value, like GLSL atomicCompSwap.
            return self.generate_glsl_memory_atomic_call("atomicCompSwap", arguments)
        plan = self.glsl_unified_builtin_plan(func_name, arguments)
        if plan is None:
            return None
        _common, _width, operands = plan
        rendered = []
        for argument, (_info, target_type) in zip(arguments, operands):
            if target_type is None:
                rendered.append(self.generate_expression(argument))
            else:
                rendered.append(self.glsl_converted_operand(argument, target_type))
        return f"{func_name}({', '.join(rendered)})"

    def generate_glsl_boolean_mix(self, arguments):
        """``mix`` with a boolean selector, splatting a scalar selector to the value width."""
        infos = [
            self.glsl_value_type_info(self.glsl_source_expression_type(argument))
            for argument in arguments
        ]
        if None in infos or infos[2]["family"] != "bool":
            return None
        width = max(info["width"] for info in infos[:2])
        x, y, selector = (self.generate_expression(argument) for argument in arguments)
        if infos[2]["width"] == 1 and width > 1:
            selector = f"bvec{width}({selector})"
        return f"mix({x}, {y}, {selector})"

    def glsl_expression_is_unsigned(self, expression):
        info = self.glsl_value_type_info(self.glsl_source_expression_type(expression))
        return info is not None and info["family"] == "uint"

    def glsl_signed_coordinate_expression(self, expression, rendered=None):
        """Render an image or texel-fetch coordinate with the signed type GLSL takes.

        GLSL image and texel-fetch functions only accept ``int``/``ivecN``
        coordinates and levels, while HLSL indexes resources with unsigned
        values; unsigned operands are converted explicitly.
        """
        if rendered is None:
            rendered = self.generate_expression(expression)
        info = self.glsl_value_type_info(self.glsl_source_expression_type(expression))
        if info is None or info["family"] != "uint" or info["bits"] != 32:
            return rendered
        width = info["width"]
        target = "int" if width == 1 else f"ivec{width}"
        return f"{target}({rendered})"
